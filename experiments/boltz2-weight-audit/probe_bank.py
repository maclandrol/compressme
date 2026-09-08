"""Actual pretrained conditioning banks, local CPU/MPS numerical audit only."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

def metric(a,b):
    diff=(a-b).abs();tol=1e-5+1e-5*a.abs()
    return {'max_abs':float(diff.max()) if diff.numel() else 0.,'relative_l2':float(torch.linalg.vector_norm(diff)/(torch.linalg.vector_norm(a)+1e-30)), 'failed_elements':int((diff>tol).sum()),'numel':a.numel(),'bitwise_equal':torch.equal(a,b),'finite':bool(torch.isfinite(b).all())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=Path(__file__).parent);ap.add_argument('--device',choices=['cpu','mps'],default='cpu');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError('Refusing to overwrite an existing probe report')
    root=args.directory;torch.set_num_threads(2);records=[];summary=[]
    for checkpoint in ('boltz2_conf.ckpt','boltz2_aff.ckpt'):
        folder=root/(checkpoint+'.selected');manifest=json.loads((folder/'manifest.json').read_text());state={}
        for r in manifest['storages']:
            raw=(folder/r['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Selected tensor hash mismatch')
            if r['offset']!=0 or r['numel']!=r['storage_numel']:raise ValueError('Only contiguous complete storage records supported')
            state[r['name']]=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).reshape(r['shape'])
        embedding={}
        for key,x in state.items():
            if key.startswith('input_embedder.'):
                embedding[key]={'numel':x.numel(),'zeros':int((x==0).sum()),'constant_rows':bool(torch.equal(x,x[0].expand_as(x))),'max_abs':float(x.abs().max())}
        for bank,branches,dim,out in [('token_trans_proj_z',24,128,16),('atom_enc_proj_z',3,16,4),('atom_dec_proj_z',3,16,4)]:
            gamma=torch.stack([state[f'diffusion_conditioning.{bank}.{i}.0.weight'] for i in range(branches)])
            beta=torch.stack([state[f'diffusion_conditioning.{bank}.{i}.0.bias'] for i in range(branches)])
            w=torch.stack([state[f'diffusion_conditioning.{bank}.{i}.1.weight'] for i in range(branches)])
            identical_norms=bool(torch.equal(gamma,gamma[0].expand_as(gamma)) and torch.equal(beta,beta[0].expand_as(beta)))
            original=gamma.numel()+beta.numel()+w.numel();fused=w.numel()+branches*out
            summary.append({'checkpoint':checkpoint,'bank':bank,'branches':branches,'dim':dim,'out_per_branch':out,'source_values':original,'folded_values':fused,'saved_values':original-fused,'identical_norm_parameters':identical_norms,'embedding_audit':embedding})
            for dtype in (torch.float64,torch.float32):
                if args.device=='mps' and dtype==torch.float64:continue
                g=gamma.to(dtype).to(args.device);b=beta.to(dtype).to(args.device);weight=w.to(dtype).to(args.device)
                # Prepare coefficients accurately on CPU, then retain the runtime dtype.
                fw=(w.double()*gamma.double().unsqueeze(1)).reshape(branches*out,dim).to(dtype).to(args.device)
                fb=torch.einsum('hod,hd->ho',w.double(),beta.double()).reshape(branches*out).to(dtype).to(args.device)
                gen=torch.Generator().manual_seed(912)
                base=torch.randn((2,17,17,dim),generator=gen,dtype=dtype).to(args.device)
                cases={'normal':base,'large_scale':base*1000,'large_offset':base*.01+1000,'near_constant':base*1e-7+1.,'constant':torch.full_like(base,3.),'empty':base[:0],'singleton':base[:1,:1,:1]}
                for label,x in cases.items():
                    source=torch.cat([F.linear(F.layer_norm(x,(dim,),g[i],b[i],1e-5),weight[i]) for i in range(branches)],-1)
                    norm=F.layer_norm(x,(dim,),None,None,1e-5)
                    folded=F.linear(norm,fw,fb)
                    shared_stats=torch.cat([F.linear(norm*g[i]+b[i],weight[i]) for i in range(branches)],-1)
                    records.append({'checkpoint':checkpoint,'bank':bank,'dtype':str(dtype),'device':args.device,'case':label,'shape':list(x.shape),'folded':metric(source,folded),'shared_statistics_only':metric(source,shared_stats)})
    result={'scope':'actual pretrained component weights on synthetic pair features, no full structure/confidence/affinity outputs validated','tolerance':{'atol':1e-5,'rtol':1e-5},'torch':torch.__version__,'banks':summary,'checks':records}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'banks':summary,'checks':len(records),'folded_failed_cases':[dict(checkpoint=r['checkpoint'],bank=r['bank'],dtype=r['dtype'],case=r['case'],**r['folded']) for r in records if r['folded']['failed_elements']],'statistics_failed_cases':sum(bool(r['shared_statistics_only']['failed_elements']) for r in records)},indent=2))
if __name__=='__main__':
    with torch.inference_mode():main()
