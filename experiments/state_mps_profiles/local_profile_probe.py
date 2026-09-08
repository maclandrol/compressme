"""Real State SE encoder shape diagnostics only; no production edits/timing."""
import argparse,hashlib,importlib.util,json,platform,sys
from pathlib import Path
import torch
from torch.nn import functional as F

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def byte_equal(a,b):
    return (a.shape==b.shape and a.dtype==b.dtype and
        torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                    b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))

def difference(a,b):
    a,b=a.detach().cpu(),b.detach().cpu()
    delta=a.double()-b.double()
    return {'shape':list(a.shape),'max_abs':float(delta.abs().max()) if delta.numel() else 0,
            'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a.double()).clamp_min(1e-30)),
            'numeric_equal':torch.equal(a,b),'byte_equal':byte_equal(a,b),
            'elementwise_atol_rtol_1e5':bool(torch.isclose(a,b,atol=1e-5,rtol=1e-5).all())}

def load_original(weights,architecture,loader):
    manifest=json.loads((architecture/'SOURCE.json').read_text())
    assert file_hash(weights)==manifest['checkpoint_sha256']
    for path,sha in manifest['sha256'].items():assert file_hash(architecture/path)==sha,path
    spec=importlib.util.spec_from_file_location('state_profile_original',loader)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.load_state_se(weights,config=architecture/'config.json',source_root=architecture).requires_grad_(False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--architecture',type=Path,required=True)
    parser.add_argument('--loader',type=Path,required=True)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    model=load_original(args.weights,args.architecture,args.loader).to(args.device).eval()
    generator=torch.Generator().manual_seed(9428)
    selected=torch.unique(torch.cat((torch.tensor([0,1,2,3,4,13,9895,19789]),
                     torch.randint(0,19790,(8,),generator=generator))))
    profile_rows=(1,2,15,16,1024)
    tables={};repeat_records=[]
    with torch.inference_mode():
        for normalized in (False,True):
            branch='normalized' if normalized else 'raw'
            for rows in profile_rows:
                results=[];inputs=[];linear=[]
                for token in selected.tolist():
                    ids=torch.full((1,rows),token,dtype=torch.long,device=args.device)
                    x=model.pe_embedding(ids)
                    if normalized:x=F.normalize(x,dim=-1)
                    y0=model.encoder[0](x);y=model.encoder[2](model.encoder[1](y0))
                    results.append(y[:,0,:].cpu())
                    inputs.append(x[:,0,:].cpu())
                    linear.append(y0[:,0,:].cpu())
                    if not byte_equal(y,y[:,:1,:].expand_as(y)):
                        repeat_records.append({'branch':branch,'rows':rows,'token':token,
                                               'repeated_position_difference':difference(y,y[:,:1,:].expand_as(y))})
                tables[(branch,rows)]={'final':torch.cat(results),'input':torch.cat(inputs),'linear':torch.cat(linear)}
                print('compiled sampled profile',branch,rows,'position_failures',len(repeat_records),flush=True)
        profile_comparisons=[]
        for branch in ('raw','normalized'):
            for rows in profile_rows:
                profile_comparisons.append({'branch':branch,'rows':rows,
                    **{stage:difference(tables[(branch,1024)][stage],tables[(branch,rows)][stage])
                       for stage in ('input','linear','final')}})
        shapes=[(1,n) for n in (1,2,3,4,5,7,8,11,12,13,14,15,16,17,32,128,2048)]
        shapes.extend([(2,1),(2,6),(2,7),(2,8),(2,128),(2,512),(4,3),(4,4),(4,32)])
        grid=[]
        for B,T in shapes:
            for strided in (False,True):
                local=torch.randint(0,len(selected),(B,T*(2 if strided else 1)),generator=generator)
                ids=selected[local].to(args.device)
                if strided:ids=ids[:,::2];local=local[:,::2]
                raw=model.pe_embedding(ids)
                raw_result=model.encoder(raw)
                norm=F.normalize(raw,dim=-1)
                norm[:,0,:]=model.cls_token.expand(B,-1)
                if model.dataset_token is not None:
                    norm=torch.cat((norm,model.dataset_token.expand(B,-1).unsqueeze(1)),dim=1)
                normalized_result=model.encoder(norm)[:,1:T,:]
                for branch,actual,indices in [('raw',raw_result,local),('normalized',normalized_result,local[:,1:T])]:
                    rec={'branch':branch,'B':B,'T':T,'ids_stride':list(ids.stride()),'strided':strided,
                         'normalization_rows':B*T if branch=='normalized' else None,
                         'encoder_rows':B*(T+int(model.dataset_token is not None)) if branch=='normalized' else B*T,
                         'comparison_profiles':{str(n):difference(actual,F.embedding(indices,tables[(branch,n)]['final'])) for n in profile_rows}}
                    rec['byte_matching_profiles']=[n for n,m in rec['comparison_profiles'].items() if m['byte_equal']]
                    grid.append(rec)
                print('shape',B,T,'strided',strided,'raw',grid[-2]['byte_matching_profiles'],
                      'normalized',grid[-1]['byte_matching_profiles'],flush=True)
    result={'scope':'Selected real gene rows; original encoder only; no whole-model compression claim',
        'device':args.device,'python':platform.python_version(),'torch':torch.__version__,
        'weights_sha256':file_hash(args.weights),'config_sha256':file_hash(args.architecture/'config.json'),
        'selected_genes':selected.tolist(),'profile_rows':profile_rows,'parameters_original':sum(p.numel() for p in model.parameters()),
        'encoder_parameters':sum(p.numel() for p in model.encoder.parameters()),
        'repeated_position_failures':repeat_records,'profile_comparisons':profile_comparisons,'shape_grid':grid}
    args.output.write_text(json.dumps(result,indent=2))
    print('wrote',args.output,flush=True)

if __name__=='__main__':main()
