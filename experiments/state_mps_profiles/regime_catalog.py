"""Map real State encoder finite-row byte regimes, including mixed positions."""
import argparse,collections,hashlib,json,platform
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from local_profile_probe import load_original,difference,byte_equal,file_hash

parser=argparse.ArgumentParser()
parser.add_argument('--weights',type=Path,required=True)
parser.add_argument('--architecture',type=Path,required=True)
parser.add_argument('--loader',type=Path,required=True)
parser.add_argument('--device',default='mps')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--sample-tables',type=Path)
args=parser.parse_args()
torch.set_num_threads(4)
model=load_original(args.weights,args.architecture,args.loader).to(args.device).eval()
rng=torch.Generator().manual_seed(9428)
selected=torch.unique(torch.cat((torch.tensor([0,1,2,3,4,13,9895,19789]),torch.randint(0,19790,(8,),generator=rng))))
counts=list(range(1,129))+[257,1024]
templates={};records=[];failures=[]
def digest(x):return hashlib.sha256(x.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
with torch.inference_mode():
    for branch in ('raw','normalized'):
        for rows in counts:
            by_stage=collections.defaultdict(list)
            for token in selected.tolist():
                ids=torch.full((1,rows),token,dtype=torch.long,device=args.device)
                x=model.pe_embedding(ids)
                if branch=='normalized':x=F.normalize(x,dim=-1)
                y0=model.encoder[0](x)
                y=model.encoder[2](model.encoder[1](y0))
                for stage,value in [('input',x),('linear',y0),('final',y)]:
                    by_stage[stage].append(value[:,0,:].cpu())
                if not byte_equal(y,y[:,:1,:].expand_as(y)):
                    failures.append({'type':'repeated_positions','branch':branch,'rows':rows,'token':token,
                                     'difference':difference(y,y[:,:1,:].expand_as(y))})
            tables={stage:torch.cat(values) for stage,values in by_stage.items()}
            templates[(branch,rows)]=tables
            records.append({'branch':branch,'rows':rows,'signatures':{k:digest(v) for k,v in tables.items()}})
            # Cover every sampled gene in mixed chunks and reverse its positions.
            for offset in (0,7):
                for start in range(0,len(selected),rows):
                    local=((torch.arange(rows)+start+offset)%len(selected)).reshape(1,rows)
                    ids=selected[local].to(args.device)
                    x=model.pe_embedding(ids)
                    if branch=='normalized':x=F.normalize(x,dim=-1)
                    y0=model.encoder[0](x);y=model.encoder[2](model.encoder[1](y0))
                    expected=F.embedding(local,tables['final'])
                    if not byte_equal(y,expected):
                        failures.append({'type':'mixed_positions','branch':branch,'rows':rows,
                                         'start':start,'offset':offset,'difference':difference(y,expected)})
            if rows%16==0 or rows>128:print(branch,rows,'failures',len(failures),flush=True)
groups={}
for branch in ('raw','normalized'):
    groups[branch]={}
    for stage in ('input','linear','final'):
        grouped=collections.defaultdict(list)
        for row in records:
            if row['branch']==branch:grouped[row['signatures'][stage]].append(row['rows'])
        groups[branch][stage]=[{'signature':sig,'rows':rs,'representative':rs[0]} for sig,rs in grouped.items()]
if args.sample_tables:
    tensors={}
    for branch in ('raw','normalized'):
        for group in groups[branch]['final']:
            rows=group['representative'];tensors[f'{branch}_{rows}']=templates[(branch,rows)]['final'].clone()
    tensors['selected_gene_ids']=selected.clone()
    save_file(tensors,str(args.sample_tables))
result={'scope':'Selected real gene-row encoder diagnostics; no full-vocabulary or whole-model acceptance claim',
    'device':args.device,'python':platform.python_version(),'torch':torch.__version__,
    'weights_sha256':file_hash(args.weights),'selected_genes':selected.tolist(),
    'tested_encoder_row_counts':counts,'groups':groups,'failures':failures,'records':records}
args.output.write_text(json.dumps(result,indent=2))
print(json.dumps({'groups':groups,'failures':len(failures)},indent=2),flush=True)
