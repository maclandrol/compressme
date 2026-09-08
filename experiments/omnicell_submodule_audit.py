"""Reproduce pinned OmniCell embedder candidates; no whole-model support claim.

Requires the original trusted model.py, backbone.pth and LMConfig.json supplied
locally. Files are checked against the audited revisions before class execution.
"""
import argparse,ast,copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from compressme.finite_lookup import compile_finite_lookup
torch.set_num_threads(4)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint",type=Path,required=True)
parser.add_argument("--config",type=Path,required=True)
parser.add_argument("--source",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
source_path=args.source
for path,expected in [
    (args.checkpoint,"12497eb1dc76985ca3bcd88845a6b6edb7fcfdbb6c6df43d3da56645d922c0c7"),
    (args.config,"bb4fd00c29d8764058d986dbe3d2c2b036f1c3d0df91d661c63de71c2ea49322"),
    (source_path,"6277b4b2bc2ceda15e3db7a9ccf859bb2e488c4322d22d95f582645c0429552c")]:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            digest.update(block)
    if digest.hexdigest()!=expected:
        raise ValueError(f"Pinned audit file hash mismatch: {path}")
# Execute only the audited original definitions that do not import FlashAttention.
tree=ast.parse(source_path.read_text())
names={'LMConfig','MoE4Embedder','Embedder'}
classes=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names],type_ignores=[])
namespace={'torch':torch,'nn':nn,'F':F}
exec(compile(classes,str(source_path),'exec'),namespace)
config=namespace['LMConfig'](**json.loads((args.config).read_text()))
state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
assert isinstance(state,dict)
weights={k.removeprefix('embedder.'):v for k,v in state.items() if k.startswith('embedder.')}
assert weights, list(state)[:10]
embedder=namespace['Embedder'](config).eval()
embedder.load_state_dict(weights,strict=True)
embedder.requires_grad_(False)
E=embedder.gene_embedding.weight
report={
 'scope':'Actual checkpoint embedder only; full transformer attention and biological outputs unvalidated',
 'source_revision':'fc2d818d1d78345f0c7ccf686f34caff5bbd846a',
 'checkpoint':{'repository':'PJSucas/OmniCell-v1','revision':'9a33a49daa919237189909e4d9ba48117f3ca12f','sha256':'12497eb1dc76985ca3bcd88845a6b6edb7fcfdbb6c6df43d3da56645d922c0c7'},
 'config':json.loads((args.config).read_text()),
 'checkpoint_keys':len(state),
 'checkpoint_tensor_values':sum(t.numel() for t in state.values()),
 'embedding_shape':list(E.shape),'embedding_dtype':str(E.dtype),
 'embedding_zero_rows':int((E==0).all(-1).sum()),
 'embedder_values_before':sum(p.numel() for p in embedder.parameters()),
}
seq=nn.Sequential(copy.deepcopy(embedder.gene_embedding),copy.deepcopy(embedder.value_embedding.router)).eval()
started=time.perf_counter()
compiled=compile_finite_lookup(seq,input_contract='token_indices_only',chunk_size=1024,validation_chunk_size=257)
report['finite_compile_seconds']=time.perf_counter()-started
report['finite_report']=compiled.report
if compiled.report['status']!='accepted_on_full_domain_validation':
 args.output.write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2));raise SystemExit(0)
table=compiled.model.weight.detach().numpy()
byte_rows=np.ascontiguousarray(table).view(np.uint8).reshape(len(table),-1)
report['unique_logit_rows_bitwise']=len(np.unique(byte_rows,axis=0))
report['router_values_before']=sum(p.numel() for p in embedder.value_embedding.router.parameters())
report['table_values_after']=table.size
report['whole_embedder_values_after_finite']=report['embedder_values_before']-report['router_values_before']+table.size

class RoutedLookupEmbedder(nn.Module):
 def __init__(self,original,lookup):
  super().__init__()
  self.gene_embedding=copy.deepcopy(original.gene_embedding)
  self.value_embedding=copy.deepcopy(original.value_embedding)
  self.value_embedding.router=nn.Identity()
  self.lookup=copy.deepcopy(lookup)
  self.load_balance_loss=0.0
  self.eval()
 def forward(self,gene,value):
  gene_embedded=self.gene_embedding(gene)
  value=value.to(gene_embedded.dtype)
  m=self.value_embedding
  shared_input=value.unsqueeze(-1)
  shared_output=sum(expert(shared_input) for expert in m.shared_experts)
  routing_logits=self.lookup(gene)
  routing_weights=F.softmax(routing_logits,dim=-1)
  topk_weights,topk_idx=torch.topk(routing_weights,m.topk,dim=-1)
  sparse_weights=torch.zeros_like(routing_weights).scatter(-1,topk_idx,topk_weights)
  expert_outputs=torch.stack([expert(shared_input) for expert in m.routing_experts],dim=2)
  routing_output=(expert_outputs*sparse_weights.unsqueeze(-1)).sum(dim=2)
  m.load_balance_loss=m._calc_balance_loss(routing_weights,sparse_weights)
  self.load_balance_loss=m.load_balance_loss
  return gene_embedded+(shared_output+routing_output)

candidate=RoutedLookupEmbedder(embedder,compiled.model)
assert sum(p.numel() for p in candidate.parameters())==report['whole_embedder_values_after_finite']
def metrics(a,b):
 diff=(a.double()-b.double()).abs()
 return {'max_abs':float(diff.max()) if diff.numel() else 0.,'mismatches':int((diff>1e-5+1e-5*a.double().abs()).sum()),'elements':a.numel(),'bitwise':torch.equal(a,b)}
records=[]
torch.manual_seed(182)
with torch.inference_mode():
 for shape in [(1,1),(1,32),(4,32),(1,2000),(2,2000)]:
  genes=torch.randint(0,config.num_gene,shape)
  values=torch.rand(shape)*10
  expected=embedder(genes,values)
  balance=embedder.load_balance_loss.clone()
  actual=candidate(genes,values)
  records.append({'shape':list(shape),'values_range':[0,10],'embedding':metrics(expected,actual),'balance':metrics(balance,candidate.load_balance_loss)})
report['full_embedder_gates']=records
report['full_embedder_all_gates_accepted']=all(r[k]['mismatches']==0 for r in records for k in ['embedding','balance'])
# Check every token's routing probabilities and exact chosen expert indices.
topk_mismatches=0;prob_max=0.;topk_gap=math.inf
with torch.inference_mode():
 for first in range(0,config.num_gene,257):
  genes=torch.arange(first,min(config.num_gene,first+257)).reshape(1,-1)
  left=F.softmax(embedder.value_embedding.router(embedder.gene_embedding(genes)),dim=-1)
  right=F.softmax(compiled.model(genes),dim=-1)
  prob_max=max(prob_max,float((left-right).abs().max()))
  topk_mismatches+=int((left.topk(config.topk,dim=-1).indices!=right.topk(config.topk,dim=-1).indices).sum())
  sorted_=left.sort(dim=-1,descending=True).values
  topk_gap=min(topk_gap,float((sorted_[...,config.topk-1]-sorted_[...,config.topk]).min()))
report['all_token_routing']={'probability_max_abs':prob_max,'ordered_topk_index_mismatches':topk_mismatches,'minimum_topk_boundary_gap':topk_gap}
boundary_records=[]
with torch.inference_mode():
 probabilities=F.softmax(compiled.model.weight,dim=-1)
 sorted_probs=probabilities.sort(dim=-1,descending=True).values
 gaps=sorted_probs[:,config.topk-1]-sorted_probs[:,config.topk]
 worst_ids=gaps.argsort()[:128]
 for identifier in worst_ids:
  genes=identifier.reshape(1,1)
  source_prob=F.softmax(embedder.value_embedding.router(embedder.gene_embedding(genes)),dim=-1)
  target_prob=F.softmax(compiled.model(genes),dim=-1)
  selected_difference=int((source_prob.topk(config.topk,dim=-1).indices.sort(-1).values != target_prob.topk(config.topk,dim=-1).indices.sort(-1).values).sum())
  expected=embedder(genes,torch.full((1,1),10.0));balance=embedder.load_balance_loss.clone()
  actual=candidate(genes,torch.full((1,1),10.0))
  boundary_records.append({'gene_id':int(identifier),'compiled_boundary_gap':float(gaps[identifier]),'selected_expert_difference':selected_difference,'embedding':metrics(expected,actual),'balance':metrics(balance,candidate.load_balance_loss)})
report['targeted_singleton_small_margin']=boundary_records
report['targeted_singleton_passed']=all(r[k]['mismatches']==0 for r in boundary_records for k in ['embedding','balance'])
# Five shared scalar Linear maps compose without retaining a finite gene table.
folded=nn.Linear(1,config.d_model,bias=False)
with torch.no_grad():folded.weight.copy_(sum(m.weight.double() for m in embedder.value_embedding.shared_experts).float())
shared_records=[]
with torch.inference_mode():
 for bound in [1.,10.,100.,1000000.]:
  values=torch.linspace(-bound,bound,4097).reshape(-1,1)
  expected=sum(expert(values) for expert in embedder.value_embedding.shared_experts)
  shared_records.append({'absolute_input_bound':bound,**metrics(expected,folded(values))})
report['shared_affine_sum']={'values_before':config.num_shared*config.d_model,'values_after':config.d_model,'validation':shared_records,'scope':'Real-arithmetic affine identity; changed FP32 sum order; error scales with continuous expression magnitude'}
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['embedding_shape','embedding_zero_rows','embedder_values_before','router_values_before','table_values_after','whole_embedder_values_after_finite','unique_logit_rows_bitwise','full_embedder_all_gates_accepted','all_token_routing','targeted_singleton_passed','shared_affine_sum']},indent=2))
print('targeted singleton failing examples',json.dumps([r for r in boundary_records if r['embedding']['mismatches'] or r['selected_expert_difference']][:5],indent=2))
