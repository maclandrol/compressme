import sys,json,time,argparse,torch
from pathlib import Path
sys.path.insert(0,str(Path('examples/nesso').resolve()))
from prepare import load_batch
from nesso.model.models.nesso1 import Nesso1
from compressme.validation import compare_outputs,seeded
import nesso.model.layers.triangular_attention.primitives as primitives
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--case',default='tiny20');a=p.parse_args()
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
m=Nesso1.from_pretrained('work/nesso/upstream/v1.0.0').eval().requires_grad_(False).to(a.device);m.use_kernels=False
m.predict_args=dict(m.predict_args or {},recycling_steps=5,refine_protein_inference=True,refine_protein_cutoff=22.,refine_protein_tokens_budget=256,affinity_protein_cutoff=15.,save_metadata=True)
batch=load_batch(Path('work/nesso/fixtures')/a.case,device=a.device)
def sync():
 if a.device=='mps':torch.mps.synchronize()
def run():
 with seeded(42),torch.no_grad():
  sync();t=time.perf_counter();f=m(batch,recycling_steps=5,refine_protein_inference=True);sync();dt=time.perf_counter()-t
  f={k:v.cpu().clone() for k,v in f.items()}
 with seeded(42),torch.no_grad():
  pred=m.predict_step(batch,0);sync();pred={k:v.cpu().clone() if isinstance(v,torch.Tensor) else v for k,v in pred.items()}
 return dt,f,pred
def sdpa(query,key,value,biases):
 bias=biases[0]
 for x in biases[1:]:bias=bias+x
 return torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=bias,dropout_p=0.0,scale=1.0)
before=run();original=primitives._attention
try:
 primitives._attention=sdpa;after=run()
finally:primitives._attention=original
report={'device':a.device,'case':a.case,'method':'SDPA mathematical equivalent; floating-point order changes', 'reference_diagnostic_seconds':before[0], 'candidate_diagnostic_seconds':after[0], 'forward':compare_outputs(before[1],after[1]),'predict':compare_outputs(before[2],after[2])}
report['bitwise_identical']=all(v['bitwise'] for key in ['forward','predict'] for v in report[key].values())
dest=Path(f'work/nesso/sdpa_probe_{a.case}_{a.device}.json');dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'saved':str(dest),'bitwise_identical':report['bitwise_identical'],'forward_max_abs':max(v['max_abs'] for v in report['forward'].values()),'forward_max_relative_l2':max(v['relative_l2'] for v in report['forward'].values()),'affinity_metrics':{k:v for k,v in report['predict'].items() if 'affinity' in k}},indent=2))
