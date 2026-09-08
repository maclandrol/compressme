"""Small actual checkpoint alias test plus PyTorch strict-load precedence."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class Gate(nn.Module):
    def __init__(self,dim):
        super().__init__();self.output_projection_linear=nn.Linear(dim,dim);self.output_projection=nn.Sequential(self.output_projection_linear,nn.Sigmoid())
    def forward(self,x):return self.output_projection(x)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=Path(__file__).parent);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.directory
    if args.output.exists():raise ValueError('Refusing to overwrite an existing probe report')
    base='input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers.0.'
    records=json.loads((root/'boltz2_conf.ckpt.tensor-metadata.json').read_text());state={};hashes={}
    selected=json.loads((root/'alias_selected/manifest.json').read_text())
    for r in selected['storages']:
        raw=(root/'alias_selected'/r['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Selected tensor hash mismatch')
        name=r['local_name'];state[name]=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).reshape(r['shape']);hashes[name]=r['sha256']
    result={'actual_first_gate_checkpoint_duplicates_equal':all(torch.equal(state['output_projection_linear.'+suffix],state['output_projection.0.'+suffix]) for suffix in ('weight','bias')),'tensor_sha256':hashes}
    model=Gate(128).eval();model.load_state_dict(state,strict=True)
    result['source_registration_is_same_module']=model.output_projection_linear is model.output_projection[0]
    result['loaded_equals_last_child_alias']=all(torch.equal(model.state_dict()['output_projection_linear.'+suffix],state['output_projection.0.'+suffix]) for suffix in ('weight','bias'))
    x=torch.randn(4,128,generator=torch.Generator().manual_seed(64));expected=torch.sigmoid(torch.nn.functional.linear(x,state['output_projection.0.weight'],state['output_projection.0.bias']))
    result['actual_first_gate_forward_bitwise_equal']=torch.equal(model(x),expected)
    altered={k:v.clone() for k,v in state.items()};altered['output_projection_linear.weight'].fill_(99);altered['output_projection_linear.bias'].fill_(99)
    model.load_state_dict(altered,strict=True);result['strict_load_child_alias_wins_if_values_differ']=torch.equal(model(x),expected)
    pairs=[];byname={r['name']:r for r in records}
    for name,r in byname.items():
        if '.output_projection_linear.' in name:
            target=name.replace('.output_projection_linear.','.output_projection.0.')
            if target in byname:pairs.append({'first':name,'last':target,'numel':r['numel'],'same_storage':r['storage_key']==byname[target]['storage_key']})
    result['architecture_alias_pairs']=pairs;result['potential_inference_duplicate_values']=sum(r['numel'] for r in pairs);result['limitation']='This probe executes only one actual gate and strict-load precedence. The complete architecture and native output checks are outside this component probe scope.'
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('architecture_alias_pairs','tensor_sha256')},indent=2))
if __name__=='__main__':
    with torch.inference_mode():main()
