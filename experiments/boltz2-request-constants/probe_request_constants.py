"""Local actual-weight gates for exact request-constant Boltz conditioning."""
import argparse,contextlib,json,sys
from functools import partial
from pathlib import Path
import torch
from safetensors.torch import load_file
from request_constants import static_atom_conditioning_scope,PreparedSingleConditioningPrefix

def compare(a,b):
    if isinstance(a,tuple):return [compare(x,y) if x is not None else {'both_none':y is None} for x,y in zip(a,b)]
    x=a.detach().cpu().contiguous();y=b.detach().cpu().contiguous();diff=(x-y).abs()
    return {'shape':list(x.shape),'bitwise_equal':torch.equal(x.view(torch.uint8),y.view(torch.uint8)),'mixed_gate':bool(torch.allclose(x,y,atol=1e-5,rtol=1e-5)),'max_abs':float(diff.max()) if diff.numel() else 0.,'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shared-state',type=Path,required=True);ap.add_argument('--safe-loader-dir',type=Path,required=True);ap.add_argument('--fixture',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',choices=['cpu','mps'],default='cpu');args=ap.parse_args()
    torch.set_grad_enabled(False);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');sys.path.insert(0,str(args.safe_loader_dir))
    from safe_state import load_tensor_state
    from boltz.model.modules.transformersv2 import AtomTransformer
    from boltz.model.modules.encodersv2 import SingleConditioning,get_indexing_matrix,single_to_keys
    state,h=load_tensor_state(args.shared_state,'boltz2_conf.ckpt');fixture={k:v.to(args.device) for k,v in load_file(str(args.fixture)).items()};device=torch.device(args.device);checks=[];budgets=[]
    W,H=32,128;N=fixture['c'].shape[1];K=N//W
    indexing=get_indexing_matrix(K,W,H,device);to_keys=partial(single_to_keys,indexing_matrix=indexing,W=W,H=H)
    for family,prefix in [('encoder','structure_module.score_model.atom_attention_encoder.atom_encoder.'),('decoder','structure_module.score_model.atom_attention_decoder.atom_decoder.')]:
        model=AtomTransformer(attn_window_queries=W,attn_window_keys=H,dim=128,dim_single_cond=128,depth=3,heads=4,activation_checkpointing=True,post_layer_norm=False).eval().requires_grad_(False)
        model.load_state_dict({k.removeprefix(prefix):v for k,v in state.items() if k.startswith(prefix)},strict=True);model.to(device)
        ids={n:id(p) for n,p in model.named_parameters(remove_duplicate=False)}
        for mult in (1,3):
            c=fixture['c'].repeat_interleave(mult,0);q=fixture['q'].repeat_interleave(mult,0);mask=fixture['atom_mask'].repeat_interleave(mult,0);bias=fixture['atom_enc_bias' if family=='encoder' else 'atom_dec_bias']
            gen=torch.Generator().manual_seed(94);noise=torch.randn(q.shape,generator=gen).to(device);queries=[q+scale*noise for scale in (0.,.25,2.5)]
            def run(value):return model(q=value,c=c,bias=bias,to_keys=to_keys,mask=mask,multiplicity=mult)
            baseline=[run(value) for value in queries]
            with static_atom_conditioning_scope(model,c) as count:
                budgets.append({'family':family,'multiplicity':mult,**count})
                for i,value in enumerate(queries):checks.append({'component':'full_atom_transformer','family':family,'multiplicity':mult,'dynamic_query_case':i,**compare(baseline[i],run(value))})
            assert ids=={n:id(p) for n,p in model.named_parameters(remove_duplicate=False)}
            checks.append({'component':'restored_atom_transformer','family':family,'multiplicity':mult,**compare(baseline[0],run(queries[0]))})
        del model
    score=h['score_model_args'];single=SingleConditioning(sigma_data=score['sigma_data'],token_s=h['token_s'],dim_fourier=score['dim_fourier'],num_transitions=score['conditioning_transition_layers']).eval().requires_grad_(False)
    prefix='structure_module.score_model.single_conditioner.';single.load_state_dict({k.removeprefix(prefix):v for k,v in state.items() if k.startswith(prefix)},strict=True);single.to(device);del state
    for mult in (1,3):
        strunk=fixture['s_trunk'].repeat_interleave(mult,0);sinputs=fixture['s_inputs'].repeat_interleave(mult,0);prepared=PreparedSingleConditioningPrefix(single,strunk,sinputs)
        for t in (-5.,-1.,0.,1.,2.,5.):
            times=torch.full((strunk.shape[0],),t,device=device)
            comparisons=compare(single(times,strunk,sinputs),prepared(times,strunk,sinputs))
            checks.append({'component':'full_single_conditioner','multiplicity':mult,'time':t,'outputs':comparisons})
        budgets.append({'family':'single_conditioner_prefix','multiplicity':mult,'prepared_values':prepared.prefix.numel(),'prepared_bytes':prepared.prefix.numel()*4,'condition_linear_calls_per_original_forward':1,'condition_layernorm_calls_per_original_forward':1})
    def passed(r):
        if 'outputs' in r:return all(x.get('both_none',x.get('bitwise_equal',False)) for x in r['outputs'])
        return r['bitwise_equal'] and r['finite']
    report={'scope':'Complete original atom-transformer submodules and complete SingleConditioning on actual checkpoint weights; request-constant partial evaluation only. No full Boltz prediction comparison here.','device':args.device,'torch':torch.__version__,'all_bitwise_equal':all(passed(r) for r in checks),'checks':checks,'operation_and_memory_counts':budgets,'gates':{'atol':1e-5,'rtol':1e-5}}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'all_bitwise_equal':report['all_bitwise_equal'],'checks':len(checks),'budgets':budgets},indent=2))
if __name__=='__main__':main()
