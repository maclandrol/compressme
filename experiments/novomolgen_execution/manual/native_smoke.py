"""Original pinned NovoMolGen native-Llama API checks; no compression or timing.

Run with the isolated Transformers4.46.2/tokenizers0.20.3 overlay on PYTHONPATH.
All model code comes from the installed Transformers wheel, not a Hub repo.
"""
import argparse,hashlib,inspect,json,os,platform,traceback,warnings
from pathlib import Path
os.environ.setdefault('HF_HOME','/private/tmp/compressme-agent-novomolgen-runtime/hf-cache')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
import transformers
import tokenizers
from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast,GenerationConfig
from transformers.cache_utils import DynamicCache,StaticCache
from safetensors.torch import load_file

WEIGHTS=Path('/private/tmp/compressme-models/novomolgen32-atomwise/model.safetensors')
NATIVE=Path('/Users/manu/Code/compressme/vendor/novomolgen_audit/sources/huggingface/NovoMolGen_32M_SMILES_AtomWise/hf-checkpoint')
SHA256='3cf10749dce97e74edbb0c4d4e609265014e90f7be2fa50d731af67977cd0699'

def load_native(device='cpu',attn_implementation='eager'):
    digest=hashlib.file_digest(WEIGHTS.open('rb'),'sha256').hexdigest()
    assert digest==SHA256,(digest,SHA256)
    config=LlamaConfig.from_json_file(str(NATIVE/'config.json'))
    config._attn_implementation=attn_implementation
    with torch.random.fork_rng(devices=[]):model=LlamaForCausalLM(config)
    model.load_state_dict(load_file(str(WEIGHTS),device='cpu'),strict=True)
    model.generation_config=GenerationConfig.from_pretrained(NATIVE,local_files_only=True)
    model=model.eval().to(device)
    tokenizer=PreTrainedTokenizerFast.from_pretrained(NATIVE,local_files_only=True)
    tokenizer.padding_side='left'
    return model,tokenizer

def stats(x):
    return {'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),
            'finite':bool(torch.isfinite(x).all()) if x.is_floating_point() else True}

def close(a,b):
    delta=(a.detach().cpu().double()-b.detach().cpu().double())
    return {'max_abs':float(delta.abs().max()) if delta.numel() else 0.0,
            'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a.detach().cpu().double()).clamp_min(1e-12)),
            'bitwise':torch.equal(a,b),
            'allclose_1e5':torch.allclose(a,b,atol=1e-5,rtol=1e-5)}

def cache_tensors(cache):
    if hasattr(cache,'to_legacy_cache'):return cache.to_legacy_cache()
    if hasattr(cache,'key_cache'):return tuple(zip(cache.key_cache,cache.value_cache))
    return cache

def cache_info(cache):
    tensors=cache_tensors(cache)
    return {'type':type(cache).__name__,'layers':len(tensors),
            'first_key':stats(tensors[0][0]),'first_value':stats(tensors[0][1]),
            'all_finite':all(bool(torch.isfinite(t).all()) for layer in tensors for t in layer)}

def native_smoke(device,attn_implementation):
    torch.set_num_threads(4)
    if device=='mps':assert torch.backends.mps.is_available(),'Run outside sandbox for MPS access'
    model,tokenizer=load_native(device,attn_implementation)
    report={'scope':'Original native Llama checkpoint API smoke only; no compression, timing or molecular-quality claim',
            'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
            'tokenizers':tokenizers.__version__,'device':device,'dtype':'float32',
            'attention_implementation':model.config._attn_implementation,
            'model_class':type(model).__name__,'class_source':inspect.getfile(type(model)),
            'forward_signature':str(inspect.signature(model.forward)),
            'checkpoint_sha256':SHA256,'unique_parameters':sum(p.numel() for p in model.parameters()),
            'stages':{},'failures':[]}
    def stage(name,fn):
        try:
            report['stages'][name]=fn()
            print(name,'passed',flush=True)
        except Exception as exc:
            report['failures'].append({'stage':name,'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
            print(name,'FAILED',type(exc).__name__,str(exc),flush=True)
    encoded=tokenizer([tokenizer.bos_token+'CCO',tokenizer.bos_token+'c1ccccc1'],
        add_special_tokens=False,padding=True,return_token_type_ids=False,return_tensors='pt').to(device)
    ids,mask=encoded['input_ids'],encoded['attention_mask']
    positions=(mask.cumsum(-1)-1).clamp_min(0)
    labels=ids.masked_fill(mask==0,-100)
    report['prompts']={'strings':['<bos>CCO','<bos>c1ccccc1'],'input_ids':ids.cpu().tolist(),
                       'attention_mask':mask.cpu().tolist(),'contains_unk':bool((ids==tokenizer.unk_token_id).any())}
    def all_outputs():
        out=model(input_ids=ids,attention_mask=mask,position_ids=positions,labels=labels,
            use_cache=True,output_hidden_states=True,output_attentions=True,return_dict=True)
        assert torch.isfinite(out.loss) and torch.isfinite(out.logits).all()
        assert len(out.hidden_states)==model.config.num_hidden_layers+1
        assert len(out.attentions)==model.config.num_hidden_layers
        assert all(a is not None and torch.isfinite(a).all() for a in out.attentions)
        return {'keys':list(out.keys()),'loss':float(out.loss),'logits':stats(out.logits),
                'hidden_states':[stats(h) for h in out.hidden_states],
                'attentions':[stats(a) for a in out.attentions],
                'cache':cache_info(out.past_key_values)}
    def tuple_outputs():
        args=dict(input_ids=ids,attention_mask=mask,position_ids=positions,labels=labels,
            use_cache=True,output_hidden_states=True,output_attentions=True)
        reference=model(**args,return_dict=True)
        output=model(**args,return_dict=False)
        assert len(output)==5
        assert torch.equal(reference.loss,output[0]) and torch.equal(reference.logits,output[1])
        return {'tuple_length':len(output),'logits_match':True,'cache':cache_info(output[2])}
    def embedded_inputs():
        embeddings=model.get_input_embeddings()(ids)
        reference=model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,output_hidden_states=True)
        output=model(inputs_embeds=embeddings,attention_mask=mask,position_ids=positions,use_cache=False,output_hidden_states=True)
        assert torch.equal(reference.logits,output.logits)
        perturbed=model(inputs_embeds=embeddings+0.01,attention_mask=mask,position_ids=positions,use_cache=False)
        assert torch.isfinite(perturbed.logits).all()
        return {'input_ids_vs_equivalent_embeds':close(reference.logits,output.logits),'arbitrary_continuous_embeds':stats(perturbed.logits)}
    def cache_roundtrip(dynamic):
        cache=(StaticCache(config=model.config,batch_size=ids.shape[0],max_cache_len=ids.shape[1]+5,
            device=torch.device(device),dtype=torch.float32) if dynamic=='static' else (DynamicCache() if dynamic else None))
        prefill=model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=True,
            past_key_values=cache,cache_position=torch.arange(ids.shape[1],device=device),output_hidden_states=True,output_attentions=True)
        before=cache_info(prefill.past_key_values)
        next_ids=torch.full((ids.shape[0],1),tokenizer.encode('N',add_special_tokens=False)[0],device=device,dtype=torch.long)
        extended_mask=torch.cat([mask,torch.ones_like(next_ids)],dim=-1)
        extended_ids=torch.cat([ids,next_ids],dim=-1)
        extended_positions=(extended_mask.cumsum(-1)-1).clamp_min(0)
        complete=model(input_ids=extended_ids,attention_mask=extended_mask,position_ids=extended_positions,use_cache=False,
            output_hidden_states=True,output_attentions=True)
        step=model(input_ids=next_ids,attention_mask=extended_mask,position_ids=extended_positions[:,-1:],
            past_key_values=prefill.past_key_values,cache_position=torch.tensor([ids.shape[1]],device=device),
            use_cache=True,output_hidden_states=True,output_attentions=True)
        assert step.logits.shape==(ids.shape[0],1,model.config.vocab_size)
        return {'prefill_cache':before,'step_cache':cache_info(step.past_key_values),
                'last_logits_vs_uncached':close(complete.logits[:,-1:],step.logits),
                'last_hidden_vs_uncached':close(complete.hidden_states[-1][:,-1:],step.hidden_states[-1]),
                'step_attention_shape':list(step.attentions[0].shape)}
    def trimmed_logits():
        kwargs=dict(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False)
        complete=model(**kwargs).logits
        output=model(**kwargs,num_logits_to_keep=1).logits
        assert output.shape==(ids.shape[0],1,model.config.vocab_size)
        return close(complete[:,-1:],output)
    def generate(kind):
        kwargs=dict(max_new_tokens=8,return_dict_in_generate=True,output_scores=True,
            output_logits=True,output_hidden_states=True,output_attentions=True)
        if kind=='beam':kwargs.update(num_beams=2,do_sample=False)
        elif kind=='sample':kwargs.update(do_sample=True,top_k=10,temperature=0.8)
        else:kwargs.update(do_sample=False)
        torch.manual_seed(319)
        output=model.generate(**encoded,**kwargs)
        assert output.sequences.ndim==2 and output.sequences.shape[0]==ids.shape[0]
        assert output.scores and all(torch.isfinite(t).all() for t in output.logits)
        summary={'keys':list(output.keys()),'sequences':output.sequences.cpu().tolist(),
                 'decoded':tokenizer.batch_decode(output.sequences,skip_special_tokens=False),
                 'steps':len(output.scores),'hidden_state_steps':len(output.hidden_states),
                 'attention_steps':len(output.attentions)}
        if output.past_key_values is not None:summary['cache']=cache_info(output.past_key_values)
        if kind=='sample':
            torch.manual_seed(319)
            repeated=model.generate(**encoded,**kwargs)
            summary['same_seed_repeats_exactly']=torch.equal(output.sequences,repeated.sequences)
        return summary
    def plain_generation(embedded=False):
        inputs={'inputs_embeds':model.get_input_embeddings()(ids),'attention_mask':mask} if embedded else dict(encoded)
        output=model.generate(**inputs,max_new_tokens=8,do_sample=False)
        assert output.ndim==2 and output.shape[0]==ids.shape[0]
        return {'sequences':output.cpu().tolist(),'shape':list(output.shape),
                'decoded':tokenizer.batch_decode(output,skip_special_tokens=False)}
    with torch.inference_mode():
        for name,fn in [('all_forward_outputs',all_outputs),('return_dict_false',tuple_outputs),
                        ('inputs_embeds',embedded_inputs),('legacy_cache',lambda:cache_roundtrip(False)),
                        ('dynamic_cache',lambda:cache_roundtrip(True)),('static_cache',lambda:cache_roundtrip('static')),
                        ('num_logits_to_keep',trimmed_logits),('plain_generation',plain_generation),
                        ('generation_from_embeds',lambda:plain_generation(True)),
                        ('greedy_generation',lambda:generate('greedy')),('beam_generation',lambda:generate('beam')),
                        ('sampled_generation',lambda:generate('sample'))]:stage(name,fn)
    report['all_stages_completed']=not report['failures']
    if device=='mps':torch.mps.synchronize()
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--attention',choices=['eager','sdpa'],default='eager');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    destination=args.output or Path(__file__).with_name(f'native-{args.device}-{args.attention}.json')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');report=native_smoke(args.device,args.attention)
    report['python_warnings']=[str(w.message) for w in captured]
    destination.write_text(json.dumps(report,indent=2))
    print('report',destination,'complete',report['all_stages_completed'],flush=True)
