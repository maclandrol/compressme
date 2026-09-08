#!/usr/bin/env python3
"""Pinned NovoMolGen original native-Llama API smoke, with explicit local inputs.

No compression, timing or molecular-quality evaluation. Runtime imports use a
fresh private temporary cache. No Hugging Face repository Python is executed.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

MODEL_ID='chandar-lab/NovoMolGen_32M_SMILES_AtomWise'
REVISION='dcd3f59261bebf84142c13617d4e129b4b0d0fdc'
CHECKPOINT_SHA256='3cf10749dce97e74edbb0c4d4e609265014e90f7be2fa50d731af67977cd0699'
CONFIG_SHA256='f3cc487205aa417f4786f50d49a4120097baaffd80a9510ccfe5c37aa4ed3a34'
LLAMA_SOURCE_SHA256='733e7625fec5abcfa416ab9b866cb91875d41007f2e31c3e90d2ae0111ae98f7'
TOKENIZER_SHA256={
    'generation_config.json':'2c13e64df63c6cd0a5e174a177c222bef64892a42105fe9f92ef6f0e59502308',
    'tokenizer.json':'4baa7905f4431811d024de231313a4935a28575f84d08793b2d258a0d6a419cd',
    'tokenizer_config.json':'d83cf3df2f4567f89fe50fa6d6c36f832258f6920d41668eb4d48e8410835a13',
    'special_tokens_map.json':'db82f8bd9b25d14f9c788e6bde64de84d42f1c2538f1c245ba6cb3e872d14b18',
}
_private_cache=None
_runtime_ready=False


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime_versions():
    versions={name:importlib.metadata.version(name) for name in ('transformers','tokenizers')}
    if versions['transformers']!='4.46.2':
        raise RuntimeError(f"This archived native API requires transformers==4.46.2; found {versions['transformers']}")
    if versions['tokenizers']!='0.20.3':
        raise RuntimeError(f"This archived tokenizer requires tokenizers==0.20.3; found {versions['tokenizers']}")
    return versions


def verify_inputs(checkpoint,config,tokenizer_source):
    paths={'checkpoint':(Path(checkpoint),CHECKPOINT_SHA256),
           'config':(Path(config),CONFIG_SHA256)}
    root=Path(tokenizer_source)
    if not root.is_dir():
        raise ValueError('--tokenizer-source must be a local directory')
    paths.update({name:(root/name,expected) for name,expected in TOKENIZER_SHA256.items()})
    verified={}
    for name,(path,expected) in paths.items():
        if not path.is_file():
            raise ValueError(f'Missing local input: {name}')
        observed=sha256(path)
        if observed!=expected:
            raise ValueError(f'SHA256 mismatch for {name}: {observed}')
        verified[name]=observed
    return {'model_id':MODEL_ID,'revision':REVISION,'sha256':verified}


def _ensure_runtime():
    """Import the pinned installed runtime without touching shared caches.

    The CLI is intended for a fresh process. Helpers require this module to
    initialize Hugging Face before any unrelated Hugging Face imports. Cache
    environment variables are restored after import; the runtime's resolved
    cache constants keep pointing to a unique temporary directory for the
    process lifetime, which TemporaryDirectory removes on normal exit.
    """
    global _private_cache,_runtime_ready
    if _runtime_ready:
        return
    verify_runtime_versions()
    if 'transformers' in sys.modules or 'huggingface_hub' in sys.modules:
        raise RuntimeError('Run this isolated native smoke in a fresh process before importing Hugging Face libraries')
    _private_cache=tempfile.TemporaryDirectory(prefix='compressme-novo-native-')
    root=Path(_private_cache.name)
    settings={
        'HF_HOME':str(root/'hf'),
        'HF_HUB_CACHE':str(root/'hf'/'hub'),
        'HUGGINGFACE_HUB_CACHE':str(root/'hf'/'hub'),
        'HF_MODULES_CACHE':str(root/'hf'/'modules'),
        'TORCH_HOME':str(root/'torch'),
        'HF_HUB_OFFLINE':'1',
        'TOKENIZERS_PARALLELISM':'false',
        'PYTORCH_PRETRAINED_BERT_CACHE':None,
        'PYTORCH_TRANSFORMERS_CACHE':None,
        'TRANSFORMERS_CACHE':None,
    }
    saved={name:os.environ.get(name) for name in settings}
    try:
        for name,value in settings.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
        import torch as _torch
        import transformers as _transformers
        import tokenizers as _tokenizers
        from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast,GenerationConfig
        from transformers.cache_utils import DynamicCache,StaticCache
        from safetensors.torch import load_file
        observed=sha256(inspect.getfile(LlamaForCausalLM))
        if observed!=LLAMA_SOURCE_SHA256:
            raise RuntimeError(f'Installed modeling_llama.py differs from the pinned source: {observed}')
        globals().update(torch=_torch,transformers=_transformers,tokenizers=_tokenizers,
                         LlamaConfig=LlamaConfig,LlamaForCausalLM=LlamaForCausalLM,
                         PreTrainedTokenizerFast=PreTrainedTokenizerFast,GenerationConfig=GenerationConfig,
                         DynamicCache=DynamicCache,StaticCache=StaticCache,load_file=load_file)
        _runtime_ready=True
    finally:
        for name,value in saved.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def load_native(checkpoint,config,tokenizer_source,*,device='cpu',attn_implementation='eager'):
    """Load verified local inputs using only the pinned installed Llama class."""
    verify_inputs(checkpoint,config,tokenizer_source)
    _ensure_runtime()
    config=LlamaConfig.from_json_file(str(config))
    config._attn_implementation=attn_implementation
    with torch.random.fork_rng(devices=[]):model=LlamaForCausalLM(config)
    model.load_state_dict(load_file(str(checkpoint),device='cpu'),strict=True)
    model.generation_config=GenerationConfig.from_pretrained(tokenizer_source,local_files_only=True)
    model=model.eval().to(device)
    tokenizer=PreTrainedTokenizerFast.from_pretrained(tokenizer_source,local_files_only=True)
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

def native_smoke(device,attn_implementation,*,checkpoint,config,tokenizer_source,threads=4):
    _ensure_runtime()
    torch.set_num_threads(threads)
    if device=='mps':assert torch.backends.mps.is_available(),'Run outside sandbox for MPS access'
    model,tokenizer=load_native(checkpoint,config,tokenizer_source,device=device,attn_implementation=attn_implementation)
    report={'scope':'Original native Llama checkpoint API smoke only; no compression, timing or molecular-quality claim',
            'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
            'tokenizers':tokenizers.__version__,'device':device,'dtype':'float32',
            'attention_implementation':model.config._attn_implementation,
            'model_class':type(model).__name__,'class_source':'transformers/models/llama/modeling_llama.py',
            'class_source_sha256':LLAMA_SOURCE_SHA256,
            'forward_signature':str(inspect.signature(model.forward)),
            'checkpoint_sha256':CHECKPOINT_SHA256,'config_sha256':CONFIG_SHA256,'unique_parameters':sum(p.numel() for p in model.parameters()),
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--tokenizer-source',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--attention',choices=['eager','sdpa'],default='eager')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--check-inputs-only',action='store_true',help='Check versions and input hashes without importing the model runtime')
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    if not args.check_inputs_only and args.output is None:parser.error('--output is required for a smoke run')
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error('Output already exists; use another path or explicitly supply --overwrite')
    try:
        versions=verify_runtime_versions()
        provenance=verify_inputs(args.checkpoint,args.config,args.tokenizer_source)
    except (ValueError,RuntimeError,importlib.metadata.PackageNotFoundError) as exc:
        parser.error(str(exc))
    if args.check_inputs_only:
        print(json.dumps({'verified':True,'versions':versions,'inputs':provenance},indent=2))
        return
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        report=native_smoke(args.device,args.attention,checkpoint=args.checkpoint,config=args.config,
                            tokenizer_source=args.tokenizer_source,threads=args.threads)
    report['python_warnings']=[str(w.message) for w in captured]
    report['input_provenance']=provenance
    report['cache_policy']='Unique per-process temporary local-only cache; no shared/default cache migration'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w' if args.overwrite else 'x') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')
    print('report',args.output,'complete',report['all_stages_completed'],flush=True)
    if not report['all_stages_completed']:
        raise SystemExit(1)


if __name__=='__main__':main()
