"""Nesso's native ESM extraction versus explicit final-backbone extraction.

Use paired mode for same-run latency and full saved-tensor byte comparisons.
Use original-only/final-only in separate processes for process high-water RSS;
that RSS includes loading and allocator caches, not just retained hidden states.
No Nesso affinity computation is included in these ESM preprocessing timings.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import time

PINNED_REVISION='08e4846e537177426273712802403f7ba8261b6c'
PINNED_WEIGHTS_SHA256='a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def sync(device):
    if device.type=='mps':torch.mps.synchronize()
    elif device.type=='cuda':torch.cuda.synchronize(device)


def rss_high_water_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def tensor_record(tensor):
    value=tensor.detach().cpu().contiguous()
    data=value.view(torch.uint8).numpy().tobytes()
    encoded=safe_bytes({'embeddings':value})
    restored=safe_load(encoded)['embeddings']
    assert torch.equal(value.view(torch.uint8),restored.view(torch.uint8))
    return {'shape':list(value.shape),'dtype':str(value.dtype),'tensor_bytes':len(data),
            'tensor_sha256':hashlib.sha256(data).hexdigest(),
            'safetensors_sha256':hashlib.sha256(encoded).hexdigest(),
            'safetensors_roundtrip_byte_equal':True}


def compare(left,right):
    a,b=left.detach().cpu().contiguous(),right.detach().cpu().contiguous()
    if a.shape!=b.shape or a.dtype!=b.dtype:return {'accepted':False,'error':'shape or dtype changed'}
    exact=torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta=a.double()-b.double()
    absolute=float(delta.abs().max()) if delta.numel() else 0.0
    relative=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a.double()).clamp_min(1e-12))
    return {'accepted':exact and finite,'byte_equal':exact,'finite':finite,'max_abs':absolute,'relative_l2':relative,
            'saved_embedding_byte_equal':safe_bytes({'embeddings':a})==safe_bytes({'embeddings':b})}


def read_sequences(paths):
    result=[];names=set()
    for path in paths:
        payload=yaml.safe_load(path.read_text())
        for index,entry in enumerate(payload.get('sequences',[])):
            if 'protein' not in entry:continue
            protein=entry['protein'];sequence=protein['sequence']
            if not isinstance(sequence,str) or not sequence:raise ValueError('Expected a nonempty protein sequence')
            name=f'{path.stem}:{protein.get("id",index)}'
            if name in names:raise ValueError(f'Duplicate protein case name: {name}')
            names.add(name);result.append((name,sequence,path))
    if not result:raise ValueError('No protein sequences found in the supplied YAML files')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--sequence-yaml',type=Path,nargs='+',required=True)
    parser.add_argument('--expected-sha256',default=PINNED_WEIGHTS_SHA256)
    parser.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    parser.add_argument('--attention',choices=['eager','sdpa'],help='Omit to preserve native automatic selection')
    parser.add_argument('--mode',choices=['paired','original-only','final-only'],default='paired')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--reference-report',type=Path,help='Optional separate original-only report for final-only byte fingerprints')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; preserve earlier reports with a new path')
    if args.reference_report and args.mode!='final-only':parser.error('--reference-report applies only to final-only mode')
    report={'accepted':False,'status':'running','mode':args.mode,'device':args.device,'cases':[]}
    try:
        global torch, AutoModelForMaskedLM, AutoTokenizer, safe_bytes, safe_load, yaml, nesso_esm, FinalEmbedding
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from safetensors.torch import save as safe_bytes, load as safe_load
        import yaml
        from nesso.data import esm as nesso_esm
        from compressme.final_embedding import FinalEmbedding
        if args.rounds<1 or args.warmup<0 or args.threads<1:raise ValueError('Invalid rounds/warmup/threads')
        device=torch.device(args.device)
        if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable')
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; no NVIDIA validation ran')
        weights=args.model_dir/'model.safetensors'
        digest=sha256(weights)
        if digest!=args.expected_sha256:raise ValueError('ESM model.safetensors SHA256 mismatch')
        if digest!=PINNED_WEIGHTS_SHA256:raise ValueError('This benchmark is pinned to the audited650M checkpoint')
        report.update(weights_sha256=digest,checkpoint_revision=PINNED_REVISION,config_sha256=sha256(args.model_dir/'config.json'),
                      python=platform.python_version(),torch=torch.__version__,transformers=__import__('transformers').__version__)
        sequences=read_sequences(args.sequence_yaml)
        torch.set_num_threads(args.threads)
        kwargs={'local_files_only':True,'trust_remote_code':False,'use_safetensors':True,'dtype':torch.float32}
        if args.attention:kwargs['attn_implementation']=args.attention
        masked_lm=AutoModelForMaskedLM.from_pretrained(args.model_dir,**kwargs).eval().to(device)
        tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
        config=masked_lm.config
        final=FinalEmbedding(masked_lm.esm,output_contract='last_hidden_state') if args.mode!='original-only' else None
        original_parameters=sum(p.numel() for p in masked_lm.parameters())
        final_parameters=sum(p.numel() for p in masked_lm.esm.parameters())
        parameter_dtypes=sorted({str(p.dtype) for p in masked_lm.parameters()})
        if parameter_dtypes!=['torch.float32']:raise ValueError('Expected unchanged float32 model weights')
        if args.mode=='final-only':del masked_lm;masked_lm=None;gc.collect()
        sync(device)
        loaded_rss=rss_high_water_bytes()
        def invoke(name,sequence):
            if name=='original':
                return nesso_esm.extract_esm_embedding(sequence,masked_lm,tokenizer)
            inputs=tokenizer(sequence,return_tensors='pt')
            inputs={key:value.to(device) for key,value in inputs.items()}
            # Same BOS/EOS-inclusive layout as native Nesso esm.py.
            return final(**inputs).squeeze(0).unsqueeze(0).cpu()
        modes=['original','final'] if args.mode=='paired' else [args.mode.removesuffix('-only')]
        rng=random.Random(7231)
        for name,sequence,path in sequences:
            tokenized=tokenizer(sequence,return_tensors='pt')
            B,T=tokenized['input_ids'].shape
            initial={kind:invoke(kind,sequence) for kind in modes};sync(device)
            repeats={kind:compare(initial[kind],invoke(kind,sequence)) for kind in modes}
            records={kind:tensor_record(value) for kind,value in initial.items()}
            gate=compare(initial['original'],initial['final']) if args.mode=='paired' else None
            if args.reference_report:
                previous=json.loads(args.reference_report.read_text())
                if previous.get('device')!=args.device or previous.get('weights_sha256')!=digest:
                    raise ValueError('Reference report backend/checkpoint differs')
                match=next(item for item in previous['cases'] if item['name']==name)
                source=match['outputs']['original'];target=records[modes[0]]
                if source!=target:raise ValueError('Separate-process saved embedding fingerprint differs')
            for _ in range(args.warmup):
                for kind in modes:invoke(kind,sequence)
            sync(device);pairs=[]
            for _ in range(args.rounds):
                ordering=list(modes);rng.shuffle(ordering);times={};values={}
                for kind in ordering:
                    sync(device);start=time.perf_counter();value=invoke(kind,sequence);sync(device)
                    times[kind]=(time.perf_counter()-start)*1000;values[kind]=value
                pair_gate=compare(values['original'],values['final']) if args.mode=='paired' else compare(initial[modes[0]],values[modes[0]])
                pairs.append({'order':ordering,'latency_ms':times,'output_gate':pair_gate})
            medians={kind:statistics.median(pair['latency_ms'][kind] for pair in pairs) for kind in modes}
            item={'name':name,'sequence_characters':len(sequence),'encoded_shape':[B,T],
                  'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),'yaml_sha256':sha256(path),
                  'outputs':records,'source_self_repeat':repeats,'same_run_gate':gate,'pairs':pairs,'median_ms':medians,
                  'analytical_extra_retained_hidden_bytes':config.num_hidden_layers*B*T*config.hidden_size*4}
            if args.mode=='paired':item['original_over_final_speed_ratio']=medians['original']/medians['final']
            report['cases'].append(item)
            print(name,medians,'byte gate',gate,flush=True)
        import transformers.models.esm.modeling_esm as modeling_esm
        import transformers.utils.output_capturing as capture
        import compressme.final_embedding as helper
        source_files=[Path(__file__),Path(nesso_esm.__file__),Path(modeling_esm.__file__),Path(capture.__file__),Path(helper.__file__)]
        report.update(weights_sha256=digest,checkpoint_revision=PINNED_REVISION,config_sha256=sha256(args.model_dir/'config.json'),
            python=platform.python_version(),torch=torch.__version__,transformers=__import__('transformers').__version__,
            attention_implementation=config._attn_implementation,parameter_dtypes=parameter_dtypes,
            threads=args.threads,rounds=args.rounds,warmup=args.warmup,
            source_hashes={str(p):sha256(p) for p in source_files},
            unique_parameters={'original_masked_lm':original_parameters,'selected_backbone':final_parameters,
                               'unique_task_head_difference':original_parameters-final_parameters},
            retention_note='Analytical retained hidden tensors: extra33*B*T*1280float32 values for this checkpoint. Not allocator usage or measured peak memory.',
            rss_high_water={'after_loading_bytes':loaded_rss,'after_inference_bytes':rss_high_water_bytes(),
              'scope':'Whole-process lifetime high-water RSS, including model loading and allocator caches. Separate original-only/final-only processes are required for any comparison; loading may dominate both.'},
            timing_scope='Nesso ESM preprocessing only: tokenizer, unchanged input transfer, selected model invocation and CPU result transfer; model/tokenizer loading excluded. No Nesso affinity-model inference.',
            pretrained_embedding_gate_passed=all(all(x['accepted'] for x in c['source_self_repeat'].values()) and all(p['output_gate']['accepted'] for p in c['pairs']) and (c['same_run_gate'] is None or c['same_run_gate']['accepted']) for c in report['cases']),
            full_nesso_validation=False)
        report['original_vs_final_comparison_ran']=args.mode=='paired' or bool(args.reference_report)
        report['accepted']=report['pretrained_embedding_gate_passed'] and report['original_vs_final_comparison_ran']
        if not report['pretrained_embedding_gate_passed']:
            report['status']='failed'
        elif report['original_vs_final_comparison_ran']:
            report['status']='passed'
        else:
            report['status']='measured_without_original_vs_final_comparison'
        report['pretrained_embedding_gate_passed']=report['accepted']
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report.get(key) for key in ('status','accepted','error')},indent=2))
    return 1 if report['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
