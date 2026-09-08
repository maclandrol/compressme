"""Interleaved original/candidate native token-API latency; no tokenizer timing."""
import argparse, gc, inspect, json, platform, random, statistics, sys, time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--device',choices=['cpu','mps'],required=True)
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--config',required=True)
parser.add_argument('--rounds',type=int,default=9)
parser.add_argument('--warmup-pairs',type=int,default=3)
parser.add_argument('--case')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).parent))
import torch, transformers
from prototype import load_reference, make_candidate, file_sha256
from probe import observable
from compressme.validation import compare_outputs

torch.set_num_threads(4)
reference=load_reference(args.checkpoint,args.config,device=args.device)
candidate=make_candidate(reference,table_device=args.device)
models=[reference,candidate]
device=torch.device(args.device)
random_order=random.Random(3729)
def synchronize():
    if args.device=='mps':torch.mps.synchronize()
def gate(left,right):
    metrics=compare_outputs(observable(left),observable(right))
    failed={k:v for k,v in metrics.items() if v['max_abs']>1e-5 or v['relative_l2']>1e-5}
    return {'accepted':not failed,'worst_max_abs':max((v['max_abs'] for v in metrics.values()),default=0),
            'tensor_outputs':len(metrics),'failures':failed}
def summary(samples):
    return {'median_ms':statistics.median(samples),'mean_ms':statistics.mean(samples),
            'sample_sd_ms':statistics.stdev(samples) if len(samples)>1 else 0,
            'min_ms':min(samples),'max_ms':max(samples),'samples_ms':samples}
def run_case(name,functions):
    if args.case is not None and name != args.case:
        return {'case':name,'timed':False,'skipped_by_filter':True}
    left,right=(fn() for fn in functions)
    comparison=gate(left,right)
    del left,right
    if not comparison['accepted']:
        return {'case':name,'gate':comparison,'timed':False}
    for _ in range(args.warmup_pairs):
        for fn in functions:
            result=fn();synchronize();del result
    samples=[[],[]]
    gc.collect()
    for _ in range(args.rounds):
        order=[0,1];random_order.shuffle(order)
        for index in order:
            synchronize();start=time.perf_counter()
            result=functions[index]()
            synchronize();elapsed=(time.perf_counter()-start)*1000
            samples[index].append(elapsed);del result
    summaries=[summary(x) for x in samples]
    record={'case':name,'gate':comparison,'timed':True,'reference':summaries[0],
            'candidate':summaries[1], 'speedup':summaries[0]['median_ms']/summaries[1]['median_ms']}
    print(name,round(summaries[0]['median_ms'],3),'→',round(summaries[1]['median_ms'],3),
          'ms',round(record['speedup'],3),'x',flush=True)
    return record

records=[]
with torch.inference_mode():
    for batch,length,all_outputs in [(1,32,False),(1,32,True),(4,32,False),(1,512,False)]:
        ids=((torch.arange(batch*length).reshape(batch,length)*7+2)%84).to(device)
        kwargs=dict(input_ids=ids,use_cache=True,output_hidden_states=all_outputs,output_attentions=all_outputs)
        functions=[lambda model=model,kwargs=kwargs:model(**kwargs) for model in models]
        records.append(run_case(f'prefill_{batch}x{length}_all_outputs_{all_outputs}',functions))
    for all_outputs in (False,True):
        prefix=torch.tensor([[2,4,4,9]+[4]*28],device=device)
        prefilled=[model(input_ids=prefix,use_cache=True).past_key_values for model in models]
        assert all(isinstance(cache,tuple) for cache in prefilled)
        functions=[]
        for model,cache in zip(models,prefilled):
            kwargs=dict(input_ids=torch.tensor([[4]],device=device),past_key_values=cache,
                attention_mask=torch.ones((1,33),dtype=torch.long,device=device),
                position_ids=torch.tensor([[32]],device=device),cache_position=torch.tensor([32],device=device),
                use_cache=True,output_hidden_states=all_outputs,output_attentions=all_outputs)
            functions.append(lambda model=model,kwargs=kwargs:model(**kwargs))
        records.append(run_case(f'decode_b1_prefix32_all_outputs_{all_outputs}',functions))
    prompt=torch.tensor([[2,4,4,9]],device=device)
    for all_outputs in (False,True):
        kwargs=dict(input_ids=prompt,attention_mask=torch.ones_like(prompt),do_sample=False,
            max_new_tokens=12,return_dict_in_generate=True,output_scores=True,output_logits=True,
            output_hidden_states=all_outputs,output_attentions=all_outputs)
        functions=[lambda model=model,kwargs=kwargs:model.generate(**kwargs) for model in models]
        records.append(run_case(f'greedy_b1_max12_all_outputs_{all_outputs}',functions))

result={'scope':'Same-backend, same-cache token-ID native model API latency; input transfer and tokenizer excluded',
    'device':args.device,'variant':'generic','backend':'eager','python':platform.python_version(),'torch':torch.__version__,
    'transformers':transformers.__version__,'cpu_threads':torch.get_num_threads(),'warmup_pairs':args.warmup_pairs,
    'rounds':args.rounds,'synchronized_mps':args.device=='mps','randomized_interleaved_order':True,
    'checkpoint_sha256':file_sha256(args.checkpoint),'config_sha256':file_sha256(args.config),
    'model_source_sha256':file_sha256(inspect.getfile(type(reference))),
    'parameters_before':sum(p.numel() for p in reference.parameters()),
    'parameters_after':sum(p.numel() for p in candidate.parameters()),
    'buffers_before':sum(b.numel() for b in reference.buffers()),
    'buffers_after':sum(b.numel() for b in candidate.buffers()),
    'tables':{k:p.numel() for k,p in candidate.named_parameters() if k.startswith('_first_qkv')},
    'records':records}
args.output.write_text(json.dumps(result,indent=2))
print('report',args.output,flush=True)
