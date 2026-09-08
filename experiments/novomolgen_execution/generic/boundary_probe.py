"""Original-weight full-output gate at shape boundaries; no timing."""
import argparse, inspect, json, platform
from pathlib import Path
import torch, transformers
from compressme.validation import compare_outputs
from prototype import load_reference, make_candidate, file_sha256
from probe import observable, tensor_byte_comparison

parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--config',required=True)
parser.add_argument('--device',choices=['cpu','mps'],default='mps')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
torch.set_num_threads(4)
source=load_reference(args.checkpoint,args.config,device=args.device)
candidate=make_candidate(source,table_device=args.device)
records=[]
rng=torch.Generator().manual_seed(9185)
with torch.inference_mode():
    for count in (2,3,4,6,8,9,12,13,14,15,16,17,18,24,27,31,32):
        for batch in range(1,count+1):
            if count%batch:continue
            length=count//batch
            for strided in (False,True):
                ids=torch.randint(0,84,(batch,length*(2 if strided else 1)),generator=rng).to(args.device)
                positions=torch.arange(13,13+length*(2 if strided else 1),device=args.device).expand(batch,-1)
                if strided:
                    ids=ids[:,::2]
                    positions=positions[:,::2]
                kwargs=dict(input_ids=ids,position_ids=positions,output_hidden_states=True,
                            output_attentions=True,use_cache=True)
                left=source(**kwargs);right=candidate(**kwargs)
                left,right=observable(left),observable(right)
                metrics=compare_outputs(left,right)
                byte_comparison=tensor_byte_comparison(left,right)
                failed={path:v for path,v in metrics.items() if v['max_abs']>1e-5 or v['relative_l2']>1e-5}
                rec={'shape':[batch,length],'strided':strided,'ids_stride':list(ids.stride()),
                     'position_stride':list(positions.stride()),'accepted':not failed and byte_comparison['accepted'],
                     'numeric_accepted':not failed,'tensor_byte_comparison':byte_comparison,
                     'max_abs':max(v['max_abs'] for v in metrics.values()),
                     'max_relative_l2':max(v['relative_l2'] for v in metrics.values()),
                     'tensor_outputs':len(metrics),'failed_outputs':failed}
                records.append(rec)
                print(batch,length,'strided',strided,'passed' if rec['accepted'] else 'FAILED',flush=True)
result={'device':args.device,'python':platform.python_version(),'torch':torch.__version__,
        'transformers':transformers.__version__,'checkpoint_sha256':file_sha256(args.checkpoint),
        'config_sha256':file_sha256(args.config),'model_source_sha256':file_sha256(inspect.getfile(type(source))),
        'parameters_before':sum(p.numel() for p in source.parameters()),
        'parameters_after':sum(p.numel() for p in candidate.parameters()),
        'tolerances':{'max_abs':1e-5,'relative_l2':1e-5},'accepted':all(r['accepted'] for r in records),
        'case_count':len(records),'worst_max_abs':max(r['max_abs'] for r in records),
        'all_tensor_bytes_equal':all(r['tensor_byte_comparison']['accepted'] for r in records),
        'byte_tensor_leaves':sum(r['tensor_byte_comparison']['tensor_leaves'] for r in records),
        'bytes_checked':sum(r['tensor_byte_comparison']['bytes_checked'] for r in records),
        'tensor_comparisons':sum(r['tensor_outputs'] for r in records),'records':records}
args.output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
