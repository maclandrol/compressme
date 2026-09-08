"""Local original QKV shape audit; exact comparisons, no timing."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from prototype import load_reference, make_candidate

base = Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--config',required=True)
parser.add_argument('--device',choices=['cpu','mps'],default='mps')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
torch.set_num_threads(4)
model = load_reference(args.checkpoint,args.config,device=args.device)
candidate = make_candidate(model, table_device=args.device)
records = []
with torch.inference_mode():
    for batch, length in [(1,n) for n in (1,2,3,4,6,8,9,12,16,17,18,24,27,31,32,33,42,64,68,84,128)] + [(b,l) for b,l in ((2,1),(3,1),(4,1),(8,1),(2,6),(3,9),(4,17),(2,42),(32,1),(1,2048))]:
        ids = (torch.arange(batch*length).reshape(batch,length)%84).to(args.device)
        x = model.model.layers[0].input_layernorm(model.model.embed_tokens(ids))
        actual = torch.cat([getattr(model.model.layers[0].self_attn,name)(x) for name in ('q_proj','k_proj','v_proj')], dim=-1)
        rec = {'batch':batch,'length':length}
        for name,table in [('bulk',candidate._first_qkv_rows),('singleton',candidate._first_qkv_singleton_rows)]:
            wanted = F.embedding(ids,table)
            rec[name] = {'equal':torch.equal(actual,wanted),'max_abs':float((actual-wanted).abs().max())}
        records.append(rec)
        print(rec,flush=True)
args.output.write_text(json.dumps(records,indent=2))
