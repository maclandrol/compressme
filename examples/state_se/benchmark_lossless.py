"""Paired latency, fresh token/count inputs; no throughput or peak-memory claim."""
import argparse, hashlib, json, random, statistics, time
from pathlib import Path
import torch
from original import load_state_se as load_original
from compressme import load_state_se
from compressme.validation import compare_outputs

def sync(device):
    if device=='mps':torch.mps.synchronize()
    elif device.startswith('cuda'):torch.cuda.synchronize()

def batch(device, tokens, generator):
    ids=torch.randint(19790,(1,tokens),generator=generator);ids[:,0]=3
    return tuple(v.to(device) for v in (ids,torch.randint(19790,(1,31),generator=generator),
      torch.rand(1,31,generator=generator),torch.zeros(1,dtype=torch.long),torch.rand(1,31,generator=generator),
      torch.zeros(1,tokens,dtype=torch.bool),torch.full((1,),4.),torch.rand(1,tokens,generator=generator)*12,
      torch.zeros(1,dtype=torch.int32)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',choices=['cpu','mps'],required=True)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4)
    original=load_original(a.checkpoint,config=a.artifact/'architecture/config.json',source_root=a.artifact/'architecture').eval().requires_grad_(False).to(a.device)
    packed=load_state_se(a.artifact,device=a.device)
    gen=torch.Generator().manual_seed(807);order=random.Random(19273);records=[]
    with torch.inference_mode():
      for tokens in (32,2048):
        for _ in range(2):
          current=batch(a.device,tokens,gen)
          original._compute_embedding_for_batch(current);packed._compute_embedding_for_batch(current)
        sync(a.device);pairs=[]
        for i in range(a.rounds):
          current=batch(a.device,tokens,gen);names=['original','packed'];order.shuffle(names);times={};outs={}
          for name in names:
            model=original if name=='original' else packed
            sync(a.device);start=time.perf_counter();out=model._compute_embedding_for_batch(current);sync(a.device)
            times[name]=(time.perf_counter()-start)*1000
            outs[name]=tuple(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for v in out)
          metrics=compare_outputs(outs['original'],outs['packed'])
          pairs.append({'order':names,'ms':times,'byte_equal':all(v['bitwise'] for v in metrics.values()),'metrics':metrics})
        before=statistics.median(p['ms']['original'] for p in pairs);after=statistics.median(p['ms']['packed'] for p in pairs)
        records.append({'B':1,'T':tokens,'Q':31,'pairs':pairs,'original_median_ms':before,'packed_median_ms':after,
                       'packed_over_original_latency':after/before,'original_over_packed_speed':before/after})
        print(records[-1],flush=True)
    result={'device':a.device,'torch':torch.__version__,'cpu_threads':4,'rounds':a.rounds,
      'scope':'Warmed interleaved original complete numerical batch API, fresh gene/count values per pair; synchronization included at timing boundaries. Loading and external input construction excluded; internal decode/transfer included.',
      'no_peak_request_memory_measurement':True,'persistent_decoded_cache_bytes':0,'cases':records,
      'all_paired_outputs_byte_equal':all(p['byte_equal'] for r in records for p in r['pairs']),
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'artifact_manifest_sha256':hashlib.sha256((a.artifact/'manifest.json').read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
