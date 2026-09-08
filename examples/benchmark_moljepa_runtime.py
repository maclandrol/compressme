import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/compressme-mpl-cache')
import argparse,copy,json,random,statistics,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from compressme import load_moljepa,compare_outputs
from compressme.moljepa_io import moljepa_factory
from compressme.smiles_runtime import accelerate_smiles

torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--report',type=Path,default=ROOT/'benchmarks/moljepa_runtime_macos.json')
args=p.parse_args()
smiles=json.loads((ROOT/'benchmarks/verification_smiles.json').read_text())['smiles']
workloads={1:[smiles[56]],4:[smiles[i] for i in [5,28,46,56]],32:smiles[::2]}
original=moljepa_factory(ROOT/'artifacts/moljepa-smiles/architecture')
original.load_state_dict(load_file(str(args.checkpoint)),strict=True)
compressed=load_moljepa(ROOT/'artifacts/moljepa-smiles',accelerate=False)
fast=accelerate_smiles(compressed,metal=False)
production=accelerate_smiles(compressed,metal=True)
models={'original':original,'compressed':compressed,'fast':fast,'production':production}
results={'cpu_threads':4,'torch':torch.__version__,'workloads':workloads,
         'method':'LIVE PACKAGE production accelerate_smiles. Float32 eval, complete fresh SMILES API; no molecule or output caching. CSR and CPU-to-MPS transfer included.10 warmups/20 shuffled interleaved synchronized rounds. Original trained checkpoint vs previous compressed vs fast(metal=False) vs production(metal=True). All64 complete outputs+attentions gated at max_abs1e-5 before timing.',
         'parameter_counts':{name:sum(p.numel() for p in model.parameters()) for name,model in models.items()},
         'state_bytes':{name:sum(t.numel()*t.element_size() for t in model.state_dict().values()) for name,model in models.items()},
         'gates':{},'devices':{}}
def save():args.report.write_text(json.dumps(results,indent=2))
for device in ['cpu','mps']:
    for model in models.values():model.to(device).eval()
    def sync():
        if device=='mps':torch.mps.synchronize()
    gates=[]
    with torch.inference_mode():
        for offset in range(0,len(smiles),4):
            values=smiles[offset:offset+4]
            for attn in [False,True]:
                expected=original(values,return_attn=attn)
                for name in ('compressed','fast','production'):
                    actual=models[name](values,return_attn=attn);errors=compare_outputs(expected,actual)
                    assert all(e['max_abs']<=1e-5 for e in errors.values()),(device,offset,attn,name,errors)
                    gates.append({'offset':offset,'return_attn':attn,'name':name,'errors':errors})
        results['gates'][device]=gates;save()
        print('ALL64 GATES',device,max(e['max_abs'] for g in gates for e in g['errors'].values()),flush=True)
        if device=='mps':
            try:
                copied=copy.deepcopy(production)
                error=compare_outputs(production(workloads[4]),copied(workloads[4]))
                assert all(e['max_abs']<=1e-5 for e in error.values())
                results['compiled_deepcopy']={'pass':True,'errors':error}
                del copied
            except Exception as exc:
                results['compiled_deepcopy']={'pass':False,'exception':repr(exc)}
            save();print('DEEPCOPY',results['compiled_deepcopy'],flush=True)
        for size,values in workloads.items():
            fns={name:lambda m=model:m(values) for name,model in models.items()}
            for _ in range(10):
                for fn in fns.values():fn()
            sync();times={name:[] for name in fns};rng=random.Random(4561+size)
            for _ in range(20):
                order=list(fns);rng.shuffle(order)
                for name in order:
                    sync();start=time.perf_counter();fns[name]();sync()
                    times[name].append(1000*(time.perf_counter()-start))
            result={name:{'median_ms':statistics.median(vals),'p10_ms':sorted(vals)[2],
                          'p90_ms':sorted(vals)[17],'min_ms':min(vals),'max_ms':max(vals),'milliseconds':vals}
                    for name,vals in times.items()}
            results['devices'].setdefault(device,{})[str(size)]=result;save()
            print(device,size,json.dumps({name:{k:v for k,v in val.items() if k!='milliseconds'} for name,val in result.items()}),flush=True)
    for model in models.values():model.cpu()
print('PARAMETERS',results['parameter_counts'],'STATE BYTES',results['state_bytes'],flush=True)
print('DONE',flush=True)
