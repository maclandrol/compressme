"""Build, verify and benchmark the pinned Mol-JEPA SMILES recipe from raw weights."""
import argparse,gc,json,random,shutil,statistics,tempfile,time
from pathlib import Path
from common import verify,write_report,device_setup,synchronize
from fetch import MOL_SHA,MOL_SIZE,MOL_REVISION


def original(directory):
    from compressme.moljepa_io import moljepa_factory
    from safetensors.torch import load_file
    root=Path(directory);verify(root/'model.safetensors',MOL_SHA,MOL_SIZE)
    model=moljepa_factory(root/'architecture')
    model.load_state_dict(load_file(str(root/'model.safetensors')),strict=True)
    return model.eval().requires_grad_(False)


def cases():
    from compressme import Example
    smiles=json.loads((Path(__file__).parent/'verification_smiles.json').read_text())['smiles']
    result=[Example((smiles[i:i+8],),{'return_attn':flag}) for flag in (False,True) for i in range(0,len(smiles),8)]
    return smiles,result


def build(args):
    import torch
    from compressme import optimize_moljepa,load_moljepa,validate
    if args.artifact.exists():raise FileExistsError('Choose a new artifact directory')
    model=original(args.original);_,examples=cases()
    result=optimize_moljepa(model,smiles_only=True,validation=examples)
    if result.report['status']!='accepted_on_validation_examples':raise RuntimeError('Compilation output gate failed; no artifact exported')
    result.report['source_revision']=MOL_REVISION
    result.report['source_checkpoint_sha256']=MOL_SHA
    result.report['scope']='SMILES-only; all predictions, CLS, latent embeddings and optional attentions; frozen float32 inference'
    args.artifact.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+args.artifact.name+'.',dir=args.artifact.parent))
    try:
        result.save(temporary,packing=args.packing)
        shutil.copytree(args.original/'architecture',temporary/'architecture',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        recovered=load_moljepa(temporary,accelerate=False)
        gate=validate(model,recovered,examples,relative_tolerance=1e-5,absolute_tolerance=1e-5)
        if not gate['accepted']:raise RuntimeError('Fresh artifact reload failed all-output gate')
        result.report['reload_validation']=gate
        write_report(temporary/'reproduction.json',result.report)
        args.artifact.mkdir();temporary.replace(args.artifact)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return {'accepted':True,'artifact':str(args.artifact),'parameters_before':result.report['parameters_before'],
            'parameters_after':result.report['parameters_after'],'reload_examples':len(examples)}


def verify_artifact(args):
    import torch
    from compressme import load_moljepa,validate
    model=original(args.original).to(args.device)
    candidate=load_moljepa(args.artifact,device=args.device,accelerate=args.accelerate)
    smiles,examples=cases()
    self_gate=validate(model,model,examples,relative_tolerance=1e-5,absolute_tolerance=1e-5)
    gate=validate(model,candidate,examples,relative_tolerance=1e-5,absolute_tolerance=1e-5)
    report={'device':args.device,'torch':torch.__version__,'dtype':'float32','precision_changed':False,
            'matmul_precision':torch.get_float32_matmul_precision(),'threads':torch.get_num_threads(),
            'checkpoint_sha256':MOL_SHA,'source_revision':MOL_REVISION,'accelerate':args.accelerate,
            'self_repeat':self_gate,'all_outputs':gate,'accepted':self_gate['accepted'] and gate['accepted']}
    if not report['accepted']:
        write_report(args.report,report);raise RuntimeError('All-output gate failed; no benchmark performed')
    if args.rounds:
        batch=smiles[:args.batch_size];models={'original':model,'compressed':candidate};timings={k:[] for k in models}
        with torch.inference_mode():
            for item in models.values():
                for _ in range(3):item(batch,return_attn=True)
            rng=random.Random(739)
            for _ in range(args.rounds):
                order=list(models);rng.shuffle(order)
                for name in order:
                    synchronize(args.device);start=time.perf_counter()
                    models[name](batch,return_attn=True)
                    synchronize(args.device);timings[name].append(time.perf_counter()-start)
        medians={k:statistics.median(v) for k,v in timings.items()}
        report['benchmark']={'protocol':'custom_first_n_smiles_attentions_3_warmups','batch_size':len(batch),'return_attn':True,'seconds':timings,'median_seconds':medians,
                             'speedup':medians['original']/medians['compressed'],
                             'scope':'Warm complete SMILES calls including featurization; no result cache, no general speed guarantee'}
    write_report(args.report,report)
    return {'accepted':True,'report':str(args.report),'benchmark':report.get('benchmark')}



def benchmark_artifact(args):
    import torch
    from compressme import load_moljepa,validate
    from compressme.smiles_runtime import accelerate_smiles
    protocol=json.loads((Path(__file__).parent/'moljepa-benchmark.json').read_text())
    compressed=load_moljepa(args.artifact,accelerate=False)
    models={'original':original(args.original),'compressed':compressed,
            'fast':accelerate_smiles(compressed,metal=False),
            'production':accelerate_smiles(compressed,metal=True)}
    models={name:model.to(args.device).eval() for name,model in models.items()}
    smiles,_=cases()
    from compressme import Example
    examples=[Example((smiles[i:i+4],),{'return_attn':flag}) for i in range(0,len(smiles),4) for flag in (False,True)]
    gates={name:validate(models['original'],model,examples,relative_tolerance=1e-5,absolute_tolerance=1e-5)
           for name,model in models.items()}
    report={'device':args.device,'torch':torch.__version__,'protocol':protocol,
            'rounds':args.rounds,'warmups':10,'recorded_round_count':args.rounds==20,
            'parameter_counts':{name:sum(p.numel() for p in model.parameters()) for name,model in models.items()},
            'matmul_precision':torch.get_float32_matmul_precision(),'dtype':'float32','cpu_threads':torch.get_num_threads(),
            'gates':gates,'accepted':all(g['accepted'] for g in gates.values()),'workloads':protocol['workloads'],'timings':{}}
    if not report['accepted']:
        write_report(args.report,report);raise RuntimeError('Complete-output benchmark gate failed; no timing performed')
    with torch.inference_mode():
        for size,values in protocol['workloads'].items():
            for _ in range(10):
                for model in models.values():model(values)
            synchronize(args.device);times={name:[] for name in models};rng=random.Random(4561+int(size))
            for _ in range(args.rounds):
                order=list(models);rng.shuffle(order)
                for name in order:
                    synchronize(args.device);start=time.perf_counter()
                    models[name](values)
                    synchronize(args.device);times[name].append(1000*(time.perf_counter()-start))
            report['timings'][size]={name:{'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v),
                'p10_ms':sorted(v)[min(len(v)-1,int(len(v)*.1))],
                'p90_ms':sorted(v)[int((len(v)-1)*.9)],'milliseconds':v} for name,v in times.items()}
    report['scope']='Recorded molecules, variants, default return_attn=False,10warmups, synchronized interleaving;20rounds reproduce the report protocol, not guaranteed identical timing. Complete-output gates both attention modes.'
    write_report(args.report,report)
    return {'accepted':True,'report':str(args.report),'timings':{size:{name:v['median_ms'] for name,v in data.items()} for size,data in report['timings'].items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('build');p.add_argument('--original',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--packing',action='store_true')
    p=commands.add_parser('verify');p.add_argument('--original',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--report',type=Path,required=True)
    p.add_argument('--accelerate',action='store_true');p.add_argument('--rounds',type=int,default=0);p.add_argument('--batch-size',type=int,default=4)
    p=commands.add_parser('benchmark',help='Recorded1/4/32 workloads, four variants,10warmups, return_attn=False')
    p.add_argument('--original',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps');p.add_argument('--report',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=20)
    args=parser.parse_args()
    if getattr(args,'rounds',0)<0 or not 1<=getattr(args,'batch_size',4)<=64:parser.error('Use nonnegative rounds and batch size 1..64')
    if args.command in ('verify','benchmark') and args.report.exists():raise FileExistsError('Report already exists')
    if args.command=='benchmark' and args.rounds<1:parser.error('Benchmark rounds must be positive')
    device_setup(getattr(args,'device','cpu'))
    print(json.dumps(build(args) if args.command=='build' else benchmark_artifact(args) if args.command=='benchmark' else verify_artifact(args),indent=2))

if __name__=='__main__':main()
