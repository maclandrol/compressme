"""Build/verify complete Boltz-2 sharing from safely converted pinned weights.

No checkpoint unpickling. Native preprocessing does consume trusted upstream
RDKit molecular pickles that the explicit fetch step verifies first.
"""
import argparse,gc,json,shutil,tempfile
from pathlib import Path
from common import device_setup,verify,write_report
from safe_state import load_tensor_state


def original(directory):
    import torch
    import compressme.boltz2_io as io
    verify(Path(directory)/'manifest.json','3c855681aa05d0d18ebeeb2335fe5850a66a03ba881eada1afcedfb60e13258c')
    root=Path(io.__file__).parent/'data/boltz2'
    bundle=io.boltz2_factory(root)
    for member,checkpoint in [('confidence','boltz2_conf.ckpt'),('affinity','boltz2_aff.ckpt')]:
        state,_=load_tensor_state(directory,checkpoint)
        # Preserve native strict load order, including original registered aliases.
        bundle[member].load_state_dict(state,strict=True)
        del state
    return bundle.eval().requires_grad_(False)


def molecular_assets(directory):
    root=Path(directory);report=json.loads((root.parent/'molecular-assets.json').read_text())
    from fetch import ASSET_SHA,ASSET_SIZE
    if report.get('archive_sha256')!=ASSET_SHA or report.get('archive_bytes')!=ASSET_SIZE:
        raise ValueError('Run the explicit pinned boltz2-molecules fetch step')
    for record in report['assets']:
        name=record['file']
        if Path(name).name!=name:raise ValueError('Molecular asset path is not a plain basename')
        verify(root/name,record['sha256'],record['bytes'])


def prepare(input_file,molecules,work,seed):
    from boltz.main import process_inputs
    from boltz.data.types import Manifest
    from pytorch_lightning import seed_everything
    from rdkit import Chem
    if work.exists():raise FileExistsError('Choose a fresh --work directory')
    molecular_assets(molecules)
    work.mkdir(parents=True);Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    seed_everything(seed)
    process_inputs(data=[input_file],out_dir=work,ccd_path=work/'unused.pkl',mol_dir=molecules,
        use_msa_server=False,msa_server_url='https://api.colabfold.com',msa_pairing_strategy='greedy',
        boltz2=True,preprocessing_threads=1)
    manifest=Manifest.load(work/'processed/manifest.json')
    if len(manifest.records)!=1 or not manifest.records[0].affinity:
        raise ValueError('This complete-pair verifier requires one complex with an affinity request')
    return manifest


def batch_for(member,manifest,molecules,work,device,seed):
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from pytorch_lightning import seed_everything
    processed=work/'processed';affinity=member=='affinity'
    dm=Boltz2InferenceDataModule(manifest=manifest,target_dir=work/'predictions' if affinity else processed/'structures',
        msa_dir=processed/'msa',mol_dir=molecules,num_workers=0,constraints_dir=processed/'constraints',
        template_dir=processed/'templates',extra_mols_dir=processed/'mols',affinity=affinity,
        override_method='other' if affinity else None)
    seed_everything(seed)
    return dm.transfer_batch_to_device(next(iter(dm.predict_dataloader())),device,0)


def clone_output(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if type(value) is dict:return {k:clone_output(v) for k,v in value.items()}
    if type(value) is list:return [clone_output(v) for v in value]
    if type(value) is tuple:return tuple(clone_output(v) for v in value)
    if value is None or type(value) in (str,bool,int,float):return value
    raise TypeError('Unsupported native output leaf '+str(type(value)))


def predict(model,batch,seed):
    import torch
    from pytorch_lightning import seed_everything
    seed_everything(seed)
    with torch.no_grad():result=model.predict_step(batch,0)
    if result.get('exception'):raise RuntimeError('Native prediction returned exception=True')
    return result


def byte_gate(reference,candidate):
    from compressme import compare_outputs
    metrics=compare_outputs(reference,candidate)
    if not metrics:raise ValueError('No tensor outputs inspected')
    return {'accepted':all(v['bitwise'] for v in metrics.values()),'tensor_outputs':len(metrics),'metrics':metrics}


def baseline(bundle,manifest,args,device):
    from boltz.data.write.writer import BoltzWriter,BoltzAffinityWriter
    outputs={};batches={};gates={}
    for member in ('confidence','affinity'):
        batch=batch_for(member,manifest,args.molecules,args.work,device,args.seed);batches[member]=batch
        result=predict(bundle[member],batch,args.seed);outputs[member]=clone_output(result)
        gates[member]=byte_gate(outputs[member],predict(bundle[member],batch,args.seed))
        writer=(BoltzAffinityWriter(data_dir=args.work/'predictions',output_dir=args.work/'predictions') if member=='affinity'
                else BoltzWriter(data_dir=args.work/'processed/structures',output_dir=args.work/'predictions',output_format='mmcif',boltz2=True))
        writer.write_on_batch_end(None,bundle[member],result,None,batch,0,0)
        del result
    if not all(g['accepted'] for g in gates.values()):raise RuntimeError('Original self-repeat is not byte-identical')
    return outputs,batches,gates


def execute(args):
    import torch
    from compressme import share_frozen_parameters,load_boltz2
    from compressme.boltz2_io import install_architecture
    if args.report.exists():raise FileExistsError('Choose a new report')
    if args.command=='build' and args.artifact.exists():raise FileExistsError('Choose a new artifact directory')
    device=device_setup(args.device)
    bundle=original(args.safe_state).to(device)
    manifest=prepare(args.input,args.molecules,args.work,args.seed)
    outputs,batches,self_repeat=baseline(bundle,manifest,args,device)
    if args.command=='build':
        result=share_frozen_parameters(bundle,inplace=True);candidate=result.model
    else:
        del bundle;gc.collect()
        if args.device=='mps':torch.mps.empty_cache()
        candidate=load_boltz2(args.artifact,device=args.device)
    gates={member:byte_gate(outputs[member],predict(candidate[member],batches[member],args.seed))
           for member in ('confidence','affinity')}
    report={'accepted':all(g['accepted'] for g in gates.values()),'device':args.device,'torch':torch.__version__,
        'dtype':'float32','precision_changed':False,'matmul_precision':torch.get_float32_matmul_precision(),
        'seed':args.seed,'self_repeat':self_repeat,'all_outputs':gates,
        'source_revision':'b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc',
        'schedules':{name:dict(candidate[name].predict_args) for name in candidate},
        'scope':'Complete native outputs for one freshly parsed protein+ligand fixture; same-device byte gate, no biological accuracy or speed claim'}
    if args.command=='build':
        report['sharing']=dict(result.report)
        if report['accepted']:
            # CPU serialisation preserves original dtypes. Reapply sharing after
            # a device move because .to(cpu) can split the storage aliases.
            if args.device!='cpu':
                candidate.cpu();result=share_frozen_parameters(candidate,inplace=True)
            result.report['complete_output_validation']=report
            args.artifact.parent.mkdir(parents=True,exist_ok=True)
            temporary=Path(tempfile.mkdtemp(prefix='.'+args.artifact.name+'.',dir=args.artifact.parent))
            try:
                result.save(temporary,packing=False);install_architecture(temporary)
                args.artifact.mkdir();temporary.replace(args.artifact)
            finally:
                if temporary.exists():shutil.rmtree(temporary)
    write_report(args.report,report)
    if not report['accepted']:raise RuntimeError('Candidate failed complete-output byte gate')
    return {'accepted':True,'tensor_outputs':sum(g['tensor_outputs'] for g in gates.values()),'report':str(args.report)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','verify'])
    parser.add_argument('--safe-state',type=Path,required=True);parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--molecules',type=Path,required=True);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--input',type=Path,default=Path(__file__).parent/'tiny_complex.yaml')
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(execute(args),indent=2))

if __name__=='__main__':main()
