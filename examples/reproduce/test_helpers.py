"""Small reproducibility-helper checks; no downloads, upstream models or GPUs."""
import argparse,hashlib,json,os,pickle,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from common import fetch,verify,write_report


def test_pinned_fetch_does_not_execute_and_refuses_changed_cache(tmp_path):
    source=tmp_path/'source.py';source.write_text('raise RuntimeError("never execute")')
    digest=hashlib.sha256(source.read_bytes()).hexdigest();output=tmp_path/'cache/source.py'
    fetch(source.as_uri(),output,digest,size=source.stat().st_size)
    assert output.read_bytes()==source.read_bytes()
    output.write_text('changed')
    with pytest.raises(ValueError):fetch(source.as_uri(),output,digest)


def test_download_size_limit_leaves_no_destination(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'1234');output=tmp_path/'dest'
    with pytest.raises(ValueError):fetch(source.as_uri(),output,'0'*64,size=3)
    assert not output.exists() and not list(tmp_path.glob('.dest.*'))


def test_report_never_overwrites(tmp_path):
    path=tmp_path/'report.json';write_report(path,{'x':1})
    with pytest.raises(FileExistsError):write_report(path,{'x':2})
    assert json.loads(path.read_text())=={'x':1}


def test_checkpoint_reducer_is_inert(tmp_path):
    import static_metadata
    marker=tmp_path/'ran'
    class Payload:
        def __reduce__(self):return os.system,('touch '+str(marker),)
    result=static_metadata.parse(pickle.dumps(Payload(),protocol=2))
    assert isinstance(result,static_metadata.Box) and not marker.exists()


def test_native_byte_gate_retains_flags_and_signed_zero():
    torch=pytest.importorskip('torch');import boltz2
    assert boltz2.byte_gate({'x':torch.tensor([0.]),'exception':False},
                           {'x':torch.tensor([-0.]),'exception':False})['accepted'] is False
    with pytest.raises(ValueError):
        boltz2.byte_gate({'x':torch.ones(1),'exception':False},{'x':torch.ones(1),'exception':True})


def test_build_orchestration_exports_and_reloads_general_sharing(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');pytest.importorskip('safetensors');import boltz2
    from compressme import load
    def factory():
        a=torch.nn.Linear(2,2);b=torch.nn.Linear(2,2);b.load_state_dict(a.state_dict())
        a.predict_args={};b.predict_args={}
        return torch.nn.ModuleDict({'confidence':a,'affinity':b}).eval().requires_grad_(False)
    def baseline(model,manifest,args,device):
        batches={k:torch.ones(3,2) for k in model}
        outputs={k:boltz2.clone_output(model[k](batches[k])) for k in model}
        return outputs,batches,{k:{'accepted':True} for k in model}
    monkeypatch.setattr(boltz2,'original',lambda path:factory())
    monkeypatch.setattr(boltz2,'prepare',lambda *args:None)
    monkeypatch.setattr(boltz2,'baseline',baseline)
    monkeypatch.setattr(boltz2,'predict',lambda model,batch,seed:model(batch))
    args=argparse.Namespace(command='build',safe_state=tmp_path,artifact=tmp_path/'artifact',
        molecules=tmp_path,work=tmp_path/'work',input=tmp_path/'input',device='cpu',seed=0,report=tmp_path/'result.json')
    assert boltz2.execute(args)['accepted']
    report=json.loads(args.report.read_text())
    assert report['sharing']['saved_resident_storage_bytes']>0
    restored=load(factory,args.artifact).model
    assert torch.equal(restored['confidence'](torch.ones(3,2)),restored['affinity'](torch.ones(3,2)))


def test_reported_moljepa_protocol_has_exact_workloads_and_call_settings():
    root=Path(__file__).resolve().parents[2]
    protocol=json.loads((Path(__file__).parent/'moljepa-benchmark.json').read_text())
    source=root/'benchmarks/moljepa_runtime_macos.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['source_report_sha256']
    assert protocol['workloads']==json.loads(source.read_text())['workloads']
    assert protocol['warmups']==10 and protocol['rounds']==20
    assert protocol['return_attn'] is False and protocol['shuffle_seed_base']==4561
    assert protocol['variants']==['original','compressed','fast','production']


def test_benchmark_orchestration_uses_reported_variants_workloads_and_calls(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    import moljepa,compressme,compressme.smiles_runtime
    calls={};created={};gates=[]
    class Probe(torch.nn.Module):
        def __init__(self,name):
            super().__init__();self.name=name;self.p=torch.nn.Parameter(torch.ones(1),requires_grad=False)
            calls[name]=[];created[name]=self
        def forward(self,smiles,return_attn=False):
            calls[self.name].append((list(smiles),return_attn));return self.p
    monkeypatch.setattr(moljepa,'original',lambda directory:Probe('original'))
    monkeypatch.setattr(compressme,'load_moljepa',lambda *args,**kwargs:Probe('compressed'))
    monkeypatch.setattr(compressme.smiles_runtime,'accelerate_smiles',lambda model,metal:Probe('production' if metal else 'fast'))
    def validate(reference,candidate,examples,**kwargs):
        gates.append((candidate.name,len(examples),[e.kwargs['return_attn'] for e in examples]))
        return {'accepted':True}
    monkeypatch.setattr(compressme,'validate',validate)
    args=argparse.Namespace(original=tmp_path,artifact=tmp_path,device='cpu',rounds=2,report=tmp_path/'timings.json')
    assert moljepa.benchmark_artifact(args)['accepted']
    protocol=json.loads((Path(__file__).parent/'moljepa-benchmark.json').read_text())
    for name in protocol['variants']:
        assert len(calls[name])==36
        assert all(flag is False for _,flag in calls[name])
        assert [calls[name][i*12][0] for i in range(3)]==list(protocol['workloads'].values())
    assert len(gates)==4 and all(n==32 and set(flags)=={False,True} for _,n,flags in gates)
    assert not json.loads(args.report.read_text())['recorded_round_count']
