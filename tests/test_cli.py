import io
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import pytest
from compressme.cli import main


def safefile(path):
    header=json.dumps({'w':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}).encode()
    path.write_bytes(len(header).to_bytes(8,'little')+header+b'\0'*8)
    return path


def test_local_inspect_report_and_repo_without_execution(tmp_path,capsys):
    weights=safefile(tmp_path/'weights.safetensors');repo=tmp_path/'source';repo.mkdir()
    (repo/'model.py').write_text('raise RuntimeError("must not execute")')
    report=tmp_path/'report.json'
    assert main(['inspect',str(weights),'--repo',str(repo),'--report',str(report)])==0
    stdout=json.loads(capsys.readouterr().out)
    assert stdout['tensor_count']==1 and stdout['repository']['python_files_inspected']==1
    assert json.loads(report.read_text())['repository']['files'][0]['file']=='model.py'
    assert main(['inspect',str(weights),'--report',str(report)])==2


def test_inspect_hf_backward_compatible(monkeypatch,capsys):
    import compressme.hub as hub
    calls=[]
    monkeypatch.setattr(hub,'inspect_huggingface',lambda *a,**k:calls.append((a,k)) or {'repo_id':a[0],'revision':'a'*40})
    assert main(['inspect','owner/model','--revision','v1','--filename','weights.safetensors'])==0
    assert calls==[(('owner/model',),{'revision':'v1','filename':'weights.safetensors'})]
    assert json.loads(capsys.readouterr().out)['repo_id']=='owner/model'


def test_targets_unchanged(capsys):
    assert main(['targets'])==0
    assert isinstance(json.loads(capsys.readouterr().out),list)


def test_inspect_and_targets_without_torch(tmp_path):
    weights=safefile(tmp_path/'weights.safetensors')
    code='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','numpy','safetensors','rdkit','boltz','transformers'):
   raise AssertionError('Forbidden import '+fullname)
sys.meta_path.insert(0,Block())
from compressme.cli import main
assert main(['inspect',sys.argv[1]])==0
assert main(['targets'])==0
'''
    result=subprocess.run([sys.executable,'-c',code,str(weights)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_pack_unpack_and_finite_cap(tmp_path,capsys):
    pytest.importorskip('zstandard');pytest.importorskip('numpy')
    source=tmp_path/'source';source.write_bytes(bytes(range(256))*21+b'x')
    packed=tmp_path/'packed';restored=tmp_path/'restored';report=tmp_path/'packing.json'
    assert main(['pack',str(source),str(packed),'--report',str(report)])==0
    assert main(['unpack',str(packed),str(restored),'--max-output-bytes',str(source.stat().st_size)])==0
    assert restored.read_bytes()==source.read_bytes()
    assert main(['unpack',str(packed),str(tmp_path/'small'),'--max-output-bytes','1'])==2
    assert not (tmp_path/'small').exists()
    assert main(['pack',str(source),str(packed)])==2
    assert main(['pack',str(source),str(packed),'--overwrite'])==0


def test_bad_cli_input_has_concise_error(tmp_path,capsys):
    assert main(['inspect',str(tmp_path/'missing.safetensors')])==2
    assert 'Traceback' not in capsys.readouterr().err


def model_fixture(tmp_path,kind='affine'):
    torch=pytest.importorskip('torch');safetensors=pytest.importorskip('safetensors.torch')
    from torch import nn
    factory=tmp_path/'factory.py';weights=tmp_path/'weights.safetensors'
    if kind=='affine':
        model=nn.Sequential(nn.Linear(4,16),nn.Linear(16,3))
        definition='return nn.Sequential(nn.Linear(4,16),nn.Linear(16,3))'
        examples='return [Example((torch.arange(12,dtype=torch.float32).reshape(3,4)/10,))]'
    elif kind=='share':
        class Twin(nn.Module):
            def __init__(self):super().__init__();self.a=nn.Linear(4,4);self.b=nn.Linear(4,4)
            def forward(self,x):return {'left':self.a(x),'right':self.b(x)}
        model=Twin();model.b.load_state_dict(model.a.state_dict())
        definition='return Twin()';examples='return [Example((torch.ones(2,4),))]'
    else:
        model=nn.Sequential(nn.Embedding(8,4),nn.Linear(4,2));model[0].weight.data.zero_()
        definition='return nn.Sequential(nn.Embedding(8,4),nn.Linear(4,2))'
        examples='return [Example((torch.tensor([0,3,7]),))]'
    source='''import torch
from torch import nn
from compressme import Example
class Twin(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Linear(4,4);self.b=nn.Linear(4,4)
 def forward(self,x):return {'left':self.a(x),'right':self.b(x)}
def make_model():
 '''+definition+'''
def make_examples():
 '''+examples+'\n'
    factory.write_text(source);safetensors.save_file(model.state_dict(),str(weights))
    return model,factory,weights


@pytest.mark.parametrize('method',['affine','constant_embeddings','share'])
def test_trusted_factory_semantic_compression_and_reload(tmp_path,capsys,method):
    torch=pytest.importorskip('torch');from compressme import load
    model,factory,weights=model_fixture(tmp_path,method)
    destination=tmp_path/'compact';report=tmp_path/'full.json'
    args=['compress',str(weights),str(destination),'--factory',str(factory)+':make_model',
          '--validation',str(factory)+':make_examples','--method',method,'--report',str(report)]
    assert main(args)==0,capsys.readouterr().err
    summary=json.loads(capsys.readouterr().out)
    assert summary['validation']['accepted'] and summary['device']=='cpu'
    full=json.loads(report.read_text())
    assert full['cli_provenance']['local_python_executed'] and not full['cli_provenance']['precision_changed']
    assert 'float32_matmul_precision' in full['cli_provenance']['execution_settings']
    assert 'cuda_matmul_allow_tf32' in full['cli_provenance']['execution_settings']
    if method=='share':
        assert full['parameters_before']==full['parameters_after']
        assert full['resident_storage_bytes_after']<full['resident_storage_bytes_before']
    else:assert full['parameters_after']<full['parameters_before']
    from compressme.cli import _local_factories
    with _local_factories(str(factory)+':make_model',str(factory)+':make_examples') as (make,examples,_):
        restored=load(make,destination).model
        from compressme.validation import validate
        assert validate(model,restored,list(examples()),relative_tolerance=1e-5,absolute_tolerance=1e-5)['accepted']
    assert main(args)==2


def test_failed_output_gate_never_exports(tmp_path,monkeypatch,capsys):
    torch=pytest.importorskip('torch')
    _,factory,weights=model_fixture(tmp_path)
    from compressme.compiler import CompressionResult
    import compressme.affine as affine
    class Wrong(torch.nn.Module):
        def forward(self,x):return torch.zeros(x.shape[0],3)
    monkeypatch.setattr(affine,'compile_affine',lambda model:CompressionResult(Wrong(),{'method':'test_bad_proposal'}))
    destination=tmp_path/'rejected';report=tmp_path/'rejected.json'
    assert main(['compress',str(weights),str(destination),'--factory',str(factory)+':make_model',
                 '--validation',str(factory)+':make_examples','--report',str(report)])==2
    assert not destination.exists()
    assert not json.loads(report.read_text())['validation']['accepted']


def test_semantic_pickle_refused_before_factory_execution(tmp_path,capsys):
    pytest.importorskip('torch');pytest.importorskip('safetensors')
    marker=tmp_path/'executed';factory=tmp_path/'factory.py'
    factory.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    weights=tmp_path/'weights.ckpt';weights.write_bytes(b'opaque')
    assert main(['compress',str(weights),str(tmp_path/'out'),'--factory',str(factory)+':make_model',
                 '--validation',str(factory)+':make_examples'])==2
    assert not marker.exists()


def test_repo_never_becomes_import_root(tmp_path,capsys):
    _,factory,weights=model_fixture(tmp_path)
    repo=tmp_path/'repository';repo.mkdir();(repo/'only_in_repo.py').write_text('value=1')
    factory.write_text('import only_in_repo\n'+factory.read_text())
    assert main(['compress',str(weights),str(tmp_path/'out'),'--factory',str(factory)+':make_model',
                 '--validation',str(factory)+':make_examples','--repo',str(repo)])==2
    assert not (tmp_path/'out').exists()


def test_module_entrypoint_semantic_subprocess(tmp_path):
    _,factory,weights=model_fixture(tmp_path)
    completed=subprocess.run([sys.executable,'-m','compressme','compress',str(weights),str(tmp_path/'out'),
        '--factory',str(factory)+':make_model','--validation',str(factory)+':make_examples'],
        capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    assert json.loads(completed.stdout)['reduction_observed']


def test_local_factory_module_registration_restored(tmp_path):
    import hashlib,types
    from compressme.cli import _local_factories
    _,factory,_=model_fixture(tmp_path)
    name='_compressme_local_'+hashlib.sha256(str(factory.resolve()).encode()).hexdigest()[:16]
    sentinel=types.ModuleType(name);sys.modules[name]=sentinel
    try:
        with _local_factories(str(factory)+':make_model',str(factory)+':make_examples'):
            assert sys.modules[name] is not sentinel
        assert sys.modules[name] is sentinel
        factory.write_text('raise RuntimeError("failure")')
        with pytest.raises(RuntimeError):
            with _local_factories(str(factory)+':make_model',str(factory)+':make_examples'):pass
        assert sys.modules[name] is sentinel
    finally:sys.modules.pop(name,None)


def test_noop_export_is_labelled(tmp_path,capsys):
    _,factory,weights=model_fixture(tmp_path)
    assert main(['compress',str(weights),str(tmp_path/'out'),'--factory',str(factory)+':make_model',
                 '--validation',str(factory)+':make_examples','--method','constant_embeddings'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['status']=='accepted_no_reduction' and not report['reduction_observed']


def test_share_requires_bytes_even_when_numerical_gate_passes(tmp_path,capsys,monkeypatch):
    torch=pytest.importorskip('torch')
    from compressme.compiler import CompressionResult
    import compressme.sharing as sharing
    _,factory,weights=model_fixture(tmp_path)
    class OneUlp(torch.nn.Module):
        def __init__(self,original):super().__init__();self.original=original
        def forward(self,x):
            result=self.original(x)
            return torch.nextafter(result,torch.full_like(result,float('inf')))
    monkeypatch.setattr(sharing,'share_frozen_parameters',lambda model:CompressionResult(OneUlp(model),{'method':'share'}))
    assert main(['compress',str(weights),str(tmp_path/'out'),'--factory',str(factory)+':make_model',
                 '--validation',str(factory)+':make_examples','--method','share'])==2
    report=json.loads(capsys.readouterr().out)
    assert report['validation']['required_bitwise'] and not report['validation']['accepted']
    assert not (tmp_path/'out').exists()
