import json
import os
import pickle
from pathlib import Path
import struct
import zipfile
import pytest
from compressme.checkpoint import inspect_checkpoint,inspect_safetensors,inspect_repository


def safefile(path,header,payload=b''):
    raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    path.write_bytes(len(raw).to_bytes(8,'little')+raw+payload)
    return path


def test_safetensors_metadata_without_value_reads(tmp_path):
    path=safefile(tmp_path/'model.safetensors',{'w':{'dtype':'F32','shape':[2,3],'data_offsets':[0,24]},'z':{'dtype':'I64','shape':[0],'data_offsets':[24,24]},'__metadata__':{'source':'fixture'}},b'\0'*24)
    result=inspect_safetensors(path,include_tensors=True,include_hash=True)
    assert result['tensor_count']==2 and result['tensor_bytes']==24
    assert result['stored_elements']==6 and result['metadata_keys']==['source']
    assert not result['tensor_values_read'] and not result['code_executed']
    assert len(result['sha256'])==64


@pytest.mark.parametrize('header,payload',[
 ({'w':{'dtype':'F32','shape':[True],'data_offsets':[0,4]}},b'\0'*4),
 ({'w':{'dtype':'F32','shape':[-1],'data_offsets':[0,4]}},b'\0'*4),
 ({'w':{'dtype':'F32','shape':[2],'data_offsets':[0,4]}},b'\0'*4),
 ({'w':{'dtype':'F32','shape':[1],'data_offsets':[1,5]}},b'\0'*5),
 ({'w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},b'\0'*5),
 ({'w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]},'v':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},b'\0'*4),
 ({'__metadata__':{'bad':3}},b''),
 ({'w':{'dtype':'F32','shape':[10**100,10**100],'data_offsets':[0,4]}},b'\0'*4),
])
def test_invalid_safetensors_rejected(tmp_path,header,payload):
    with pytest.raises(ValueError):inspect_safetensors(safefile(tmp_path/'model.safetensors',header,payload))


def test_duplicate_and_huge_header_rejected(tmp_path):
    path=tmp_path/'model.safetensors';raw=b'{"x":{},"x":{}}';path.write_bytes(len(raw).to_bytes(8,'little')+raw)
    with pytest.raises(ValueError,match='Duplicate'):inspect_safetensors(path)
    path.write_bytes(((16<<20)+1).to_bytes(8,'little'))
    with pytest.raises(ValueError,match='header'):inspect_safetensors(path)


def test_unknown_dtype_remains_explicitly_unverified(tmp_path):
    result=inspect_safetensors(safefile(tmp_path/'model.safetensors',{'w':{'dtype':'FUTURE','shape':[1],'data_offsets':[0,1]}},b'x'))
    assert result['unverified_dtypes']==['FUTURE']


def test_pickle_is_disassembled_never_executed(tmp_path):
    marker=tmp_path/'executed'
    class Dangerous:
        def __reduce__(self):return (os.system,('touch '+str(marker),))
    path=tmp_path/'model.ckpt'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('archive/data.pkl',pickle.dumps(Dangerous(),protocol=2))
        archive.writestr('archive/data/0',b'\0'*32)
    result=inspect_checkpoint(path)
    assert not marker.exists() and not result['pickle_deserialized']
    assert result['archive_entries']==2
    assert result['pickle_metadata']['status']=='disassembled_only'


def test_legacy_checkpoint_stays_opaque(tmp_path):
    path=tmp_path/'model.pt';path.write_bytes(pickle.dumps({'a':1}))
    assert inspect_checkpoint(path)['kind']=='opaque_checkpoint'


def test_zip_directory_cap_precedes_zipfile_allocation(tmp_path,monkeypatch):
    path=tmp_path/'model.ckpt'
    path.write_bytes(b'PK\x03\x04'+struct.pack('<4s4H2LH',b'PK\x05\x06',0,0,1,1,(16<<20)+1,0,0))
    monkeypatch.setattr(zipfile,'ZipFile',lambda *a,**k:pytest.fail('Archive allocation must not run'))
    with pytest.raises(ValueError,match='directory'):inspect_checkpoint(path)


def test_local_source_inventory_does_not_execute(tmp_path):
    marker=tmp_path/'executed';(tmp_path/'model.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\nclass Model:\n def forward(self,x):\n  return self.weight @ x\n')
    (tmp_path/'large.py').write_text('x'*2000)
    result=inspect_repository(tmp_path,max_file_bytes=1024)
    assert not marker.exists() and not result['code_executed']
    assert result['python_files_inspected']==1 and result['truncated']
    assert result['counts']['weight_attribute_accesses']==1


def test_repo_url_is_only_provenance():
    result=inspect_repository('https://github.com/example/model')
    assert not result['fetched'] and not result['code_executed']
    for value in ('https://user:password@example.com/repo','https://example.com/repo?token=secret','ftp://example.com/repo'):
        with pytest.raises(ValueError):inspect_repository(value)


def test_source_symlinks_not_followed(tmp_path):
    outside=tmp_path/'outside';outside.mkdir();(outside/'ignored.py').write_text('class Hidden:pass')
    source=tmp_path/'source';source.mkdir();(source/'linked').symlink_to(outside,target_is_directory=True)
    (source/'linked.py').symlink_to(outside/'ignored.py')
    assert inspect_repository(source)['python_files_inspected']==0


def test_source_inventory_skips_named_virtual_environments(tmp_path):
    from compressme.checkpoint import inspect_repository
    for name in ('.venv', '.venv-boltz', '.venv-state', '.venv_gpu'):
        environment=tmp_path/name
        environment.mkdir()
        (environment/'dependency.py').write_text('class ShouldNotCount: pass')
    (tmp_path/'model.py').write_text('class ActualModel: pass')
    report=inspect_repository(tmp_path,max_files=1)
    assert [item['file'] for item in report['files']]==['model.py']
    assert report['counts']['classes']==1
