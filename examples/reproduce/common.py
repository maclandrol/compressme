"""Small first-party reproduction utilities; no model imports at module import."""
from pathlib import Path
import hashlib,json,os,tempfile,urllib.request


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def verify(path,expected,size=None):
    path=Path(path)
    if not path.is_file() or (size is not None and path.stat().st_size!=size) or sha256(path)!=expected:
        raise ValueError('Pinned file hash/size mismatch: '+str(path))
    return path


def fetch(url,path,expected,size=None,maximum=16<<20):
    path=Path(path)
    if path.exists():return verify(path,expected,size)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        total=0
        with os.fdopen(fd,'wb') as output,urllib.request.urlopen(url,timeout=120) as response:
            while block:=response.read(1<<20):
                total+=len(block)
                if total>(size if size is not None else maximum):raise ValueError('Download exceeds pinned size/budget')
                output.write(block)
        verify(temporary,expected,size)
        os.link(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return path


def write_report(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def device_setup(name,threads=4):
    import torch
    if name=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable')
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(threads);torch.set_float32_matmul_precision('highest')
    return torch.device(name)


def synchronize(device):
    import torch
    if str(device)=='mps':torch.mps.synchronize()
    elif str(device).startswith('cuda'):torch.cuda.synchronize(device)
