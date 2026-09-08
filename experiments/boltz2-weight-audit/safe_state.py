"""Read this audit's pure safetensors state mapping, with no checkpoint pickle."""
import hashlib,json
from pathlib import Path

def load_tensor_state(directory,checkpoint,*,verify_file=True):
    """Return the original named tensor state and JSON hyperparameters.

    Every tensor is a CPU mmap view. Equal byte blobs share backing storage;
    source shapes and original state names are reconstructed. This does not
    instantiate Boltz or promise that model.load_state_dict will retain sharing.
    The caller should construct the pinned class/config and load strict=True.
    """
    from safetensors import safe_open
    root=Path(directory).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if manifest['format']!='compressme-boltz2-shared-state-audit-v1':raise ValueError('Unknown manifest format')
    if checkpoint not in manifest['models']:raise ValueError('Unknown checkpoint name')
    path=(root/manifest['tensor_file']).resolve()
    if path.parent!=root:raise ValueError('Tensor file must be inside artifact directory')
    if verify_file:
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            while block:=handle.read(4*1024*1024):digest.update(block)
        if digest.hexdigest()!=manifest['tensor_file_sha256']:raise ValueError('Tensor file hash mismatch')
    spec=manifest['models'][checkpoint];config_path=(root/spec['hyper_parameters']).resolve()
    if config_path.parent!=root:raise ValueError('Config must be inside artifact directory')
    raw=config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['source_hparams_sha256'][checkpoint]:raise ValueError('Config hash mismatch')
    state={};cache={}
    with safe_open(path,framework='pt',device='cpu') as handle:
        for name,binding in spec['bindings'].items():
            key=binding['tensor']
            if key not in cache:cache[key]=handle.get_tensor(key)
            state[name]=cache[key].reshape(binding['shape'])
    return state,json.loads(raw)
