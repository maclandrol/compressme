"""Download pinned public Nesso-1 assets into local, ignored working storage."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

NESSO_REVISION = '499ed12b0343918ab01b2519226390cf8eca038a'
ESM_REVISION = '08e4846e537177426273712802403f7ba8261b6c'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('work/nesso/upstream'))
    p.add_argument('--with-preprocessing', action='store_true',
                   help='Also download the 413 MB CCD asset and 2.61 GB ESM-2 checkpoint')
    args = p.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['v1.0.0/hparams.json', 'v1.0.0/model.safetensors', 'LICENSE']
    if args.with_preprocessing:
        names.append('ccd.pkl')
    report = {'nesso_repository': 'recursionpharma/nesso', 'nesso_revision': NESSO_REVISION, 'files': {}}
    for name in names:
        path = Path(hf_hub_download('recursionpharma/nesso', name,
                                  revision=NESSO_REVISION, local_dir=args.output, token=False))
        report['files'][name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    if args.with_preprocessing:
        snapshot = snapshot_download('facebook/esm2_t33_650M_UR50D', revision=ESM_REVISION,
                                     allow_patterns=['*.json', '*.txt', '*.safetensors'], token=False)
        report['esm'] = {'repository': 'facebook/esm2_t33_650M_UR50D', 'revision': ESM_REVISION,
                         'snapshot': snapshot,
                         'weights_sha256': sha256(Path(snapshot) / 'model.safetensors')}
    (args.output / 'download.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
