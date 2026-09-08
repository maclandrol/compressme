"""Verify every included archive file against the archival hash manifest."""
import argparse,hashlib,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=Path(__file__).parent);args=ap.parse_args();root=args.directory.resolve()
    manifest=json.loads((root/'SHA256.json').read_text());actual={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and p.name!='SHA256.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if set(actual)!=set(manifest['files']):raise ValueError('Archive file inventory differs from manifest')
    for name,p in actual.items():
        if p.is_symlink() or not p.resolve().is_relative_to(root):raise ValueError('Unexpected file link')
        raw=p.read_bytes();expected=manifest['files'][name]
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise ValueError(f'Archive file hash mismatch: {name}')
    print(json.dumps({'verified_files':len(actual),'verified_bytes':sum(r['bytes'] for r in manifest['files'].values()),'manifest_itself_excluded':True},indent=2))
if __name__=='__main__':main()
