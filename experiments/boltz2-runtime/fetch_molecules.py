"""Fetch pinned upstream molecular data; extract only its canonical molecule files."""
import ast
import hashlib
import importlib.util
import json
import pathlib
import pickletools
import tarfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
REVISION = "6fdef46d763fee7fbb83ca5501ccceff43b85607"
SHA256 = "39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7"
SIZE = 1855662080
URL = f"https://huggingface.co/boltz-community/boltz-2/resolve/{REVISION}/mols.tar"

def main():
    archive = ROOT / "mols.tar"
    if not archive.exists():
        digest = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(URL, timeout=90) as src, (ROOT / "mols.tar.partial").open("wb") as dst:
            while chunk := src.read(4 * 1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
                count += len(chunk)
                if count % (64 * 1024 * 1024) == 0:
                    print(f"Molecular archive: {count / SIZE:.1%}", flush=True)
        if count != SIZE or digest.hexdigest() != SHA256:
            raise RuntimeError(f"Archive integrity mismatch: {count} {digest.hexdigest()}")
        (ROOT / "mols.tar.partial").rename(archive)
    else:
        digest = hashlib.sha256()
        with archive.open("rb") as src:
            while chunk := src.read(4 * 1024 * 1024):
                digest.update(chunk)
        if archive.stat().st_size != SIZE or digest.hexdigest() != SHA256:
            raise RuntimeError("Existing archive failed published SHA256")
    source = pathlib.Path(importlib.util.find_spec("boltz.data.const").origin)
    tree = ast.parse(source.read_text())
    canonical = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "canonical_tokens" for t in n.targets))
    wanted = {f"mols/{token}.pkl" for token in canonical}
    output = ROOT / "mols"
    output.mkdir(exist_ok=True)
    records = []
    with tarfile.open(archive, "r:") as src:
        for member in src:
            if member.name not in wanted:
                continue
            if not member.isfile() or member.size > 2 * 1024 * 1024:
                raise ValueError("Unexpected canonical molecular asset")
            data = src.extractfile(member).read()
            path = output / pathlib.PurePosixPath(member.name).name
            path.write_bytes(data)
            globals_ = [arg for op, arg, _ in pickletools.genops(data) if op.name == "GLOBAL"]
            records.append({"name": member.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "pickle_globals": globals_})
    if {r["name"] for r in records} != wanted:
        raise RuntimeError("Missing canonical assets")
    report = {"url": URL, "revision": REVISION, "archive_sha256": SHA256, "archive_bytes": SIZE, "assets": records}
    (ROOT / "molecular-assets.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified archive and extracted {len(records)} canonical molecule assets", flush=True)

if __name__ == "__main__":
    main()
