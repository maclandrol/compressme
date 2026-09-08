"""Build original State SE with lossless block embedding storage, CPU or MPS."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from compressme import CompressionResult
from compressme.packed_embedding import PackedFrozenEmbedding
from compressme.finite_lookup import _resident_storage_bytes
from original import load_state_se


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build(checkpoint, architecture, output, *, block_rows=16):
    checkpoint, architecture, output = map(Path, (checkpoint, architecture, output))
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((architecture / 'SOURCE.json').read_text())
    if sha256(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Original State checkpoint hash mismatch')
    for name, expected in manifest['sha256'].items():
        if sha256(architecture / name) != expected:
            raise ValueError(f'Pinned State architecture hash mismatch: {name}')
    torch.set_num_threads(4)
    model = load_state_se(checkpoint, config=architecture/'config.json', source_root=architecture).eval().requires_grad_(False)
    before = _resident_storage_bytes(model)
    source = model.pe_embedding
    packed = PackedFrozenEmbedding(source, block_rows=block_rows)
    # Exhaustive reconstruction gate; no model output or calibration cache.
    with torch.no_grad():
        for start in range(0, source.num_embeddings, 257):
            ids = torch.arange(start, min(start + 257, source.num_embeddings))
            a, b = source(ids), packed(ids)
            if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                raise ValueError('Lossless embedding reconstruction failed')
    model.pe_embedding = packed
    after = _resident_storage_bytes(model)
    if after >= before:
        raise ValueError('This block encoding does not reduce total resident state')
    report = {'target': 'State SE-100M', 'representation': 'lossless_original_embedding',
              'method': 'Original frozen embedding rows, independently byte-shuffled and Zstandard-compressed',
              'validated_devices': ['cpu', 'mps'], 'validated_dtype': 'float32',
              'checkpoint_sha256': sha256(checkpoint), 'known_genes': source.num_embeddings,
              'original_registered_bytes': before, 'candidate_registered_bytes': after,
              'packed_host_bytes': packed.payload.numel() + packed.offsets.numel()*8,
              'persistent_decoded_cache_bytes': 0, 'block_rows': block_rows,
              'contract': 'Original State numerical APIs and raw5120-vector encoder remain; frozen inference only. Original name helper receives exact decoded raw rows.',
              'verification': 'Every19790 original embedding row is byte-checked during this build. Run the provided backend verifier for complete model and portable-reload evidence.',
              'runtime': 'CPU decompression and host-to-device copies add overhead; no inference acceleration claim.',
              'packed_embedding_source_sha256': sha256(Path(__import__('compressme.packed_embedding', fromlist=['__file__']).__file__))}
    CompressionResult(model, report).save(output)
    shutil.copytree(architecture, output/'architecture', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (output/'build-result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--architecture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--block-rows', type=int, default=16)
    args = parser.parse_args()
    build(args.checkpoint, args.architecture, args.output, block_rows=args.block_rows)
