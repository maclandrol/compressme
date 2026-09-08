"""Portable standalone full-file codec audit; no torch or original model import."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import sys
import time


def sha256_file(path, block_bytes=1 << 20):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def token(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def run(args):
    source = args.source.resolve()
    codec_path = args.codec_dir.resolve() / 'packing_files.py'
    if sha256_file(codec_path) != args.codec_sha256:
        raise ValueError('Codec source SHA256 mismatch')
    initial = token(source)
    if initial[2] != args.expected_size:
        raise ValueError('Source file size mismatch')
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh output directory: ' + str(args.output_dir))
    args.output_dir.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('audited_packing_files', codec_path)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    import numpy
    import zstandard

    report = {'status': 'running', 'source_file': source.name,
              'source_bytes': initial[2], 'expected_source_sha256': args.expected_sha256,
              'codec_file': codec_path.name, 'codec_sha256': args.codec_sha256,
              'scope': 'lossless disk/transport bytes; no tensor interpretation, resident-model reduction or inference-speed claim',
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'numpy': numpy.__version__, 'zstandard': zstandard.__version__,
                              'zstandard_backend': zstandard.backend},
              'timing_scope': 'diagnostic wall times; other model validation may run concurrently',
              'stages': {}}
    started = time.perf_counter()
    stage = time.perf_counter()
    digest = sha256_file(source)
    if digest != args.expected_sha256:
        raise ValueError('Original source SHA256 mismatch')
    report['stages']['source_verification'] = {'seconds': time.perf_counter()-stage, 'sha256': digest}
    print('Original source SHA256 verified', flush=True)

    packed = args.output_dir / 'model.cmprpack'
    restored = args.output_dir / 'model.restored.safetensors'
    stage = time.perf_counter()
    packed_result = codec.pack_file(source, packed, level=args.level)
    if packed_result['sha256'] != args.expected_sha256:
        raise AssertionError('Packing read different source bytes')
    report['stages']['packing'] = dict(packed_result, seconds=time.perf_counter()-stage,
                                       cumulative_peak_rss_bytes=peak_rss_bytes())
    print('Packed', packed_result['packed_bytes'], 'bytes; peak RSS', peak_rss_bytes(), flush=True)

    stage = time.perf_counter()
    unpacked_result = codec.unpack_file(packed, restored, max_output_bytes=args.expected_size)
    if unpacked_result['sha256'] != args.expected_sha256:
        raise AssertionError('Unpacking reconstruction SHA256 mismatch')
    report['stages']['unpacking'] = dict(unpacked_result, seconds=time.perf_counter()-stage,
                                         cumulative_peak_rss_bytes=peak_rss_bytes())
    print('Restored complete file and verified codec SHA256', flush=True)

    stage = time.perf_counter()
    checked = 0
    source_digest, restored_digest = hashlib.sha256(), hashlib.sha256()
    with source.open('rb') as original, restored.open('rb') as candidate:
        while True:
            left, right = original.read(1 << 20), candidate.read(1 << 20)
            if left != right:
                raise AssertionError('Byte mismatch in block beginning at offset ' + str(checked))
            if not left:
                break
            checked += len(left)
            source_digest.update(left)
            restored_digest.update(right)
    if checked != args.expected_size or token(source) != initial:
        raise AssertionError('Source size or state changed during audit')
    if source_digest.hexdigest() != args.expected_sha256 or restored_digest.hexdigest() != args.expected_sha256:
        raise AssertionError('Independent streaming comparison hashes disagree')
    report['stages']['full_byte_comparison'] = {
        'seconds': time.perf_counter()-stage, 'all_bytes_equal': True,
        'bytes_compared': checked, 'source_sha256': source_digest.hexdigest(),
        'restored_sha256': restored_digest.hexdigest(), 'source_stat_unchanged': True}

    stage = time.perf_counter()
    packed_sha = sha256_file(packed)
    report['stages']['packed_hash'] = {'seconds': time.perf_counter()-stage, 'sha256': packed_sha}
    report['status'] = 'passed'
    report['packing_level'] = args.level
    report['packed_fraction_of_source'] = packed_result['packed_bytes'] / checked
    report['disk_bytes_saved'] = checked-packed_result['packed_bytes']
    report['disk_percent_saved'] = 100*(1-report['packed_fraction_of_source'])
    report['peak_process_rss_bytes'] = peak_rss_bytes()
    report['rss_measurement'] = 'resource.getrusage(RUSAGE_SELF).ru_maxrss; bytes on macOS, KiB converted to bytes elsewhere; fresh standalone process; includes imports and codec workspace'
    report['total_diagnostic_seconds'] = time.perf_counter()-started
    report['torch_imported'] = 'torch' in sys.modules
    report['outputs'] = {'packed': {'file': packed.name, 'bytes': packed.stat().st_size, 'sha256': packed_sha},
                         'restored': {'file': restored.name, 'bytes': restored.stat().st_size, 'sha256': args.expected_sha256}}
    report_path = args.output_dir / 'full-result.json'
    with report_path.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    manifest = {'full-result.json': sha256_file(report_path),
                packed.name: packed_sha, restored.name: args.expected_sha256,
                'audit_script_sha256': sha256_file(Path(__file__))}
    with (args.output_dir / 'hashes.json').open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'packed_bytes': packed.stat().st_size,
                      'disk_percent_saved': report['disk_percent_saved'],
                      'peak_process_rss_bytes': report['peak_process_rss_bytes'],
                      'all_bytes_equal': True, 'report': str(report_path)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--codec-dir', type=Path, required=True)
    parser.add_argument('--codec-sha256', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-size', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--level', type=int, default=9)
    run(parser.parse_args())
