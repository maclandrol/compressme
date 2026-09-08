"""Trusted provider for examples/validate_backends.py; no original code edits."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from validate_backends import BackendCase, BackendSuite
sys.path.insert(0, str(Path(__file__).resolve().parent))
from original import load_state_se as load_original
from lossless_checks import cases, outputs
from compressme import load_state_se


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def provider(device, options):
    artifact = Path(options['artifact'])
    checkpoint = Path(options['checkpoint'])
    architecture = Path(options.get('architecture', artifact/'architecture'))
    source = json.loads((architecture/'SOURCE.json').read_text())
    if sha256(checkpoint) != source['checkpoint_sha256']:
        raise ValueError('Reference State checkpoint hash mismatch')
    for name, expected in source['sha256'].items():
        if sha256(architecture/name) != expected:
            raise ValueError(f'Reference State source hash mismatch: {name}')
    names = json.loads((architecture/'vocabulary.json').read_text())
    selected = [0, 3, 123, 19789]
    def reference(destination):
        model = load_original(checkpoint, config=architecture/'config.json', source_root=architecture).to(destination)
        model.protein_embeds = {names[i]:model.pe_embedding.weight[i].detach().cpu().clone() for i in selected}
        return model
    def candidate(destination):
        with patch('torch.load', side_effect=AssertionError('Pickle loading forbidden during portable reload')):
            return load_state_se(artifact, device=destination)
    def probes(model):
        for index, (description, batch, raw) in enumerate(cases(device)):
            name = f"full-{index:02}-B{description['B']}-T{description['T']}-Q{description['Q']}"
            yield BackendCase(name, lambda model, d, batch=batch, raw=raw: outputs(model,batch,raw))
        yield BackendCase('original-name-helper', lambda model,d: model.get_gene_embedding([names[i] for i in selected]+['not-a-real-gene']))
        # Reading .weight intentionally materializes all original rows. This is
        # a transient snapshot, not a persistent decoded table or mutable alias.
        yield BackendCase('complete-original-weight-snapshot', lambda model,d: model.pe_embedding.weight)
    files = [Path(__file__), Path(__file__).with_name('original.py'), Path(__file__).with_name('lossless_checks.py'),
             artifact/'manifest.json', architecture/'SOURCE.json']
    import compressme.packed_embedding, compressme.state_se_io
    files += [Path(compressme.packed_embedding.__file__), Path(compressme.state_se_io.__file__)]
    return BackendSuite(reference,candidate,probes,
        {'target':'State SE-100M lossless original embedding', 'checkpoint_sha256':source['checkpoint_sha256'],
         'scope':'25 complete model cases; arbitrary raw5120 vectors; all decoder/dataset heads; original gene-name helper; all19790x5120 weight bytes',
         'artifact':str(artifact),'no_precision_change':True,'no_shape_profiles':True},tuple(files))
