#!/usr/bin/env python3
"""Verify portable State SE AnnData encoding through the public compressme API.

Smoke (no original package, checkpoint or protein dictionary):
  python verify_state_se_anndata.py --artifact ARTIFACT --mode smoke --report RESULT.json

Full upstream parity (actual original safetensors plus a pinned State checkout):
  python verify_state_se_anndata.py --artifact ARTIFACT --mode parity \
      --upstream-source STATE_CHECKOUT --checkpoint ORIGINAL.safetensors --report RESULT.json

The optional --tokenizer-source overrides ARTIFACT/tokenizer_source. Temporary
synthetic .h5ad files are created in a fresh private directory and removed after
verification. --work-dir only selects their parent directory. Existing user
inputs are never modified. These are software equivalence probes using real
pretrained weights, not biological-quality benchmarks.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import torch


def forbid_torch_load(*args, **kwargs):
    raise AssertionError('Original torch checkpoints and protein dictionaries are forbidden in smoke mode')


# Spawned macOS workers import this module before reconstructing the dataset.
# Inherit the guard there as well as in the parent; no upstream package imports
# occur at module level or anywhere in smoke mode.
if os.environ.get('COMPRESSME_VERIFY_NO_TORCH_LOAD') == '1':
    torch.load = forbid_torch_load

from compressme import load_state_se_encoder
from compressme.state_anndata import StateSEAnnDataEncoder, VocabularyKeys
from compressme.validation import compare_outputs


def seed_all(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fingerprint(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def fixtures(directory, names, *, all_cases=True):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    names = list(names[:12])
    counts = np.array([[3,0,1,0,2,0,2,0,1,0,1,0],
                       [0,2,0,0,1,0,1,0,2,0,0,0],
                       [1,0,1,0,3,0,0,0,1,0,0,0]], dtype=np.float32)
    logged = np.log1p(counts)
    def make(value, index=None, columns=None):
        value = ad.AnnData(value,
            obs=pd.DataFrame({'cell_type':['a','b','a'], 'row_value':[8,2,9]},
                             index=['cell3','cell1','cell2']),
            var=pd.DataFrame(columns or {'kind':['gene']*12}, index=index or names))
        value.uns['fixture'] = {'note':'Preserve metadata'}
        value.obsm['existing'] = np.arange(6,dtype=np.float32).reshape(3,2)
        return value
    raw = counts*50
    raw[0,11] = -3
    examples = {'dense_log': make(logged)}
    if all_cases:
        examples.update({
        'csr_log': make(sparse.csr_matrix(logged)),
        'csc_raw_negative': make(sparse.csc_matrix(raw)),
        'column_fallback': make(logged, ['ENSG_TEST_'+str(i) for i in range(12)],
                                {'gene_symbol':names, 'kind':['gene']*12}),
        'partial_index_precedence': make(logged, names[:4]+['unknown'+str(i) for i in range(8)],
                                         {'gene_symbol':names}),
        'duplicate_names': make(logged, names[:10]+[names[0],names[1]]),
        'zero_counts': make(np.zeros_like(logged)),
        })
    for name, value in examples.items():
        value.write_h5ad(directory/(name+'.h5ad'))
    return examples


def collect(loader):
    try:
        return list(loader)
    finally:
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()


def equal_fields(left, right):
    assert len(left) == len(right) == 9
    for a, b in zip(left, right):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert compare_outputs(a, b)['output']['bitwise']


def exception_of(operation):
    try:
        operation()
    except Exception as error:
        return type(error).__name__, str(error)
    raise AssertionError('Expected the documented upstream failure')


def verify_source_checkout(source_root, artifact, tokenizer_source):
    source_emb = source_root/'src/state/emb'
    architecture = json.loads((artifact/'architecture/SOURCE.json').read_text())
    tokenizer = json.loads((tokenizer_source/'SOURCE.json').read_text())
    for metadata in (architecture, tokenizer):
        for name in metadata['unmodified_upstream_files']:
            actual = hashlib.sha256((source_emb/name).read_bytes()).hexdigest()
            if actual != metadata['sha256'][name]:
                raise ValueError(f'Upstream source differs from pinned artifact: {name}')
    return architecture


def original_reference(args, tokenizer_source, vocabulary):
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    metadata = verify_source_checkout(args.upstream_source, args.artifact, tokenizer_source)
    checksum = hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    if digest != metadata['checkpoint_sha256']:
        raise ValueError('Original checkpoint differs from the artifact provenance')
    sys.path.insert(0, str(args.upstream_source/'src'))
    from state.emb.nn.model import StateEmbeddingModel
    from state.emb.inference import Inference
    from state.emb.data import create_dataloader
    config_path = args.config or args.artifact/'architecture/config.json'
    cfg = OmegaConf.load(config_path)
    tensors = load_file(str(args.checkpoint), device='cpu')
    # The released model head is14418-wide; the published YAML says14420.
    # Derive that one dimension from the actual tensor, then strict-load all.
    cfg.dataset[cfg.dataset.current].num_datasets = tensors['dataset_encoder.4.weight'].shape[0]
    for name, value in tensors.items():
        if name.startswith('gene_embedding_layer.'):
            tied = 'encoder.'+name.removeprefix('gene_embedding_layer.')
            assert tied in tensors and torch.equal(value,tensors[tied])
    with torch.random.fork_rng(devices=[]):
        model = StateEmbeddingModel(token_dim=cfg.tokenizer.token_dim,
            d_model=cfg.model.emsize, nhead=cfg.model.nhead, d_hid=cfg.model.d_hid,
            nlayers=cfg.model.nlayers, output_dim=cfg.model.output_dim, dropout=0.0,
            compiled=False, max_lr=cfg.optimizer.max_lr,
            emb_cnt=tensors['pe_embedding.weight'].shape[0],
            emb_size=tensors['pe_embedding.weight'].shape[1], cfg=cfg)
        model.pe_embedding = torch.nn.Embedding.from_pretrained(tensors['pe_embedding.weight'],freeze=True)
    model.load_state_dict(tensors,strict=True)
    model.eval()
    assert model.encoder is model.gene_embedding_layer
    # This input is explicitly vocabulary metadata. Source preprocessing only
    # reads keys; it cannot be used by either model's raw-vector name helper.
    reference = Inference(cfg=copy.deepcopy(cfg),protein_embeds=VocabularyKeys(vocabulary))
    reference.model = model
    return reference, create_dataloader, metadata


def seeded_and_roundtrip_checks(encoder, directory):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from contextlib import ExitStack
    from unittest.mock import patch
    input_path = directory/'dense_log.h5ad'
    seed_all(933)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    # CPU-only encoding must leave unrelated accelerator RNG untouched. These
    # guards test that without requiring a CUDA device or initializing MPS.
    with ExitStack() as guards:
        guards.enter_context(patch('torch.cuda.manual_seed_all',
            side_effect=AssertionError('CPU encoding touched CUDA RNG')))
        if hasattr(torch,'mps') and hasattr(torch.mps,'manual_seed'):
            guards.enter_context(patch('torch.mps.manual_seed',
                side_effect=AssertionError('CPU encoding touched MPS RNG')))
        first = encoder.encode_adata(input_path,batch_size=2,seed=2932)
    after = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    assert before[0] == after[0]
    assert all(np.array_equal(a,b) for a,b in zip(before[1],after[1]))
    assert torch.equal(before[2],after[2])
    second = encoder.encode_adata(input_path,batch_size=2,seed=2932)
    np.testing.assert_array_equal(first,second)
    output_path = directory/'encoded_roundtrip.h5ad'
    written = encoder.encode_adata(input_path,output_path,batch_size=2,seed=2932)
    source, loaded = ad.read_h5ad(input_path), ad.read_h5ad(output_path)
    pd.testing.assert_frame_equal(source.obs,loaded.obs)
    pd.testing.assert_frame_equal(source.var,loaded.var)
    np.testing.assert_array_equal(source.X,loaded.X)
    np.testing.assert_array_equal(source.obsm['existing'],loaded.obsm['existing'])
    np.testing.assert_array_equal(written,loaded.obsm['X_emb'])
    assert source.uns == loaded.uns
    assert exception_of(lambda:encoder.encode_adata(input_path,output_path))[0] == 'FileExistsError'
    assert exception_of(lambda:encoder.encode_adata(input_path,input_path))[0] == 'FileExistsError'
    wrong = list(encoder.model.gene_names)
    wrong[0],wrong[1] = wrong[1],wrong[0]
    assert exception_of(lambda:StateSEAnnDataEncoder(encoder.model,
        source_directory=encoder.source_directory,gene_names=wrong))[0] == 'ValueError'
    return first, {'explicit_seed_reproducible':True,'caller_rng_preserved':True,
                   'unrelated_accelerator_seed_functions_not_called':True,
                   'h5ad_metadata_roundtrip':True,'existing_files_protected':True,
                   'wrong_vocabulary_order_rejected':True}


def smoke(args, directory):
    os.environ['COMPRESSME_VERIFY_NO_TORCH_LOAD'] = '1'
    torch.load = forbid_torch_load
    encoder = load_state_se_encoder(args.artifact,tokenizer_source=args.tokenizer_source)
    fixtures(directory,encoder.model.gene_names,all_cases=False)
    result, checks = seeded_and_roundtrip_checks(encoder,directory)
    expected = json.loads(args.expected.read_text())
    actual = fingerprint(result)
    for name in ('shape','dtype','sha256'):
        assert actual[name] == expected[name], (name,actual[name],expected[name])
    provenance = json.loads((args.artifact/'architecture/SOURCE.json').read_text())
    assert provenance['checkpoint_sha256'] == expected['checkpoint_sha256']
    assert not any(name == 'state' or name.startswith('state.') for name in sys.modules)
    return {'mode':'smoke','fresh_process':True,'torch_load_forbidden_in_parent_and_workers':True,
            'original_state_package_not_imported':True,'public_loader':'compressme.load_state_se_encoder',
            'output':actual,'matches_recorded_original_output_bitwise':True,**checks}


def parity(args,directory):
    import anndata as ad
    import numpy as np
    from scipy import sparse
    tokenizer_source = args.tokenizer_source or args.artifact/'tokenizer_source'
    encoder = load_state_se_encoder(args.artifact,tokenizer_source=args.tokenizer_source)
    reference, create_dataloader, provenance = original_reference(args,tokenizer_source,encoder.model.gene_names)
    examples = fixtures(directory,encoder.model.gene_names)
    report = {'mode':'parity','source_revision':provenance['source_revision'],
              'actual_pretrained_weights':True,'synthetic_fixtures':True,'biological_quality_benchmark':False,
              'device':'cpu','dtype':'float32','workers':1,'pad_length':2048,
              'preprocessing':[],'encoding':[],
              'equality_method':'Explicit contiguous logical value bytes; signed zero distinguished'}
    assert encoder.model.cfg.dataset.pad_length == 2048
    for name, example in examples.items():
        example = example.copy()
        if sparse.issparse(example.X) and not isinstance(example.X,sparse.csr_matrix):
            example.X = sparse.csr_matrix(example.X)
        cfg = copy.deepcopy(reference._vci_conf)
        cfg.model.batch_size = 2
        seed_all(7342)
        left = collect(create_dataloader(cfg,adata=example,adata_name=name,
            shape_dict={name:example.shape},shuffle=False,protein_embeds=reference.protein_embeds,
            precision=torch.float32,gene_column=reference._auto_detect_gene_column(example)))
        seed_all(7342)
        right = collect(encoder.make_dataloader(example,dataset_name=name,batch_size=2))
        assert len(left) == len(right) == 2
        for a,b in zip(left,right):
            equal_fields(a,b)
        report['preprocessing'].append({'case':name,'batches':2,'nine_fields_bitwise_equal':True,
            'selected_column':encoder._auto_detect_gene_column(example),
            'cell_order':torch.cat([batch[3] for batch in right]).tolist()})
        path = directory/(name+'.h5ad')
        seed_all(2932)
        expected = reference.encode_adata(str(path),batch_size=2)
        seed_all(2932)
        actual = encoder.encode_adata(path,batch_size=2)
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
        assert actual.shape == (3,1034) and actual.dtype == np.float32
        byte_equal = (expected.shape == actual.shape and expected.dtype == actual.dtype
                      and np.ascontiguousarray(expected).tobytes() == np.ascontiguousarray(actual).tobytes())
        assert byte_equal, f'Output bytes changed in {name}'
        report['encoding'].append({'case':name,'shape':list(actual.shape),
            'maximum_absolute_error':float(np.abs(expected-actual).max()),
            'bitwise_equal':byte_equal,'tensor_value_bytes_compared':int(actual.nbytes),'allclose_1e5':True})
        print(f'Passed {name}',flush=True)
    _, checks = seeded_and_roundtrip_checks(encoder,directory)
    report.update(checks)
    original = ad.read_h5ad(directory/'dense_log.h5ad')
    invalid = {
        'no_overlap':original.copy(),
        'zero_cells':original[:0,:].copy(),
        'no_padding_candidates':original.copy(),
    }
    invalid['no_overlap'].var_names = ['MISSING'+str(i) for i in range(12)]
    invalid['no_padding_candidates'].X = np.ones(original.shape,dtype=np.float32)
    expected_exceptions = {'no_overlap':'AssertionError','zero_cells':'ValueError','no_padding_candidates':'RuntimeError'}
    for name,value in invalid.items():
        path = directory/(name+'.h5ad')
        value.write_h5ad(path)
        a = exception_of(lambda:reference.encode_adata(str(path),batch_size=2))
        b = exception_of(lambda:encoder.encode_adata(path,batch_size=2))
        assert a[0] == b[0] == expected_exceptions[name]
        if name == 'no_padding_candidates':
            assert 'random_' in a[1] and 'random_' in b[1]
        report[name+'_reference_exception'] = a[0]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--tokenizer-source',type=Path)
    parser.add_argument('--mode',choices=['smoke','parity'],default='smoke')
    parser.add_argument('--expected',type=Path,default=Path(__file__).with_name('state_se_anndata_expected.json'))
    parser.add_argument('--upstream-source',type=Path,help='Pinned State repository root; required only for parity')
    parser.add_argument('--checkpoint',type=Path,help='Original SE-100M safetensors; required only for parity')
    parser.add_argument('--config',type=Path,help='Optional original YAML; defaults to artifact architecture/config.json')
    parser.add_argument('--work-dir',type=Path,help='Parent for disposable generated fixtures')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    if args.mode == 'parity' and (args.upstream_source is None or args.checkpoint is None):
        parser.error('--mode parity requires --upstream-source and --checkpoint')
    if args.threads < 1:
        parser.error('--threads must be positive')
    args.artifact = args.artifact.resolve()
    if args.tokenizer_source is not None:
        args.tokenizer_source = args.tokenizer_source.resolve()
    if args.upstream_source is not None:
        args.upstream_source = args.upstream_source.resolve()
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    with tempfile.TemporaryDirectory(prefix='compressme-state-anndata-',dir=args.work_dir) as temporary:
        result = (smoke if args.mode == 'smoke' else parity)(args,Path(temporary))
    result['torch_version'] = torch.__version__
    result['threads'] = torch.get_num_threads()
    text = json.dumps(result,indent=2)+'\n'
    if args.report is not None:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
