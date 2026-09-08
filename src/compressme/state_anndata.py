"""Explicit CPU AnnData encoding for a compiled State SE numerical model.

Uses the pinned, unmodified dataset and collator. Their gene ordering, count
heuristics, stochastic tie breaking and padding remain observable. Only ordered
vocabulary keys are provided; no raw protein vectors are fabricated. The public
name helpers and expression decoder are separate and are not provided here.
"""
from __future__ import annotations
from collections.abc import Mapping
import contextlib
import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import types
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VocabularyKeys(Mapping):
    """Ordered names only; deliberately cannot supply a protein vector."""
    def __init__(self, names):
        self.names = tuple(names)
        if (not self.names or len(set(self.names)) != len(self.names)
                or not all(isinstance(name,str) for name in self.names)):
            raise ValueError('Expected a nonempty ordered unique gene vocabulary')
    def __iter__(self):
        return iter(self.names)
    def __len__(self):
        return len(self.names)
    def __getitem__(self, key):
        raise TypeError('This mapping contains vocabulary metadata only, not raw protein vectors')


def pinned_loader(source_directory):
    root=Path(source_directory).resolve()
    manifest=json.loads((root/'SOURCE.json').read_text())
    for name, expected in manifest['sha256'].items():
        file=(root/name).resolve()
        if root not in file.parents or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'State tokenizer source hash mismatch: {name}')
    identity=hashlib.sha256(json.dumps(manifest['sha256'],sort_keys=True).encode()).hexdigest()[:16]
    package='compressme_state_tokenizer_'+identity
    if package not in sys.modules:
        pkg=types.ModuleType(package); pkg.__path__=[str(root)]; sys.modules[package]=pkg
        saved_path=list(sys.path)
        try:
            with warnings.catch_warnings():
                importlib.import_module(package+'.data.loader')
        except Exception:
            for name in tuple(sys.modules):
                if name==package or name.startswith(package+'.'):
                    del sys.modules[name]
            raise
        finally:
            sys.path[:]=saved_path
    return sys.modules[package+'.data.loader']


class _PinnedObject:
    def __init__(self, source_object, source_directory):
        self._source_directory=str(Path(source_directory).resolve())
        self._class_name=type(source_object).__name__
        self._source_object=source_object
    def __getstate__(self):
        return {'source_directory':self._source_directory,'class_name':self._class_name,
                'source_state':self._source_object.__dict__}
    def __setstate__(self, state):
        self._source_directory=state['source_directory']; self._class_name=state['class_name']
        cls=getattr(pinned_loader(self._source_directory),self._class_name)
        self._source_object=cls.__new__(cls)
        self._source_object.__dict__.update(state['source_state'])
        # Restoring during multiprocessing unpickle happens before PyTorch's
        # worker RNG seeding; imports cannot perturb the collator random stream.


class _PinnedDataset(_PinnedObject, Dataset):
    def __len__(self):
        return len(self._source_object)
    def __getitem__(self, index):
        return self._source_object[index]


class _PinnedCollator(_PinnedObject):
    def __call__(self, batch):
        return self._source_object(batch)


@contextlib.contextmanager
def _seeded(seed):
    if seed is None:
        yield
        return
    py_state=random.getstate(); np_state=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed); np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state)


class StateSEAnnDataEncoder:
    """Encode .h5ad files into 1024 cell plus 10 dataset features on CPU float32.

    `seed=None` preserves the source's use of ambient RNG. An explicit seed
    reproduces worker sampling and restores caller RNG state after the call.
    Model inputs/outputs are never cached. Only the model's parameter-only
    constant cache is reused. Existing output files are never overwritten.
    """
    def __init__(self, model, *, source_directory, gene_names=None):
        self.model=model
        self.source_directory=str(Path(source_directory).resolve())
        self._cfg=copy.deepcopy(model.cfg)
        self.vocabulary=VocabularyKeys(model.gene_names if gene_names is None else gene_names)
        expected=model.cfg.embeddings[model.cfg.embeddings.current].num
        if len(self.vocabulary)!=expected:
            raise ValueError('Gene vocabulary length must match the published model token domain')
        if getattr(model,'gene_names',()) and self.vocabulary.names!=tuple(model.gene_names):
            raise ValueError('Gene vocabulary order must match the model artifact')
        self._check_runtime()
        pinned_loader(self.source_directory)

    def _check_runtime(self):
        if self.model.training or self.model.device.type!='cpu' or next(self.model.parameters()).dtype!=torch.float32:
            raise ValueError('This State SE AnnData adapter is validated only for frozen CPU float32 inference')
        if hasattr(self.model,'_validate_token_runtime'):
            self.model._validate_token_runtime()

    def _auto_detect_gene_column(self, adata):
        # Same precedence and strict-greater tie rule as pinned Inference.
        protein_genes=set(self.vocabulary.keys())
        best_column=None; best_overlap=0
        if hasattr(adata.var,'index'):
            overlap=len(protein_genes.intersection(set(adata.var.index)))
            if overlap>best_overlap:
                best_overlap=overlap
        for col in adata.var.columns:
            col_genes=set(adata.var[col].dropna().astype(str))
            overlap=len(protein_genes.intersection(col_genes))
            if overlap>best_overlap:
                best_overlap=overlap; best_column=col
        return best_column

    def make_dataloader(self, adata, *, dataset_name='inference', batch_size=None, data_dir=None):
        self._check_runtime()
        from omegaconf import OmegaConf
        cfg=OmegaConf.create(OmegaConf.to_container(self._cfg,resolve=True))
        if batch_size is not None:
            cfg.model.batch_size=int(batch_size)
        module=pinned_loader(self.source_directory)
        original=module.create_dataloader(
            cfg,adata=adata,adata_name=dataset_name,shape_dict={dataset_name:adata.shape},
            data_dir=data_dir,shuffle=False,protein_embeds=self.vocabulary,precision=torch.float32,
            gene_column=self._auto_detect_gene_column(adata))
        # Preserve the source DataLoader settings while making dynamic-source
        # objects reconstructible under macOS multiprocessing spawn.
        return DataLoader(_PinnedDataset(original.dataset,self.source_directory),
                          batch_size=original.batch_size,shuffle=False,
                          collate_fn=_PinnedCollator(original.collate_fn,self.source_directory),
                          num_workers=original.num_workers,persistent_workers=original.persistent_workers)

    def encode(self, dataloader):
        self._check_runtime()
        with torch.no_grad():
            for batch in dataloader:
                _,_,_,cell,dataset=self.model._compute_embedding_for_batch(batch)
                yield cell.detach().cpu().float().numpy(), self.model.dataset_embedder(dataset).detach().cpu().float().numpy()

    def encode_adata(self, input_adata_path, output_adata_path=None, emb_key='X_emb',
                     dataset_name=None, batch_size=None, *, seed=None):
        self._check_runtime()
        import anndata
        from scipy.sparse import csr_matrix,issparse
        input_path=Path(input_adata_path)
        output_path=Path(output_adata_path) if output_adata_path is not None else None
        if output_path is not None and (output_path.exists() or output_path.resolve()==input_path.resolve()):
            raise FileExistsError('The AnnData adapter never overwrites an existing output or its input file')
        adata=anndata.read_h5ad(input_path)
        if issparse(adata.X) and not isinstance(adata.X,csr_matrix):
            adata.X=csr_matrix(adata.X)
        with _seeded(seed):
            loader=self.make_dataloader(adata,dataset_name=dataset_name or input_path.stem,
                                        batch_size=batch_size,data_dir=str(input_path.parent))
            cells=[]; datasets=[]
            for cell,dataset in self.encode(loader):
                cells.append(cell)
                if dataset is not None:
                    datasets.append(dataset)
            # Keep upstream behavior for an empty AnnData: concatenate raises.
            result=np.concatenate(cells,axis=0).astype(np.float32)
            if datasets:
                result=np.concatenate((result,np.concatenate(datasets,axis=0).astype(np.float32)),axis=-1)
        if output_path is not None:
            adata.obsm[emb_key]=result
            adata.write_h5ad(output_path)
        return result
