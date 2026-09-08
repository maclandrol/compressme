"""Explicit frozen gene-ID adapter for State SE's complete numerical outputs.

The projected_lookup is a generic module mapping (gene_ids, normalize=bool) to
the shared encoder's affine output. State-specific code only connects its two
gene branches and fixed CLS/dataset rows to the unchanged downstream network.

Contract: the upstream numeric gene-ID batch API is preserved. By default the
original encoder is retained, preserving arbitrary raw-vector forward and
gene_embedding_layer inputs too. The full raw pe_embedding.weight cannot be
reconstructed. The StateModel gene-name helper needs the original protein
dictionary supplied explicitly to retain its raw-vector zero-sum guard.
Training is outside this frozen adapter.
The upstream attention ignores its mask; this adapter preserves that behavior.
"""
from __future__ import annotations
import copy
import math
from types import MappingProxyType
import torch
from torch import nn
from torch.nn import functional as F


class FinalBranchLookup(nn.Module):
    """Two generic finite-domain results for a shared normalized/raw call site."""
    def __init__(self, raw, normalized):
        super().__init__()
        self.raw = raw
        self.normalized = normalized
        self.eval()

    def forward(self, ids, normalize=False):
        if type(normalize) is not bool:
            raise ValueError("Expected an explicit normalization branch")
        return self.normalized(ids) if normalize else self.raw(ids)


class StateSETokenAdapter(nn.Module):
    def __init__(self, source, projected_lookup, *, copy_downstream=True,
                 preserve_raw_forward=True, retain_protein_dictionary=False):
        super().__init__()
        if source.training or source.dropout != 0.0:
            raise ValueError("State SE adapter requires deterministic eval with inference dropout=0")
        if source.encoder is not source.gene_embedding_layer:
            raise ValueError("Expected the shared upstream gene encoder")
        if ([type(layer) for layer in source.encoder] != [nn.Linear, nn.LayerNorm, nn.SiLU]
                or len(source.encoder[1].normalized_shape) != 1):
            raise ValueError("Unexpected State SE encoder topology")
        self.projected_lookup = projected_lookup
        self.preserve_raw_forward = preserve_raw_forward
        if preserve_raw_forward:
            self.encoder = copy.deepcopy(source.encoder)
            self.gene_embedding_layer = self.encoder
            self.post_projection = self.encoder[1:]
        else:
            self.post_projection = copy.deepcopy(source.encoder[1:])
        self.cls_token = nn.Parameter(source.cls_token.detach().clone(), requires_grad=False)
        self.dataset_token = (nn.Parameter(source.dataset_token.detach().clone(), requires_grad=False)
                              if source.dataset_token is not None else None)
        self.dropout = source.dropout
        self.d_model = source.d_model
        self.supported_token_devices = ("cpu",)
        self.supported_token_dtypes = (torch.float32,)
        self._gene_names = ()
        self._gene_name_index = {}
        self.cfg = copy.deepcopy(source.cfg)
        self.z_dim = source.z_dim
        self.z_dim_rd = source.z_dim_rd
        self.z_dim_ds = source.z_dim_ds
        self._has_dataset_token = source.dataset_token is not None
        self.constant_mode = "precomputed"
        self._shape_constant_cache = None
        with torch.no_grad():
            cls = source.encoder(source.cls_token).detach().clone()
            dataset = (source.encoder(source.dataset_token).detach().clone()
                       if source.dataset_token is not None else None)
        self.register_buffer("encoded_cls", cls)
        self.register_buffer("encoded_dataset", dataset)
        for name in ("transformer_encoder", "decoder", "binary_decoder", "count_encoder", "bin_encoder",
                     "dataset_embedder", "dataset_encoder", "dataset_loss"):
            if hasattr(source, name):
                module = getattr(source, name)
                setattr(self, name, copy.deepcopy(module) if copy_downstream else module)
        self._resize_batch_function = type(source).resize_batch
        self._original_forward_function = type(source).forward
        self._original_gene_function = type(source).get_gene_embedding
        self.protein_embeds = source.protein_embeds if retain_protein_dictionary else None
        self.eval()
        self.requires_grad_(False)

    @property
    def device(self):
        return self.encoded_cls.device

    @property
    def gene_names(self):
        return self._gene_names

    @property
    def gene_to_index(self):
        return MappingProxyType(self._gene_name_index)

    def set_gene_vocabulary(self, names):
        names = tuple(names)
        expected = self.cfg.embeddings[self.cfg.embeddings.current].num
        if len(names) != expected or len(set(names)) != len(names) or not all(isinstance(n, str) for n in names):
            raise ValueError("Gene vocabulary must contain one unique name for every published token row")
        self._gene_names = names
        self._gene_name_index = {name: index for index, name in enumerate(names)}

    def train(self, mode=True):
        if mode:
            raise ValueError("Compiled State gene lookup is frozen inference only")
        return super().train(False)

    def forward(self, src, mask, counts=None, dataset_nums=None):
        if not self.preserve_raw_forward:
            raise ValueError("Raw embedding forward is outside this gene-ID adapter; use forward_gene_ids or the original numeric batch API")
        return self._original_forward_function(self, src, mask, counts=counts, dataset_nums=dataset_nums)

    def get_gene_embedding(self, genes):
        if not self.preserve_raw_forward or self.protein_embeds is None:
            raise ValueError("The upstream raw-vector zero-sum guard requires the original protein dictionary supplied explicitly; use get_gene_embedding_by_id")
        return self._original_gene_function(self, genes)

    def get_gene_embedding_by_id(self, ids):
        self._validate_token_runtime()
        return self.post_projection(self.projected_lookup(ids, normalize=False))

    def _validate_token_runtime(self):
        if self.device.type not in self.supported_token_devices or self.encoded_cls.dtype not in self.supported_token_dtypes:
            raise ValueError("State SE token lookup is validated only for CPU float32; this device/dtype did not pass its numerical gate")

    def forward_gene_ids(self, src, mask, counts=None, dataset_nums=None):
        self._validate_token_runtime()
        if self.training or any(m.training for m in self.modules()):
            raise ValueError("Compiled State gene lookup is frozen inference only")
        if src.ndim != 2 or src.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected the upstream two-dimensional integer gene-ID sentences")
        src = src.to(self.device)
        encoded = self.post_projection(self.projected_lookup(src, normalize=True))
        # Empty batches contain no constant values to encode. Preserve the
        # upstream empty-output shapes without asking the cache for zero rows.
        if self.constant_mode == "source_shape" and src.shape[0] != 0:
            if not self.preserve_raw_forward:
                raise ValueError("Shape-matched constants require the retained source encoder")
            if self._shape_constant_cache is None:
                from compressme.shape_constants import ShapeMatchedConstantRows
                self._shape_constant_cache = ShapeMatchedConstantRows(max_entries=8)
            B, T = src.shape
            length = T + int(self._has_dataset_token)
            assignments = []
            for batch_index in range(B):
                assignments.append((batch_index * length, self.cls_token))
                if self._has_dataset_token:
                    assignments.append((batch_index * length + length - 1, self.dataset_token))
            constants = self._shape_constant_cache(self.encoder, (B, length, self.cls_token.shape[-1]), assignments)
            constants = constants.reshape(B, 2 if self._has_dataset_token else 1, self.d_model)
            encoded[:, 0, :] = constants[:, 0, :]
            if self._has_dataset_token:
                encoded = torch.cat((encoded, constants[:, 1:2, :]), dim=1)
        else:
            encoded[:, 0, :] = self.encoded_cls.expand(encoded.size(0), -1)
            if self._has_dataset_token:
                encoded = torch.cat((encoded, self.encoded_dataset.expand(encoded.size(0), -1).unsqueeze(1)), dim=1)
        encoded = encoded * math.sqrt(self.d_model)
        if counts is not None:
            counts = counts.to(self.device).unsqueeze(-1)
            bin_weights = F.softmax(self.count_encoder(counts), dim=-1)
            bin_embeddings = self.bin_encoder(torch.arange(10, device=self.device))
            count_emb = torch.matmul(bin_weights, bin_embeddings)
            if self._has_dataset_token:
                dataset_count_emb = torch.zeros(count_emb.size(0), 1, count_emb.size(2), device=self.device)
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)
            encoded = encoded + count_emb
        # Original State forward intentionally passes no padding mask.
        output = self.transformer_encoder(encoded, src_key_padding_mask=None)
        gene_output = self.decoder(output)
        embedding = F.normalize(gene_output[:, 0, :], dim=1)
        dataset_emb = gene_output[:, -1, :] if self._has_dataset_token else None
        return gene_output, embedding, dataset_emb

    def _compute_embedding_for_batch(self, batch):
        batch_sentences = batch[0].to(self.device)
        task_ids = batch[1].to(self.device)
        Y, batch_weights = batch[2], batch[4]
        mask = batch[5].to(torch.bool)
        counts = batch[7]
        if counts is not None:
            counts = counts.to(self.device)
        dataset_nums = batch[8]
        if dataset_nums is not None:
            dataset_nums = dataset_nums.to(self.device)
        _, embedding, dataset_emb = self.forward_gene_ids(batch_sentences, mask, counts, dataset_nums)
        X = self.get_gene_embedding_by_id(task_ids)
        return X, Y, batch_weights, embedding, dataset_emb

    def resize_batch(self, *args, **kwargs):
        return self._resize_batch_function(*args, **kwargs)
