"""Exact sparse execution of Mol-JEPA's fixed default graph feature schema.

This avoids constructing pair features for non-edges and skips generic PyG
collation. It never caches molecules, graphs, or model outputs. Descriptor
definitions and datamol parsing remain the installed upstream implementations.
The caller must provide the unmodified upstream GraphFeaturizer factory for
fallback; unsupported/invalid input is replayed there to preserve its errors.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch


_ATOM_NAMES = (
    "atom_chiral_tag_one_hot", "atom_degree_one_hot", "atom_formal_charge",
    "atom_hybridization_one_hot", "atom_implicit_valence_one_hot",
    "atom_is_aromatic", "atom_is_chiral_center", "atom_is_in_ring",
    "atom_num_radical_electrons", "atom_one_hot", "atom_total_num_H_one_hot",
)
_BOND_NAMES = (
    "bond_direction_one_hot", "bond_is_conjugated", "bond_is_in_ring",
    "bond_stereo_one_hot", "bond_type_one_hot",
)


class SparseMolJEPAFeaturizer:
    """Fixed 82-atom/17-edge schema; explicit H and canonical atom order.

    For each selected adjacent atom pair, the unweighted shortest-path distance
    is exactly one. Ring co-membership is the same union over GetSymmSSSR rings
    as upstream. Bond features are copied to both directed edges, which are
    sorted by (source, destination), matching dense_to_sparse traversal.

    No conformer generation or N-by-N feature allocation is needed. A fallback
    factory is required because malformed-input exceptions are upstream API.
    """

    def __init__(self, legacy_factory):
        import datamol as dm
        from rdkit import Chem
        from molfeat.calc.atom import AtomCalculator
        from molfeat.calc.bond import BondCalculator, EdgeMatCalculator
        from molfeat.calc import _atom_bond_features as original

        if tuple(sorted(AtomCalculator.DEFAULT_FEATURIZER)) != _ATOM_NAMES:
            raise ValueError("Unsupported Mol-JEPA atom feature schema")
        if tuple(sorted(BondCalculator.DEFAULT_FEATURIZER)) != _BOND_NAMES:
            raise ValueError("Unsupported Mol-JEPA bond feature schema")
        expected_pair = {
            "pairwise_2D_dist": original.pairwise_2D_dist,
            "pairwise_ring_membership": original.pairwise_ring_membership,
        }
        if EdgeMatCalculator.DEFAULT_PAIRWISE_FEATURIZER != expected_pair:
            raise ValueError("Unsupported Mol-JEPA pair feature schema")
        self._atom_functions = tuple(AtomCalculator.DEFAULT_FEATURIZER[x] for x in _ATOM_NAMES)
        self._bond_functions = tuple(BondCalculator.DEFAULT_FEATURIZER[x] for x in _BOND_NAMES)
        self._to_mol = dm.to_mol
        self._get_rings = Chem.GetSymmSSSR
        self._legacy_factory = legacy_factory
        self._legacy = None
        self.fallback_count = 0

    def __deepcopy__(self, memo):
        # RDKit Boost.Python functions cannot be pickled. Rebuild the fixed
        # descriptor definitions; no parsed molecules or outputs are cached.
        result = type(self)(self._legacy_factory)
        memo[id(self)] = result
        result.fallback_count = self.fallback_count
        return result

    def _fallback(self, smiles):
        self.fallback_count += 1
        if self._legacy is None:
            self._legacy = self._legacy_factory()
        return self._legacy.encode(smiles)

    def encode(self, smiles):
        try:
            return self._encode(smiles)
        except Exception:
            # Preserve the original parser/empty-molecule failure behaviour.
            return self._fallback(smiles)

    def _encode(self, smiles):
        mol = self._to_mol(smiles, add_hs=True, ordered=True)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("Use upstream handling for invalid/empty molecules")
        n = mol.GetNumAtoms()
        atoms = np.asarray([
            [value for fn in self._atom_functions for value in fn(atom)]
            for atom in mol.GetAtoms()
        ], dtype=np.float32)
        if atoms.shape != (n, 82):
            raise ValueError("Unexpected atom feature width")

        bonds = list(mol.GetBonds())
        nb = len(bonds)
        pairs = np.empty((2, 2 * nb), dtype=np.int32)
        features = np.empty((2 * nb, 17), dtype=np.float32)
        # Match upstream timing of ring discovery: after atom/bond descriptors.
        for i, bond in enumerate(bonds):
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row = [value for fn in self._bond_functions for value in fn(bond)]
            if len(row) != 15:
                raise ValueError("Unexpected bond feature width")
            pairs[:, 2 * i] = (a, b)
            pairs[:, 2 * i + 1] = (b, a)
            features[2 * i:2 * i + 2, :15] = row
        rings_by_atom = [set() for _ in range(n)]
        for ring_id, ring in enumerate(self._get_rings(mol)):
            for atom in ring:
                rings_by_atom[atom].add(ring_id)
        for i, bond in enumerate(bonds):
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            features[2 * i:2 * i + 2, 15] = 1.0
            features[2 * i:2 * i + 2, 16] = bool(rings_by_atom[a] & rings_by_atom[b])
        # np.lexsort is integer-only here: ordering, not changed arithmetic.
        order = np.lexsort((pairs[1], pairs[0]))
        pairs = np.ascontiguousarray(pairs[:, order])
        features = np.ascontiguousarray(features[order])
        return {"edge_index": torch.from_numpy(pairs), "x": torch.from_numpy(atoms),
                "edge_attr": torch.from_numpy(features)}


def batch_graph_features(graphs, *, data_class=None):
    """Collate the fixed homogeneous graph schema, preserving PyG metadata.

    The returned Batch has all five fields the original Mol-JEPA forward reads.
    When its original MultimodalData class is supplied, get_example and
    to_data_list also reconstruct the same individual graphs.
    """
    graphs = list(graphs)
    if not graphs:
        # This is the original custom collator's error for empty data_list.
        raise IndexError("list index out of range")
    counts = np.asarray([g["x"].shape[0] for g in graphs], dtype=np.int64)
    edge_counts = np.asarray([g["edge_index"].shape[1] for g in graphs], dtype=np.int64)
    ptr = torch.from_numpy(np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts))))
    edge_ptr = torch.from_numpy(np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(edge_counts))))
    x = torch.cat([g["x"] for g in graphs], 0)
    edge_attr = torch.cat([g["edge_attr"] for g in graphs], 0)
    edge_index = torch.cat([
        g["edge_index"].long() + int(offset)
        for g, offset in zip(graphs, ptr[:-1])
    ], 1)
    assignment = torch.from_numpy(np.repeat(np.arange(len(graphs), dtype=np.int64), counts))
    batch = Batch(_base_cls=data_class) if data_class is not None else Batch()
    batch.graph_x = x
    batch.graph_edge_index = edge_index
    batch.graph_edge_attr = edge_attr
    batch.graph_x_batch = assignment
    batch.graph_x_ptr = ptr
    batch._num_graphs = len(graphs)
    batch._slice_dict = {"graph_x": ptr, "graph_edge_index": edge_ptr, "graph_edge_attr": edge_ptr}
    zeros = torch.zeros(len(graphs), dtype=torch.long)
    batch._inc_dict = {"graph_x": zeros, "graph_edge_index": ptr[:-1], "graph_edge_attr": zeros}
    return batch


def smiles_to_batch(smiles_list, *, featurizer, data_class=None):
    """Fresh parsing/featurization and direct batching for every input call."""
    return batch_graph_features((featurizer.encode(s) for s in smiles_list), data_class=data_class)
