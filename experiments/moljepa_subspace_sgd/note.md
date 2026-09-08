# Mol-JEPA: preserving Q/K fine-tuning in smaller fixed subspaces

The actual first graph-attention layer admits a substantial **restricted fine-tuning** reduction: its query/key weights and biases can be represented by 243,024 trainable coefficients instead of 4,202,496. The tested contract freezes the preceding input projection and the shared edge projection, and trains only Q/K with ordinary SGD. This is a component experiment, not an unrestricted whole-model training replacement or an additional reduction of the existing inference-optimized artifact.

## Actual source and consumers

The audited Mol-JEPA checkpoint has 82 node features, a biased 82→512 input projection, eight attention heads of width 512, and 17 edge features. `GraphEncoder` keeps the512-dimensional projected input for its residual. Each first-layer `TransformerConv` also uses separate query, key, value and skip projections. The [pinned model source](https://huggingface.co/Flogrammer/Mol-JEPA/blob/4c912b450175f31b5ba913a5dc921c03b27b985a/modeling_moljepa.py) and checkpoint were already downloaded and audited locally; the online browser could not reopen this pinned URL during this subtask.

The same edge projection contributes to keys **and** values. The implementation averages heads, retains the skip path, and scales logits by the original \(1/\sqrt{512}\). These semantics were checked directly in installed PyG 2.8.0.post1 and agree with the [official TransformerConv source documentation](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/transformer_conv.html). The current online documentation identifies itself as 2.9.0; the numerical experiment uses the installed 2.8.0.post1 implementation.

## Construction and SGD argument

Write the frozen node projection as \(x=P a+b\), with raw features \(a\in\mathbb R^{82}\). Augment the bias coordinate:

\[
\bar x=\begin{bmatrix}x\\1\end{bmatrix}
=F\begin{bmatrix}a\\1\end{bmatrix},\qquad
F=\begin{bmatrix}P&b\\0&1\end{bmatrix}\in\mathbb R^{513\times83}.
\]

Take an orthonormal basis \(U\) for this fixed input subspace. For each head, include query/key biases in matrices \(\bar Q,\bar K\in\mathbb R^{512\times513}\), and let \(E\in\mathbb R^{512\times17}\) be its edge map. An orthonormal basis \(R\) for the columns of

\[
[\bar Q U,\ \bar K U,\ E]
\]

needs at most \(83+83+17=183\) columns. Store

\[
Q_c=R^T\bar Q U,\quad K_c=R^T\bar K U,\quad E_c=R^TE,
\quad z=U^TF[a;1].
\]

Then the complete edge-conditioned score is unchanged in real arithmetic:

\[
\frac{q_i^T(k_j+Ee_{ij})}{\sqrt{512}}
=\frac{(Q_cz_i)^T(K_cz_j+E_ce_{ij})}{\sqrt{512}}.
\]

The reduced score dimension does not change the scale. All 512 value coordinates, the original value-edge contribution, head averaging and skip computation remain intact.

For a loss observing Q/K only through these scores, query-output gradients are linear combinations of keys plus edges, and key-output gradients are linear combinations of queries. They therefore remain in \(R\). Inputs remain in \(U\), so the augmented weight gradients satisfy

\[
\nabla_{\bar Q}L=R(\nabla_{Q_c}L)U^T,
\qquad
\nabla_{\bar K}L=R(\nabla_{K_c}L)U^T.
\]

Ordinary simultaneous SGD commutes with this fixed orthonormal change of coordinates. Biases are included and use the same learning rate. The argument imposes no discrete restriction on the 82-dimensional raw features or 17-dimensional edge features. It requires the frozen projections and the stated consumer contract. Neither Adam equivalence nor alternative parameter-group learning rates were tested.

The prototype uses reduced QR without a singular-value threshold: it keeps all 83 input and 183 head coordinates. Basis construction uses float64; retained coefficients use the same dtype as their reference run. The large verification bases \(U,R\) are not part of stored module state. Deployment needs only the 6,889-value raw-coordinate map and 24,888-value projected-edge table.

## Measured scope and results

The reference uses the actual pretrained first-layer weights and original PyG convolution. Twenty SGD steps cycle through ethanol, benzene, alanine and aspirin, with original Mol-JEPA graph featurization, explicit hydrogens and a changing synthetic downstream target. Only Q/K are updated, at learning rate 0.05. Both float64 and float32 runs check convolution outputs, attention probabilities, raw logits, raw-node gradients, edge-feature gradients and projected parameters.

| Quantity | Float64 maximum absolute difference | Float32 maximum absolute difference |
|---|---:|---:|
| Convolution output | 7.11e-15 | 1.43e-6 |
| Attention probabilities | 2.66e-15 | 4.77e-7 |
| Raw attention logits | 3.38e-14 | 5.72e-6 |
| Raw-node input gradient | 1.53e-16 | 5.59e-8 |
| Edge-feature input gradient | 1.08e-17 | 3.63e-9 |

All 20 molecular steps pass `atol=rtol=1e-10` for float64 or `1e-5` for float32. This is a trajectory comparison with actual pretrained weights, not evidence of downstream biological quality after fine-tuning.

Additional COO checks cover duplicate directed edges, a self-loop, an isolated node and an empty edge set. Empty-edge outputs are bitwise equal. The float64 mixed-COO test passes throughout. With arbitrary Gaussian raw/edge features, the float32 convolution output and attention probabilities pass, but one raw-logit element fails the mixed 1e-5 gate. The largest raw-logit absolute difference in that case is 5.34e-5; this maximum is not necessarily the failing near-zero element. The failure is recorded in `results.json`. The tolerance was not widened, and a universal float32 score guarantee is not claimed.

## Storage and the boundary on trainability

| Stored quantity | Original | Reduced |
|---|---:|---:|
| Trainable Q/K coefficients | 4,202,496 | 243,024 |
| Complete first convolution plus input projection | 6,678,528 | 2,750,833 |

The component saves 3,927,695 retained scalar values, or 15,710,780 bytes in float32. The second row includes all frozen value, edge, skip and input weights, plus the new constant maps. Gradients, activations, temporary products and the rest of Mol-JEPA are excluded. No wall-clock timing or full-model checkpoint export was performed. The existing bilinear inference representation uses a different parameterization and may store less; it does not establish this restricted original-SGD equivalence.

The frozen-projection condition is substantive. In a real-weight gradient probe on aspirin, 74.7% of the shared edge-weight gradient norm lies outside the retained score subspace, because the same edge weights serve the full-width value path. Unfreezing the input projection also generates a query-response direction with 4.51% of its norm outside that subspace. These are explicit obstructions to claiming unrestricted whole-model SGD equivalence. Recomputing bases during training would require a separate state-preservation analysis.

The experiment contains no production edits. `probe.py` takes `--checkpoint` and `--source-file` paths and checks pinned hashes before source execution. New runs default to `replay-results.json`; the archived evidence remains in `results.json`. Dependencies are PyTorch, PyG, safetensors and the original featurizer dependencies molfeat/RDKit. The [linear-subspace/Gram-SGD note](../gram_sgd/gram-sgd-note.md) gives the simpler factorized-linear construction and its primary-source context.

The underlying invariance of query/key dot products under shared orthogonal coordinates is established in [Collaborative Multi-Head Attention](https://arxiv.org/abs/2006.16362); [KQ-SVD](https://arxiv.org/abs/2512.05916) also compresses joint attention interactions, using a low-rank cache objective. Neither is a proof that this frozen input-and-edge subspace preserves the original Q/K SGD trajectory. [Tarmoun et al. (2021)](https://proceedings.mlr.press/v139/tarmoun21a.html) provide related symmetry and Gramian analysis for linear-model gradient flow. The specific result here is the fixed-subspace argument for this attention component and its tested SGD trajectories. Its novelty has not been established, and those trajectories do not validate unrestricted biological-model fine-tuning.
