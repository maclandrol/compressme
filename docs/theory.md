# What the compressor can guarantee

`compressme` aims to reduce the computation and weights needed to evaluate an existing model while preserving its input and output contract. It uses algebraic rewrites and, when requested, bounded approximations. Neither operation requires a student model, training labels or distillation.

Three claims must remain separate:

| Transformation | Guarantee | Required condition |
| --- | --- | --- |
| Exact affine rewrite | Same mathematical function | The compiler proves the relevant dataflow and affine operations |
| Input-domain specialisation | Same mathematical function on the declared input domain | The removed branch is unreachable for every supported input |
| Spectral truncation | A stated local error bound | The bound's input-domain, arithmetic and parameter assumptions hold |

An exact identity over real numbers does not imply bit-identical floating-point outputs. A local activation bound does not establish a bound on final molecular coordinates, affinity scores or experimental predictive quality.

## Compute consumers before an unnecessary expansion

A learned representation can have hundreds of coordinates while carrying an exactly lower-dimensional affine input. Suppose a producer and its consumers are

\[
h=A x+a,\qquad y_i=W_i h+b_i,
\]

where \(x\in\mathbb R^d\), \(h\in\mathbb R^n\), and consumer \(i\) has \(m_i\) outputs. Then

\[
\boxed{y_i=(W_iA)x+(W_i a+b_i).}
\]

This identity holds for every input. It needs neither a low-rank approximation nor evidence that training produced small singular values. The reachable representation is contained in \(a+\operatorname{range}(A)\), whose dimension is at most \(d\). Each consumer only needs its action on that affine set.

The producer may remain in the graph for a residual connection or another consumer. Eligible linear consumers can still read the original \(x\) through their fused weights. The public input, residual representation and output shapes remain unchanged.

Let \(M=\sum_i m_i\). If the producer must remain, the original and fused weight counts, excluding unchanged biases, are

\[
N_{\rm original}=nd+Mn,\qquad
N_{\rm fused}=nd+Md.
\]

Dense matrix-vector multiply counts follow the same expressions. The saving is exactly \(M(n-d)\). Therefore this rewrite saves dense weight entries and multiply-accumulates precisely when \(d<n\). If \(d\ge n\), retain the original consumers unless another independent transformation reduces cost. Among these two representations, the minimum count is

\[
nd+M\min(n,d).
\]

If all uses of the producer disappear, its cost can also be removed; then compare \(Md\) directly with \(nd+Mn\). These are counts for the stated dense implementations, not lower bounds over every possible matrix algorithm.

The audited Mol-JEPA graph encoder provides a concrete candidate: an \(82\rightarrow512\) affine input projection followed directly by query, key and value projections with \(4096\) outputs each, plus a \(512\)-output skip projection. Thus \(M=12{,}800\). Keeping the producer for its residual use changes these consumer weights from \(12{,}800\times512\) to \(12{,}800\times82\), a reduction from 6,553,600 to 1,049,600 entries. The 5,504,000 saved weights are approximately 12.12% of the loaded model's 45,406,721 parameters. The checkpoint additionally contains 51 float32 buffer entries and one int64 buffer entry. This is a structural calculation for the [pinned Mol-JEPA implementation](https://huggingface.co/Flogrammer/Mol-JEPA/blob/4c912b450175f31b5ba913a5dc921c03b27b985a/modeling_moljepa.py) and [configuration](https://huggingface.co/Flogrammer/Mol-JEPA/blob/4c912b450175f31b5ba913a5dc921c03b27b985a/config.json), not a measured runtime result.

A safe compiler must prove the edge, rather than infer it from module names or one observed execution. A nonlinearity, normalisation, gate, input-dependent matrix, changed tensor, or in-place mutation between producer and consumer invalidates simple affine composition. Shared modules and tensors require accounting for every call site. Inference evaluation mode and the supported control-flow path are part of the rewrite contract.

### Fine-tuning without reducing this block's function class

If \(A\) has full column rank and every consumer has a freely trainable bias, independently training the fused weights does not reduce the affine function class at this block. Given any fused consumer \(y_i=C_i x+c_i\), choose

\[
W_i=C_iA^+,\qquad b_i=c_i-W_i a,
\]

where \(A^+A=I_d\). Then \(W_i(Ax+a)+b_i=C_i x+c_i\) for every input. The producer can still supply the original residual branch. This establishes equivalence of the attainable functions while \(A\) remains full column rank; it does not establish equivalence of gradient updates, regularisation or training trajectories.

The redundant directions can be written explicitly. Changing \(W_i\) to \(W_i+N_i\), with \(N_iA=0\), and changing its bias to \(b_i-N_i a\), preserves the function. Their dimension is \(m_i(n-d)\) for full-column-rank \(A\), matching the removed weight count. This is an exact quotient of redundant parameters, rather than an approximate low-rank restriction on the function.

For the audited Mol-JEPA projection, numerical SVD found rank 82, with singular values approximately 0.5494 to 6.6903 and condition number 12.18. These measurements support the full-column-rank premise for that checkpoint. They do not ensure that an unconstrained fine-tuning run will preserve it.

## Contract graph-attention scores before expanding features

The audited graph attention has a second exact opportunity. For receiver node \(i\), source node \(j\), edge features \(e_{ji}\), and one head of width \(c\), write

\[
q_i=Qx_i+b_q,\qquad k_j=Kx_j+b_k,\qquad g_{ji}=Ee_{ji}.
\]

Its logit is \(q_i^\top(k_j+g_{ji})/\sqrt c\). Define

\[
B=K^\top Q,\quad u=K^\top b_q,\qquad
D=E^\top Q,\quad v=E^\top b_q.
\]

Then

\[
q_i^\top(k_j+g_{ji})
=(Bx_i+u)^\top x_j+(Dx_i+v)^\top e_{ji}+q_i^\top b_k.
\]

The last term is constant over the incoming edges of receiver \(i\) within a head. Subtracting it leaves the receiver-wise softmax unchanged. The replacement therefore computes

\[
\boxed{\alpha_{ji}=
\operatorname{softmax}_{j\to i}
\left(
\frac{(Bx_i+u)^\top x_j+(Dx_i+v)^\top e_{ji}}{\sqrt c}
\right).}
\]

This retains the original head-width scale \(\sqrt c\), even when dot products now use smaller features. It removes \(Q,K,b_q,b_k\) from the deployed score computation. The edge projection \(E\) remains necessary for values. An edge projection bias, if present, also contributes a receiver-constant score shift but must still be accounted for in values.

For equal source and receiver feature width \(d\), edge width \(e\), and biased query/key projections, the original query/key parameter count per head is \(2c(d+1)\). The contracted maps and offsets contain \((d+e)(d+1)\) parameters. The saving per head is

\[
(d+1)(2c-d-e),
\]

which is positive when \(d+e<2c\). For the audited \(H=8,c=512,e=17\) configuration, the first layer after affine fusion has \(d=82\), giving 614,200 fewer parameters. A later layer with \(d=512\) saves 2,031,480. These counts retain the edge-value projection and exclude unchanged value, skip and normalisation parameters. If the edge width or source feature width is too large, retain the original score implementation.

Query-key product compression is established prior art; this use performs full contraction without rank truncation. The structural opportunity is that the audited graph attention expands each head to a width equal to or larger than its input features.

### Aggregate values before their linear expansion

Let \(\widetilde\alpha_{ji,h}\) denote the attention coefficient actually used for messages, including attention dropout, and let

\[
s_{i,h}=\sum_{j\to i}\widetilde\alpha_{ji,h}x_j,\qquad
t_{i,h}=\sum_{j\to i}\widetilde\alpha_{ji,h}e_{ji},\qquad
a_{i,h}=\sum_{j\to i}\widetilde\alpha_{ji,h}.
\]

Linearity gives the exact per-head message

\[
m_{i,h}=V_hs_{i,h}+E_ht_{i,h}+b_{v,h}a_{i,h}.
\]

This moves linear projections after graph aggregation. It can reduce scattered intermediate features from width \(c\) to \(d+e+1\) per head, while retaining the same value parameters. It is most attractive for the first fused layer; it is not automatically smaller for later layers with \(d=c\).

Retain \(a_{i,h}\) explicitly. It is zero for nodes without incoming edges and need not equal one after dropout. If the edge projection has a bias, add it to the value bias in this formula. Head averaging can occur after these per-head computations. Different heads generally have different attention coefficients, so their raw-feature aggregates cannot be merged into one common aggregate.

### Attention semantics and numerical scope

The [PyG TransformerConv implementation](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/transformer_conv.html) uses queries from target nodes, keys and values from source nodes, and softmax grouped by target. It returns attention weights from before dropout, even when `return_attention_weights=False` is passed as a boolean. A compatible adapter must preserve this behaviour, edge order, root/skip and optional beta mixing, and its declared adjacency representation. Sparse or bipartite inputs require explicit support or rejection.

For training-mode equality, use the same dropout coefficients on the same ordered edge/head entries. At evaluation time dropout is inactive. Coordinate reassociation and subtraction of constant logit shifts can change floating-point results, so neither mode is guaranteed bit-identical.

An independent float64 calculation checked the contracted scores and aggregated values on a graph with duplicate edges, self-loops, isolated nodes, query/key/value biases and a fixed dropout mask. Maximum discrepancies were approximately \(3.3\times10^{-16}\) for attention and \(1.8\times10^{-15}\) for values; input and edge gradient discrepancies were below \(2.6\times10^{-14}\). This verifies the derivation on that test, not the complete model or every backend.

The original model couples \(B,D\) through \(Q\), and couples the score-side \(D\) to the value-side \(E\). Its stacked affine score map \(\begin{bmatrix}B&u\\D&v\end{bmatrix}=\begin{bmatrix}K^\top\\E^\top\end{bmatrix}\begin{bmatrix}Q&b_q\end{bmatrix}\) has rank at most \(c\). Independently training contracted parameters can change these constraints; for \(d=c\), the new map has \(c+1\) columns and can acquire a rank unavailable to the original factorisation. The full-rank affine-family equivalence above does not automatically apply to this contraction. Input derivatives remain mathematically equal at corresponding fixed weights, while gradients with respect to the two parameterisations need the appropriate chain rule and generally follow different optimisation trajectories.

## Remove inactive branches only under an explicit input contract

If a model contains encoders for several modalities but its deployed API accepts only SMILES, encoders that cannot execute for any valid SMILES request can be excluded from that specialised artifact. This is elimination of unreachable computation and unused state under a restricted domain.

The Mol-JEPA audit identified 15,462,401 parameters in inactive modality encoders on the default SMILES-only route, approximately 34.05% of the model's 45,406,721 parameters. The percentage is an audit result for the [pinned checkpoint implementation](https://huggingface.co/Flogrammer/Mol-JEPA/blob/4c912b450175f31b5ba913a5dc921c03b27b985a/modeling_moljepa.py), not a general property of Mol-JEPA variants. Record the checkpoint identifier, exact counts and selected entry point alongside the exported artifact. In particular, a specialised export must reject requests that supply optional data for the removed modalities.

The guarantee is

\[
\forall x\in\mathcal D_{\rm SMILES},\quad
F_{\rm specialised}(x)=F_{\rm original}(x)
\]

in real arithmetic, provided the removed branches have no observable side effects and all supported outputs remain available. This export must reject unsupported modalities clearly. One successful SMILES trace does not prove that a branch is unreachable for every valid input.

Report this reduction separately from approximate weight compression. An encoder that was already inactive does not contribute inference FLOPs on that route, so removing its parameters primarily reduces storage and resident model state.

## Bound the approximation after normalisation

A standalone linear operator has no finite uniform absolute approximation bound over all unbounded inputs unless the replacement is exact. LayerNorm supplies a bounded intermediate domain that makes a useful local statement possible.

Consider fixed affine LayerNorm over the last feature dimension \(d\), followed directly by \(W\in\mathbb R^{m\times d}\) and bias \(b\). Define

\[
P=I-\frac{\mathbf1\mathbf1^\top}{d},\qquad
z(x)=\frac{Px}{\sqrt{\|Px\|_2^2/d+\epsilon}},\qquad \epsilon>0.
\]

The original block is

\[
f(x)=W\operatorname{diag}(\gamma)z(x)+W\beta+b.
\]

Since \(Pz=z\), set

\[
A=W\operatorname{diag}(\gamma)P,\qquad c=W\beta+b.
\]

Then \(f(x)=Az(x)+c\), \(A\mathbf1=0\), and \(\|z(x)\|_2<\sqrt d\). Every centred vector with norm below \(\sqrt d\) is reachable: for such a vector \(z\), choose

\[
x=\frac{\sqrt\epsilon\,z}{\sqrt{1-\|z\|_2^2/d}}.
\]

Thus LayerNorm removes one direction and bounds the others. It does not, by itself, imply a much smaller reachable subspace. The definition follows [Layer Normalization](https://arxiv.org/abs/1607.06450) and [PyTorch's LayerNorm semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).

Write \(A=U\Sigma V^\top\), with singular values in descending order, and retain

\[
A_r=U_r\Sigma_rV_r^\top.
\]

The replacement computes \(f_r(x)=A_rz(x)+c\) through two dense projections. In real arithmetic,

\[
\boxed{\sup_x\|f(x)-f_r(x)\|_2=\sqrt d\,\sigma_{r+1}(A).}
\]

The upper bound follows from the operator norm of the residual. If the discarded singular value is positive, its right singular vector is centred. Scaling the input along that vector makes the normalised vector approach norm \(\sqrt d\), proving equality of the supremum. Zero residual gives zero error.

More generally, for any rank-at-most-\(r\) replacement \(B\), the uniform error with the same normalisation and fused bias is \(\sqrt d\,\|(A-B)P\|_2\). Optimal low-rank approximation makes truncated SVD minimax within this class. This uses classical [low-rank matrix approximation](https://doi.org/10.1007/BF02288367); it is not a new SVD theorem or an optimum over all possible nonlinear replacement networks.

Choose the smallest rank satisfying the requested local tolerance, then require \(r(m+d)<md\) before accepting a parameter-saving factorisation. Include all biases and retained buffers in the actual artifact comparison. A flat spectrum can force a rank too large to save memory. Returning the original block is then the correct outcome.

This fusion requires fixed \(\gamma,\beta\) and a normalised shape matching the linear input dimension. Input-dependent adaptive normalisation and normalisation across additional tensor dimensions need separate derivations.

### An input-dependent local bound

Using \(V_r^\top\) as the first projection makes \(t=V_r^\top z\) available during inference. Orthogonality gives

\[
\|f(x)-f_r(x)\|_2
\le\sigma_{r+1}(A)
\sqrt{\|z\|_2^2-\|t\|_2^2}.
\]

The extra arithmetic consists of norms over existing vectors. This can improve the bound when a particular input lies near the retained subspace. It does not compare outputs of two full networks whose earlier representations have already diverged.

For a generic linear layer, the corresponding statement is only

\[
\|(W-W_r)x\|_2\le\|W-W_r\|_2\,\|x\|_2.
\]

A finite calibration set can measure typical input norms or reconstruction errors. It does not establish a universal input bound.

## Real arithmetic, stored weights and execution

Affine composition changes the association of multiplications and additions. In floating point, evaluating \(W(Ax+a)+b\) and evaluating a precomputed \((WA)x+(Wa+b)\) generally produces slightly different results. The rewrite is algebraically exact; numerical agreement must be measured at the deployed dtype and backend.

For approximate factors stored as \(\widehat A\) and fused bias \(\widehat c\), the local real-arithmetic bound against the original coefficients is

\[
\sqrt d\,\|(A-\widehat A)P\|_2+\|c-\widehat c\|_2.
\]

This residual includes factor and bias storage error if evaluated against the original coefficients. It still excludes rounding while computing normalisation and matrix products. An ordinary numerical SVD or spectral-norm estimate is not an outward-rounded, machine-verified upper bound. Describe such results as analytical bounds evaluated numerically, with their scope and dtype recorded.

The input-dependent formula also assumes exact orthogonality. Subtracting nearly equal squared norms can produce a false zero after rounding; clamping a negative result alone does not certify the error. Formal numerical certification would require conservative arithmetic error bounds or interval calculations covering both preprocessing and execution.

Exported certificates belong to specific parameter values. Subsequent fine-tuning invalidates them unless recomputed. Training the fused architecture also changes its optimisation parameterisation: gradient descent on \(WA\) is not generally equivalent to independently updating \(W\) and \(A\).

## What remains to establish for structural models

For a composition of Lipschitz blocks, local perturbation bounds can be propagated by a telescoping argument. If block \(\ell\) has a uniform replacement error \(\delta_\ell\), a valid end-to-end bound is

\[
\|F(x)-\widehat F(x)\|
\le\sum_\ell\delta_\ell\prod_{j>\ell}L_j,
\]

using valid downstream Lipschitz constants on all intermediate states reached by the compared compositions. Large constants, recurrent sampling and input-dependent interactions can make this bound too loose to guide Boltz compression. A scalar block bound must therefore remain separate from measured coordinate error, pose validity, ensemble coverage, binding calibration and ligand ranking.

Fewer parameters or multiply-accumulates do not guarantee lower wall time. Two small matrix products can incur more launch overhead than one larger product. Apple Silicon evaluation needs representative sequence and atom counts, synchronised MPS timing, warm-up, peak memory and a CPU baseline. Removing CUDA-specific dependencies and kernels is a separate portability task.

## Relationship to existing compression work

The exact affine rewrite is standard algebra applied to proven model dataflow. Eliminating inactive modalities is program specialisation. Neither should be presented as a new compression theorem.

Activation-aware and training-free low-rank methods already include [SVD-LLM](https://arxiv.org/abs/2403.07378), [BALF](https://arxiv.org/abs/2509.25136) and [Swift-SVD](https://arxiv.org/abs/2604.01609). [KQ-SVD](https://arxiv.org/abs/2512.05916) treats query-key interactions directly, while [SAFE-SVD](https://arxiv.org/abs/2605.17985) studies output-function sensitivity and physical fidelity in scientific foundation models. Joint attention factorisation or output-aware rank allocation alone would duplicate substantial prior work.

A different theoretical route, [width-independent compression of analytic networks](https://arxiv.org/abs/2608.21752), constructs smaller networks through derivative matching and reweighting. Its width dependence on the effective input dimension limits direct application to large structural models; it does not supply a ready general-purpose Boltz compressor.

The proposed package contribution is a compiler that discovers valid reductions, preserves a declared API, records exact versus approximate guarantees, and rejects transformations that fail its error or cost requirements. Whether these ingredients deliver substantial compression on Boltz-2 and Mol-JEPA is an empirical question for their actual checkpoints and supported execution paths.
