# Preserving a factorized model's SGD trajectory with less state

Two consecutive bias-free linear maps can sometimes be compressed **during fine-tuning**, while preserving their effective SGD update in real arithmetic. Keeping only their product is generally insufficient. Keeping the product and two Gram matrices is sufficient. This is an isolated research result with synthetic numerical checks, not a production optimizer or a biological-model validation.

Let \(A\in\mathbb R^{m\times h}\), \(B\in\mathbb R^{h\times n}\), and \(C=AB\). Assume every loss term and consumer observes the pair only through \(C\). Nonlinear computation after the pair is allowed. Write \(G=\partial L/\partial C\), \(P=AA^T\), and \(Q=B^TB\).

Simultaneous plain SGD gives

\[
A'=A-\eta GB^T,\qquad B'=B-\eta A^TG.
\]

Multiplying these updates gives the closed state update

\[
\begin{aligned}
C'&=C-\eta(GQ+PG)+\eta^2GC^TG,\\
P'&=P-\eta(GC^T+CG^T)+\eta^2GQG^T,\\
Q'&=Q-\eta(C^TG+G^TC)+\eta^2G^TPG.
\end{aligned}
\]

Every right-hand side uses the old state. The quadratic terms are necessary for discrete SGD. Dropping them gives a different update.

A compact proof stacks \(S=[A;B^T]\). Its update is \(S'=M(G)S\), where

\[
M(G)=\begin{bmatrix}I_m&-\eta G\\-\eta G^T&I_n\end{bmatrix}.
\]

Consequently \(K=SS^T=\begin{bmatrix}P&C\\C^T&Q\end{bmatrix}\) satisfies \(K'=MKM^T\). This requires neither balanced initialization nor a particular differentiable loss. Initialize the state from actual factors; in real arithmetic, its positive semidefiniteness and rank bound are then preserved.

The missing state matters. In a scalar example, \((A,B)=(1,1)\) and \((2,1/2)\) have the same product. With \(G=1\) and \(\eta=0.1\), simultaneous factor SGD gives products \(0.81\) and \(0.585\), respectively. Ordinary SGD on their shared product would give \(0.9\).

## When one factor is frozen

If \(B\) is frozen, \(Q\) is constant and only

\[
C'=C-\eta GQ
\]

is needed. This is right-preconditioned SGD, becoming ordinary product SGD when \(Q=I\).

For factor momentum \(V_A\), retain its projection \(D=V_AB\). With coupled weight decay \(\lambda\) on \(A\), momentum \(\mu\), zero dampening and no Nesterov:

\[
D'=\mu D+GQ+\lambda C,\qquad C'=C-\eta D'.
\]

The experiment starts with zero momentum and compares against these precise [PyTorch SGD semantics](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/optim/sgd.py). It does not preserve access to the original factors, or a separately reported penalty depending on their unobserved components.

## Numerical evidence

The attached CPU experiment uses actual `torch.optim.SGD`, changing minibatches, a nonlinear downstream loss, \((m,h,n)=(7,64,5)\), batch size 13 and 20 steps. It checks outputs, input gradients and effective state after each step. Both a two-stage affine forward and an explicitly multiplied-product forward are tested. The latter separates update rounding from some forward reassociation effects.

| Method | Stored values before → after | Worst output error, float64 | Worst output error, float32 |
|---|---:|---:|---:|
| Both factors train, Gram state | 768 → 109 | 2.22e-15 | 2.15e-6 |
| Frozen B | 768 → 60 | 2.66e-15 | 9.54e-7 |
| Frozen B, momentum 0.9 and decay 0.02 | 1,216 → 95 | 2.22e-15 | 9.54e-7 |

All 12 trajectories pass `atol=rtol=1e-10` in float64 or `1e-5` in float32. An independent rational-arithmetic check verifies all three identities exactly. Storage includes retained weights and optimizer state, excluding gradients, temporary products, activations and the surrounding model. The maximum float32 Gram-state discrepancy is 6.68e-6. These observations do not establish a bound for arbitrary training lengths, learning rates or data.

## Storage and computation

Original factors store \(h(m+n)\) values. Full Gram state stores \(mn+m^2+n^2\), so it saves storage precisely when

\[
h(m+n)>mn+m^2+n^2.
\]

Packing the symmetric Grams would reduce this to \((m+n)(m+n+1)/2\) values. The prototype does not implement packing. For \(m=n=512,h=2048\), full state is 786,432 values instead of 2,097,152, a 2.67-fold reduction. For a LoRA-like width \(h=8\), the same Gram state is **96 times larger** than the factors. This is a possible compressor for sufficiently wide linear expansions, not a general low-rank adapter compressor.

Given \(G\), the two factor-gradient products cost approximately \(2mnh\) multiply-accumulates; the shared-product Gram implementation costs \(3mn(m+n)+mn\min(m,n)\). The frozen update costs \(mn^2\). These estimates omit gradient computation and memory traffic. Fusing the forward can save hidden activations, but the dense Gram updates can consume that benefit. No wall-clock speedup was measured.

The frozen state needs \(mn+n^2\) values, or \(2mn+n^2\) with projected momentum. Its savings assume that the original \(B\) can be removed; if another consumer needs \(B\), its storage must still be counted.

## A simpler alternative: keep reduced factors

The columns of \([A^T,B]\) span a hidden subspace of dimension \(r\le m+n\). Choose an orthonormal basis \(R\in\mathbb R^{h\times r}\), and initialize

\[
A_r=AR,\qquad B_r=R^TB.
\]

Because \(A=A_rR^T\) and \(B=RB_r\), the reduced factors preserve \(C,P,Q\). Their ordinary SGD updates commute with this fixed change of coordinates. Scalar momentum and weight decay also commute, provided any existing momentum lies in the retained subspace. Fresh zero momentum satisfies this condition. Otherwise its directions must also be included in the basis.

The addendum retains every reduced-QR column, with no numerical rank threshold or singular-value truncation. It reduces width 64 → 12 and weights 768 → 144; with momentum, retained state is 1,536 → 288. Four 20-step trajectories pass the same mixed tolerances, including momentum and decay. Worst output error is 4.88e-15 in float64 and 1.43e-6 in float32. The maximum float32 \(Q\) discrepancy is 1.14e-5, which passes the combined absolute/relative gate; it is not an absolute-error-below-1e-5 claim. The basis is retained only to verify the experiment and need not be stored for subsequent reduced-factor training.

This alternative trades somewhat larger state than the Gram formulation for ordinary factor SGD and smaller affine kernels. Its numerical QR errors and training drift still require validation.

## Scope and prior art

An intervening activation, normalization or dropout, external use of the hidden representation, factor-specific losses, or tied factor consumers violates the stated contract. Biases are not handled by these prototypes. Adam and other coordinatewise optimizer state do not generally commute with hidden rotations and are not covered. Floating-point products change evaluation order, so neither approach promises bitwise equality. The Gram implementation can also accumulate symmetry or positive-semidefiniteness error. No biological-model fine-tuning trajectory has been validated here.

The product-and-Gram recurrences follow by expanding the ordinary SGD updates. The broader use of lifted matrices and effective dynamics has established precedent. [Gunasekar et al. (2017)](https://papers.nips.cc/paper_files/paper/2017/file/58191d2a914c6dae66371c9dcdc91b41-Paper.pdf) use a positive-semidefinite lift of rectangular matrix factorization and derive factor-independent gradient-flow dynamics. [Arora, Cohen and Hazan (2018)](https://arxiv.org/abs/1802.06509) derive end-to-end preconditioning under balanced-initialization and gradient-flow assumptions. [Tarmoun et al. (2021)](https://proceedings.mlr.press/v139/tarmoun21a.html) analyse Gramian conservation and Riccati-type gradient flow; Section 5 also gives a scalar discrete recurrence containing the quadratic step-size term. These sources provide context, but do not directly establish the full finite-step matrix construction above. Its contribution here is an explicit derivation, complete state accounting and numerical checks. Priority for that construction or the reduced-factor variant has not been established.

Run `python verify_gram_sgd.py` and `python verify_hidden_subspace_sgd.py` in this directory. Both require PyTorch only and write complete JSON records beside the scripts. The checked runtime was PyTorch 2.14.0 on CPU. These files are research artifacts and make no production package changes.
