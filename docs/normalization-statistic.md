# Exact contraction across LayerNorm

LayerNorm appears to interrupt affine composition, but its denominator only needs one scalar norm. In an affine expansion followed by LayerNorm and another affine map, we can compute that norm from the narrow input and contract the remaining maps in advance. The expanded representation no longer has to be constructed.

Let

\[
h=Ax+a,\quad
f(x)=W\operatorname{LN}_{\gamma,\beta,\epsilon}(h)+b,
\]

with input dimension \(d\), intermediate dimension \(n\), output dimension \(m\), fixed affine LayerNorm over those \(n\) features, and \(\epsilon>0\). Define

\[
P=I-\mathbf1\mathbf1^\top/n,\quad
\bar x=[x;1],\quad T=[PA\;Pa].
\]

Take the full reduced QR factorisation \(T=UR\), retaining every row of \(R\). Its dimension is \(k\times(d+1)\), where \(k=\min(n,d+1)\). No estimated numerical rank or truncation threshold is used. Since \(U^\top U=I\),

\[
\|Ph\|^2=\|T\bar x\|^2=\|R\bar x\|^2.
\]

Set \(D=W\operatorname{diag}(\gamma)T\) and \(c=W\beta+b\). Then

\[
\boxed{
f(x)=\frac{D\bar x}{\sqrt{\|R\bar x\|^2/n+\epsilon}}+c.
}
\]

The formula evaluates the same mathematical function for every input using only \(D,R,c\). It replaces the expanded producer and original LayerNorm parameters. A Gram matrix \(T^\top T\) would give an equivalent quadratic form; QR avoids forming that matrix and evaluating potentially cancelling cross terms.

With all original biases and LayerNorm affine parameters present, the isolated block has

\[
N_{\rm old}=nd+n+2n+mn+m=n(d+m+3)+m
\]

parameters. The contracted block has

\[
N_{\rm new}=m(d+1)+k(d+1)+m.
\]

For \(d=82,n=512,m=512\), these counts are 306,176 and 49,897, respectively, approximately 83.7% fewer parameters. These counts describe a synthetic architecture. Any source modules needed by other consumers must be included in the net storage comparison. Several eligible downstream affine consumers can share the same normalisation denominator.

The supported block contains one-dimensional standard LayerNorm directly between standard `nn.Linear` modules, with matching finite materialised parameters. The original divisor \(n\) and epsilon are retained. Custom hooks and instance forwards are rejected; intervening GELU, other elementwise nonlinearities, dropout, residual additions or input-dependent LayerNorm affine parameters need separate analysis.

Exactness is in real arithmetic. QR, centering, coefficient casting and runtime reassociation introduce numerical differences. Offline conversion runs in float64 and rejects coefficients that overflow their storage dtype. Float16/bfloat16 coefficients keep their storage type while this block uses float32 arithmetic internally. That changes numerical evaluation relative to the original, so bit identity is not promised. Automatic mixed precision is unaudited.

Twenty-four focused tests check output and input-gradient agreement in float64/float32, biases and absent biases, nonuniform signed LayerNorm scales, absent LayerNorm affine parameters, zero and constant producers, rank-deficient and wide matrices, float16/bfloat16 storage, frozen parameters and invalid-input rejection. The method has not been applied to an actual Mol-JEPA block with this exact topology.

Independent fine-tuning of \(D,R,c\) changes the original coupling between numerator and denominator. Inference equivalence at conversion does not establish the same optimisation trajectory or function class during training.

The underlying scalar-statistic idea has close precedent in [QK-Normed MLA](https://arxiv.org/abs/2606.16310), which preserves an exact latent attention path with post-projection RMSNorm. This prototype applies the algebra to generic affine–LayerNorm–affine chains, with centering, biases and a QR denominator. The contribution is this exact compiler transformation; the underlying normalisation identity is not a new theorem.
