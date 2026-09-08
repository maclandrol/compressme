# Exact contraction across LayerNorm

An affine expansion followed by LayerNorm and another affine map can be evaluated without constructing the expanded representation. The nonlinearity only needs one scalar norm of that representation. The remaining computation can be contracted in advance.

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

Take the **full reduced QR factorisation** \(T=UR\), retaining every row of \(R\). Its dimension is \(k\times(d+1)\), where \(k=\min(n,d+1)\). No estimated numerical rank or truncation threshold is used. Since \(U^\top U=I\),

\[
\|Ph\|^2=\|T\bar x\|^2=\|R\bar x\|^2.
\]

Set \(D=W\operatorname{diag}(\gamma)T\) and \(c=W\beta+b\). Then

\[
\boxed{
f(x)=\frac{D\bar x}{\sqrt{\|R\bar x\|^2/n+\epsilon}}+c.
}
\]

This is the same mathematical function for every input. It stores \(D,R,c\), and needs neither the original expanded producer nor the original LayerNorm parameters. A Gram matrix \(T^\top T\) gives an equivalent quadratic form, but QR avoids forming a Gram matrix and evaluating potentially cancelling cross terms.

With all original biases and LayerNorm affine parameters present, the isolated block has

\[
N_{\rm old}=nd+n+2n+mn+m=n(d+m+3)+m
\]

parameters. The contracted block has

\[
N_{\rm new}=m(d+1)+k(d+1)+m.
\]

For \(d=82,n=512,m=512\), these counts are 306,176 and 49,897, respectively, approximately 83.7% fewer parameters. This is a synthetic architectural example. A compiler must count source modules retained for other consumers before reporting a net saving. The normalisation denominator can be shared across several eligible downstream affine consumers.

The implementation supports one-dimensional standard LayerNorm between standard `nn.Linear` modules, requires matching finite materialised parameters, and rejects custom hooks or instance forwards. It does not cross intervening GELU, other elementwise nonlinearities, dropout, residual additions or input-dependent LayerNorm affine parameters without further analysis. It retains the original divisor \(n\) and epsilon.

Exactness is in real arithmetic. QR, centering, coefficient casting and runtime reassociation introduce numerical differences. The prototype converts offline in float64 and rejects coefficients that overflow their storage dtype. Float16/bfloat16 coefficients retain their storage type but use float32 arithmetic internally for this block. That path also changes numerical evaluation relative to the original; it does not promise bit identity. Automatic mixed precision is not audited.

Twenty-four focused tests check output and input-gradient agreement in float64/float32, biases and absent biases, nonuniform signed LayerNorm scales, absent LayerNorm affine parameters, zero and constant producers, rank-deficient and wide matrices, float16/bfloat16 storage, frozen parameters and invalid-input rejection. The method has not been applied to an actual Mol-JEPA block with this exact topology.

Independent fine-tuning of \(D,R,c\) changes the original coupling between numerator and denominator. Inference equivalence at conversion does not establish the same optimisation trajectory or function class during training.

The underlying scalar-statistic idea has close precedent in [QK-Normed MLA](https://arxiv.org/abs/2606.16310), which preserves an exact latent attention path with post-projection RMSNorm. This prototype applies the algebra to generic affine–LayerNorm–affine chains, with centering, biases and a QR denominator. It should be presented as an exact compiler transformation, without claiming a new normalisation theorem.
