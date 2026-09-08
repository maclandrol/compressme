# Scalar hinge normal form: rejected State SE experiment

This prototype compresses a scalar-input block `Linear(1,h) → ReLU/LeakyReLU → Linear(h,m)` without training. The algebra preserves its function over all real scalar inputs. The actual State SE candidate failed the required numerical tests and is **not enabled in the production model**.

Write the block as

\[
f(x)=d+\sum_j C_j L_\alpha(a_jx+b_j),\qquad
L_\alpha(z)=\alpha z+(1-\alpha)\max(z,0).
\]

For each nonzero slope, set \(t_j=-b_j/a_j\),
\(q_j=\alpha\) when \(a_j>0\), and \(q_j=1\) when \(a_j<0\). Then

\[
L_\alpha(a_jx+b_j)
=q_j(a_jx+b_j)+(1-\alpha)|a_j|\max(x-t_j,0).
\]

Zero-slope units contribute constants. Summing gives

\[
f(x)=c+sx+\sum_j J_j\max(x-t_j,0),\qquad
J_j=(1-\alpha)|a_j|C_j.
\]

Negative first-layer slopes are included through their affine correction. No assumption that counts are positive is required. This is the standard hinge representation of a piecewise-linear function; the experiment tests whether it is useful as a compressor.

With biases in both original layers and no removable units, storage changes from \(mh+2h+m\) to \(mh+h+2m\) values, saving \(h-m\). The implementation counts knots, every output coefficient, slope and intercept. It retains no source weights. A cached interval evaluator would generally need both slopes and intercepts for each interval, approximately \(2mh+h+2m\) values. That larger representation is not used.

The derivation is exact before coefficient rounding. The implementation stores coefficients in the original dtype and computes an exact-rational residual certificate for those stored values. For output \(i\), with \(\Delta\) denoting stored minus ideal coefficients,

\[
|\widehat f_i(x)-f_i(x)|\le E_{0,i}+E_{1,i}|x|,
\]

\[
E_{1,i}=|\Delta s_i|+\sum_j|\Delta J_{ij}|,
\qquad
E_{0,i}=|\Delta c_i|+
\sum_j\left(|\Delta J_{ij}|\,|t_j|+
|\widehat J_{ij}|\,|\Delta t_j|\right).
\]

The certificate follows from the 1-Lipschitz property of ReLU and
\(\max(x-t,0)\le |x|+|t|\). Reported bound coefficients are rounded upward. It covers real evaluation of the rounded coefficients. **It excludes floating-point execution rounding and overflow in both implementations**, so it does not establish a universal output tolerance for PyTorch. Numerical probes include signed inputs, all knots and their neighborhoods, empty tensors, and different leading dimensions. Failure returns an unchanged copy of the source block.

State SE's `1→512→10` count encoder would shrink from 6,154 to 5,652 values, saving 502 values or 2,008 float32 bytes. Its strict local gate failed: maximum absolute error was `0.00067138671875`, although relative L2 error was about `5.05e-7`. The limit remained `1e-5`; see [the local report](state_se_scalar_local.json).

A separate diagnostic inserted the rejected candidate into the complete model without accepting it. Four of 105 ordinary output comparisons failed, with maximum absolute error `2.664327621459961e-5`. Failures included dataset logits and all-token outputs at long sequence lengths. Additional signed and large-count probes passed their 39 comparisons, and the empty batch matched exactly. Those passes do not override the observed failures; see [the complete diagnostic](state_se_scalar_whole_diagnostic.json). This experiment remains preserved for analysis, with no production promotion or speed claim.

The implementation and 12 focused tests are in [scalar_piecewise.py](scalar_piecewise.py) and [test_scalar_piecewise.py](tests/test_scalar_piecewise.py). From the repository root, run `python -m pytest experiments/tests/test_scalar_piecewise.py`. CPU float32 and float64 were tested. Numerical error calculations explicitly transfer outputs to CPU before converting to float64, avoiding unsupported float64 work on MPS; this change does not constitute an MPS validation. The operator supports frozen inference only because PyTorch's chosen input derivatives at activation knots can change under the rewrite.
