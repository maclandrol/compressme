# Clarification of historical equality terminology

The archived manual reports use `max_abs=0` and `torch.equal`, which establish
equal numerical tensor values. Their original notes called this "bitwise",
but those checks do not distinguish positive and negative zero. The thirty
indexed historical files remain unchanged so their evidence can be inspected.

Treat those historical claims as exact numerical equality. The later generic
compiler reports explicitly compare tensor bytes after copying logical tensor
values to contiguous CPU storage, including scalar, integer and Boolean leaves.
That new evidence is recorded separately; it is not retroactively attributed to
the manual prototype or its timing gates.
