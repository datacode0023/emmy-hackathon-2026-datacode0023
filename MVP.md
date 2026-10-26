# MVP — the minimum bar for a valid submission

Your compiler, invoked as

```bash
python compile.py <kernel>.ir.json > out.ptx
```

passes **every problem in categories 1 and 2** through `submit.py`:

```
p00_scalar_add  p01_vector_add  p02_scale_shift  p03_relu_mul       (pointwise)
r01_array_sum   r02_big_sum     r03_mean         r04_row_sum        (reductions)
```

"Passes" means: `ptxas` accepts the PTX, the entry point is named as the IR names it, and the
outputs match `expected.npz` within the problem's tolerance under the launch geometry in
`meta.json`.

That's eight problems and three real ideas: straight-line pointwise PTX with
unrolled bodies and vector stores, constants as kernel args, and the
reduction machinery (strided loops, warp shuffles, shared memory, barriers). Achievable in the
week by someone who has never written PTX — but category 1 is a real day of work, not an hour.

Categories 3–5 (matmul, fused, attention-like) are how submissions differentiate — judges weigh
how far up the ladder a compiler gets, how it handles the hidden set, and code quality.

A complete submission is the source code plus a 3 to 5 minute video — see
[Submitting](README.md#submitting).

Notes:

- The MVP is judged by running **your compiler**, not PTX files you generated once. Hardcoded
  outputs for the open problems score nothing on the hidden set.
- Partial credit is real: a clean compiler that only does category 1 correctly beats a messy one
  that half-does three categories.
