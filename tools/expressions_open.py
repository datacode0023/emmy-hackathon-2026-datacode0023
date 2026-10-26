"""The OPEN problem set: torch expressions per category, easy to hard.

Each spec:
  id        problem folder name (category prefix orders them)
  code      the expression handed to `emmy compile -c` (this defines the Kernel IR)
  ref       numpy function computing the expected output(s) from the seeded inputs,
            taking arrays in the order of the traced graph's `inputs` list
  rtol/atol correctness tolerance (looser for reductions: summation order differs
            between a GPU tree reduction and numpy, and neither is "wrong")
  knobs     optional EMMY_KNOBS schedule pins, for problems whose category depends on
            the schedule emmy would otherwise pick

Rules when adding expressions:
  - `x.T` does not trace - pre-transpose instead.
  - Constants (e.g. `* 2.0`) become ConstantOp kernel args; resolved automatically.
  - Trace every new expression once via tools/generate.py before committing it.

The HIDDEN set lives OUTSIDE this repo on purpose - same categories, different
shapes/ops. Submissions are evaluated against it after the event.
"""

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


OPEN = [
    # ---- category 1: pointwise -------------------------------------------
    # p00: smallest size -> simplest IR. p01 up have unrolled bodies and vector stores.
    dict(id="p00_scalar_add",
         code="torch.randn(128) + torch.randn(128)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="p01_vector_add",
         code="torch.randn(1024) + torch.randn(1024)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="p02_scale_shift",
         code="torch.randn(4096) * 2.0 + 1.0",
         ref=lambda x: [x * np.float32(2.0) + np.float32(1.0)], rtol=1e-6, atol=1e-6),
    dict(id="p03_relu_mul",
         code="x = torch.randn(2048); y = torch.randn(2048); torch.relu(x) * y",
         ref=lambda x, y: [np.maximum(x, 0) * y], rtol=0, atol=0),

    # ---- category 2: reductions ------------------------------------------
    # Pinned to one block of 128 cooperating threads. Unpinned, emmy may split a
    # reduction across blocks with an atomic add - a category-3 concept.
    dict(id="r01_array_sum",
         code="torch.randn(1024).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="r02_big_sum",
         code="torch.randn(65536).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)], rtol=1e-3, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="r03_mean",
         code="torch.randn(32768).mean()",
         ref=lambda x: [x.astype(np.float64).mean().astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="r04_row_sum",
         code="torch.randn(64, 512).sum(-1)",
         ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),

    # ---- category 3: matmul ----------------------------------------------
    dict(id="m01_matmul_512",
         code="torch.randn(512, 512) @ torch.randn(512, 512)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),
    dict(id="m02_matmul_rect",
         code="torch.randn(128, 256) @ torch.randn(256, 64)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),
    # No matvec: emmy currently mis-traces (M,K) @ (K,) - re-add if fixed upstream.

    # ---- category 4: fused -----------------------------------------------
    dict(id="f01_rmsnorm",
         code="x = torch.randn(64, 1024); w = torch.randn(1024); "
              "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         ref=lambda x, w: [(x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5)
                            * w).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="f02_row_softmax",
         code="torch.softmax(torch.randn(64, 256), dim=-1)",
         ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-5),
    dict(id="f03_l2norm",
         code="x = torch.randn(64, 512); x / torch.sqrt((x * x).sum(-1, keepdim=True))",
         ref=lambda x: [(x / np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True))).astype(np.float32)],
         rtol=1e-3, atol=1e-4),

    # ---- category 5: attention-like --------------------------------------
    # K is supplied PRE-TRANSPOSED: `q @ k.T` does not trace.
    dict(id="a01_attention",
         code="q = torch.randn(64, 64); kt = torch.randn(64, 64); v = torch.randn(64, 64); "
              "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 8.0)
                                @ v.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="a02_attention_128",
         code="q = torch.randn(128, 32); kt = torch.randn(32, 128); v = torch.randn(128, 32); "
              "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 5.656854)
                                @ v.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
]
