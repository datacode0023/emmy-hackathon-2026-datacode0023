# The task

Write a compiler: **Kernel IR in, PTX out.**

```bash
python compile.py path/to/<kernel>.ir.json > out.ptx
```

That exact invocation is the submission contract. Your repo must contain a `compile.py` at its
root that reads one Kernel IR json path as `argv[1]` and writes PTX to stdout. We run every
submission mechanically against a hidden set of IRs — if your compiler needs a different
invocation, it cannot be evaluated.

**One rule: the translation must be yours.** Emmy itself contains a Kernel IR → CUDA C++
compiler (`render_kernelop`), and chaining it with `nvcc -ptx` produces working PTX for any IR.
Submissions that lower the IR through emmy's renderer, `emmy compile`, `nvcc`, or any other
existing compiler are not eligible — the task is writing the IR → PTX translation, and judges
read every submission's code. Using emmy to *load* the IR (`Graph.from_dict`) is of course fine;
that's what it's required for.

## What your PTX must satisfy

- **Entry name** = the kernel's `name` in the IR (`k_sum_1_reduce`, not your own naming).
- **Parameters** = the kernel's `arg_order` **from `meta.json`** (shipped with every problem,
  hidden set included; buffers are `.u64` pointers). In today's problems it happens to equal the
  node's `inputs` followed by its outputs, but emmy may reorder arguments between pipeline stages,
  so `meta.json` is the contract — not the IR node's `inputs` list.
- **Correct under the prescribed launch.** `meta.json` gives grid, block and shared-memory size,
  derived from the IR's own schedule. Your PTX runs under *that* geometry — e.g. if the IR tiles
  1024 elements over one block of 128 threads, each thread must cover 8 elements.
- **Correct results** within the per-problem tolerance (`rtol`/`atol` in `meta.json`). Exact
  equality is deliberately not required: reductions sum in a different order than numpy, and both
  answers are legitimate.

## The categories

| # | category | anchor problem | what it forces your compiler to handle |
|---|---|---|---|
| 1 | pointwise | `p00_scalar_add` | `Tile`, `Load`, elementwise ops, `Write`; unrolled bodies + **vector stores/loads**; constants as kernel args (`p02`) |
| 2 | reductions | `r01_array_sum` | `StridedLoop`, `Accum`, `WarpShuffle`, `Smem`, `Sync`, `TreeHalve`; longer strides (`r02`), 2-D rows (`r04`) |
| 3 | matmul | `m01_matmul_512` | 2-D indexing, accumulator tiles, `cp.async` copies into shared memory; **atomics + `zero_outputs`** (`m02`) |
| 4 | fused | `f01_rmsnorm` | a reduction and an elementwise pass in one kernel; `f02` is full online softmax with two coupled accumulators — the hardest kernel before category 5 |
| 5 | attention-like | `a01_attention` | softmax fused into a matmul, in one kernel |

Every hidden-set problem is a variation within these categories — different shapes, sizes and op
mixes. Nothing outside them.

## Rough structure of a compiler

Every working submission has roughly these stages:

1. **Load** — read the IR with `Graph.from_dict`, find the `KernelOp`, take the entry name from
   it and the parameter order from `meta.json`'s `arg_order`.
2. **Prologue** — emit `.version`/`.target`, the signature, param loads, `cvta.to.global`, the
   thread id, and the bounds guard derived from the Tile's `N`.
3. **Walk** — visit the body statements in order; keep a map from IR value names to registers.
   Index expressions are small trees (`Var`/`Literal`/`BinaryExpr`) — compile them recursively.
4. **Emit** — one translation per statement kind (the table below).

**Registers:** hand out fresh virtual registers (`%f1, %f2, ...`) and declare the counts at the
top — `ptxas` performs the real register allocation downstream. Do not build liveness analysis;
it is not where the points are.

## The IR vocabulary

The common statement kinds below are the starting point. The worked example
(`examples/k_sum_1_reduce.ptx`) shows each translation in context.

| IR node | meaning | typical PTX |
|---|---|---|
| `Tile (N=...)` | the block: N cells of work | launch geometry + `%tid.x` |
| `StridedLoop` | thread strides an axis | label + `setp` + `@%p bra` + stride add |
| `Load` | read global memory | `ld.global.f32` |
| `Accum` | running accumulator in a register | `add.f32 %f, %f, %x` |
| `Assign` / elementwise | arithmetic | `mul.f32`, `fma`, `ex2.approx`, ... |
| `WarpShuffle` | combine across a warp via registers | `shfl.sync.bfly.b32` ladder |
| `Smem` | shared scratch per block | `.shared` + `st.shared` / `ld.shared` |
| `Cond` | predicated section | `setp` + `@%p bra` |
| `Sync` | block barrier | `bar.sync 0` |
| `TreeHalve` | combine per-warp partials | shared-memory tree |
| `Write` | store a result | `st.global.f32`; a grouped write `buf[i] = (v0, v1, v2, v3)` is a **vector store** `st.global.v4.f32` (v2 for pairs); `red.global.add.f32` when it accumulates |

The table above is the working set. The complete reference follows: every Kernel IR statement and expression,
its arguments, and what it does. Operations marked `*` appear in **no** open or hidden problem — a compiler that
handles the unmarked ones covers every evaluation task. The short table also leaves out `Loop` (matmul and
attention), `Let` (softmax and attention bind the literal `1.0` with it), and, in matmul only, `Init` and the three
`CpAsync*` copies (a `cp.async` copy into shared memory, its commit, and its wait).

## Kernel IR reference

### Program

| Operation | Arguments | Description |
| --- | --- | --- |
| `KernelOp` | `name`, `body`, `zero_delegated`, `serial` | One GPU kernel whose parameters are derived from the body; `zero_delegated` lists outputs whose zero-init a predecessor kernel performs. `serial` lists axes launched one coordinate at a time, with coordinates passed as runtime integer parameters; it is empty in every problem. |

### Scalar compute (shared with Loop IR and Tile IR)

| Operation | Arguments | Description |
| --- | --- | --- |
| `Load` | `names`, `input`, `index`, `dtype` | Read one value, or several consecutive values, from a buffer into SSA names. |
| `Assign` | `name`, `op`, `args`, `dtype` | Bind `name = op(args)` for one elementwise operation over SSA names. |
| `Let` | `name`, `value`, `dtype` | Bind one pure expression to an SSA name: a scalar literal, a precomputed integer index, or a `FlatIndex` buffer offset. |
| `Accum` | `name`, `value`, `op`, `dtype`, `axes`, `base` | Fold `value` into a reduce accumulator that starts at the operation's identity and is visible after the loop. |
| `Pre`* | `name`, `carrier`, `index` | Read one cell of carried state: the previous step's value inside its carrying loop (the seed on the first step), or the final value after the loop. |
| `Carry`* | `name`, `value`, `index`, `seed`, `dtype` | Set a carried-state cell's next value. Steps run in order; reads within a step see the previous state, and skipped cells retain their values. |
| `Init` | `name`, `identity`, `dtype` | Declare carried state seeded with a literal, so a masked tail can select the identity by name or a chunk loop can update it in place. |
| `Select`* | `name`, `branches` (`value`, `select`) | Bind `name` to the value of the branch whose coordinate predicate holds. |
| `Write` | `output`, `index`, `values`, `value_dtype`, `atomic`, `swizzle` | Store one value, or several consecutive values, into a buffer, optionally as an atomic add. |
| `ZeroPrologue`* | `dst`, `words` | Zero another kernel's atomic accumulator from an earlier kernel on the same stream, replacing a per-launch memset. |
| `Loop` | `axis`, `body`, `unroll`, `seed` | Run `body` once per value of `axis`, folding any accumulators inside. |
| `StridedLoop` | `axis`, `start`, `step`, `body`, `unroll`, `end`, `seed` | Loop from `start` in steps of `step`, the form cooperative threads use to stride an axis. |
| `Cond` | `cond`, `body`, `else_body` | Run `body` when the predicate holds, otherwise `else_body`. |

### Launch, shared memory and barriers

| Operation | Arguments | Description |
| --- | --- | --- |
| `Tile` | `axes`, `body`, `block_threads`, `aux_threads`, `raster_axes`, `raster_group`, `raster_orient` | Map the iteration space onto the thread grid, decode each thread's axis indices, and run `body` once per cell. |
| `Smem` | `name`, `extents`, `dtype`, `align` | Declare one `__shared__` array for the CTA, with optional byte alignment. |
| `Sync` | `barrier_id`, `count`, `warp` | Emit a barrier: CTA-wide `__syncthreads()`, warp-scope `__syncwarp()`, or a named barrier over `count` threads. |

### Transport into shared memory

| Operation | Arguments | Description |
| --- | --- | --- |
| `CpAsyncCopy` | `smem`, `smem_index`, `src`, `src_index`, `nbytes`, `swizzle`, `lane_index`, `lane_rows` | Copy 4, 8 or 16 bytes from a global buffer straight into shared memory with one `cp.async` instruction. |
| `CpAsyncCommit` | — | Close the thread's preceding `cp.async` copies into one commit group. |
| `CpAsyncWait` | `group` | Block until at most `group` `cp.async` groups remain in flight. |
| `TmaDescriptor`* | `name`, `src_buf`, `src_shape`, `box_extents`, `swizzle`, `dtype` | Declare the host-built tensor-map descriptor a TMA box copy reads; renders nothing in the kernel. |
| `TmaLoad`* | `smem`, `smem_index`, `desc`, `coords`, `mbar`, `mbar_slot` | Copy one box of a global buffer into shared memory with a single-thread `cp.async.bulk.tensor` instruction. |
| `MbarrierInit`* | `mbar`, `count`, `slot` | Initialize one mbarrier, or one slot of an mbarrier array, once in the kernel prologue. |
| `MbarrierArriveExpectTx`* | `mbar`, `bytes_`, `slot` | Announce the byte count of the TMA copy about to arrive on this mbarrier. |
| `MbarrierArrive`* | `mbar`, `slot` | Signal arrival without a byte count, used by consumer warps to mark a slot empty. |
| `MbarrierWait`* | `mbar`, `phase`, `slot` | Block until the mbarrier's parity flips for `phase`, so the copied data is visible. |
| `SetMaxNReg`* | `count`, `direction` | Shrink or grow the calling warp's register budget (sm_90 and newer). |

### Cross-thread combine

| Operation | Arguments | Description |
| --- | --- | --- |
| `WarpShuffle` | `state`, `state_b`, `combine_states`, `length`, `dtype` | Fold per-lane partial states over `length` lanes with a register-only XOR shuffle butterfly. |
| `TreeHalve` | `bufs`, `state`, `state_b`, `combine_states`, `length`, `tid_var`, `dtype`, `barrier_id`, `barrier_count`, `inner` | Reduce a power-of-two shared-memory tree of partial states and broadcast the result to every thread, for combines wider than a warp. |

### Tensor core: register fragments (`mma.sync`)

| Operation | Arguments | Description |
| --- | --- | --- |
| `RegFragment`* | `name`, `role`, `shape`, `dtype`, `count`, `nregs` | Declare a per-thread register array for the A, B or C operand of one atom cell; C starts at zero. |
| `LdmatrixLoad`* | `frag`, `src_buffer`, `src_index`, `role`, `ldm`, `swizzle`, `staged`, `gmem_guard`, `k_zero`, `b_trans`, `pair_frag`, `byte_slab`, `scale_buffer`, `scale_index`, `scale_ldm`, `fragment_layout`, `fragment_index` | Load one operand fragment into registers from shared memory, or straight from global memory, with guards and optional byte-slab dequantization. |
| `BlockScaleLoad`* | `frag`, `src_buffer`, `src_index`, `role`, `ldm` | Load one lane's block scales for the block-scaled fp4 `mma`. |
| `MmaSyncPtx`* | `c_frag`, `a_frag`, `b_frag`, `shape`, `ab_dtype`, `c_dtype`, `b_row_major`, `sfa_frag`, `sfb_frag` | Issue one `mma.sync` instruction computing `c = a · b + c` over the fragments. |
| `FragmentPromote`* | `dst`, `src` | Add a packed f16 accumulator into its f32 shadow fragment and zero the f16 one. |
| `FragmentApply`* | `out`, `op`, `args`, `kinds`, `in_place`, `layout`, `post`, `row_base`, `col_base` | Apply one elementwise operation to every element of a C fragment; each argument is a fragment, a per-row pair, a uniform scalar, a predicate over the element's coordinates (a mask) or a global-memory load at those coordinates (a bias). |
| `FragmentRowReduce`* | `top`, `bot`, `frags`, `op`, `layout`, `dtype` | Reduce a warp's C fragments per row, producing the row pair `FragmentApply` broadcasts. |
| `FragmentRepack`* | `frag`, `srcs`, `ab_dtype`, `fragment_layout`, `part` | Convert C fragments into one 16-bit A fragment, so a result can feed the next contraction. |
| `RegStore`* | `dst_buffer`, `dst_index`, `frag`, `shape`, `ldm`, `epilogue`, `extra_frags`, `m_guard`, `n_guard`, `atomic`, `fragment_layout`, `volta_interleaved`, `fragment_index`, `swizzle`, `row_dim`, `col_dim` | Store a C fragment to the output buffer per lane, after an optional fused pointwise epilogue and a downconvert. |

`RegStore.epilogue` is a pure `Lambda`* (`params`, `body`, `results`): the projection tail's own `Load`, `Assign` and
`Select` statements, evaluated per fragment element with the leading params bound to `frag` and `extra_frags`.
`FragLayout`* (`n_elems`, `elem_row`, `reduce_xors`, `lane_decl`, `lane_names`, `row_off`, `col_off`) is the per-atom
geometry the fragment operations read.

### Tensor core: warp groups (`wgmma`)

| Operation | Arguments | Description |
| --- | --- | --- |
| `WgmmaDescriptor`* | `name`, `smem`, `smem_index`, `swizzle`, `lbo_bytes`, `sbo_bytes` | Build the 64-bit shared-memory matrix descriptor a `wgmma` operand is read through. |
| `WgmmaMma`* | `c_frags`, `b_desc`, `shape`, `ab_dtype`, `a_desc`, `a_frag`, `scale_d`, `trans_a`, `trans_b` | Issue one asynchronous `wgmma.mma_async` cell across the four warps of a group into their C fragments. |
| `WgmmaFence`* | — | Order the group's earlier register and shared-memory accesses before the next cell. |
| `WgmmaCommit`* | — | Close the cells issued since the last commit into one group. |
| `WgmmaWait`* | `group` | Block until at most `group` committed groups remain in flight. |

### Expressions

Indices, predicates and loop bounds are expressions, not statements.

| Expression | Arguments | Description |
| --- | --- | --- |
| `Var` | `name` | Reference an axis variable or an SSA name. |
| `Literal` | `value`, `dtype` | A numeric constant. |
| `BinaryExpr` | `op`, `left`, `right` | An arithmetic, comparison or logical operation on two expressions. |
| `Builtin`* | `name` | A GPU built-in variable such as `threadIdx.x` or `blockIdx.x`. |
| `FuncCallExpr`* | `name`, `args` | A call to a registered elementwise function such as `exp` or `maximum`. |
| `TernaryExpr` | `cond`, `if_true`, `if_false` | Choose between two expressions by a predicate. |
| `CastExpr`* | `dtype`, `expr` | Convert an expression to another scalar type. |
| `FlatIndex`* | `buffer`, `index` | The row-major element offset of a coordinate in a buffer, flattened against the buffer's shape at render time. |

## Facts that will save you a day

- **Guard your threads.** The launch may carry more threads than the Tile has cells (`p00`: 64
  cells, 256 threads). The guard `if (thread_id < N)` is NOT in the IR — derive it from the
  Tile's `N`, or the extra threads read out of bounds.
- `pip install emmy-ml==0.3.11` is required to *read* the IR: the kernel body deserializes through
  Emmy's classes. `Graph.from_dict` gives you typed objects; never parse the body string yourself.
  Emmy alone is enough (no torch) — torch is only needed if you generate your own IRs.
- The graph level of the json is plain: `inputs`, `outputs`, per-node `shape`/`dtype` — your
  signatures come from there for free.
- `ConstantOp` nodes carry `"value"` — they arrive as 1-element buffer args at launch.
- PTX rejects **non-ASCII anywhere**, comments included.
- Every PTX file must start with `.version` / `.target` / `.address_size` — and the version must
  be new enough for the target: **`sm_89` needs `.version 7.8` or later** (use `8.0`). ptxas's
  error for this is clear, but only after your first submission bounces.
- Loads can be **vectorized** too: `a, b, c, d = load x[i]` in the IR is one `ld.global.v4.f32`.
- Shared memory declared in the PTX (static) needs no `smem_bytes` at launch; `meta.json` already
  accounts for this.
- Want more practice inputs? Install torch as well, then
  `EMMY_DUMP_DIR=out emmy compile -c "<torch expr>" --ir kernel` generates IRs for your own
  expressions. Known limitation: `x.T` does not trace — pre-transpose.

## What is explicitly NOT required

No optimization. No register-allocation cleverness (`ptxas` does that). No dynamic shapes — every
problem is static. No TMA / tensor-core instructions. Correctness under the given launch is the
whole game; speed is a tiebreaker for judges, not a gate.
