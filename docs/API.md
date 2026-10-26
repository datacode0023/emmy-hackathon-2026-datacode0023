# The runner API

`submit.py` wraps all of this — read on only if you want to call the server directly.
Base URL: `http://runner.riftstack.ai:48701` (the default in `submit.py`). GPU: RTX 5090 (`sm_120`).

## GET /health

    { "ok": true, "can_compile": true, "can_run": true,
      "ptxas": "/usr/local/cuda/bin/ptxas",
      "gpu": "NVIDIA GeForce RTX 5090 (compute capability 12.0, sm_120)" }

## POST /compile

    in : { "ptx": "<source text>", "arch": "sm_89", "flags": [] }   // arch defaults to the runner's GPU
    out: { "ok": true, "cubin_id": "<sha256>", "bytes": 3360,
           "entry_points": ["k_sum_1_reduce"],
           "registers": { "k_sum_1_reduce": 10 } }
    or : { "ok": false, "stage": "ptxas", "stderr": "<verbatim, untruncated>" }

A failed compile is HTTP **200** with `ok:false` — the request was well-formed, the PTX wasn't.
A malformed request is 422. `stderr` is ptxas's output verbatim: it is your compiler feedback.
`cubin_id` is content-addressed over (ptx, arch, ptxas version, flags) — identical submissions
compile once, and the id is your handle for `/run`. `registers` is per-entry register usage
from `ptxas -v` — lower is generally better for occupancy.

## POST /run

    in : { "cubin_id": "...",            // or "ptx" to compile-and-run in one call
           "entry": "k_sum_1_reduce",    // which kernel in the module
           "grid": [1,1,1], "block": [128,1,1], "smem_bytes": 16,
           "zero_outputs": [],           // buffers re-zeroed before EVERY launch
           "warmup": 0, "iters": 1,      // timed reps; median is reported
           "max_return": 4096,           // cap on returned output values
           "args": [ {"name":"x","dtype":"f32","shape":[1024],"seed":0},
                     {"name":"sum_1","dtype":"f32","shape":[1],"output":true} ] }
    out: { "ok": true, "median_us": 69.6, "iters": 1,
           "outputs": { "sum_1": { "shape":[1], "dtype":"f32", "sum": -50.37,
                                   "values": [-50.37], "truncated": false } } }

All launch values come from the problem's `meta.json` — you never invent them.

### Argument forms

| written as | meaning |
|---|---|
| `"seed": 0` | input buffer, filled with `np.random.default_rng(seed).standard_normal(shape)` |
| `"data": [..]` | input buffer with exact values (constants arrive this way) |
| `"output": true` | allocated zeroed, copied back after the launch |
| `"scalar": true, "value": v` | passed by value, not as a pointer |

`dtype` ∈ `f32, f16, f64, i32, i64`. Args are **positional** — order must match the kernel
signature. `zero_outputs` matters for kernels that accumulate via atomics: without it, each
launch of a timing loop adds into the previous result.

### Failure stages

`ptxas` (your PTX didn't assemble) · `lookup` (unknown cubin_id) · `run` (launch failed — e.g.
wrong entry name, or the launch timed out) · `request` (neither ptx nor cubin_id given) ·
`busy` (too many launches queued — retry in a few seconds; `submit.py` does this for you).

### Limits

| | |
|---|---|
| `arch` | defaults to the runner's GPU (`default_arch` in `/health`); `submit.py` uses that default |
| `flags` | at most 8, from `-O0`..`-O3`, `--maxrregcount=N`, `-lineinfo` |
| `ptx` | at most 2,000,000 characters, in `/compile` and `/run`; `ptxas` gets 30 s |
| a launch | 20 s for warmup, timed runs and copies together; a launch that runs longer is stopped and reported as a `run` failure |
| `warmup` / `iters` | at most 20 / 200 |
| buffers | at most 2^26 elements across all args of one launch, counting `data` |
| `max_return` | at most 2^22 values across all outputs of one launch |
| `smem_bytes` | at most 99 KiB (the per-block limit of the runner's GPU) |

Out-of-range values are a 422. Launches run one at a time. A launch that fails retires its
worker process, so an illegal memory access in one kernel never affects the next launch.

Timing note: `median_us` is indicative (no clock locking) — good feedback, not a benchmark.
