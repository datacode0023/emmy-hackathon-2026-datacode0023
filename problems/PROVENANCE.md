# Problem set provenance

Generated with the published `emmy-ml==0.3.11` wheel for the reference GPU (RTX 4090, sm_89),
using Python 3.12, Torch 2.14.0+cpu and NumPy 2.5.3:

    python tools/generate.py tools/expressions_open.py --emmy <emmy> --out problems

The IRs depend on the GPU emmy schedules against — a different target can change them
structurally (kernel counts, fusion), not just cosmetically. `--target` (default `sm_89`) sets
it, and the generator runs emmy with no machine-local tuning data (tune DB, online prior,
goldens), so one emmy release gives the same IR on any machine. The category-2 reductions pin
their schedule with `knobs`; unpinned, emmy may split them across blocks with an atomic add.

Packaging fills `KernelOp.name` from the CUDA launch entry when necessary, so the IR, filenames
and `meta.json` agree. `meta.json.smem_bytes` is the **dynamic** allocation: zero for kernels
whose shared arrays are static (up to 48 KiB), and the pool size above that limit, matching
Emmy 0.3.11's renderer and runtime.

Checked on 2026-09-19 with CUDA 13.0 on an RTX 5090: every problem's reference kernel, built
from its `.ref.cu`, passes `submit.py --arch sm_120` under the launch in `meta.json`, and every
one also assembles for `sm_89`. Every IR loads with 0.3.11 in an environment without Torch.
The pointwise mock compiler and the worked reduction example pass as well. `expected.npz`
is byte-identical to the previous set, and the pinned reductions' IR text is unchanged.
The set has not been run on an RTX 4090.

The hidden set uses the same tool with `--hidden` (IR + meta only) against a private
expression file.
