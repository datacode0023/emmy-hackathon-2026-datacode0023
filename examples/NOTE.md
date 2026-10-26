# Worked example

`k_sum_1_reduce.ptx` — hand-written PTX for `problems/r01_array_sum`. Entry name and signature
follow the contract; comments map each section to the IR statement it implements (strided loop,
warp shuffle, shared memory, barrier, tree, guarded write).

    python submit.py r01_array_sum examples/k_sum_1_reduce.ptx

It shows what your compiler's *output* should look like — the task is the program that produces
this automatically, for any IR.
