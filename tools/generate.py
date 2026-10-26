#!/usr/bin/env python3
"""Generate problem packages from a list of torch expressions.

For each expression this runs `emmy compile -c <code> --ir cuda` with a dump dir,
then packages, per problem:

    problems/<id>/
        <kernel>.ir.json    the Kernel IR  - the INPUT to a participant's compiler
        <kernel>.ir.txt     the same, pretty-printed for humans
        <kernel>.ref.cu     emmy's generated CUDA C++ - the reference answer
        meta.json           launch descriptor(s) + args + tolerance - what submit.py uses
        expected.npz        expected output arrays, computed from the seeded inputs

Usage:
    python tools/generate.py --emmy /path/to/emmy --out problems tools/expressions_open.py
    python tools/generate.py --only p01_vector_add ...

The IRs depend on the target GPU. Regenerate on (or for) the machine the server
will run on before publishing - see PROVENANCE in the output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def load_specs(path: Path):
    spec = importlib.util.spec_from_file_location("exprs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OPEN


def flat(dimspec) -> int:
    """CudaOp grid/block dims are lists of factors; multiply them out."""
    if isinstance(dimspec, int):
        return dimspec
    total = 1
    for f in dimspec:
        if not isinstance(f, (int, float)):
            raise ValueError(f"symbolic launch dim {f!r} - dynamic shapes not supported here")
        total *= int(f)
    return total


def trace(emmy: str, spec: dict, dump: Path, target: str) -> None:
    # No machine-local evidence (tune DB, online prior, goldens), so one emmy release gives the
    # same IR on any machine. A spec's `knobs` pins the schedule where its category depends on it.
    env = {
        **os.environ,
        "EMMY_DUMP_DIR": str(dump),
        "EMMY_TUNE_DB": str(dump.parent / "autotune.db"),
        "EMMY_ONLINE_FILE": str(dump.parent / "online.json"),
        "EMMY_GOLDEN_FILE": "",
        "EMMY_KNOBS": spec.get("knobs", ""),
    }
    subprocess.run(
        [emmy, "compile", "-c", spec["code"], "--ir", "cuda", "--target", target],
        env=env, check=True, capture_output=True, text=True, timeout=900,
    )


def package(spec: dict, dump: Path, outdir: Path, hidden: bool = False) -> dict:
    """Assemble one problem folder from a dump directory. Returns a summary row."""
    # Find the stages by name: their numeric prefix moves whenever emmy adds a pass.
    kernel_json = next(dump.glob("*_lowering_kernel.json"))
    graph = json.loads(kernel_json.read_text())
    cuda_graph = json.loads(next(dump.glob("*_lowering_cuda.json")).read_text())
    kdir = kernel_json.with_suffix(".kernels")

    # ---- per-kernel launch descriptors from the CudaOp nodes -------------
    kernels = []
    for nid, node in cuda_graph["nodes"].items():
        if node["op"] != "CudaOp":
            continue
        of = node["op_fields"]
        # CudaOp records the total shared-memory footprint. Emmy 0.3.11's
        # renderer uses static arrays through 48 KiB and a dynamic pool above
        # it; its runtime passes zero at launch for the static case. Passing
        # the total as dynamic memory too can exceed the per-block limit.
        smem_bytes = of.get("smem_bytes", 0)
        kernels.append(dict(
            node=nid,
            entry=of["kernel_name"],
            arg_order=of["arg_order"],
            grid=[flat(d) for d in of["grid"]],
            block=[flat(d) for d in of["block"]],
            smem_bytes=smem_bytes if smem_bytes > 48 * 1024 else 0,
            zero_outputs=of.get("zero_outputs", []),
        ))
    if not kernels:
        raise ValueError("no CudaOp in the cuda stage")

    # ---- resolve every arg name against the graph ------------------------
    nodes = graph["nodes"]

    # every buffer name -> its tensor record (node ids double as their primary
    # output's buffer name in emmy, so KernelOp outputs are found here too)
    tensor = {}
    for n in nodes.values():
        for t in n.get("outputs", []):
            tensor[t["name"]] = t

    def tinfo(name: str):
        t = tensor.get(name)
        if t is None:
            raise ValueError(f"arg {name!r} not found in any node's outputs")
        return t["dtype"], (t["shape"] or [1])

    graph_outputs = set(graph["outputs"])
    for k in kernels:
        outs = {t["name"] for t in cuda_graph["nodes"][k["node"]]["outputs"]}
        args = []
        for name in k["arg_order"]:
            dtype, shape = tinfo(name)
            n = nodes.get(name)
            if name in outs:  # this kernel writes it
                args.append(dict(name=name, dtype=dtype, shape=shape, output=True))
            elif n is not None and n["op"] == "ConstantOp":
                args.append(dict(name=name, dtype=dtype, shape=shape,
                                 data=[n["op_fields"]["value"]]))
            elif n is not None and n["op"] == "InputOp":
                args.append(dict(name=name, dtype=dtype, shape=shape,
                                 seed=graph["inputs"].index(name)))
            else:  # produced by another kernel: chained at submit time
                args.append(dict(name=name, dtype=dtype, shape=shape,
                                 intermediate=True))
        k["args"] = args
        k["final"] = bool(outs & graph_outputs)

    # ---- expected outputs from the seeded inputs -------------------------
    DT = {"f32": np.float32, "f16": np.float16, "f64": np.float64,
          "i32": np.int32, "i64": np.int64}
    seeded = {}
    for name in graph["inputs"]:
        t = nodes[name]["outputs"][0]
        rng = np.random.default_rng(graph["inputs"].index(name))
        seeded[name] = rng.standard_normal([d for d in t["shape"]] or [1]).astype(DT[t["dtype"]])
    refs = spec["ref"](*[seeded[n] for n in graph["inputs"]])
    expected = {out: np.asarray(r) for out, r in zip(graph["outputs"], refs)}

    # ---- write the folder -------------------------------------------------
    outdir.mkdir(parents=True, exist_ok=True)
    # Name IR files by ENTRY name, not the dump's stem: the dump keys per-kernel
    # sub-graphs by node id (matmul_1) while the entry carries a k_ prefix
    # (k_matmul_1), and submit.py / the evaluator look them up by entry.
    written = {"meta.json"}
    for k in kernels:
        src = kdir / f"{k['node']}.json"
        if not src.exists():                      # fall back to the entry-named dump
            src = kdir / f"{k['entry']}.json"
        if not src.exists():
            raise ValueError(f"no per-kernel IR dump for {k['entry']} (node {k['node']})")
        kernel_ir = json.loads(src.read_text())
        kernel_nodes = [n for n in kernel_ir["nodes"].values() if n["op"] == "KernelOp"]
        if len(kernel_nodes) != 1:
            raise ValueError(f"expected one KernelOp for {k['entry']}, got {len(kernel_nodes)}")
        # CUDA lowering supplies a fallback when KernelOp.name is empty (e.g.
        # split-K matmuls). Participants need the actual launch entry in the IR.
        kernel_nodes[0]["op_fields"]["name"] = k["entry"]
        ir_name = f"{k['entry']}.ir.json"
        (outdir / ir_name).write_text(json.dumps(kernel_ir, indent=2))
        written.add(ir_name)
        txt = src.with_suffix(".txt")
        if txt.exists():
            txt_name = f"{k['entry']}.ir.txt"
            shutil.copy(txt, outdir / txt_name)
            written.add(txt_name)
    if not hidden:  # reference CUDA out of the CudaOp itself
        for k in kernels:
            src = cuda_graph["nodes"][k["node"]]["op_fields"].get("kernel_source", "")
            (outdir / f"{k['entry']}.ref.cu").write_text(src)
            written.add(f"{k['entry']}.ref.cu")
    meta = dict(id=spec["id"], code=spec["code"], rtol=spec["rtol"], atol=spec["atol"],
                graph_inputs=graph["inputs"], graph_outputs=graph["outputs"],
                kernels=kernels)
    if hidden:
        # The torch expression would let a submission re-trace the problem through
        # emmy's own CUDA backend and nvcc the reference - same cheat, another door.
        meta.pop("code")
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    if not hidden:
        np.savez(outdir / "expected.npz", **expected)
        written.add("expected.npz")

    # Regenerating in place can rename or fuse kernels. Remove their old files,
    # and strip references if this directory is being repackaged with --hidden.
    for pattern in ("*.ir.json", "*.ir.txt", "*.ref.cu", "expected.npz"):
        for stale in outdir.glob(pattern):
            if stale.name not in written:
                stale.unlink()

    return dict(id=spec["id"], kernels=len(kernels),
                entries=[k["entry"] for k in kernels], ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("expressions")
    ap.add_argument("--emmy", default="emmy")
    ap.add_argument("--out", default="problems")
    ap.add_argument("--only", default=None)
    ap.add_argument("--target", default="sm_89",
                    help="GPU the IR is scheduled for (default: sm_89, which the problems target)")
    ap.add_argument("--hidden", action="store_true",
                    help="IR only: omit .ref.cu and expected.npz so a submission "
                         "cannot pass by shelling out to nvcc on the reference")
    args = ap.parse_args()

    specs = load_specs(Path(args.expressions))
    if args.only:
        specs = [s for s in specs if s["id"] == args.only]

    rows, failed = [], []
    for spec in specs:
        with tempfile.TemporaryDirectory() as td:
            dump = Path(td) / "dump"
            try:
                trace(args.emmy, spec, dump, args.target)
                row = package(spec, dump, Path(args.out) / spec["id"], hidden=args.hidden)
                rows.append(row)
                print(f"  OK    {spec['id']:<20} {row['kernels']} kernel(s): {', '.join(row['entries'])}")
            except subprocess.CalledProcessError as e:
                failed.append(spec["id"])
                tail = (e.stderr or "").strip().splitlines()[-1:] or ["?"]
                print(f"  FAIL  {spec['id']:<20} trace: {tail[0][:100]}")
            except Exception as e:  # noqa: BLE001 - report and continue vetting
                failed.append(spec["id"])
                print(f"  FAIL  {spec['id']:<20} package: {e}")

    print(f"\n{len(rows)} ok, {len(failed)} failed" + (f": {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
