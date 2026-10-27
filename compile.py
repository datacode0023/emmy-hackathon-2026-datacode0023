#!/usr/bin/env python3
"""A first Kernel IR -> PTX compiler, covering f32 pointwise add/multiply/ReLU.

Usage: python compile.py path/to/kernel.ir.json > out.ptx

Emmy only loads the IR. All PTX instructions below are emitted by this file.
Supported: one scalar 1-D Tile, integer indices, Load, add/multiply/relu, Write.
Constant arguments are read from buffers through Load, just like other inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Tile
from emmy.compiler.ir.stmt import Assign, Load, Write


class CompileError(ValueError):
    """An IR feature this first compiler does not support yet."""


class PTXCompiler:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.counts = {"r": 0, "rd": 0, "f": 0, "p": 0}
        self.axes: dict[str, str] = {}
        self.values: dict[str, str] = {}
        self.pointers: dict[str, str] = {}

    def reg(self, kind: str) -> str:
        """Allocate a virtual register; ptxas handles physical registers."""
        number = self.counts[kind]
        self.counts[kind] += 1
        return f"%{kind}{number}"

    def emit(self, instruction: str) -> None:
        self.lines.append(f"    {instruction}")

    def index(self, expr) -> str:
        """Recursively lower an integer index expression, e.g. a0 * 2 + 1."""
        if isinstance(expr, Literal) and type(expr.value) is int:
            return str(expr.value)
        if isinstance(expr, Var):
            if expr.name not in self.axes:
                raise CompileError(f"Unknown index variable: {expr.name}")
            return self.axes[expr.name]
        if isinstance(expr, BinaryExpr):
            instructions = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32"}
            if expr.op not in instructions:
                raise CompileError(f"Unsupported index operation: {expr.op}")
            left = self.index(expr.left)
            right = self.index(expr.right)
            result = self.reg("r")
            self.emit(f"{instructions[expr.op]} {result}, {left}, {right};")
            return result
        raise CompileError(f"Unsupported index expression: {type(expr).__name__}")

    def address(self, buffer: str, indices: tuple) -> str:
        """An IR offset counts elements; a global PTX address counts bytes."""
        if len(indices) != 1:
            raise CompileError("Only 1-D buffer indices are supported")
        if buffer not in self.pointers:
            raise CompileError(f"Buffer missing from meta.json arg_order: {buffer}")
        offset = self.index(indices[0])
        # mul.wide requires a register for its first source operand.
        if not offset.startswith("%"):
            literal = self.reg("r")
            self.emit(f"mov.s32 {literal}, {offset};")
            offset = literal
        byte_offset = self.reg("rd")
        address = self.reg("rd")
        self.emit(f"mul.wide.s32 {byte_offset}, {offset}, 4;")
        self.emit(f"add.u64 {address}, {self.pointers[buffer]}, {byte_offset};")
        return address

    @staticmethod
    def require_f32(dtype) -> None:
        if dtype is None or dtype.name != "f32":
            raise CompileError("Only f32 values are supported")

    @staticmethod
    def vector_operand(registers: list[str]) -> tuple[str, str]:
        """A grouped IR load/store accesses consecutive values, not one value."""
        width = len(registers)
        if width == 1:
            return "", registers[0]
        if width in (2, 4):
            return f".v{width}", "{" + ", ".join(registers) + "}"
        raise CompileError(f"Unsupported load/store width: {width}")

    def statement(self, stmt) -> None:
        # Keep the emitted PTX easy to compare with the input IR.
        self.emit(f"// IR: {type(stmt).__name__}")
        if isinstance(stmt, Load):
            self.require_f32(stmt.dtype)
            address = self.address(stmt.input, stmt.index)
            registers = [self.reg("f") for _ in stmt.names]
            suffix, operand = self.vector_operand(registers)
            self.emit(f"ld.global{suffix}.f32 {operand}, [{address}];")
            self.values.update(zip(stmt.names, registers))
        elif isinstance(stmt, Assign):
            self.require_f32(stmt.dtype)
            # ReLU has one IR operand; add and multiply each have two.
            instructions = {
                "add": ("add.rn.f32", 2),
                "multiply": ("mul.rn.f32", 2),
                "relu": ("max.f32", 1),
            }
            if stmt.op.name not in instructions:
                raise CompileError(f"Unsupported elementwise operation: {stmt.op.name}")
            instruction, arity = instructions[stmt.op.name]
            if len(stmt.args) != arity:
                raise CompileError(f"{stmt.op.name} requires {arity} operands")
            operands = [self.values[name] for name in stmt.args]
            if stmt.op.name == "relu":
                operands.append("0f00000000")  # Exact f32 +0: relu(x) = max(x, 0).
            result = self.reg("f")
            self.emit(f"{instruction} {result}, {', '.join(operands)};")
            self.values[stmt.name] = result
        elif isinstance(stmt, Write):
            self.require_f32(stmt.value_dtype)
            if stmt.atomic or stmt.swizzle != "NONE":
                raise CompileError("Atomic or swizzled writes are not supported")
            address = self.address(stmt.output, stmt.index)
            registers = [self.values[name] for name in stmt.values]
            suffix, operand = self.vector_operand(registers)
            self.emit(f"st.global{suffix}.f32 [{address}], {operand};")
        else:
            raise CompileError(f"Unsupported statement: {type(stmt).__name__}")

    def compile(self, kernel: KernelOp, launch: dict) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", kernel.name):
            raise CompileError(f"Unsupported PTX entry name: {kernel.name!r}")
        if kernel.serial or len(kernel.body) != 1 or not isinstance(kernel.body[0], Tile):
            raise CompileError("Expected one non-serial Tile kernel")
        tile = kernel.body[0]
        if (len(tile.axes) != 1 or tile.block_threads is not None
                or tile.aux_threads or tile.raster_axes is not None
                or tile.raster_group is not None):
            raise CompileError("Only scalar 1-D Tiles are supported")
        extent = tile.axes[0].extent.as_static()
        if not 0 <= extent < 2**31:
            raise CompileError("Tile extent must fit in a nonnegative signed 32-bit index")
        if launch["grid"][1:] != [1, 1] or launch["block"][1:] != [1, 1]:
            raise CompileError("Only 1-D launches are supported")
        if any(arg["dtype"] != "f32" or arg.get("scalar") for arg in launch["args"]):
            raise CompileError("Only f32 buffer parameters are supported")

        # Numeric parameter names avoid depending on the spelling of IR names.
        parameters = []
        for number, name in enumerate(launch["arg_order"]):
            parameter = f"param_{number}"
            parameters.append(f"    .param .u64 {parameter}")
            pointer = self.reg("rd")
            self.emit(f"ld.param.u64 {pointer}, [{parameter}];")
            self.emit(f"cvta.to.global.u64 {pointer}, {pointer};")
            self.pointers[name] = pointer

        # p00 launches 256 threads for only 64 cells. Each cell handles 2 values.
        self.emit("// IR: Tile -- global thread id and bounds guard")
        tid, block, threads, gid = (self.reg("r") for _ in range(4))
        self.emit(f"mov.u32 {tid}, %tid.x;")
        self.emit(f"mov.u32 {block}, %ctaid.x;")
        self.emit(f"mov.u32 {threads}, %ntid.x;")
        self.emit(f"mad.lo.u32 {gid}, {block}, {threads}, {tid};")
        outside = self.reg("p")
        self.emit(f"setp.ge.u32 {outside}, {gid}, {extent};")
        self.emit(f"@{outside} bra DONE;")
        self.axes[tile.axes[0].name] = gid

        for stmt in tile.body:
            self.statement(stmt)
        self.lines.append("DONE:")
        self.emit("ret;")

        # Counts are known after walking the body; declarations precede code.
        register_types = {"r": ".b32", "rd": ".b64", "f": ".f32", "p": ".pred"}
        declarations = [
            f"    .reg {register_types[kind]} %{kind}<{count}>;"
            for kind, count in self.counts.items() if count
        ]
        return "\n".join([
            ".version 8.0", ".target sm_89", ".address_size 64", "",
            f".visible .entry {kernel.name}(", ",\n".join(parameters), ")", "{",
            *declarations, "", *self.lines, "}", "",
        ])


def compile_file(ir_path: Path) -> str:
    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    kernels = [node.op for node in graph.nodes.values() if isinstance(node.op, KernelOp)]
    if len(kernels) != 1:
        raise CompileError("Expected exactly one KernelOp in the input IR")
    kernel = kernels[0]
    meta = json.loads(ir_path.with_name("meta.json").read_text())
    launches = [launch for launch in meta["kernels"] if launch["entry"] == kernel.name]
    if len(launches) != 1:
        raise CompileError(f"Expected one meta.json entry for {kernel.name}")
    return PTXCompiler().compile(kernel, launches[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path, help="Kernel IR JSON (beside its meta.json)")
    args = parser.parse_args()
    try:
        ptx = compile_file(args.ir)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    # Emit only after the entire compilation succeeds; stdout is exclusively PTX.
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
