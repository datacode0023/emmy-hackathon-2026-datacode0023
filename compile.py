#!/usr/bin/env python3
"""Kernel IR -> PTX for f32 pointwise operations and 1-D sum reductions.

Usage: python compile.py path/to/kernel.ir.json > out.ptx

Emmy only loads the IR. All PTX instructions below are emitted by this file.
Supported: one 1-D Tile, integer indices, Load, add/multiply/relu, Write;
StridedLoop, sum Accum, WarpShuffle, Smem, Cond, Sync, and TreeHalve.
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
from emmy.compiler.ir.kernel.ir import KernelOp, Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, StridedLoop, Write


class CompileError(ValueError):
    """An IR feature this first compiler does not support yet."""


class PTXCompiler:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.counts = {"r": 0, "rd": 0, "f": 0, "p": 0}
        self.axes: dict[str, str] = {}
        self.values: dict[str, str] = {}
        self.pointers: dict[str, str] = {}
        self.shared: dict[str, tuple[str, int]] = {}
        self.shared_declarations: list[str] = []
        self.label_count = 0
        self.block_threads = 0
        self.cooperative = False
        self.conditional_depth = 0
        self.loop_depth = 0

    def reg(self, kind: str) -> str:
        """Allocate a virtual register; ptxas handles physical registers."""
        number = self.counts[kind]
        self.counts[kind] += 1
        return f"%{kind}{number}"

    def emit(self, instruction: str) -> None:
        self.lines.append(f"    {instruction}")

    def label(self, prefix: str) -> str:
        self.label_count += 1
        return f"{prefix}_{self.label_count}"

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
        """An empty index denotes a scalar; each f32 element occupies 4 bytes."""
        if len(indices) > 1:
            raise CompileError("Only scalar and 1-D buffer indices are supported")
        if buffer not in self.pointers and buffer not in self.shared:
            raise CompileError(f"Buffer missing from meta.json arg_order: {buffer}")
        offset = self.index(indices[0]) if indices else "0"
        # mul.wide requires a register for its first source operand.
        if not offset.startswith("%"):
            literal = self.reg("r")
            self.emit(f"mov.s32 {literal}, {offset};")
            offset = literal
        if buffer in self.shared:
            symbol, _ = self.shared[buffer]
            base, byte_offset, address = (self.reg("r") for _ in range(3))
            self.emit(f"mov.u32 {base}, {symbol};")
            self.emit(f"mul.lo.s32 {byte_offset}, {offset}, 4;")
            self.emit(f"add.u32 {address}, {base}, {byte_offset};")
            return address
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

    def assign(self, stmt: Assign, default_dtype=None) -> None:
        # Combine programs inherit their enclosing reduction's dtype.
        self.require_f32(stmt.dtype if stmt.dtype is not None else default_dtype)
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
            operands.append("0f00000000")
        result = self.reg("f")
        self.emit(f"{instruction} {result}, {', '.join(operands)};")
        self.values[stmt.name] = result

    def predicate(self, expr) -> str:
        comparisons = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
        if not isinstance(expr, BinaryExpr) or expr.op not in comparisons:
            raise CompileError("Only integer comparison predicates are supported")
        left, right = self.index(expr.left), self.index(expr.right)
        result = self.reg("p")
        self.emit(f"setp.{comparisons[expr.op]}.s32 {result}, {left}, {right};")
        return result

    def conditional(self, stmt: Cond) -> None:
        otherwise, done = self.label("ELSE"), self.label("ENDIF")
        condition = self.predicate(stmt.cond)
        self.emit(f"@!{condition} bra {otherwise};")
        saved_values = self.values.copy()
        self.conditional_depth += 1
        for child in stmt.body:
            self.statement(child)
        self.values = saved_values.copy()
        self.emit(f"bra {done};")
        self.lines.append(f"{otherwise}:")
        for child in stmt.else_body:
            self.statement(child)
        self.conditional_depth -= 1
        self.values = saved_values
        self.lines.append(f"{done}:")

    def strided_loop(self, stmt: StridedLoop) -> None:
        if not isinstance(stmt.step, Literal) or type(stmt.step.value) is not int or stmt.step.value <= 0:
            raise CompileError("StridedLoop requires a positive literal integer step")
        saved_values, saved_axes = self.values.copy(), self.axes.copy()
        accumulators = {}
        for child in stmt.body:
            if isinstance(child, Accum):
                self.require_f32(child.dtype)
                if child.op.name != "add" or child.base is not None:
                    raise CompileError("Only ordinary sum accumulators are supported")
                if child.name not in accumulators:
                    if stmt.seed:
                        register = self.reg("f")
                        self.emit(f"mov.f32 {register}, 0f00000000;")
                    else:
                        register = self.values[child.name]
                    accumulators[child.name] = register
                    self.values[child.name] = register

        start = self.index(stmt.start)
        end = self.index(stmt.end if stmt.end is not None else stmt.axis.extent.expr)
        induction = self.reg("r")
        self.emit(f"mov.s32 {induction}, {start};")
        self.axes[stmt.axis.name] = induction
        head, done = self.label("LOOP"), self.label("LOOP_END")
        outside = self.reg("p")
        self.lines.append(f"{head}:")
        self.emit(f"setp.ge.s32 {outside}, {induction}, {end};")
        self.emit(f"@{outside} bra {done};")
        self.loop_depth += 1
        for child in stmt.body:
            self.statement(child)
        self.loop_depth -= 1
        self.emit(f"add.s32 {induction}, {induction}, {stmt.step.value};")
        self.emit(f"bra {head};")
        self.lines.append(f"{done}:")
        self.axes = saved_axes
        self.values = saved_values
        self.values.update(accumulators)

    @staticmethod
    def power_of_two(length: int) -> bool:
        return length > 0 and length & (length - 1) == 0

    def require_collective(self) -> None:
        if not self.cooperative or self.conditional_depth or self.loop_depth:
            raise CompileError("Collectives require a cooperative Tile outside conditions and loops")

    def barrier(self, barrier_id: int, count: int | None = None) -> None:
        self.require_collective()
        if not 0 <= barrier_id < 16 or count not in (None, self.block_threads):
            raise CompileError("Only full-CTA barriers with ids 0..15 are supported")
        suffix = f", {count}" if count is not None else ""
        self.emit(f"bar.sync {barrier_id}{suffix};")

    def warp_shuffle(self, stmt: WarpShuffle) -> None:
        self.require_collective()
        self.require_f32(stmt.dtype)
        if not self.power_of_two(stmt.length) or stmt.length > 32 or len(stmt.state) != len(stmt.state_b):
            raise CompileError("WarpShuffle requires matching states and power-of-two length <= 32")
        distance = stmt.length // 2
        while distance:
            saved_values = self.values.copy()
            # Snapshot all components before executing the combine program.
            for name, other in zip(stmt.state, stmt.state_b):
                raw, shuffled = self.reg("r"), self.reg("r")
                value = self.reg("f")
                self.emit(f"mov.b32 {raw}, {self.values[name]};")
                self.emit(f"shfl.sync.bfly.b32 {shuffled}, {raw}, {distance}, 0x1f, 0xffffffff;")
                self.emit(f"mov.b32 {value}, {shuffled};")
                self.values[other] = value
            for combine in stmt.combine_states:
                self.assign(combine, default_dtype=stmt.dtype)
            combined = {name: self.values[name] for name in stmt.state}
            self.values = saved_values
            self.values.update(combined)
            distance //= 2

    def tree_halve(self, stmt: TreeHalve) -> None:
        self.require_collective()
        self.require_f32(stmt.dtype)
        if stmt.tid_var != "warp" or stmt.inner is not None:
            raise CompileError("Only per-warp shared-memory trees without inner groups are supported")
        if (not self.power_of_two(stmt.length) or stmt.length > self.block_threads // 32
                or not len(stmt.bufs) == len(stmt.state) == len(stmt.state_b)):
            raise CompileError("Invalid shared-memory tree length or state counts")
        for buffer in stmt.bufs:
            if buffer not in self.shared or self.shared[buffer][1] < stmt.length:
                raise CompileError("Shared-memory tree buffer is missing or too small")

        distance = stmt.length // 2
        while distance:
            skip = self.label("TREE_NEXT")
            inactive = self.reg("p")
            # One writer per logical warp index prevents duplicate shared writes.
            self.emit(f"setp.ne.u32 {inactive}, {self.axes['lane']}, 0;")
            self.emit(f"@{inactive} bra {skip};")
            self.emit(f"setp.ge.u32 {inactive}, {self.axes['warp']}, {distance};")
            self.emit(f"@{inactive} bra {skip};")
            saved_values = self.values.copy()
            partner = BinaryExpr("+", Var("warp"), Literal(distance, "int"))
            addresses = []
            for buffer, name, other in zip(stmt.bufs, stmt.state, stmt.state_b):
                address = self.address(buffer, (Var("warp"),))
                partner_address = self.address(buffer, (partner,))
                left, right = self.reg("f"), self.reg("f")
                self.emit(f"ld.shared.f32 {left}, [{address}];")
                self.emit(f"ld.shared.f32 {right}, [{partner_address}];")
                self.values[name], self.values[other] = left, right
                addresses.append(address)
            for combine in stmt.combine_states:
                self.assign(combine, default_dtype=stmt.dtype)
            for address, name in zip(addresses, stmt.state):
                self.emit(f"st.shared.f32 [{address}], {self.values[name]};")
            self.values = saved_values
            self.lines.append(f"{skip}:")
            # Every thread reaches the barrier, including non-writer lanes.
            self.barrier(stmt.barrier_id, stmt.barrier_count)
            distance //= 2

        # TreeHalve broadcasts the final result, even if a later Write is guarded.
        for buffer, name in zip(stmt.bufs, stmt.state):
            address = self.address(buffer, ())
            result = self.reg("f")
            self.emit(f"ld.shared.f32 {result}, [{address}];")
            self.values[name] = result

    def statement(self, stmt) -> None:
        # Keep the emitted PTX easy to compare with the input IR.
        self.emit(f"// IR: {type(stmt).__name__}")
        if isinstance(stmt, Load):
            self.require_f32(stmt.dtype)
            address = self.address(stmt.input, stmt.index)
            registers = [self.reg("f") for _ in stmt.names]
            suffix, operand = self.vector_operand(registers)
            space = "shared" if stmt.input in self.shared else "global"
            self.emit(f"ld.{space}{suffix}.f32 {operand}, [{address}];")
            self.values.update(zip(stmt.names, registers))
        elif isinstance(stmt, Assign):
            self.assign(stmt)
        elif isinstance(stmt, Accum):
            self.require_f32(stmt.dtype)
            if stmt.op.name != "add" or stmt.base is not None:
                raise CompileError("Only ordinary sum accumulators are supported")
            result = self.values[stmt.name]
            self.emit(f"add.rn.f32 {result}, {result}, {self.values[stmt.value]};")
        elif isinstance(stmt, Write):
            self.require_f32(stmt.value_dtype)
            if stmt.atomic or stmt.swizzle != "NONE":
                raise CompileError("Atomic or swizzled writes are not supported")
            address = self.address(stmt.output, stmt.index)
            registers = [self.values[name] for name in stmt.values]
            suffix, operand = self.vector_operand(registers)
            space = "shared" if stmt.output in self.shared else "global"
            self.emit(f"st.{space}{suffix}.f32 [{address}], {operand};")
        elif isinstance(stmt, StridedLoop):
            self.strided_loop(stmt)
        elif isinstance(stmt, Cond):
            self.conditional(stmt)
        elif isinstance(stmt, WarpShuffle):
            self.warp_shuffle(stmt)
        elif isinstance(stmt, Smem):
            if stmt.dtype != "float" or len(stmt.extents) != 1 or stmt.extents[0] <= 0:
                raise CompileError("Only 1-D f32 shared buffers are supported")
            if stmt.name in self.shared or stmt.name in self.pointers:
                raise CompileError(f"Duplicate shared buffer: {stmt.name}")
            symbol = f"shared_{len(self.shared)}"
            alignment = max(4, stmt.align)
            if not self.power_of_two(alignment):
                raise CompileError("Shared-memory alignment must be a power of two")
            size = stmt.extents[0]
            self.shared[stmt.name] = (symbol, size)
            self.shared_declarations.append(f"    .shared .align {alignment} .b8 {symbol}[{size * 4}];")
        elif isinstance(stmt, Sync):
            if stmt.warp:
                raise CompileError("Warp-scope barriers are not supported yet")
            self.barrier(stmt.barrier_id, stmt.count)
        elif isinstance(stmt, TreeHalve):
            self.tree_halve(stmt)
        else:
            raise CompileError(f"Unsupported statement: {type(stmt).__name__}")

    def compile(self, kernel: KernelOp, launch: dict) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", kernel.name):
            raise CompileError(f"Unsupported PTX entry name: {kernel.name!r}")
        if kernel.serial or len(kernel.body) != 1 or not isinstance(kernel.body[0], Tile):
            raise CompileError("Expected one non-serial Tile kernel")
        tile = kernel.body[0]
        if (len(tile.axes) != 1 or tile.aux_threads or tile.raster_axes is not None
                or tile.raster_group is not None):
            raise CompileError("Only ordinary 1-D Tiles are supported")
        extent = tile.axes[0].extent.as_static()
        if not 0 <= extent < 2**31:
            raise CompileError("Tile extent must fit in a nonnegative signed 32-bit index")
        if launch["grid"][1:] != [1, 1] or launch["block"][1:] != [1, 1]:
            raise CompileError("Only 1-D launches are supported")
        self.block_threads = launch["block"][0]
        self.cooperative = tile.block_threads is not None
        if self.cooperative and (
            tile.block_threads != self.block_threads or self.block_threads % 32
            or extent != launch["grid"][0] * self.block_threads
        ):
            raise CompileError("Cooperative Tiles require complete CTAs and complete warps")
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

        # Derive the bounds guard from the Tile, including overlaunched pointwise work.
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
        if self.cooperative:
            lane, warp = self.reg("r"), self.reg("r")
            self.emit(f"and.b32 {lane}, {tid}, 31;")
            self.emit(f"shr.u32 {warp}, {tid}, 5;")
            self.axes.update(lane=lane, warp=warp)

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
            *declarations, *self.shared_declarations, "", *self.lines, "}", "",
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
