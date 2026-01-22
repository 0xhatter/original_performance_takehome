"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


def get_rw(engine, slot):
    reads = set()
    writes = set()
    mem_read = False
    mem_write = False

    if engine == 'alu':
        # (op, dest, a1, a2)
        op, dest, a1, a2 = slot
        writes.add(dest)
        reads.add(a1)
        reads.add(a2)

    elif engine == 'valu':
        op = slot[0]
        if op == 'vbroadcast':
            # (vbroadcast, dest, src)
            _, dest, src = slot
            for i in range(VLEN):
                writes.add(dest + i)
            reads.add(src)
        elif op == 'multiply_add':
            # (multiply_add, dest, a, b, c)
            _, dest, a, b, c = slot
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(a + i)
                reads.add(b + i)
                reads.add(c + i)
        else:
            # (op, dest, a1, a2)
            _, dest, a1, a2 = slot
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(a1 + i)
                reads.add(a2 + i)

    elif engine == 'load':
        op = slot[0]
        if op == 'load':
            # (load, dest, addr)
            _, dest, addr = slot
            writes.add(dest)
            reads.add(addr)
            mem_read = True
        elif op == 'load_offset':
            # (load_offset, dest, addr, offset)
            _, dest, addr, offset = slot
            writes.add(dest + offset)
            reads.add(addr + offset)
            mem_read = True
        elif op == 'vload':
            # (vload, dest, addr)
            _, dest, addr = slot
            for i in range(VLEN):
                writes.add(dest + i)
            reads.add(addr)
            mem_read = True
        elif op == 'const':
            # (const, dest, val)
            _, dest, val = slot
            writes.add(dest)

    elif engine == 'store':
        op = slot[0]
        if op == 'store':
            # (store, addr, src)
            _, addr, src = slot
            reads.add(addr)
            reads.add(src)
            mem_write = True
        elif op == 'vstore':
            # (vstore, addr, src)
            _, addr, src = slot
            reads.add(addr)
            for i in range(VLEN):
                reads.add(src + i)
            mem_write = True

    elif engine == 'flow':
        op = slot[0]
        if op == 'select':
            # (select, dest, cond, a, b)
            _, dest, cond, a, b = slot
            writes.add(dest)
            reads.add(cond)
            reads.add(a)
            reads.add(b)
        elif op == 'add_imm':
            # (add_imm, dest, a, imm)
            _, dest, a, imm = slot
            writes.add(dest)
            reads.add(a)
        elif op == 'vselect':
            # (vselect, dest, cond, a, b)
            _, dest, cond, a, b = slot
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(cond + i)
                reads.add(a + i)
                reads.add(b + i)
        elif op == 'cond_jump':
            # (cond_jump, cond, addr)
            _, cond, addr = slot
            reads.add(cond)
        elif op == 'cond_jump_rel':
            # (cond_jump_rel, cond, offset)
            _, cond, offset = slot
            reads.add(cond)
        elif op == 'jump_indirect':
            # (jump_indirect, addr)
            _, addr = slot
            reads.add(addr)
        elif op == 'trace_write':
            # (trace_write, val)
            _, val = slot
            reads.add(val)
        elif op == 'coreid':
            # (coreid, dest)
            _, dest = slot
            writes.add(dest)
        # jump, halt, pause: no regs

    elif engine == 'debug':
        op = slot[0]
        if op == 'compare':
            # (compare, val, key)
            _, val, key = slot
            reads.add(val)
        elif op == 'vcompare':
            # (vcompare, val, keys)
            _, val, keys = slot
            for i in range(VLEN):
                reads.add(val + i)

    return reads, writes, mem_read, mem_write


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        # Greedy slot packing respecting dependencies and slot limits
        instrs = []

        if not vliw:
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        current_bundle = defaultdict(list)
        bundle_reads = set()
        bundle_writes = set()
        bundle_mem_write = False

        for engine, slot in slots:
            reads, writes, mem_read, mem_write = get_rw(engine, slot)

            # Check for conflicts
            conflict = False

            # 1. Slot limits
            if len(current_bundle[engine]) >= SLOT_LIMITS[engine]:
                conflict = True

            # 2. RAW: Reads intersect with bundle writes
            if not conflict and not reads.isdisjoint(bundle_writes):
                conflict = True

            # 3. WAW: Writes intersect with bundle writes
            if not conflict and not writes.isdisjoint(bundle_writes):
                conflict = True

            # 4. Memory RAW: Reading memory after a store in same bundle
            if not conflict and mem_read and bundle_mem_write:
                conflict = True

            # 5. Memory WAW: Writing memory after a store in same bundle
            if not conflict and mem_write and bundle_mem_write:
                conflict = True

            if conflict:
                # Flush current bundle
                if current_bundle:
                    instrs.append(dict(current_bundle))
                current_bundle = defaultdict(list)
                bundle_reads = set()
                bundle_writes = set()
                bundle_mem_write = False

            # Add to bundle
            current_bundle[engine].append(slot)
            bundle_reads.update(reads)
            bundle_writes.update(writes)
            if mem_write:
                bundle_mem_write = True

        # Flush final bundle
        if current_bundle:
            instrs.append(dict(current_bundle))

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
