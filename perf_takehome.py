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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.const_vec_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        return self.pack_slots(slots)

    def _get_rw(self, engine, slot):
        reads = set()
        writes = set()
        op = slot[0]

        if engine == "alu":
            # (op, dest, a1, a2)
            _, dest, a1, a2 = slot
            reads.add(a1); reads.add(a2)
            writes.add(dest)
        elif engine == "valu":
            if op == "vbroadcast":
                _, dest, src = slot
                reads.add(src)
                for k in range(VLEN): writes.add(dest + k)
            elif op == "multiply_add":
                _, dest, a, b, c = slot
                for k in range(VLEN):
                    reads.add(a + k); reads.add(b + k); reads.add(c + k)
                    writes.add(dest + k)
            else:
                _, dest, a1, a2 = slot
                for k in range(VLEN):
                    reads.add(a1 + k); reads.add(a2 + k)
                    writes.add(dest + k)
        elif engine == "load":
            if op == "load":
                _, dest, addr = slot
                reads.add(addr)
                writes.add(dest)
            elif op == "vload":
                _, dest, addr = slot
                reads.add(addr)
                for k in range(VLEN): writes.add(dest + k)
            elif op == "const":
                _, dest, val = slot
                writes.add(dest)
            elif op == "load_offset":
                _, dest, addr, off = slot
                reads.add(addr + off)
                writes.add(dest + off)
        elif engine == "store":
            if op == "store":
                _, addr, src = slot
                reads.add(addr); reads.add(src)
            elif op == "vstore":
                _, addr, src = slot
                reads.add(addr)
                for k in range(VLEN): reads.add(src + k)
        elif engine == "flow":
            if op == "select":
                _, dest, cond, a, b = slot
                reads.add(cond); reads.add(a); reads.add(b)
                writes.add(dest)
            elif op == "vselect":
                _, dest, cond, a, b = slot
                for k in range(VLEN):
                    reads.add(cond + k); reads.add(a + k); reads.add(b + k)
                    writes.add(dest + k)
            elif op == "add_imm":
                _, dest, a, imm = slot
                reads.add(a)
                writes.add(dest)
            elif op == "cond_jump":
                _, cond, addr = slot
                reads.add(cond)
            elif op == "cond_jump_rel":
                _, cond, offset = slot
                reads.add(cond)
            elif op == "jump_indirect":
                _, addr = slot
                reads.add(addr)
            elif op == "trace_write":
                _, val = slot
                reads.add(val)
            elif op == "coreid":
                _, dest = slot
                writes.add(dest)
            # jump, halt, pause: no scratch rw
        elif engine == "debug":
            if op == "compare":
                _, loc, key = slot
                reads.add(loc)
            elif op == "vcompare":
                _, loc, keys = slot
                for k in range(VLEN): reads.add(loc + k)

        return reads, writes

    def pack_slots(self, slots):
        instrs = []
        current_instr = defaultdict(list)
        written_in_bundle = set()

        for engine, slot in slots:
            reads, writes = self._get_rw(engine, slot)

            # Check dependency
            has_dependency = not reads.isdisjoint(written_in_bundle)

            # Check limits
            limit = SLOT_LIMITS.get(engine, 0)
            current_count = len(current_instr[engine])
            limit_reached = current_count >= limit

            if has_dependency or limit_reached:
                # Flush current instruction
                if current_instr:
                    instrs.append(dict(current_instr))
                current_instr = defaultdict(list)
                written_in_bundle = set()

            current_instr[engine].append(slot)
            written_in_bundle.update(writes)

        if current_instr:
            instrs.append(dict(current_instr))

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

    def alloc_scratch_vec(self, name=None):
        return self.alloc_scratch(name, length=VLEN)

    def scratch_const_vec(self, val, name=None):
        if val not in self.const_vec_map:
            vec_addr = self.alloc_scratch_vec(name)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.const_vec_map[val] = vec_addr
        return self.const_vec_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            # slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vector(self, val_hash_vec, tmp1_vec, tmp2_vec, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const_vec(val1)
            c3 = self.scratch_const_vec(val3)
            slots.append(("valu", (op1, tmp1_vec, val_hash_vec, c1)))
            slots.append(("valu", (op3, tmp2_vec, val_hash_vec, c3)))
            slots.append(("valu", (op2, val_hash_vec, tmp1_vec, tmp2_vec)))
            # keys = [(round, i + k, "hash_stage", hi) for k in range(VLEN)]
            # slots.append(("debug", ("vcompare", val_hash_vec, keys)))
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

        # Vector constants
        zero_vec = self.scratch_const_vec(0)
        one_vec = self.scratch_const_vec(1)
        two_vec = self.scratch_const_vec(2)
        n_nodes_vec = self.scratch_const_vec(n_nodes)

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

        # Temp vectors for unrolling
        UNROLL_FACTOR = 4
        tmp_node_vals = [self.alloc_scratch_vec(f"tmp_node_val_{k}") for k in range(UNROLL_FACTOR)]
        tmp1s = [self.alloc_scratch_vec(f"tmp1_{k}") for k in range(UNROLL_FACTOR)]
        tmp2s = [self.alloc_scratch_vec(f"tmp2_{k}") for k in range(UNROLL_FACTOR)]
        tmp3s = [self.alloc_scratch_vec(f"tmp3_{k}") for k in range(UNROLL_FACTOR)]

        # Address registers for unrolled gather (scalar)
        tmp_addrs_pool = [self.alloc_scratch(f"tmp_addr_{k}") for k in range(UNROLL_FACTOR * VLEN)]

        # Buffers in scratch to avoid memory traffic
        idx_buf = self.alloc_scratch("idx_buf", length=batch_size)
        val_buf = self.alloc_scratch("val_buf", length=batch_size)

        vector_loops = batch_size // VLEN
        unrolled_loops = vector_loops // UNROLL_FACTOR
        remainder_start = unrolled_loops * UNROLL_FACTOR * VLEN

        # Prologue: Load all data from memory to scratch buffers
        prologue = []
        for vi in range(vector_loops):
            i = vi * VLEN
            i_const = self.scratch_const(i)
            # vload idx
            prologue.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
            prologue.append(("load", ("vload", idx_buf + i, tmp_addr)))
            # vload val
            prologue.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
            prologue.append(("load", ("vload", val_buf + i, tmp_addr)))

        for i in range(remainder_start, batch_size):
            i_const = self.scratch_const(i)
            prologue.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
            prologue.append(("load", ("load", idx_buf + i, tmp_addr)))
            prologue.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
            prologue.append(("load", ("load", val_buf + i, tmp_addr)))

        self.instrs.extend(self.pack_slots(prologue))

        for round in range(rounds):
            for vui in range(unrolled_loops):
                vi_base = vui * UNROLL_FACTOR

                ops_addrs = []
                ops_loads = []
                ops_hashes = []
                ops_updates = []

                for u in range(UNROLL_FACTOR):
                    vi = vi_base + u
                    i = vi * VLEN

                    # Pointers to current vector in scratch buffer
                    curr_idx_vec = idx_buf + i
                    curr_val_vec = val_buf + i

                    # Temps for this unrolled iteration
                    tmp_node_val_u = tmp_node_vals[u]
                    tmp1_u = tmp1s[u]
                    tmp2_u = tmp2s[u]
                    tmp3_u = tmp3s[u]

                    # node_val_vec gather
                    for k in range(VLEN):
                        # Use unique addr reg
                        addr_reg = tmp_addrs_pool[u * VLEN + k]
                        ops_addrs.append(("alu", ("+", addr_reg, self.scratch["forest_values_p"], curr_idx_vec + k)))
                        ops_loads.append(("load", ("load", tmp_node_val_u + k, addr_reg)))

                    # Vectorized hash
                    ops_hashes.append(("valu", ("^", curr_val_vec, curr_val_vec, tmp_node_val_u)))
                    ops_hashes.extend(self.build_hash_vector(curr_val_vec, tmp1_u, tmp2_u, round, i))

                    # Vectorized index update
                    ops_updates.append(("valu", ("&", tmp1_u, curr_val_vec, one_vec)))
                    ops_updates.append(("valu", ("==", tmp1_u, tmp1_u, zero_vec)))
                    ops_updates.append(("flow", ("vselect", tmp3_u, tmp1_u, one_vec, two_vec)))

                    ops_updates.append(("valu", ("*", curr_idx_vec, curr_idx_vec, two_vec)))
                    ops_updates.append(("valu", ("+", curr_idx_vec, curr_idx_vec, tmp3_u)))

                    ops_updates.append(("valu", ("<", tmp1_u, curr_idx_vec, n_nodes_vec)))
                    ops_updates.append(("flow", ("vselect", curr_idx_vec, tmp1_u, curr_idx_vec, zero_vec)))

                body.extend(ops_addrs)
                body.extend(ops_loads)
                body.extend(ops_hashes)
                body.extend(ops_updates)

            # Scalar remainder loop
            for i in range(remainder_start, batch_size):
                curr_idx = idx_buf + i
                curr_val = val_buf + i

                # body.append(("debug", ("compare", curr_idx, (round, i, "idx"))))
                # body.append(("debug", ("compare", curr_val, (round, i, "val"))))

                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], curr_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                # body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))

                body.append(("alu", ("^", curr_val, curr_val, tmp_node_val)))
                body.extend(self.build_hash(curr_val, tmp1, tmp2, round, i))
                # body.append(("debug", ("compare", curr_val, (round, i, "hashed_val"))))

                body.append(("alu", ("%", tmp1, curr_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", curr_idx, curr_idx, two_const)))
                body.append(("alu", ("+", curr_idx, curr_idx, tmp3)))
                # body.append(("debug", ("compare", curr_idx, (round, i, "next_idx"))))

                body.append(("alu", ("<", tmp1, curr_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", curr_idx, tmp1, curr_idx, zero_const)))
                # body.append(("debug", ("compare", curr_idx, (round, i, "wrapped_idx"))))

        # Epilogue: Store scratch to mem
        epilogue = []
        for vi in range(vector_loops):
            i = vi * VLEN
            i_const = self.scratch_const(i)
            epilogue.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
            epilogue.append(("store", ("vstore", tmp_addr, idx_buf + i)))
            epilogue.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
            epilogue.append(("store", ("vstore", tmp_addr, val_buf + i)))

        for i in range(remainder_start, batch_size):
            i_const = self.scratch_const(i)
            epilogue.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
            epilogue.append(("store", ("store", tmp_addr, idx_buf + i)))
            epilogue.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
            epilogue.append(("store", ("store", tmp_addr, val_buf + i)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        self.instrs.extend(self.pack_slots(epilogue))
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
