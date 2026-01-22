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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
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

    def make_bundle(self, slots):
        bundle = {}
        for engine, slot in slots:
            if engine not in bundle:
                bundle[engine] = []
            bundle[engine].append(slot)
        return bundle

    def scratch_vec_const(self, val):
        key = f"vec_{val}"
        if key not in self.const_map:
             addr = self.alloc_scratch(key, VLEN)
             const_addr = self.scratch_const(val)
             self.add("valu", ("vbroadcast", addr, const_addr))
             self.const_map[key] = addr
        return self.const_map[key]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
        return slots

    def build_hash_vec(self, val_hash_addr, tmp1, tmp2, round, i_base):
        instrs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
             # multiply_add optimization: a * (1 + 2^val3) + val1
             if op1 == '+' and op2 == '+' and op3 == '<<':
                 factor = 1 + (1 << val3)
                 factor_vec = self.scratch_vec_const(factor)
                 val1_vec = self.scratch_vec_const(val1)

                 instrs.append(self.make_bundle([
                     ("valu", ("multiply_add", val_hash_addr, val_hash_addr, factor_vec, val1_vec))
                 ]))
                 instrs.append({"debug": [("vcompare", val_hash_addr, tuple((round, i_base + k, "hash_stage", hi) for k in range(VLEN)))]})
                 continue

             val1_vec = self.scratch_vec_const(val1)
             val3_vec = self.scratch_vec_const(val3)

             # Cycle 1: op1 and op3
             b1 = self.make_bundle([
                 ("valu", (op1, tmp1, val_hash_addr, val1_vec)),
                 ("valu", (op3, tmp2, val_hash_addr, val3_vec))
             ])
             instrs.append(b1)
             # Cycle 2: op2
             b2 = self.make_bundle([
                 ("valu", (op2, val_hash_addr, tmp1, tmp2))
             ])
             instrs.append(b2)

             # Debug (0 cycles)
             instrs.append({"debug": [("vcompare", val_hash_addr, tuple((round, i_base + k, "hash_stage", hi) for k in range(VLEN)))]})
        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Vectorized and cached implementation.
        """
        n_vecs = batch_size // VLEN
        # Alloc vectors for full state
        idx_vecs = [self.alloc_scratch(f"idx_{i}", VLEN) for i in range(n_vecs)]
        val_vecs = [self.alloc_scratch(f"val_{i}", VLEN) for i in range(n_vecs)]

        # Temp vectors
        tmp1 = self.alloc_scratch("tmp1", VLEN)
        tmp2 = self.alloc_scratch("tmp2", VLEN)
        tmp3 = self.alloc_scratch("tmp3", VLEN)
        tmp_node_val = self.alloc_scratch("tmp_node_val", VLEN)
        tmp_addr_vec = self.alloc_scratch("tmp_addr_vec", VLEN)

        tmp_scalar = self.alloc_scratch("tmp_scalar", 1)

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
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        # Constants
        zero_vec = self.scratch_vec_const(0)
        one_vec = self.scratch_vec_const(1)
        two_vec = self.scratch_vec_const(2)
        n_nodes_vec_addr = self.alloc_scratch("n_nodes_vec", VLEN)
        self.add("valu", ("vbroadcast", n_nodes_vec_addr, self.scratch["n_nodes"]))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        # Load inputs to scratch
        for i in range(n_vecs):
            offset = i * VLEN
            i_const = self.scratch_const(offset)
            self.instrs.append(self.make_bundle([
                ("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], i_const)),
                ("alu", ("+", tmp3, self.scratch["inp_values_p"], i_const))
            ]))
            self.instrs.append(self.make_bundle([
                ("load", ("vload", idx_vecs[i], tmp_scalar)),
                ("load", ("vload", val_vecs[i], tmp3))
            ]))

        for round in range(rounds):
            for i in range(n_vecs):
                # Standard Gather (optimized later)
                adds = []
                for k in range(VLEN):
                    adds.append(("alu", ("+", tmp_addr_vec + k, self.scratch["forest_values_p"], idx_vecs[i] + k)))
                self.instrs.append(self.make_bundle(adds))

                for k in range(0, VLEN, 2):
                    loads = []
                    loads.append(("load", ("load", tmp_node_val + k, tmp_addr_vec + k)))
                    loads.append(("load", ("load", tmp_node_val + k + 1, tmp_addr_vec + k + 1)))
                    self.instrs.append(self.make_bundle(loads))

                self.instrs.append({"debug": [("vcompare", tmp_node_val, tuple((round, i*VLEN+k, "node_val") for k in range(VLEN)))]})

                # Hash
                self.instrs.append(self.make_bundle([
                    ("valu", ("^", val_vecs[i], val_vecs[i], tmp_node_val))
                ]))
                self.instrs.extend(self.build_hash_vec(val_vecs[i], tmp1, tmp2, round, i*VLEN))

                self.instrs.append({"debug": [("vcompare", val_vecs[i], tuple((round, i*VLEN+k, "hashed_val") for k in range(VLEN)))]})

                # Logic
                self.instrs.append(self.make_bundle([
                     ("valu", ("%", tmp1, val_vecs[i], two_vec)),
                     ("valu", ("*", idx_vecs[i], idx_vecs[i], two_vec))
                ]))
                self.instrs.append(self.make_bundle([
                    ("valu", ("==", tmp1, tmp1, zero_vec))
                ]))
                self.instrs.append(self.make_bundle([
                    ("flow", ("vselect", tmp3, tmp1, one_vec, two_vec))
                ]))
                self.instrs.append(self.make_bundle([
                    ("valu", ("+", idx_vecs[i], idx_vecs[i], tmp3))
                ]))

                self.instrs.append({"debug": [("vcompare", idx_vecs[i], tuple((round, i*VLEN+k, "next_idx") for k in range(VLEN)))]})

                # Wrap
                self.instrs.append(self.make_bundle([
                    ("valu", ("<", tmp1, idx_vecs[i], n_nodes_vec_addr))
                ]))
                self.instrs.append(self.make_bundle([
                    ("flow", ("vselect", idx_vecs[i], tmp1, idx_vecs[i], zero_vec))
                ]))

                self.instrs.append({"debug": [("vcompare", idx_vecs[i], tuple((round, i*VLEN+k, "wrapped_idx") for k in range(VLEN)))]})

            # Tail handling (scalar)
            for i in range(n_vecs * VLEN, batch_size):
                i_const = self.scratch_const(i)
                # Load inputs directly from memory (no caching for tail)
                self.instrs.append(self.make_bundle([
                    ("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], i_const)),
                    ("alu", ("+", tmp3, self.scratch["inp_values_p"], i_const))
                ]))
                self.instrs.append(self.make_bundle([
                    ("load", ("load", tmp1, tmp_scalar)), # Reuse tmp1 (vector) as scalar? No, it's VLEN size.
                                                          # But alu/load works on scalar address.
                                                          # tmp1 points to start of vector.
                                                          # Safe to use as scalar register.
                    ("load", ("load", tmp2, tmp3))        # tmp2 as scalar val
                ]))

                # Gather node_val
                # addr = forest_values_p + idx
                self.instrs.append(self.make_bundle([
                    ("alu", ("+", tmp3, self.scratch["forest_values_p"], tmp1))
                ]))
                self.instrs.append(self.make_bundle([
                    ("load", ("load", tmp_node_val, tmp3)) # Reuse tmp_node_val
                ]))

                # Hash
                # val = val ^ node_val
                self.instrs.append(self.make_bundle([
                    ("alu", ("^", tmp2, tmp2, tmp_node_val))
                ]))

                # Scalar hash
                self.instrs.extend(self.build(self.build_hash(tmp2, tmp_scalar, tmp3, round, i)))

                # Logic
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # tmp_scalar = val % 2
                self.instrs.append(self.make_bundle([
                    ("alu", ("%", tmp_scalar, tmp2, two_vec)) # two_vec used as scalar const?
                                                             # two_vec points to broadcasted vector.
                                                             # load/alu reads scalar at address.
                                                             # Yes, it reads first element (2). Safe.
                ]))
                self.instrs.append(self.make_bundle([
                    ("alu", ("==", tmp_scalar, tmp_scalar, zero_vec))
                ]))
                self.instrs.append(self.make_bundle([
                    ("flow", ("select", tmp3, tmp_scalar, one_vec, two_vec))
                ]))
                self.instrs.append(self.make_bundle([
                    ("alu", ("*", tmp1, tmp1, two_vec))
                ]))
                self.instrs.append(self.make_bundle([
                    ("alu", ("+", tmp1, tmp1, tmp3))
                ]))

                # Wrap
                # idx = 0 if idx >= n_nodes
                # tmp3 = idx < n_nodes
                self.instrs.append(self.make_bundle([
                    ("alu", ("<", tmp3, tmp1, self.scratch["n_nodes"])) # n_nodes scalar
                ]))
                self.instrs.append(self.make_bundle([
                    ("flow", ("select", tmp1, tmp3, tmp1, zero_vec))
                ]))

                # Store back
                self.instrs.append(self.make_bundle([
                    ("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], i_const)),
                    ("alu", ("+", tmp3, self.scratch["inp_values_p"], i_const))
                ]))
                self.instrs.append(self.make_bundle([
                    ("store", ("store", tmp_scalar, tmp1)),
                    ("store", ("store", tmp3, tmp2))
                ]))

        # Store outputs
        for i in range(n_vecs):
            offset = i * VLEN
            i_const = self.scratch_const(offset)
            self.instrs.append(self.make_bundle([
                ("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], i_const)),
                ("alu", ("+", tmp3, self.scratch["inp_values_p"], i_const))
            ]))
            self.instrs.append(self.make_bundle([
                ("store", ("vstore", tmp_scalar, idx_vecs[i])),
                ("store", ("vstore", tmp3, val_vecs[i]))
            ]))

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
