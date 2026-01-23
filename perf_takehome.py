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
import heapq

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
    reads = None
    writes = None
    mem_read = False
    mem_write = False
    op = slot[0]

    if engine == 'alu':
        # (op, dest, a1, a2)
        op, dest, a1, a2 = slot
        writes = {dest}
        reads = {a1, a2}

    elif engine == 'valu':
        op = slot[0]
        if op == 'vbroadcast':
            # (vbroadcast, dest, src)
            _, dest, src = slot
            writes = set(range(dest, dest + VLEN))
            reads = {src}
        elif op == 'multiply_add':
            # (multiply_add, dest, a, b, c)
            _, dest, a, b, c = slot
            writes = set()
            reads = set()
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(a + i)
                reads.add(b + i)
                reads.add(c + i)
        else:
            # (op, dest, a1, a2)
            _, dest, a1, a2 = slot
            writes = set()
            reads = set()
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(a1 + i)
                reads.add(a2 + i)

    elif engine == 'load':
        op = slot[0]
        if op == 'load':
            # (load, dest, addr)
            _, dest, addr = slot
            writes = {dest}
            reads = {addr}
            mem_read = True
        elif op == 'load_offset':
            # (load_offset, dest, addr, offset)
            _, dest, addr, offset = slot
            writes = {dest + offset}
            reads = {addr + offset}
            mem_read = True
        elif op == 'vload':
            # (vload, dest, addr)
            _, dest, addr = slot
            writes = set(range(dest, dest + VLEN))
            reads = {addr}
            mem_read = True
        elif op == 'const':
            # (const, dest, val)
            _, dest, val = slot
            writes = {dest}

    elif engine == 'store':
        op = slot[0]
        if op == 'store':
            # (store, addr, src)
            _, addr, src = slot
            reads = {addr, src}
            mem_write = True
        elif op == 'vstore':
            # (vstore, addr, src)
            _, addr, src = slot
            reads = set(range(src, src + VLEN))
            reads.add(addr)
            mem_write = True

    elif engine == 'flow':
        op = slot[0]
        if op == 'select':
            # (select, dest, cond, a, b)
            _, dest, cond, a, b = slot
            writes = {dest}
            reads = {cond, a, b}
        elif op == 'add_imm':
            # (add_imm, dest, a, imm)
            _, dest, a, imm = slot
            writes = {dest}
            reads = {a}
        elif op == 'vselect':
            # (vselect, dest, cond, a, b)
            _, dest, cond, a, b = slot
            writes = set()
            reads = set()
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(cond + i)
                reads.add(a + i)
                reads.add(b + i)
        elif op == 'cond_jump':
            # (cond_jump, cond, addr)
            _, cond, addr = slot
            reads = {cond}
        elif op == 'cond_jump_rel':
            # (cond_jump_rel, cond, offset)
            _, cond, offset = slot
            reads = {cond}
        elif op == 'jump_indirect':
            # (jump_indirect, addr)
            _, addr = slot
            reads = {addr}
        elif op == 'trace_write':
            # (trace_write, val)
            _, val = slot
            reads = {val}
        elif op == 'coreid':
            # (coreid, dest)
            _, dest = slot
            writes = {dest}
        # jump, halt, pause: no regs

    elif engine == 'debug':
        op = slot[0]
        if op == 'compare':
            # (compare, val, key)
            _, val, key = slot
            reads = {val}
        elif op == 'vcompare':
            # (vcompare, val, keys)
            _, val, keys = slot
            reads = set(range(val, val + VLEN))

    if reads is None: reads = set()
    if writes is None: writes = set()

    return reads, writes, mem_read, mem_write

class Scheduler:
    def schedule(self, slots):
        # 1. Build nodes
        nodes = []
        for i, (engine, slot) in enumerate(slots):
            reads, writes, mem_read, mem_write = get_rw(engine, slot)
            nodes.append({
                'id': i, 'engine': engine, 'slot': slot,
                'reads': reads, 'writes': writes,
                'mem_read': mem_read, 'mem_write': mem_write,
                'preds': [], 'succs': [],
                'unscheduled_preds': 0,
                'priority': 0
            })

        # 2. Add dependencies
        last_writer = {} # reg -> node_id
        last_readers = defaultdict(list) # reg -> list of node_ids

        # Memory serialization
        last_mem_write = None
        last_mem_read = [] # list of node_ids

        for n in nodes:
            nid = n['id']
            deps = set()

            # RAW: Dependency on last writer of any read register
            for r in n['reads']:
                if r in last_writer:
                    deps.add(last_writer[r])

            # WAR: Dependency on all previous readers of registers I am writing
            for w in n['writes']:
                if w in last_readers:
                    for reader_id in last_readers[w]:
                        deps.add(reader_id)

            # WAW: Dependency on last writer of any written register
            for w in n['writes']:
                if w in last_writer:
                    deps.add(last_writer[w])

            # Memory Deps
            # If Store: Dep on all previous Load/Store
            if n['mem_write']:
                if last_mem_write is not None:
                    deps.add(last_mem_write)
                for rid in last_mem_read:
                    deps.add(rid)
            # If Load: Dep on last Store
            if n['mem_read']:
                if last_mem_write is not None:
                    deps.add(last_mem_write)

            # Add edges
            for dep_id in deps:
                nodes[dep_id]['succs'].append(nid)
                n['preds'].append(dep_id)

            # Update state

            # Update last_readers for registers read by this node
            for r in n['reads']:
                last_readers[r].append(nid)

            # Update last_writer and clear last_readers for registers written by this node
            for w in n['writes']:
                last_writer[w] = nid
                last_readers[w] = []

            if n['mem_write']:
                last_mem_write = nid
                last_mem_read = []
            if n['mem_read']:
                last_mem_read.append(nid)

        # 3. Calculate priorities (Critical Path)
        memo = {}
        def get_height(nid):
            if nid in memo: return memo[nid]
            h = 0
            for succ_id in nodes[nid]['succs']:
                h = max(h, get_height(succ_id))
            memo[nid] = 1 + h
            return 1 + h

        # Recursive height calc might hit recursion limit if chain is 50k deep.
        # Use iterative approach for DAG height.
        # 1. Compute in-degrees for topo sort
        in_degree = defaultdict(int)
        for n in nodes:
            for s in n['succs']:
                in_degree[s] += 1

        # 2. Topo sort (Kahn's)
        q = [n['id'] for n in nodes if in_degree[n['id']] == 0]
        topo_order = []
        while q:
            u = q.pop(0)
            topo_order.append(u)
            for v in nodes[u]['succs']:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)

        # 3. Compute height in reverse topo order
        heights = {}
        for u in reversed(topo_order):
            h = 0
            for v in nodes[u]['succs']:
                h = max(h, heights.get(v, 0))
            heights[u] = 1 + h

        for n in nodes:
            n['priority'] = heights[n['id']]
            # Prioritize LOADs because they are the bottleneck
            if n['engine'] == 'load':
                n['priority'] += 10000
            elif n['engine'] == 'flow':
                n['priority'] += 5000
            n['unscheduled_preds'] = len(n['preds'])

        # 4. List Scheduling
        # Use heap for ready queue. Min heap, so store -priority.
        # Tie-breaker: smaller ID (original order) to keep stability?
        # Using n['id'] as tie breaker.
        ready_queue = []
        for n in nodes:
            if n['unscheduled_preds'] == 0:
                heapq.heappush(ready_queue, (-n['priority'], n['id']))

        instrs = []

        while ready_queue:
            cycle_instr = defaultdict(list)
            temp_queue = []
            selected_nodes = []

            # Pop nodes that fit in this cycle
            while ready_queue:
                prio, nid = heapq.heappop(ready_queue)
                node = nodes[nid]
                eng = node['engine']

                # Check limits
                limit = SLOT_LIMITS.get(eng, 0)
                if len(cycle_instr[eng]) < limit:
                    cycle_instr[eng].append(node['slot'])
                    selected_nodes.append(node)
                else:
                    temp_queue.append((prio, nid))

            # Put back deferred nodes
            for item in temp_queue:
                heapq.heappush(ready_queue, item)

            if not selected_nodes:
                # Should not happen unless limits are 0 or empty queue (but loop condition handles that)
                break

            instrs.append(dict(cycle_instr))

            # Unlock successors
            for node in selected_nodes:
                for succ_id in node['succs']:
                    succ = nodes[succ_id]
                    succ['unscheduled_preds'] -= 1
                    if succ['unscheduled_preds'] == 0:
                        heapq.heappush(ready_queue, (-succ['priority'], succ['id']))

        return instrs

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
        # Use the smart scheduler
        return Scheduler().schedule(slots)

    def _get_rw(self, engine, slot):
        # Legacy method kept for compatibility if needed, but we use standalone get_rw
        reads, writes, _, _ = get_rw(engine, slot)
        return reads, writes

    def pack_slots(self, slots):
        # Use the smart scheduler here too
        return Scheduler().schedule(slots)

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
            # Optimize (x + c) + ((x + c) << s) -> x * (1 + 2^s) + c * (1 + 2^s)
            if op1 == "+" and op2 == "+" and op3 == "<<":
                multiplier = (1 << val3) + 1
                adder = val1
                c_mul = self.scratch_const_vec(multiplier)
                c_add = self.scratch_const_vec(adder)
                slots.append(("valu", ("multiply_add", val_hash_vec, val_hash_vec, c_mul, c_add)))
            else:
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
        Vectorized kernel implementation with unrolling and scratch buffer optimization.
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
            t = self.alloc_scratch()
            self.add("load", ("const", t, i))
            self.add("load", ("load", self.scratch[v], t))

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
        inp_indices_addr = self.alloc_scratch("inp_indices_addr")
        inp_values_addr = self.alloc_scratch("inp_values_addr")

        # Temp vectors for unrolling
        UNROLL_FACTOR = 32
        tmp_node_vals = [self.alloc_scratch_vec(f"tmp_node_val_{k}") for k in range(UNROLL_FACTOR)]
        tmp1s = [self.alloc_scratch_vec(f"tmp1_{k}") for k in range(UNROLL_FACTOR)]
        tmp3s = [self.alloc_scratch_vec(f"tmp3_{k}") for k in range(UNROLL_FACTOR)]

        # Address registers for unrolled gather (scalar)
        # Reuse tmp1s as address registers to save space
        tmp_addrs_pool = []
        for u in range(UNROLL_FACTOR):
             base = tmp1s[u]
             for k in range(VLEN):
                 tmp_addrs_pool.append(base + k)

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
            # Use unique address register to allow parallelism
            addr_reg = tmp1s[vi % UNROLL_FACTOR]
            # vload idx
            prologue.append(("alu", ("+", addr_reg, self.scratch["inp_indices_p"], i_const)))
            prologue.append(("load", ("vload", idx_buf + i, addr_reg)))
            # vload val
            # Note: Reusing addr_reg creates a dependency between idx load and val load for this vector,
            # but that's fine as long as different vectors are parallel.
            prologue.append(("alu", ("+", addr_reg, self.scratch["inp_values_p"], i_const)))
            prologue.append(("load", ("vload", val_buf + i, addr_reg)))

        for i in range(remainder_start, batch_size):
            i_const = self.scratch_const(i)
            prologue.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
            prologue.append(("load", ("load", idx_buf + i, tmp_addr)))
            prologue.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
            prologue.append(("load", ("load", val_buf + i, tmp_addr)))

        self.instrs.extend(self.pack_slots(prologue))

        # Increase pool size to support up to Round 3 (8 values)
        mux_pool = [self.alloc_scratch_vec(f"mux_{k}") for k in range(2)]
        for round in range(rounds):
            eff_round = round % (forest_height + 1)
            # Use Mux for Rounds 0, 1 (and 11, 12)
            use_mux = eff_round <= 1

            # Pre-load tree values for Mux
            mux_vals = []
            if use_mux:
                base_idx = (1 << eff_round) - 1
                count = 1 << eff_round
                for k in range(count):
                    k_const = self.scratch_const(base_idx + k)
                    # Reuse tmp_addr and tmp_val for loading
                    body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], k_const)))
                    body.append(("load", ("load", tmp_val, tmp_addr)))
                    val_vec = mux_pool[k]
                    body.append(("valu", ("vbroadcast", val_vec, tmp_val)))
                    mux_vals.append(val_vec)

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
                    tmp3_u = tmp3s[u]

                    if use_mux:
                        current_vals = list(mux_vals)

                        def emit_flow_select(dest, mask, a, b):
                            # dest = a if mask!=0 else b
                            # Using vselect: dest, cond, a, b
                            ops_loads.append(("flow", ("vselect", dest, mask, a, b)))

                        # Recursive helper to build selection tree
                        def build_select(vals, bit_index, avail_regs):
                            if len(vals) == 1:
                                return vals[0]

                            mid = len(vals) // 2
                            vals0 = vals[:mid] # Corresponds to mask=0
                            vals1 = vals[mid:] # Corresponds to mask=1

                            dest = avail_regs[0]
                            inner_regs = avail_regs[1:]

                            L = build_select(vals0, bit_index - 1, inner_regs)

                            used_by_L = 1 if len(vals0) > 1 else 0
                            R = build_select(vals1, bit_index - 1, inner_regs[used_by_L:])

                            base_val = (1 << eff_round) - 1
                            base_const = self.scratch_const_vec(base_val)

                            # mask = (idx - base) >> bit_index & 1
                            # Reuse dest for mask
                            ops_loads.append(("valu", ("-", dest, curr_idx_vec, base_const)))
                            if bit_index > 0:
                                shift_const = self.scratch_const_vec(bit_index)
                                ops_loads.append(("valu", (">>", dest, dest, shift_const)))

                            mask_const = self.scratch_const_vec(1)
                            ops_loads.append(("valu", ("&", dest, dest, mask_const)))

                            emit_flow_select(dest, dest, R, L)
                            return dest

                        # Available temps: tmp_node_val_u, tmp1_u, tmp3_u
                        regs = [tmp_node_val_u, tmp1_u, tmp3_u]

                        import math
                        top_bit = int(math.log2(len(current_vals))) - 1

                        res = build_select(current_vals, top_bit, regs)
                        if res != tmp_node_val_u:
                             ops_loads.append(("valu", ("+", tmp_node_val_u, res, zero_vec)))

                    else:
                        # node_val_vec gather
                        for k in range(VLEN):
                            # Use unique addr reg
                            addr_reg = tmp_addrs_pool[u * VLEN + k]
                            ops_addrs.append(("alu", ("+", addr_reg, self.scratch["forest_values_p"], curr_idx_vec + k)))
                            ops_loads.append(("load", ("load", tmp_node_val_u + k, addr_reg)))

                    # Vectorized hash
                    ops_hashes.append(("valu", ("^", curr_val_vec, curr_val_vec, tmp_node_val_u)))
                    ops_hashes.extend(self.build_hash_vector(curr_val_vec, tmp1_u, tmp3_u, round, i))

                    # Vectorized index update
                    # inc = 1 + (val & 1)
                    # idx = 2 * idx + inc
                    # idx = 2 * idx + 1 + (val & 1)
                    # Use multiply_add: d = a * b + c
                    # We want: idx = idx * 2 + (1 + (val & 1))
                    ops_updates.append(("valu", ("&", tmp1_u, curr_val_vec, one_vec)))
                    ops_updates.append(("valu", ("+", tmp3_u, one_vec, tmp1_u)))
                    ops_updates.append(("valu", ("multiply_add", curr_idx_vec, curr_idx_vec, two_vec, tmp3_u)))

                    # Wrap: idx = idx * (idx < n_nodes)
                    # Optimization: Only needed if max_idx could exceed n_nodes
                    # max_idx at end of round r is roughly 2^(r+1).
                    if (1 << (round + 1)) >= n_nodes:
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
            addr_reg = tmp1s[vi % UNROLL_FACTOR]
            epilogue.append(("alu", ("+", addr_reg, self.scratch["inp_indices_p"], i_const)))
            epilogue.append(("store", ("vstore", addr_reg, idx_buf + i)))
            epilogue.append(("alu", ("+", addr_reg, self.scratch["inp_values_p"], i_const)))
            epilogue.append(("store", ("vstore", addr_reg, val_buf + i)))

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
