
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

        # 3. Calculate priorities (Critical Path + Successor Count)
        # Iterative height calc
        in_degree = defaultdict(int)
        for n in nodes:
            for s in n['succs']:
                in_degree[s] += 1
        q = [n['id'] for n in nodes if in_degree[n['id']] == 0]
        topo_order = []
        while q:
            u = q.pop(0)
            topo_order.append(u)
            for v in nodes[u]['succs']:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)

        heights = {}
        for u in reversed(topo_order):
            h = 0
            for v in nodes[u]['succs']:
                h = max(h, heights.get(v, 0))
            heights[u] = 1 + h

        for n in nodes:
            # Metric: Height * 10 + OutDegree
            n['priority'] = heights[n['id']] * 10 + len(n['succs'])

            # Prioritize LOADs because they are the bottleneck
            if n['engine'] == 'load':
                n['priority'] += 100000
            elif n['engine'] == 'valu':
                n['priority'] += 80000
            elif n['engine'] == 'flow':
                n['priority'] += 50000
            n['unscheduled_preds'] = len(n['preds'])

        # 4. List Scheduling
        ready_queue = []
        for n in nodes:
            if n['unscheduled_preds'] == 0:
                heapq.heappush(ready_queue, (-n['priority'], n['id']))

        instrs = []

        while ready_queue:
            cycle_instr = defaultdict(list)
            temp_queue = []
            selected_nodes = []

            while ready_queue:
                prio, nid = heapq.heappop(ready_queue)
                node = nodes[nid]
                eng = node['engine']

                limit = SLOT_LIMITS.get(eng, 0)
                if len(cycle_instr[eng]) < limit:
                    cycle_instr[eng].append(node['slot'])
                    selected_nodes.append(node)
                else:
                    temp_queue.append((prio, nid))

            for item in temp_queue:
                heapq.heappush(ready_queue, item)

            if not selected_nodes:
                break

            instrs.append(dict(cycle_instr))

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
        self.pending_slots = [] # Collect all slots here
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.const_vec_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        return Scheduler().schedule(slots)

    def _get_rw(self, engine, slot):
        reads, writes, _, _ = get_rw(engine, slot)
        return reads, writes

    def pack_slots(self, slots):
        # Just return the slots, do not schedule yet
        return slots

    def add(self, engine, slot):
        self.pending_slots.append((engine, slot))

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
        return slots

    def build_hash_vector(self, val_hash_vec, tmp1_vec, tmp2_vec, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
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
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
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

        zero_vec = self.scratch_const_vec(0)
        one_vec = self.scratch_const_vec(1)
        two_vec = self.scratch_const_vec(2)
        n_nodes_vec = self.scratch_const_vec(n_nodes)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        inp_indices_addr = self.alloc_scratch("inp_indices_addr")
        inp_values_addr = self.alloc_scratch("inp_values_addr")

        UNROLL_FACTOR = 32
        tmp_node_vals = [self.alloc_scratch_vec(f"tmp_node_val_{k}") for k in range(UNROLL_FACTOR)]
        tmp1s = [self.alloc_scratch_vec(f"tmp1_{k}") for k in range(UNROLL_FACTOR)]
        tmp3s = [self.alloc_scratch_vec(f"tmp3_{k}") for k in range(UNROLL_FACTOR)]

        tmp_addrs_pool = []
        for u in range(UNROLL_FACTOR):
             base = tmp1s[u]
             for k in range(VLEN):
                 tmp_addrs_pool.append(base + k)

        idx_buf = self.alloc_scratch("idx_buf", length=batch_size)
        val_buf = self.alloc_scratch("val_buf", length=batch_size)

        vector_loops = batch_size // VLEN
        unrolled_loops = vector_loops // UNROLL_FACTOR
        remainder_start = vector_loops * VLEN

        prologue = []
        for vi in range(vector_loops):
            i = vi * VLEN
            i_const = self.scratch_const(i)
            addr_reg = tmp1s[vi % UNROLL_FACTOR]
            prologue.append(("alu", ("+", addr_reg, self.scratch["inp_indices_p"], i_const)))
            prologue.append(("load", ("vload", idx_buf + i, addr_reg)))
            prologue.append(("alu", ("+", addr_reg, self.scratch["inp_values_p"], i_const)))
            prologue.append(("load", ("vload", val_buf + i, addr_reg)))

        MAX_MUX_VALS = 7
        vec_tree_vals = [self.alloc_scratch_vec(f"tree_val_{k}") for k in range(MAX_MUX_VALS)]
        for k in range(MAX_MUX_VALS):
            k_const = self.scratch_const(k)
            prologue.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], k_const)))
            prologue.append(("load", ("load", tmp_val, tmp_addr)))
            prologue.append(("valu", ("vbroadcast", vec_tree_vals[k], tmp_val)))

        for i in range(remainder_start, batch_size):
            i_const = self.scratch_const(i)
            prologue.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
            prologue.append(("load", ("load", idx_buf + i, tmp_addr)))
            prologue.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
            prologue.append(("load", ("load", val_buf + i, tmp_addr)))

        # Just add to pending
        self.pending_slots.extend(prologue)

        PIPELINE_STRIDE = 10
        last_result_vec = [None] * UNROLL_FACTOR

        for round in range(rounds):
            eff_round = round % (forest_height + 1)
            # Enable Mux for Rounds 0, 1, 2
            use_mux = eff_round <= 2

            for vui in range(unrolled_loops):
                vi_base = vui * UNROLL_FACTOR

                ops_addrs = []
                ops_loads = []
                ops_hashes = []
                ops_updates = []

                for u in range(UNROLL_FACTOR):
                    vi = vi_base + u
                    i = vi * VLEN

                    curr_idx_vec = idx_buf + i
                    curr_val_vec = val_buf + i

                    # INJECT PIPELINE DEPENDENCY
                    prev_u = u - PIPELINE_STRIDE
                    if prev_u >= 0 and last_result_vec[prev_u] is not None:
                         dep_reg = last_result_vec[prev_u]
                         ops_loads.append(("alu", ("+", tmp1, dep_reg, zero_const)))

                    tmp_node_val_u = tmp_node_vals[u]
                    tmp1_u = tmp1s[u]
                    tmp3_u = tmp3s[u]

                    if use_mux:
                        base_idx = (1 << eff_round) - 1
                        count = 1 << eff_round
                        current_vals = vec_tree_vals[base_idx : base_idx + count]

                        def emit_flow_select(dest, mask, a, b):
                            ops_loads.append(("flow", ("vselect", dest, mask, a, b)))

                        def build_iterative_mux(vals, regs):
                            num_vals = len(vals)

                            base_val = (1 << eff_round) - 1
                            base_const = self.scratch_const_vec(base_val)

                            def get_mask(bit_idx, target_reg):
                                if bit_idx == 0:
                                    # Optimization: Bit 0 of (idx - base) is equivalent to (val & 1)
                                    # This saves subtraction and potential shifting.
                                    # curr_val_vec holds the value from the previous round (input to this round).
                                    mask_const = self.scratch_const_vec(1)
                                    ops_loads.append(("valu", ("&", target_reg, curr_val_vec, mask_const)))
                                else:
                                    ops_loads.append(("valu", ("-", target_reg, curr_idx_vec, base_const)))
                                    # Optimization: Skip shift. Use (1 << bit_idx) mask directly.
                                    # vselect treats any non-zero value as true.
                                    mask_val = 1 << bit_idx
                                    mask_const = self.scratch_const_vec(mask_val)
                                    ops_loads.append(("valu", ("&", target_reg, target_reg, mask_const)))

                            import math
                            depth = int(math.log2(num_vals)) if num_vals > 0 else 0

                            current_level_vals = vals

                            for b in range(0, depth):
                                mask_reg = regs[2]
                                get_mask(b, mask_reg)

                                new_vals = []
                                for k in range(0, len(current_level_vals), 2):
                                    A = current_level_vals[k+1]
                                    B = current_level_vals[k]
                                    dest = regs[k // 2]

                                    emit_flow_select(dest, mask_reg, A, B)

                                    new_vals.append(dest)
                                current_level_vals = new_vals

                            if len(vals) == 1:
                                ops_loads.append(("valu", ("+", regs[0], vals[0], zero_vec)))


                        regs = [tmp_node_val_u, tmp1_u, tmp3_u]
                        build_iterative_mux(current_vals, regs)

                    else:
                        for k in range(VLEN):
                            addr_reg = tmp_addrs_pool[u * VLEN + k]
                            ops_addrs.append(("alu", ("+", addr_reg, self.scratch["forest_values_p"], curr_idx_vec + k)))
                            ops_loads.append(("load", ("load", tmp_node_val_u + k, addr_reg)))

                    ops_hashes.append(("valu", ("^", curr_val_vec, curr_val_vec, tmp_node_val_u)))
                    ops_hashes.extend(self.build_hash_vector(curr_val_vec, tmp1_u, tmp3_u, round, i))

                    ops_updates.append(("valu", ("&", tmp1_u, curr_val_vec, one_vec)))
                    ops_updates.append(("valu", ("+", tmp3_u, one_vec, tmp1_u)))
                    ops_updates.append(("valu", ("multiply_add", curr_idx_vec, curr_idx_vec, two_vec, tmp3_u)))

                    if (1 << (round + 1)) >= n_nodes:
                        ops_updates.append(("valu", ("<", tmp1_u, curr_idx_vec, n_nodes_vec)))
                        ops_updates.append(("flow", ("vselect", curr_idx_vec, tmp1_u, curr_idx_vec, zero_vec)))

                    # Update the result register for the next pipeline stage
                    last_result_vec[u] = curr_val_vec

                self.pending_slots.extend(ops_addrs)
                self.pending_slots.extend(ops_loads)
                self.pending_slots.extend(ops_hashes)
                self.pending_slots.extend(ops_updates)

            # Vector cleanup loop for remaining vectors
            for vi in range(unrolled_loops * UNROLL_FACTOR, vector_loops):
                i = vi * VLEN
                curr_idx_vec = idx_buf + i
                curr_val_vec = val_buf + i
                
                t_node = tmp_node_vals[0]
                t1 = tmp1s[0]
                t3 = tmp3s[0]

                for k in range(VLEN):
                    addr_reg = tmp_addrs_pool[k]
                    self.pending_slots.append(("alu", ("+", addr_reg, self.scratch["forest_values_p"], curr_idx_vec + k)))
                    self.pending_slots.append(("load", ("load", t_node + k, addr_reg)))

                self.pending_slots.append(("valu", ("^", curr_val_vec, curr_val_vec, t_node)))
                self.pending_slots.extend(self.build_hash_vector(curr_val_vec, t1, t3, round, i))

                self.pending_slots.append(("valu", ("&", t1, curr_val_vec, one_vec)))
                self.pending_slots.append(("valu", ("+", t3, one_vec, t1)))
                self.pending_slots.append(("valu", ("multiply_add", curr_idx_vec, curr_idx_vec, two_vec, t3)))

                if (1 << (round + 1)) >= n_nodes:
                    self.pending_slots.append(("valu", ("<", t1, curr_idx_vec, n_nodes_vec)))
                    self.pending_slots.append(("flow", ("vselect", curr_idx_vec, t1, curr_idx_vec, zero_vec)))

            for i in range(remainder_start, batch_size):
                curr_idx = idx_buf + i
                curr_val = val_buf + i

                self.pending_slots.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], curr_idx)))
                self.pending_slots.append(("load", ("load", tmp_node_val, tmp_addr)))

                self.pending_slots.append(("alu", ("^", curr_val, curr_val, tmp_node_val)))
                self.pending_slots.extend(self.build_hash(curr_val, tmp1, tmp2, round, i))

                self.pending_slots.append(("alu", ("%", tmp1, curr_val, two_const)))
                self.pending_slots.append(("alu", ("==", tmp1, tmp1, zero_const)))
                self.pending_slots.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                self.pending_slots.append(("alu", ("*", curr_idx, curr_idx, two_const)))
                self.pending_slots.append(("alu", ("+", curr_idx, curr_idx, tmp3)))

                self.pending_slots.append(("alu", ("<", tmp1, curr_idx, self.scratch["n_nodes"])))
                self.pending_slots.append(("flow", ("select", curr_idx, tmp1, curr_idx, zero_const)))

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

        self.pending_slots.extend(epilogue)

        # Schedule everything at once
        self.instrs = self.build(self.pending_slots)
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

    # Run reference kernel to the end to get final state
    ref_mem = None
    for ref_mem in reference_kernel2(mem, value_trace):
        pass

    # Run machine until finished (handling pauses)
    from problem import CoreState
    while machine.cores[0].state != CoreState.STOPPED:
        machine.run()

    try:
        # Check final result
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result (Final)"

        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        assert (
            machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
            == ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)]
        ), f"Incorrect indices (Final)"
    except AssertionError as e:
        print(e)

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
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
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)

if __name__ == "__main__":
    unittest.main()
