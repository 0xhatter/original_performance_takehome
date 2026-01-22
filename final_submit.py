import sys

with open('perf_takehome.py', 'r') as f:
    content = f.read()

content = content.replace('UNROLL_FACTOR = 32', 'UNROLL_FACTOR = 16')
content = content.replace('UNROLL_FACTOR = 8', 'UNROLL_FACTOR = 16')

start = '        # Temp vectors for unrolling'
end = '        # Address registers'
new_alloc = '''        # Temp vectors for unrolling
        UNROLL_FACTOR = 16
        tmp_node_vals = [self.alloc_scratch_vec(f"tmp_node_val_{k}") for k in range(UNROLL_FACTOR)]
        tmp1s = [self.alloc_scratch_vec(f"tmp1_{k}") for k in range(UNROLL_FACTOR)]
        tmp2s = [self.alloc_scratch_vec(f"tmp2_{k}") for k in range(UNROLL_FACTOR)]
        tmp3s = [self.alloc_scratch_vec(f"tmp3_{k}") for k in range(UNROLL_FACTOR)]
        tmp4s = [self.alloc_scratch_vec(f"tmp4_{k}") for k in range(UNROLL_FACTOR)]

'''
s_idx = content.find(start)
e_idx = content.find(end)
if s_idx != -1 and e_idx != -1:
    content = content[:s_idx] + new_alloc + content[e_idx:]

start_loop = '                    tmp_node_val_u = tmp_node_vals[u]'
end_loop = '                    if use_mux:'
new_loop = '''                    tmp_node_val_u = tmp_node_vals[u]
                    tmp1_u = tmp1s[u]
                    tmp2_u = tmp2s[u]
                    tmp3_u = tmp3s[u]
                    tmp4_u = tmp4s[u]

'''
s_idx = content.find(start_loop)
e_idx = content.find(end_loop)
if s_idx != -1 and e_idx != -1:
    content = content[:s_idx] + new_loop + content[e_idx:]

content = content.replace('range(8)', 'range(4)')
content = content.replace('use_mux = round <= 3', 'use_mux = round <= 2')

start = '                        # Round 3: 8 vals.'
end = '                    else:'
s_idx = content.find(start)
e_idx = content.find(end, s_idx)
if s_idx != -1 and e_idx != -1:
    content = content[:s_idx] + content[e_idx:]

with open('perf_takehome.py', 'w') as f:
    f.write(content)
