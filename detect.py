import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.cluster import AgglomerativeClustering

# --- Simulation of two small FSM agents ---
class FSM:
    def __init__(self, n_states, n_regs, id):
        self.n_states = n_states
        self.state = 0
        self.regs = np.zeros(n_regs, dtype=int)
        self.id = id
    def step(self):
        # simple rule: toggle a register and change state cyclically
        reg_idx = self.state % len(self.regs)
        self.regs[reg_idx] ^= 1
        self.state = (self.state + 1) % self.n_states
        return self.state, self.regs.copy()

# generate trace
def generate_trace(steps=1000):
    fsm1 = FSM(n_states=3, n_regs=4, id='A')
    fsm2 = FSM(n_states=2, n_regs=4, id='B')
    trace = []  # list of dicts {var: value}
    for t in range(steps):
        s1, r1 = fsm1.step()
        s2, r2 = fsm2.step()
        rec = {}
        # record state and regs as separate variables
        rec['A_state'] = s1
        rec.update({f'A_r{i}': r1[i] for i in range(len(r1))})
        rec['B_state'] = s2
        rec.update({f'B_r{i}': r2[i] for i in range(len(r2))})
        trace.append(rec)
    return trace

# --- Detection algorithm ---
# 1. Compute activity variance and threshold
# 2. Pairwise correlation clustering
# 3. Simple blanket test via mutual information

def detect_agents(trace, var_threshold=0.1, corr_threshold=0.1, mi_threshold=0.01):
    # prepare data matrix
    vars = list(trace[0].keys())
    T = len(trace)
    data = np.array([[rec[v] for v in vars] for rec in trace])

    # 1. activity variance
    vars_var = np.var(data, axis=0)
    active_idx = [i for i,vv in enumerate(vars_var) if vv >= var_threshold]
    active_vars = [vars[i] for i in active_idx]

    # 2. correlation-based clustering
    sub_data = data[:, active_idx]
    # compute correlation matrix
    corr = np.corrcoef(sub_data.T)
    # affinity for clustering (absolute corr)
    affinity = np.abs(corr)
    # cluster into two agents
    # clustering = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
    # --- correlation-based clustering (fixed) ------------------
    corr     = np.corrcoef(sub_data.T)
    distance = 1.0 - np.abs(corr)          # symmetric, zero-diagonal

    clustering = AgglomerativeClustering(
        n_clusters = 2,
        metric   = 'precomputed',        # use 'metric' if sklearn ≥ 1.4
        linkage    = 'average'
    )
    labels = clustering.fit_predict(distance)

    clusters = defaultdict(list)
    for var, lbl in zip(active_vars, labels):
        clusters[lbl].append(var)

    # 3. blanket test: for each cluster, compute MI between cluster internals and externals
    results = {}
    for lbl, members in clusters.items():
        # define blanket as reads/writes (all members here), externals = others
        internal_data = sub_data[:, [active_vars.index(m) for m in members]]
        external_idx = [i for i in range(sub_data.shape[1]) if vars[active_idx[i]] not in members]
        ext_data = sub_data[:, external_idx]
        # approximate MI by summing pairwise MI
        mi_sum = 0.0
        for idata in internal_data.T:
            for edata in ext_data.T:
                mi_sum += mutual_info_score(idata, edata)
        results[lbl] = {'members': members, 'blanket_violation': mi_sum}

    return clusters, results

# Example run
if __name__ == '__main__':
    trace = generate_trace(steps=2000)
    clusters, results = detect_agents(trace)
    for lbl, info in results.items():
        print(f'Agent {lbl}:')
        print(' Variables:', info['members'])
        print(' Blanket violation MI sum:', info['blanket_violation'])

