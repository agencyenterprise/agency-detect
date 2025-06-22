#!/usr/bin/env python
# detect_async_agents.py
import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.cluster import AgglomerativeClustering

# ---------- 1. Minimal toy world --------------------------------------------
class FSM:
    def __init__(self, n_states, n_regs, name):
        self.n_states = n_states
        self.state = 0
        self.regs = np.zeros(n_regs, dtype=int)
        self.name = name

    def tick(self):                              # advance exactly once
        #r = self.state % len(self.regs)
        #self.regs[r] ^= 1                        # toggle one register
        #self.state = (self.state + 1) % self.n_states
        #return self.state, self.regs.copy()
        
        # --- make the internal registers *represent* the state -----------
        # r_i = 1  ⟺  state == i            (one-hot encoding)
        # the last register never changes → becomes an "environment" bit
        for i in range(self.n_states):
            if i < len(self.regs) - 1:
                self.regs[i] = int(self.state == i)
        self.state = (self.state + 1) % self.n_states
        return self.state, self.regs.copy()

def generate_trace_async(steps=5000, p_a=0.7, p_b=0.5):
    """Each FSM advances independently with probability p∈(0,1) per world-tick."""
    a, b = FSM(3, 4, 'A'), FSM(2, 4, 'B')
    trace = []

    # ---- initialise *both* machines independently ------------------------
    last_a = a.tick()           # (state, regs)
    last_b = b.tick()           # (state, regs)
    for _ in range(steps):
        if np.random.rand() < p_a: last_a = a.tick()
        if np.random.rand() < p_b: last_b = b.tick()
        rec = {
            'A_state': last_a[0],
            **{f'A_r{i}': last_a[1][i] for i in range(len(last_a[1]))},
            'B_state': last_b[0],
            **{f'B_r{i}': last_b[1][i] for i in range(len(last_b[1]))},
        }
        trace.append(rec)
    return trace

# ---------- 2. Detection pipeline -------------------------------------------
def lagmax_mi(x, y, max_lag=3):
    best = 0.0
    for τ in range(-max_lag, max_lag + 1):       # keep τ = 0
        if τ == 0:
            w = mutual_info_score(x, y)          # synchronous influence
            best = max(best, w)
            continue
        if τ > 0: xi, yi = x[:-τ], y[τ:]
        else:     xi, yi = x[-τ:], y[:τ]
        best = max(best, mutual_info_score(xi, yi))
    return best

def detect_async_agents(trace, n_agents=2, max_lag=3, weak_thresh=0.75):
    vars_ = list(trace[0].keys())
    data  = np.array([[rec[v] for v in vars_] for rec in trace])

    # ── 1-A Remove variables that never change ────────────────────────────
    var          = data.var(axis=0)
    active_idx   = np.where(var > 0.0)[0]
    inactive_idx = np.where(var == 0.0)[0]        # "frozen" → environment
    vars_active  = [vars_[i] for i in active_idx]
    data         = data[:, active_idx]            # cluster only actives

    # pair-wise lag-max MI → similarity matrix
    D = data.shape[1]
    sim = np.zeros((D, D))
    for i, j in combinations(range(D), 2):
        w = lagmax_mi(data[:, i], data[:, j], max_lag=max_lag)
        sim[i, j] = sim[j, i] = w
    dist = 1.0 - sim / (sim.max() + 1e-12)       # normalise → distance

    clustering = AgglomerativeClustering(
        n_clusters=n_agents, metric='precomputed', linkage='complete')
    labels = clustering.fit_predict(dist)

    clusters = defaultdict(list)
    for v, lbl in zip(vars_active, labels):
        clusters[lbl].append(v)

    # ── 2-B Eject weakly connected variables → environment ────────────────
    env_bucket = []                              # collects strays and frozen

    for lbl in list(clusters.keys()):            # we may delete empty ones
        mem = clusters[lbl]
        if len(mem) <= 1:                        # singletons handled later
            continue
        idx = [vars_active.index(v) for v in mem]
        sub_sim = sim[np.ix_(idx, idx)]

        # average similarity between all pairs inside this cluster
        mean_intra = (sub_sim.sum() - np.trace(sub_sim)) / (len(idx)*(len(idx)-1))

        # test every variable against that average
        for v in mem[:]:                         # iterate over a *copy*
            j = vars_active.index(v)
            sim_to_cluster = (sub_sim[idx.index(j)].sum() - 0) / (len(idx)-1)
            if sim_to_cluster < weak_thresh * mean_intra:
                mem.remove(v)
                env_bucket.append(v)

        if not mem:                              # cluster became empty
            del clusters[lbl]

    # ── 1-B  Add the frozen zero-variance variables  ───────────────────────
    for i in inactive_idx:                       # they were never clustered
        env_bucket.append(vars_[i])

    if env_bucket:
        clusters['env'] = env_bucket

    return clusters

# ---------- 3. Demonstration -------------------------------------------------
if __name__ == '__main__':
    np.random.seed(None)               # different world every run
    trace    = generate_trace_async(steps=5000)
    clusters = detect_async_agents(trace)

    #for lbl, members in clusters.items():
    #    print(f'Agent {lbl}:', members)
    for lbl in sorted(k for k in clusters if k != 'env'):
        print(f'Agent {lbl}:', clusters[lbl])
    if 'env' in clusters:
        print('Environment:', clusters['env'])
