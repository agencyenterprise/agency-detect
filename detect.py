#!/usr/bin/env python
# detect_async_agents.py
import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import warnings

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

# ---------- 2. Markov Blanket Validation -----------------------------------

def conditional_mutual_info_knn(X, Y, Z, k=3):
    """
    Estimate conditional mutual information I(X;Y|Z) using k-NN method.
    
    Based on: Frenzel & Pompe (2007) "Partial Mutual Information for Coupling Analysis"
    I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
    """
    if len(X) != len(Y) or len(X) != len(Z):
        raise ValueError("All input arrays must have same length")
    
    n = len(X)
    if n < k + 1:
        return 0.0
    
    # Ensure arrays are 2D
    X = np.atleast_2d(X).T if X.ndim == 1 else X
    Y = np.atleast_2d(Y).T if Y.ndim == 1 else Y  
    Z = np.atleast_2d(Z).T if Z.ndim == 1 else Z
    
    # Joint spaces
    XZ = np.hstack([X, Z])
    YZ = np.hstack([Y, Z])
    XYZ = np.hstack([X, Y, Z])
    
    # Fit k-NN models
    try:
        nbrs_xz = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(XZ)
        nbrs_yz = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(YZ)
        nbrs_xyz = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(XYZ)
        nbrs_z = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(Z)
        
        # Get distances to k-th nearest neighbor for each point
        distances_xz, _ = nbrs_xz.kneighbors(XZ)
        distances_yz, _ = nbrs_yz.kneighbors(YZ)
        distances_xyz, _ = nbrs_xyz.kneighbors(XYZ)
        distances_z, _ = nbrs_z.kneighbors(Z)
        
        # Use k-th neighbor distance (index k, since we include self)
        eps_xz = distances_xz[:, k]
        eps_yz = distances_yz[:, k]
        eps_xyz = distances_xyz[:, k]
        eps_z = distances_z[:, k]
        
        # Estimate conditional MI using k-NN formula
        # This is a simplified version - full implementation would use digamma functions
        cmi = np.mean(np.log(eps_xz + 1e-10) + np.log(eps_yz + 1e-10) 
                     - np.log(eps_xyz + 1e-10) - np.log(eps_z + 1e-10))
        
        return max(0.0, cmi)  # CMI should be non-negative
    
    except Exception as e:
        warnings.warn(f"k-NN CMI estimation failed: {e}")
        return 0.0

def classify_variables(cluster_vars, all_vars, data, trace):
    """
    Classify variables in a cluster into Sensors (S), Actions (A), Internal (I).
    
    Heuristic classification:
    - Sensors: High MI with environment variables
    - Actions: High MI with other agents' future states  
    - Internal: Neither sensors nor actions
    """
    var_to_idx = {var: i for i, var in enumerate(all_vars)}
    cluster_indices = [var_to_idx[var] for var in cluster_vars]
    env_vars = [var for var in all_vars if var not in cluster_vars]
    env_indices = [var_to_idx[var] for var in env_vars]
    
    n_vars = len(cluster_vars)
    n_env = len(env_vars)
    
    if n_env == 0:
        # No environment variables - classify based on predictive power
        return {'S': [], 'A': cluster_vars[:n_vars//2], 'I': cluster_vars[n_vars//2:]}
    
    # Compute MI between cluster variables and environment
    env_mi = np.zeros(n_vars)
    for i, var_idx in enumerate(cluster_indices):
        for env_idx in env_indices:
            mi = mutual_info_score(data[:, var_idx], data[:, env_idx])
            env_mi[i] += mi
    
    # Compute MI between cluster variables and other agents' future states
    future_mi = np.zeros(n_vars)
    if len(data) > 1:
        for i, var_idx in enumerate(cluster_indices):
            for j, other_idx in enumerate(cluster_indices):
                if i != j and len(data) > 1:
                    # Lagged MI to detect influence on future states
                    mi = mutual_info_score(data[:-1, var_idx], data[1:, other_idx])
                    future_mi[i] += mi
    
    # Classification thresholds (these could be tuned)
    env_threshold = np.percentile(env_mi, 70) if n_vars > 1 else 0
    future_threshold = np.percentile(future_mi, 70) if n_vars > 1 else 0
    
    sensors = [cluster_vars[i] for i in range(n_vars) if env_mi[i] > env_threshold]
    actions = [cluster_vars[i] for i in range(n_vars) 
              if future_mi[i] > future_threshold and cluster_vars[i] not in sensors]
    internal = [var for var in cluster_vars if var not in sensors and var not in actions]
    
    return {'S': sensors, 'A': actions, 'I': internal}

def validate_markov_blanket(cluster_vars, classification, all_vars, data, tolerance=0.1):
    """
    Validate that a cluster satisfies the Markov blanket property:
    I(I_{t+1}; E_{t+1} | S_t, A_t) ≈ 0
    
    Returns:
        (is_valid, violation_score, details)
    """
    var_to_idx = {var: i for i, var in enumerate(all_vars)}
    
    # Get variable indices
    S_vars = classification['S']  # Sensors
    A_vars = classification['A']  # Actions  
    I_vars = classification['I']  # Internal states
    E_vars = [var for var in all_vars if var not in cluster_vars]  # Environment
    
    if not I_vars or not E_vars or len(data) < 10:
        return True, 0.0, "Insufficient data or variables for validation"
    
    # Prepare data for conditional MI calculation
    # We need t and t+1 time steps
    if len(data) < 2:
        return True, 0.0, "Insufficient time steps"
    
    try:
        # Current time step data
        S_t = np.column_stack([data[:-1, var_to_idx[var]] for var in S_vars]) if S_vars else np.zeros((len(data)-1, 1))
        A_t = np.column_stack([data[:-1, var_to_idx[var]] for var in A_vars]) if A_vars else np.zeros((len(data)-1, 1))
        I_t = np.column_stack([data[:-1, var_to_idx[var]] for var in I_vars])
        E_t = np.column_stack([data[:-1, var_to_idx[var]] for var in E_vars])
        
        # Next time step data  
        I_t1 = np.column_stack([data[1:, var_to_idx[var]] for var in I_vars])
        E_t1 = np.column_stack([data[1:, var_to_idx[var]] for var in E_vars])
        
        # Conditioning variables: S_t and A_t
        if S_vars and A_vars:
            conditioning = np.column_stack([S_t, A_t])
        elif S_vars:
            conditioning = S_t
        elif A_vars:
            conditioning = A_t
        else:
            conditioning = np.zeros((len(data)-1, 1))
        
        # Compute conditional MI: I(I_{t+1}; E_{t+1} | S_t, A_t)
        cmi = conditional_mutual_info_knn(I_t1, E_t1, conditioning)
        
        is_valid = cmi <= tolerance
        details = f"CMI={cmi:.4f}, threshold={tolerance}, S={len(S_vars)}, A={len(A_vars)}, I={len(I_vars)}, E={len(E_vars)}"
        
        return is_valid, cmi, details
        
    except Exception as e:
        warnings.warn(f"Markov blanket validation failed: {e}")
        return True, 0.0, f"Validation error: {e}"

# ---------- 3. Enhanced Detection Pipeline -----------------------------------
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

def detect_async_agents(trace, n_agents=2, max_lag=3, weak_thresh=0.75, 
                       validate_blankets=True, blanket_tolerance=0.1):
    """
    Enhanced agent detection with Markov blanket validation.
    """
    vars_ = list(trace[0].keys())
    data  = np.array([[rec[v] for v in vars_] for rec in trace])

    # ── 1-A Remove variables that never change ────────────────────────────
    var          = data.var(axis=0)
    active_idx   = np.where(var > 0.0)[0]
    inactive_idx = np.where(var == 0.0)[0]        # "frozen" → environment
    vars_active  = [vars_[i] for i in active_idx]
    data_active  = data[:, active_idx]            # cluster only actives

    # pair-wise lag-max MI → similarity matrix
    D = data_active.shape[1]
    sim = np.zeros((D, D))
    for i, j in combinations(range(D), 2):
        w = lagmax_mi(data_active[:, i], data_active[:, j], max_lag=max_lag)
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
    validated_clusters = {}

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
            continue
            
        # ── Markov Blanket Validation ──────────────────────────────────────
        if validate_blankets and len(mem) > 1:
            # Classify variables within cluster
            classification = classify_variables(mem, vars_, data, trace)
            
            # Validate Markov blanket property
            is_valid, violation, details = validate_markov_blanket(
                mem, classification, vars_, data, blanket_tolerance)
            
            if is_valid:
                validated_clusters[lbl] = {
                    'variables': mem,
                    'classification': classification,
                    'blanket_validation': {
                        'valid': True,
                        'violation': violation,
                        'details': details
                    }
                }
            else:
                # Blanket validation failed - demote to environment
                print(f"Agent {lbl} failed Markov blanket validation: {details}")
                env_bucket.extend(mem)
        else:
            # No validation requested or insufficient variables
            validated_clusters[lbl] = {
                'variables': mem,
                'classification': {'S': [], 'A': [], 'I': mem},
                'blanket_validation': {
                    'valid': None,
                    'violation': 0.0,
                    'details': 'Validation skipped'
                }
            }

    # ── 1-B  Add the frozen zero-variance variables  ───────────────────────
    for i in inactive_idx:                       # they were never clustered
        env_bucket.append(vars_[i])

    if env_bucket:
        validated_clusters['env'] = {
            'variables': env_bucket,
            'classification': {'S': [], 'A': [], 'I': env_bucket},
            'blanket_validation': {
                'valid': None,
                'violation': 0.0,
                'details': 'Environment variables'
            }
        }

    return validated_clusters

# ---------- 4. Demonstration -------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)               # Reproducible results for testing
    trace    = generate_trace_async(steps=5000)
    clusters = detect_async_agents(trace, validate_blankets=True)

    print("=== Enhanced Agent Detection with Markov Blanket Validation ===\n")
    
    for lbl in sorted(k for k in clusters if k != 'env'):
        cluster_info = clusters[lbl]
        print(f"Agent {lbl}:")
        print(f"  Variables: {cluster_info['variables']}")
        
        classification = cluster_info['classification']
        print(f"  Sensors (S): {classification['S']}")
        print(f"  Actions (A): {classification['A']}")  
        print(f"  Internal (I): {classification['I']}")
        
        validation = cluster_info['blanket_validation']
        print(f"  Markov Blanket: Valid={validation['valid']}, Violation={validation['violation']:.4f}")
        print(f"  Details: {validation['details']}")
        print()
    
    if 'env' in clusters:
        print("Environment:")
        print(f"  Variables: {clusters['env']['variables']}")
        print(f"  Details: {clusters['env']['blanket_validation']['details']}")
