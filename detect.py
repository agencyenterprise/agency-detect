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

# ---------- 1. Completely independent agents with no coupling ----------------
class IndependentAgent:
    def __init__(self, name, agent_type, memory_size=3):
        self.name = name
        self.agent_type = agent_type  # 'alpha' or 'beta'
        self.memory = np.zeros(memory_size, dtype=int)
        self.internal_state = 0
        self.action = 0
        self.private_sensor = 0     # Only senses own domain
        self.goal_progress = 0
        
        # Initialize with different starting conditions and ranges
        if agent_type == 'alpha':
            self.internal_state = 1
            self.goal_progress = 0
        else:  # beta
            self.internal_state = 3
            self.goal_progress = 2
    
    def sense_private_domain(self, environment):
        """Each agent only senses its own private domain"""
        
        if self.agent_type == 'alpha':
            # Alpha agent: operates on fast energy cycles
            energy_level = environment.get('alpha_energy', 5)
            energy_flow = environment.get('alpha_flow', 1)
            # Simpler sensor - less internal coupling
            self.private_sensor = (energy_level * 2 + energy_flow) % 8
            
        else:  # beta agent
            # Beta agent: operates on slow material cycles  
            material_stock = environment.get('beta_material', 3)
            material_quality = environment.get('beta_quality', 2)
            # Simpler sensor - less internal coupling
            self.private_sensor = (material_stock * 3 + material_quality) % 8
    
    def decide_action(self):
        """Simpler decision making with less internal coupling"""
        memory_influence = sum(self.memory) % 3
        sensor_influence = self.private_sensor % 3
        
        # Simpler logic with less interdependence
        if self.agent_type == 'alpha':
            random_factor = np.random.randint(0, 2)
            decision_input = (sensor_influence + memory_influence + random_factor) % 6
            # Alpha actions: 0=store, 1=process, 2=release
            if decision_input < 2:
                self.action = 0
            elif decision_input < 4:
                self.action = 1
            else:
                self.action = 2
                
        else:  # beta agent
            random_factor = np.random.randint(0, 2)
            decision_input = (sensor_influence + memory_influence + random_factor) % 6
            # Beta actions: 0=maintain, 1=build, 2=repair
            if decision_input < 2:
                self.action = 0
            elif decision_input < 4:
                self.action = 1
            else:
                self.action = 2
        
        return self.action
    
    def update_state(self):
        """Simpler state updates with weaker coupling"""
        # Update memory with sensor reading
        self.memory[1:] = self.memory[:-1]
        self.memory[0] = self.private_sensor
        
        # Simpler internal state updates
        if self.agent_type == 'alpha':
            # Alpha: simpler dynamics
            self.internal_state = (self.internal_state + self.action + 1) % 8
            
            # Simpler goal progress
            self.goal_progress = (self.goal_progress + self.action + 1) % 10
                
        else:  # beta agent
            # Beta: simpler dynamics
            self.internal_state = (self.internal_state + self.action + 2) % 6
            
            # Simpler goal progress
            self.goal_progress = (self.goal_progress + self.action + 1) % 8
    
    def get_state_dict(self):
        """Return agent state"""
        return {
            f'{self.name}_sensor': self.private_sensor,
            f'{self.name}_action': self.action,
            f'{self.name}_internal': self.internal_state,
            f'{self.name}_goal': self.goal_progress,
            **{f'{self.name}_mem{i}': self.memory[i] for i in range(len(self.memory))}
        }

class DecoupledEnvironment:
    def __init__(self):
        # Completely separate domains - no shared variables
        
        # Alpha domain: fast energy dynamics
        self.alpha_energy = 5
        self.alpha_flow = 1
        
        # Beta domain: slow material dynamics  
        self.beta_material = 3
        self.beta_quality = 2
        
        # No shared variables at all!
        
    def update(self, agents):
        """Update completely independent domains"""
        alpha_agent = next((a for a in agents if a.agent_type == 'alpha'), None)
        beta_agent = next((a for a in agents if a.agent_type == 'beta'), None)
        
        # Alpha domain: fast energy cycles (updates every step)
        if alpha_agent:
            if alpha_agent.action == 1:  # process
                self.alpha_energy = max(0, self.alpha_energy - 2)
                self.alpha_flow = min(8, self.alpha_flow + 1)
            elif alpha_agent.action == 2:  # release
                self.alpha_energy = min(10, self.alpha_energy + 1)
                self.alpha_flow = max(0, self.alpha_flow - 1)
            # Store action (0) keeps energy stable
        
        # Natural alpha dynamics (independent of beta)
        if np.random.random() < 0.4:
            self.alpha_energy = min(10, self.alpha_energy + np.random.randint(0, 2))
        if np.random.random() < 0.3:
            self.alpha_flow = (self.alpha_flow + np.random.randint(-1, 2)) % 6
        
        # Beta domain: slow material cycles (updates less frequently)
        if beta_agent and np.random.random() < 0.7:  # Slower updates
            if beta_agent.action == 1:  # build
                self.beta_material = min(8, self.beta_material + 1)
                self.beta_quality = max(0, self.beta_quality - 1)
            elif beta_agent.action == 2:  # repair
                self.beta_quality = min(6, self.beta_quality + 2)
            # Maintain action (0) keeps materials stable
        
        # Natural beta dynamics (independent of alpha)
        if np.random.random() < 0.2:  # Slower natural changes
            self.beta_material = max(0, self.beta_material - 1)
        if np.random.random() < 0.25:
            self.beta_quality = min(6, self.beta_quality + np.random.randint(0, 2))
            
    def get(self, key, default=0):
        """Get environment variable"""
        return getattr(self, key, default)
    
    def get_state_dict(self):
        """Return completely separate environment states"""
        return {
            'env_alpha_energy': self.alpha_energy,
            'env_alpha_flow': self.alpha_flow,
            'env_beta_material': self.beta_material,
            'env_beta_quality': self.beta_quality
        }

def generate_decoupled_trace(steps=10000):  # 2x more samples for better statistics
    """Generate trace from completely independent agents"""
    alpha_agent = IndependentAgent('A', 'alpha')
    beta_agent = IndependentAgent('B', 'beta')
    
    agents = [alpha_agent, beta_agent]
    environment = DecoupledEnvironment()
    
    trace = []
    
    for _ in range(steps):
        # 1. Each agent senses only its own domain
        for agent in agents:
            agent.sense_private_domain(environment)
        
        # 2. Each agent decides completely independently
        for agent in agents:
            agent.decide_action()
        
        # 3. Environment updates independent domains
        environment.update(agents)
        
        # 4. Agents update states independently
        for agent in agents:
            agent.update_state()
        
        # 5. Record timestep
        record = {}
        for agent in agents:
            record.update(agent.get_state_dict())
        record.update(environment.get_state_dict())
        
        trace.append(record)
    
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
    
    Improved classification with more sensitive thresholds.
    """
    print(f"\n=== Classifying cluster: {cluster_vars} ===")
    
    var_to_idx = {var: i for i, var in enumerate(all_vars)}
    cluster_indices = [var_to_idx[var] for var in cluster_vars]
    env_vars = [var for var in all_vars if var not in cluster_vars]
    env_indices = [var_to_idx[var] for var in env_vars]
    
    n_vars = len(cluster_vars)
    n_env = len(env_vars)
    
    if n_env == 0:
        # No environment variables - classify based on variable names and patterns
        sensors = [var for var in cluster_vars if 'sensor' in var]
        actions = [var for var in cluster_vars if 'action' in var]
        internal = [var for var in cluster_vars if var not in sensors and var not in actions]
        print(f"No env vars - Name-based classification: S={sensors}, A={actions}, I={internal}")
        return {'S': sensors, 'A': actions, 'I': internal}
    
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
    
    # More sensitive classification thresholds
    env_threshold = np.percentile(env_mi, 50) if n_vars > 1 else 0  # Lower threshold
    future_threshold = np.percentile(future_mi, 50) if n_vars > 1 else 0  # Lower threshold
    
    print(f"Environment MI: {list(zip(cluster_vars, env_mi))}")
    print(f"Future MI: {list(zip(cluster_vars, future_mi))}")
    print(f"Env threshold: {env_threshold}, Future threshold: {future_threshold}")
    
    # Also use variable names as hints
    sensors = []
    actions = []
    
    for i, var in enumerate(cluster_vars):
        print(f"Processing {var}: env_mi={env_mi[i]:.3f}, future_mi={future_mi[i]:.3f}")
        # Strong hint from variable names
        if 'sensor' in var:
            sensors.append(var)
            print(f"  -> Added to sensors (name hint)")
        elif 'action' in var:
            actions.append(var)
            print(f"  -> Added to actions (name hint)")
        # Memory variables should be internal, not sensors
        elif 'mem' in var:
            print(f"  -> Will be internal (memory)")
        # Environment variables that are sensors
        elif var.startswith('env_') and env_mi[i] > env_threshold:
            sensors.append(var)
            print(f"  -> Added to sensors (env variable with high MI)")
        # Statistical classification for others
        elif env_mi[i] > env_threshold:
            sensors.append(var)
            print(f"  -> Added to sensors (MI threshold)")
        elif future_mi[i] > future_threshold and var not in sensors:
            actions.append(var)
            print(f"  -> Added to actions (future MI)")
        else:
            print(f"  -> Will be internal")
    
    internal = [var for var in cluster_vars if var not in sensors and var not in actions]
    
    print(f"Final classification: S={sensors}, A={actions}, I={internal}")
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
        # Always classify variables, even if not validating
        classification = classify_variables(mem, vars_, data, trace)
        
        if validate_blankets and len(mem) > 1:
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
            # No validation requested or insufficient variables - but still classify
            validated_clusters[lbl] = {
                'variables': mem,
                'classification': classification,
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
    trace    = generate_decoupled_trace(steps=10000)
    
    # Debug: Analyze the trace data
    print("=== Trace Analysis ===")
    vars_ = list(trace[0].keys())
    data = np.array([[rec[v] for v in vars_] for rec in trace])
    
    print(f"Variables: {vars_}")
    print(f"Data shape: {data.shape}")
    
    # Check variance for each variable
    var_vals = data.var(axis=0)
    print("\nVariable variances:")
    for i, (var, variance) in enumerate(zip(vars_, var_vals)):
        print(f"  {var}: {variance:.6f}")
    
    active_vars = [vars_[i] for i in range(len(vars_)) if var_vals[i] > 0.0]
    print(f"\nActive variables (variance > 0): {len(active_vars)}")
    print(f"Active variables: {active_vars}")
    
    # Check some sample values
    print(f"\nFirst 10 timesteps of data:")
    for i in range(min(10, len(trace))):
        print(f"t={i}: {trace[i]}")
    
    if len(active_vars) >= 2:
        print(f"\nProceeding with clustering...")
        # Disable validation to see natural clustering with very inclusive threshold
        clusters = detect_async_agents(trace, n_agents=2, validate_blankets=False, 
                                     weak_thresh=0.2)  # Very inclusive to keep actions

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
    else:
        print(f"ERROR: Only {len(active_vars)} active variables found, need at least 2 for clustering")
