#!/usr/bin/env python
"""
Main entry point for unsupervised agent discovery.

This refactored version demonstrates clean separation of concerns:
- Agent simulation (agents.py)
- Markov blanket validation (markov_blanket.py)  
- Detection algorithm (detection.py)
- Configuration (config.py)
"""

import numpy as np
from config import SimulationConfig, DetectionConfig
from agents import generate_decoupled_trace
from detection import AgentDetector


def analyze_trace(trace):
    """Analyze and print basic trace statistics."""
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
    
    return len(active_vars) >= 2


def main():
    """Main demonstration of agent detection."""
    # Set random seed for reproducible results
    np.random.seed(SimulationConfig.RANDOM_SEED)
    
    # Generate simulation data with specified configuration:
    # - 3 solar panels
    # - 1 wood factory
    # - 1 steel factory 
    # - 3 corn factories
    print("Generating simulation data...")
    
    factory_materials = ['wood', 'steel', 'corn', 'corn', 'corn']
    trace = generate_decoupled_trace(n_solar_panels=3, factory_materials=factory_materials)
    
    total_agents = 3 + len(factory_materials)  # 3 solar + 5 factories = 8 agents
    print(f"Created {total_agents} agents total")
    
    # Analyze trace
    if not analyze_trace(trace):
        print("ERROR: Insufficient active variables for clustering")
        return
    
    print(f"\nProceeding with clustering...")
    
    # Create detector and run detection
    detector = AgentDetector()
    clusters = detector.detect_agents(trace)
    
    # Print results
    detector.print_results(clusters)


# Backward compatibility function
def detect_async_agents(trace, n_agents=2, max_lag=3, weak_thresh=0.75, 
                       validate_blankets=True, blanket_tolerance=0.1):
    """
    Backward compatibility wrapper for the old API.
    
    Args:
        trace: Time series data
        n_agents: Number of agents to detect
        max_lag: Maximum lag for temporal dependencies
        weak_thresh: Threshold for filtering weak connections
        validate_blankets: Whether to validate Markov blankets
        blanket_tolerance: Tolerance for blanket validation
        
    Returns:
        Detection results in the same format as before
    """
    # Create custom config
    config = DetectionConfig()
    config.N_AGENTS = n_agents
    config.MAX_LAG = max_lag
    config.WEAK_THRESHOLD = weak_thresh
    config.VALIDATE_BLANKETS = validate_blankets
    config.BLANKET_TOLERANCE = blanket_tolerance
    
    # Run detection
    detector = AgentDetector(config)
    return detector.detect_agents(trace)

# ---------- 4. Demonstration -------------------------------------------------
if __name__ == '__main__':
    main()
