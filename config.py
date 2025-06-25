"""
Configuration parameters for unsupervised agent discovery.

Split into separate classes for simulation and detection to allow
independent configuration of each component.
"""


class SimulationConfig:
    """Configuration for agent simulation parameters."""
    
    # Simulation parameters
    SIMULATION_STEPS = 15000
    RANDOM_SEED = 42
    
    # Agent design parameters
    MEMORY_SIZE = 3
    
    # Solar Panel agent parameters
    SOLAR_PANEL_ENERGY_RANGE = 10
    SOLAR_PANEL_FLOW_RANGE = 6
    SOLAR_PANEL_STATE_RANGE = 8
    SOLAR_PANEL_GOAL_RANGE = 10
    
    # Factory agent parameters  
    FACTORY_MATERIAL_RANGE = 8
    FACTORY_QUALITY_RANGE = 6
    FACTORY_STATE_RANGE = 6
    FACTORY_GOAL_RANGE = 8
    
    # Environment update probabilities
    SOLAR_PANEL_ENV_UPDATE_PROB = 0.4
    SOLAR_PANEL_FLOW_UPDATE_PROB = 0.3
    FACTORY_ENV_UPDATE_PROB = 0.2
    FACTORY_QUALITY_UPDATE_PROB = 0.25
    FACTORY_ACTION_EFFECT_PROB = 0.7  # Probability factory actions affect environment


class DetectionConfig:
    """Configuration for agent detection algorithm."""
    
    # Clustering parameters
    N_AGENTS = 3
    MAX_LAG = 3
    WEAK_THRESHOLD = 0.2  # Lowered to keep action variables
    
    # Classification parameters
    ENV_MI_PERCENTILE = 50  # Percentile for environment MI threshold
    FUTURE_MI_PERCENTILE = 50  # Percentile for future MI threshold
    
    # Markov blanket validation parameters
    VALIDATE_BLANKETS = True  # Can be enabled for strict validation
    BLANKET_TOLERANCE = 1.0  # Adjusted for discrete CMI estimation (was 5.0 for broken k-NN)
    CMI_SMOOTHING_ALPHA = 0.1  # Laplace smoothing parameter for discrete CMI


# Default configurations for backward compatibility
DEFAULT_SIMULATION_CONFIG = SimulationConfig()
DEFAULT_DETECTION_CONFIG = DetectionConfig() 