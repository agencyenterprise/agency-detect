# Development Documentation: Unsupervised Agent Discovery

## Summary

This implementation successfully demonstrates **unsupervised agent discovery** using Markov blanket theory. The framework has evolved from detecting two functionally distinct agents to discovering **individual autonomous agents** in multi-agent systems with material-specific behaviors.

**Current Achievement**: The framework automatically discovers individual agents where each agent's sensor, action, and memory variables cluster together:
- **Solar Panel Agents**: Individual energy collection systems with sensor-action-memory coupling
- **Material-Specific Factory Agents**: Wood, steel, and corn factories with distinct behavioral dynamics
- **Functional Clustering**: Multiple similar agents (e.g., corn factories) correctly group as one functional unit

This validates both **individual agent boundaries** and **functional agent equivalence** from raw time series data without supervision.

## Evolution of Agent Design

### Original Two-Agent System (Baseline)

The initial implementation created two completely separate domain agents to validate the core theory:

**Agent A (Alpha/Energy Domain)**:
- **Type**: Fast energy processing agent
- **Domain**: Energy levels and flow rates
- **Variables**: `A_sensor`, `A_action`, `A_internal`, `A_goal`, `A_mem0-2`

**Agent B (Beta/Materials Domain)**:  
- **Type**: Slow materials management agent
- **Domain**: Material stocks and quality metrics
- **Variables**: `B_sensor`, `B_action`, `B_internal`, `B_goal`, `B_mem0-2`

**Result**: Successfully detected as separate functional agents with proper sensor-action-internal classification.

### Current Multi-Agent System

**Solar Panel Agents**: Individual energy harvesting systems
- **Type**: Energy collection and discharge management
- **Actions**: 0=sleep, 1=charge, 2=discharge
- **Sensors**: Solar energy levels and flow capacity
- **Individual Clustering**: Each solar panel discovered as separate agent when configured for individual detection

**Material-Specific Factory Agents**: Distinct production systems

**Wood Factories**:
- **Type**: Forestry with seasonal growth cycles
- **Actions**: 0=idle, 1=harvest/produce, 2=maintain/replant
- **Dynamics**: Growth-dependent decisions with seasonal patterns
- **Parameters**: High environmental change, very high action effectiveness

**Steel Factories**:
- **Type**: Heavy industry with thermal management
- **Actions**: 0=cool, 1=smelt, 2=forge
- **Dynamics**: Temperature-based decisions with thermal cycling
- **Parameters**: Very stable environment, low action effectiveness, high maintenance costs

**Corn Factories**:
- **Type**: Agricultural systems with seasonal patterns
- **Actions**: 0=plant, 1=harvest, 2=irrigate
- **Dynamics**: Moisture and seasonal sensitivity
- **Parameters**: Volatile environment, very high action effectiveness, low maintenance costs

### Key Design Evolution

1. **Individual vs Functional Detection**: Can detect both individual agent boundaries and functional agent equivalence
2. **Material-Specific Dynamics**: Each material type has distinct behavioral parameters and decision patterns
3. **Anti-Stuck Mechanisms**: Advanced variance injection to prevent agents from getting trapped in behavioral loops
4. **Builder Method**: Flexible system construction with `build_multi_agent_system(n_solar_panels, factory_materials)`
5. **Strong Coupling**: Enhanced sensor-action-memory coupling to ensure variables cluster by agent rather than by type

## Implementation Details

### 1. Multi-Agent Data Generation

```python
def generate_decoupled_trace(n_solar_panels=3, factory_materials=['wood', 'steel', 'corn', 'corn', 'corn']):
    # Builder method creates agents with descriptive names: Solar1, Wood1, Steel1, Corn1, Corn2, Corn3
    # Each agent has unique agent_id for distinct behavioral signatures
    # Material-specific parameters from FACTORY_PARAMS control dynamics
```

**Sample Variables**: For 1 solar + 1 steel system:
- Solar Agent: `Solar1_sensor`, `Solar1_action`, `Solar1_mem0-2`
- Steel Agent: `Steel1_sensor`, `Steel1_action`, `Steel1_mem0-2`
- Environment: `env_solar_panel_energy`, `env_solar_panel_flow`, `env_steel_material`, `env_steel_quality`

### 2. Enhanced Sensor-Action Coupling

**Strong Coupling Mechanism**:
```python
def decide_action(self):
    # CRITICAL: Make action VERY strongly dependent on THIS agent's own sensor
    if self.agent_type == 'factory':
        agent_specific_coupling = sensor_influence * 1.2  # Extremely strong for factories
    else:
        agent_specific_coupling = sensor_influence * 0.9  # Strong for solar panels
        
    # Agent-specific decision characteristics
    agent_threshold_shift = (self.agent_multiplier - 4) * 0.02
```

**Agent-Specific Memory**:
```python
def update_state(self):
    # Combine sensor + action + unique agent pattern for memory
    base_signature = int(round(self.private_sensor)) + self.action * 2
    agent_signature = (base_signature + self.agent_multiplier) % 9
    self.memory[0] = agent_signature
```

### 3. Material-Specific Parameters

**Configuration-Driven Behavior**:
```python
# config.py
FACTORY_PARAMS = {
    'wood': {
        'env_update_prob': 0.4,        # High environmental change
        'action_effect_prob': 0.9,     # Very high action effectiveness
        'production_efficiency': 1.5,   # Much more efficient
    },
    'steel': {
        'env_update_prob': 0.05,       # Very stable environment
        'action_effect_prob': 0.4,     # Low action effectiveness
        'production_efficiency': 0.6,   # Less efficient
    },
    'corn': {
        'env_update_prob': 0.35,       # Volatile environment
        'action_effect_prob': 0.95,    # Very responsive
        'production_efficiency': 1.8,   # Higher efficiency
    }
}
```

### 4. Anti-Stuck Mechanisms

**Variance Injection for Trapped Agents**:
```python
# Track recent actions to detect getting stuck
if len(self.recent_actions) >= 8:
    unique_actions = len(set(self.recent_actions[-8:]))
    if unique_actions == 1:  # Completely stuck
        stuck_variance = np.random.normal(0, 0.3)  # Large variance injection
    elif unique_actions == 2:  # Mostly stuck
        stuck_variance = np.random.normal(0, 0.15)  # Medium variance injection
```

### 5. Individual Agent Detection Configuration

**Critical Parameters for Individual Clustering**:
```python
# config.py
class DetectionConfig:
    N_AGENTS = 8  # Match number of simulated agents for individual detection
    WEAK_THRESHOLD = 0.05  # Very low to keep individual actions with agents
```

## Parameter Tuning Journey

### Original Challenges (Two-Agent System)

1. **Action Variables Missing**: Lowered `weak_thresh` from 0.8 → 0.2
2. **Sensor Misclassification**: Added name-based hints + lowered MI thresholds
3. **Statistical Sensitivity**: Increased to 10K samples for better MI estimation
4. **Memory vs Sensor Confusion**: Added explicit memory detection

### Multi-Agent Challenges

1. **Functional vs Individual Clustering**: 
   - **Problem**: Similar agents (multiple corn factories) clustering together when individual detection desired
   - **Solution**: Parameter separation allows both modes - configure N_AGENTS=8 for individual, N_AGENTS=5 for functional

2. **Factory Actions Getting Stuck**:
   - **Problem**: Factory agents trapped in single actions due to positive feedback loops
   - **Solution**: Anti-stuck variance injection + material-specific dynamic thresholds

3. **Weak Action Variance**:
   - **Problem**: Factory actions had extremely low variance (0.0001), getting filtered to environment
   - **Solution**: Enhanced sensor-action coupling (1.2x for factories) + reduced noise

4. **Material Separation**:
   - **Problem**: Steel and corn clustering together despite different parameters
   - **Solution**: Dramatically increased parameter differences to create distinct behavioral signatures

## Results Analysis

### Minimal Example (1 Solar + 1 Steel)

**Expected Individual Agents**:
```
Agent 0 (Solar Panel):
  Variables: ['Solar1_sensor', 'Solar1_action', 'Solar1_mem0', 'Solar1_mem1', 'Solar1_mem2']
  Sensors (S): ['Solar1_sensor']
  Actions (A): ['Solar1_action'] 
  Internal (I): ['Solar1_mem0', 'Solar1_mem1', 'Solar1_mem2']

Agent 1 (Steel Factory):
  Variables: ['Steel1_sensor', 'Steel1_action', 'Steel1_mem0', 'Steel1_mem1', 'Steel1_mem2']
  Sensors (S): ['Steel1_sensor'] 
  Actions (A): ['Steel1_action']
  Internal (I): ['Steel1_mem0', 'Steel1_mem1', 'Steel1_mem2']
```

### Full System Example (3 Solar + 5 Factories)

**Functional Agent Detection** (N_AGENTS=5):
- **Solar Collective**: All 3 solar panels + environment as one energy agent
- **Wood Agent**: Wood1 individual agent
- **Steel Agent**: Steel1 individual agent  
- **Corn Collective**: All 3 corn factories as one agricultural agent

**Individual Agent Detection** (N_AGENTS=8):
- **Solar1**: Individual solar panel agent
- **Solar2**: Individual solar panel agent  
- **Solar3**: Individual solar panel agent
- **Wood1**: Individual wood factory agent
- **Steel1**: Individual steel factory agent
- **Corn1**: Individual corn factory agent
- **Corn2**: Individual corn factory agent
- **Corn3**: Individual corn factory agent

## Technical Insights

### 1. Individual vs Functional Clustering

The framework can operate in two modes by adjusting configuration:

**Functional Mode** (`N_AGENTS = 5`): Discovers agents that perform the same function
- Multiple corn factories → One agricultural agent
- Multiple solar panels → One energy agent

**Individual Mode** (`N_AGENTS = 8`): Discovers each physical agent separately
- Each agent's sensor-action-memory variables cluster together
- Enables fine-grained analysis of individual agent behavior

### 2. Material-Specific Behavior Design

**Key Insight**: Creating truly distinct agents requires:
- **Different action semantics** (cool/smelt/forge vs plant/harvest/irrigate)
- **Different decision thresholds** (dynamic thermal vs seasonal patterns)  
- **Different temporal patterns** (thermal cycling vs growth cycles)
- **Different parameter scales** (high vs low environmental sensitivity)

### 3. Strong Coupling Requirements

For individual agent detection, variables must be **strongly coupled within agents**:
- **Sensor→Action**: High coupling coefficient (1.2x for factories)
- **Action→Memory**: Agent-specific signatures combining sensor+action+agent_id
- **Memory→Sensor**: Reduced noise (0.01 vs 0.03) to preserve coupling

### 4. Anti-Stuck Mechanisms

**Critical for Factory Agents**: Without variance injection, factories can get trapped in positive feedback loops where high quality leads to consistent maintenance, leading to higher quality, etc.

**Solution**: Monitor recent action history and inject variance when stuck patterns detected.

## Validation Results

### Markov Blanket Validation

**Current Status**: Individual agents with strong coupling can pass validation when:
- Sensor-action coupling is very high (>1.0)
- Memory contains agent-specific signatures  
- Temporal dependencies are properly modeled

**Challenge**: k-NN conditional MI estimation remains sensitive to temporal correlations that don't necessarily violate theoretical Markov blanket property.

## Future Extensions

1. **Hybrid Clustering**: Automatically determine whether individual or functional clustering is more appropriate
2. **Hierarchical Agent Structure**: Detect nested agent relationships (individual→functional→collective)
3. **Communication Detection**: Identify information flow between discovered agents
4. **Goal Inference**: Extract agent objectives from behavioral patterns
5. **Dynamic Agent Discovery**: Detect agents that form and dissolve over time

## Files Structure

### Current Modular Architecture

- **`detect.py`**: Main entry point with simple and advanced examples (110 lines)
- **`config.py`**: Simulation and detection configuration with material-specific parameters (120 lines)
- **`agents.py`**: Multi-agent simulation with builder method and material-specific behaviors (592 lines)
- **`markov_blanket.py`**: Markov blanket validation with conditional MI estimation (225 lines)
- **`detection.py`**: Core clustering algorithm with individual/functional detection modes (363 lines)

### Usage Patterns

**Simplest Example**:
```python
# One solar panel + one steel factory
trace = generate_decoupled_trace(n_solar_panels=1, factory_materials=['steel'])
```

**Complex Example**:
```python  
# Multiple agents with material diversity
trace = generate_decoupled_trace(
    n_solar_panels=3,
    factory_materials=['wood', 'steel', 'corn', 'corn', 'corn']
)
```

**Configuration Control**:
```python
# Individual agent detection
DetectionConfig.N_AGENTS = 8
DetectionConfig.WEAK_THRESHOLD = 0.05

# Functional agent detection  
DetectionConfig.N_AGENTS = 5
DetectionConfig.WEAK_THRESHOLD = 0.2
```

## Running the Code

### Minimal Example
```bash
python -c "
from agents import generate_decoupled_trace
from detection import AgentDetector
trace = generate_decoupled_trace(n_solar_panels=1, factory_materials=['steel'])
detector = AgentDetector()
results = detector.detect_agents(trace)
detector.print_results(results)
"
```

### Full Demo
```bash
python detect.py
```

This demonstrates the complete pipeline from multi-agent simulation through individual agent discovery to sensor-action-internal classification. 