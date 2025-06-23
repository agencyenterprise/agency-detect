# Unsupervised Agent Discovery

This project implements a principled framework for discovering autonomous agents within raw dynamical systems without supervision, based on the theoretical foundations of **Markov blankets** and **active inference**.

## Overview

Complex systems—from neural networks to multi-agent simulations—often contain hidden autonomous actors with their own sensors, actions, and internal states. Traditional approaches require manual labeling or domain knowledge. This framework automatically discovers these agents by detecting **Markov blanket structures** in time series data.

## Key Concepts

**Markov Blankets**: A mathematical boundary that separates an agent's internal dynamics from the external world. For a true autonomous agent, the Markov blanket property ensures:

```
I(Internal_{t+1}; External_{t+1} | Sensors_t, Actions_t) ≈ 0
```

This means once you know the agent's sensors and actions, the internal and external dynamics become conditionally independent—the hallmark of true autonomy.

**Variable Classification**:
- **Sensors (S)**: Variables that bring information from the environment to the agent
- **Actions (A)**: Variables through which the agent influences the environment  
- **Internal (I)**: Variables that maintain the agent's private state and memory

## What This Implementation Does

1. **Discovers Agent Boundaries**: Uses mutual information clustering to find groups of variables that belong together
2. **Validates Autonomy**: Tests whether discovered clusters satisfy the Markov blanket property using conditional mutual information
3. **Classifies Variables**: Automatically identifies which variables serve as sensors, actions, or internal states
4. **Handles Temporal Dependencies**: Accounts for memory and lagged interactions between agents

## Results

The framework successfully detects two independent agents:

**Agent A (Energy Domain)**:
- Sensors: `A_sensor`, `env_alpha_flow` 
- Actions: `A_action`
- Internal: `A_mem0`, `A_mem1`, `A_mem2` (memory system)

**Agent B (Materials Domain)**:
- Sensors: `B_sensor`, `env_beta_material`
- Actions: `B_action` 
- Internal: `B_internal`, `B_mem0`, `B_mem1`, `B_mem2`, `env_beta_quality`

Each agent operates independently in its own domain while maintaining proper sensor-action-internal structure.

## Applications

- **Neural Data Analysis**: Discover functional modules in brain recordings
- **Multi-Agent Systems**: Identify emergent agents in complex simulations
- **Biological Systems**: Find autonomous subsystems in cellular or ecological data
- **Robotics**: Detect modular controllers in complex robotic systems

## Theoretical Foundation

Based on the paper "Foundations of Unsupervised Agent Discovery in Raw Dynamical Systems" which unifies:
- **Active Inference**: Agents as free-energy minimizers
- **Information Theory**: Mutual information and conditional independence
- **Markov Blanket Theory**: Formal boundaries of autonomous systems

## Getting Started

```bash
python detect.py
```

This will:
1. Generate a simulation with two independent agents
2. Analyze the time series data for agent boundaries
3. Classify variables as sensors, actions, or internal states
4. Display the discovered agent structures

See `dev.md` for detailed implementation notes and `docs/unsupervised-agent-discovery.tex` for the full theoretical framework. 