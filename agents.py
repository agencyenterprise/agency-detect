"""
Agent simulation classes for generating test data.

This module contains completely independent agents operating in separate domains
to demonstrate clean agent boundary detection.
"""

import numpy as np
from config import SimulationConfig


class IndependentAgent:
    """
    An autonomous agent operating in a specific domain.
    
    Each agent has:
    - Private sensors that only read from its domain
    - Actions that only affect its domain  
    - Internal state and memory system
    - Goal-directed behavior
    """
    
    def __init__(self, name, agent_type, memory_size=None):
        if memory_size is None:
            memory_size = SimulationConfig.MEMORY_SIZE
            
        self.name = name
        self.agent_type = agent_type  # 'alpha' or 'beta'
        self.memory = np.zeros(memory_size, dtype=int)
        self.internal_state = 0
        self.action = 0
        self.private_sensor = 0
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
            self.internal_state = (self.internal_state + self.action + 1) % SimulationConfig.ALPHA_STATE_RANGE
            
            # Simpler goal progress
            self.goal_progress = (self.goal_progress + self.action + 1) % SimulationConfig.ALPHA_GOAL_RANGE
                
        else:  # beta agent
            # Beta: simpler dynamics
            self.internal_state = (self.internal_state + self.action + 2) % SimulationConfig.BETA_STATE_RANGE
            
            # Simpler goal progress
            self.goal_progress = (self.goal_progress + self.action + 1) % SimulationConfig.BETA_GOAL_RANGE
    
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
    """
    Environment with completely separate domains for each agent.
    
    This design ensures clean agent separation by having no shared variables
    or cross-domain interactions.
    """
    
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
                self.alpha_flow = min(SimulationConfig.ALPHA_FLOW_RANGE, self.alpha_flow + 1)
            elif alpha_agent.action == 2:  # release
                self.alpha_energy = min(SimulationConfig.ALPHA_ENERGY_RANGE, self.alpha_energy + 1)
                self.alpha_flow = max(0, self.alpha_flow - 1)
            # Store action (0) keeps energy stable
        
        # Natural alpha dynamics (independent of beta)
        if np.random.random() < SimulationConfig.ALPHA_ENV_UPDATE_PROB:
            self.alpha_energy = min(SimulationConfig.ALPHA_ENERGY_RANGE, 
                                   self.alpha_energy + np.random.randint(0, 2))
        if np.random.random() < SimulationConfig.ALPHA_FLOW_UPDATE_PROB:
            self.alpha_flow = (self.alpha_flow + np.random.randint(-1, 2)) % SimulationConfig.ALPHA_FLOW_RANGE
        
        # Beta domain: slow material cycles (updates less frequently)
        if beta_agent and np.random.random() < SimulationConfig.BETA_ACTION_EFFECT_PROB:
            if beta_agent.action == 1:  # build
                self.beta_material = min(SimulationConfig.BETA_MATERIAL_RANGE, self.beta_material + 1)
                self.beta_quality = max(0, self.beta_quality - 1)
            elif beta_agent.action == 2:  # repair
                self.beta_quality = min(SimulationConfig.BETA_QUALITY_RANGE, self.beta_quality + 2)
            # Maintain action (0) keeps materials stable
        
        # Natural beta dynamics (independent of alpha)
        if np.random.random() < SimulationConfig.BETA_ENV_UPDATE_PROB:
            self.beta_material = max(0, self.beta_material - 1)
        if np.random.random() < SimulationConfig.BETA_QUALITY_UPDATE_PROB:
            self.beta_quality = min(SimulationConfig.BETA_QUALITY_RANGE, 
                                  self.beta_quality + np.random.randint(0, 2))
            
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


def generate_decoupled_trace(steps=None):
    """
    Generate trace from completely independent agents.
    
    Args:
        steps: Number of simulation steps (default from config)
        
    Returns:
        List of dictionaries containing variable states at each timestep
    """
    if steps is None:
        steps = SimulationConfig.SIMULATION_STEPS
        
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