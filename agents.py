"""
Agent simulation classes for generating test data.

This module contains independent Solar Panel and Factory agents operating
in separate domains to demonstrate clean agent boundary detection.

Solar Panel agents manage energy collection and discharge cycles.
Factory agents manage production and equipment maintenance.
"""

import numpy as np
from config import SimulationConfig


class IndependentAgent:
    """
    An autonomous agent operating in a specific resource domain.
    
    Each agent has:
    - Private sensors that monitor its domain (energy levels or production capacity)
    - Actions that affect its domain (sleep/charge/discharge or idle/produce/maintain)
    - Internal state representing system condition
    - Goal-directed behavior with realistic dynamics
    """
    
    def __init__(self, name, agent_type, memory_size=None):
        if memory_size is None:
            memory_size = SimulationConfig.MEMORY_SIZE
            
        self.name = name
        self.agent_type = agent_type  # 'solar_panel' or 'factory'
        self.memory = np.zeros(memory_size, dtype=int)
        self.internal_state = 0
        self.action = 0
        self.private_sensor = 0
        self.goal_progress = 0
        
        # Initialize with different starting conditions and ranges
        if agent_type == 'solar_panel':
            self.internal_state = 1
            self.goal_progress = 0
        else:  # factory
            self.internal_state = 3
            self.goal_progress = 2
    
    def sense_private_domain(self, environment):
        """Each agent senses its domain with realistic sensor functions"""
        
        if self.agent_type == 'solar_panel':
            # Solar Panel: Energy level sensor with noise and saturation
            energy_level = environment.get('solar_panel_energy', 5)
            energy_flow = environment.get('solar_panel_flow', 1)
            
            # Realistic sensor combining energy state and flow rate
            base_reading = energy_level + energy_flow * 0.5
            sensor_noise = np.random.normal(0, 0.2)
            raw_sensor = base_reading + sensor_noise
            
            # Store as float but will be discretized in get_state_dict()
            self.private_sensor = max(0, min(8, raw_sensor))
            
        else:  # factory agent
            # Factory: Production capacity sensor
            material_stock = environment.get('factory_material', 3)
            material_quality = environment.get('factory_quality', 2)
            
            # Realistic sensor for production capacity
            capacity_indicator = material_stock * 0.6 + material_quality * 1.2
            sensor_noise = np.random.normal(0, 0.15)
            raw_sensor = capacity_indicator + sensor_noise
            
            # Store as float but will be discretized in get_state_dict()
            self.private_sensor = max(0, min(8, raw_sensor))
    
    def decide_action(self):
        """Realistic decision making with properly normalized influences"""
        # Normalize both influences to [0,1] scale
        memory_influence = np.mean(self.memory) / 8.0 if len(self.memory) > 0 else 0
        sensor_influence = self.private_sensor / 8.0  # Already normalized
        
        # Add realistic noise and combine influences
        noise = np.random.normal(0, 0.1)
        demand_signal = sensor_influence + memory_influence * 0.3 + noise
        
        if self.agent_type == 'solar_panel':
            # Solar Panel: Energy harvesting agent with different thresholds
            # Actions: 0=sleep, 1=charge, 2=discharge
            if demand_signal < 0.3:
                self.action = 0  # sleep
            elif demand_signal < 0.7:
                self.action = 1  # charge
            else:
                self.action = 2  # discharge
                
        else:  # factory agent
            # Factory: Resource production agent with different thresholds
            # Actions: 0=idle, 1=produce, 2=maintain
            if demand_signal < 0.4:
                self.action = 0  # idle/wait
            elif demand_signal < 0.8:
                self.action = 1  # produce resources
            else:
                self.action = 2  # maintain equipment
        
        return self.action
    
    def update_state(self):
        """State updates with momentum and saturation using floats"""
        # Update memory with recent sensor reading (convert to int for memory)
        self.memory[1:] = self.memory[:-1]
        self.memory[0] = int(round(self.private_sensor))
        
        # More realistic internal state updates with momentum
        if self.agent_type == 'solar_panel':
            # Solar Panel: Energy system internal state with momentum
            state_change = (self.action - 1) * 0.3 + np.random.normal(0, 0.1)
            self.internal_state = max(0, min(SimulationConfig.SOLAR_PANEL_STATE_RANGE, 
                                           self.internal_state + state_change))
            
            # Goal progress with realistic advancement/decay
            if self.action == 1:  # charging advances goal
                progress_gain = min(0.5, self.private_sensor / 10.0)
                self.goal_progress = min(SimulationConfig.SOLAR_PANEL_GOAL_RANGE, 
                                       self.goal_progress + progress_gain)
            else:
                # Other actions cause slight goal decay
                self.goal_progress = max(0, self.goal_progress - 0.1)
                
        else:  # factory agent
            # Factory: Production system with wear-based state changes
            wear_factor = max(0.1, self.action / 3.0)  # more action = more wear
            state_change = self.action * 0.2 - wear_factor + np.random.normal(0, 0.05)
            self.internal_state = max(0, min(SimulationConfig.FACTORY_STATE_RANGE, 
                                           self.internal_state + state_change))
            
            # Goal progress based on production efficiency
            if self.action == 1:  # producing advances goal
                efficiency = max(0.1, self.internal_state / SimulationConfig.FACTORY_STATE_RANGE)
                progress_gain = efficiency * 0.4
                self.goal_progress = min(SimulationConfig.FACTORY_GOAL_RANGE, 
                                       self.goal_progress + progress_gain)
            elif self.action == 2:  # maintenance slightly advances goal
                self.goal_progress = min(SimulationConfig.FACTORY_GOAL_RANGE, 
                                       self.goal_progress + 0.1)
            # Idle (action=0) causes no goal change
    
    def get_state_dict(self):
        """Return agent state with all values discretized to integers"""
        return {
            f'{self.name}_sensor': int(round(self.private_sensor)),
            f'{self.name}_action': int(self.action),
            f'{self.name}_internal': int(round(self.internal_state)),
            f'{self.name}_goal': int(round(self.goal_progress)),
            **{f'{self.name}_mem{i}': int(self.memory[i]) for i in range(len(self.memory))}
        }


class DecoupledEnvironment:
    """
    Environment with separate energy and resource management domains.
    
    Solar Panel domain: Energy system with collection/discharge dynamics
    - solar_panel_energy: Available energy level (0 to SOLAR_PANEL_ENERGY_RANGE)
    - solar_panel_flow: Energy flow capacity (0 to SOLAR_PANEL_FLOW_RANGE)
    
    Factory domain: Resource production system with maintenance
    - factory_material: Raw material stock (0 to FACTORY_MATERIAL_RANGE)  
    - factory_quality: Equipment condition (0 to FACTORY_QUALITY_RANGE)
    
    This design ensures clean agent separation with no shared variables.
    """
    
    def __init__(self):
        # Separate domains with no shared variables
        
        # Solar Panel domain: Energy collection system
        self.solar_panel_energy = 5    # Current stored energy
        self.solar_panel_flow = 1      # Energy flow/collection capacity
        
        # Factory domain: Resource production system  
        self.factory_material = 3   # Raw material inventory
        self.factory_quality = 2    # Equipment condition/efficiency
        
        # Domains are completely independent!
        
    def update(self, agents):
        """Update with realistic resource dynamics using float computations"""
        solar_panel_agent = next((a for a in agents if a.agent_type == 'solar_panel'), None)
        factory_agent = next((a for a in agents if a.agent_type == 'factory'), None)
        
        # Solar Panel domain: Energy system with realistic dynamics
        if solar_panel_agent:
            if solar_panel_agent.action == 0:  # sleep
                # Slow energy decay when sleeping
                self.solar_panel_energy = max(0, self.solar_panel_energy - 0.1)
                self.solar_panel_flow = self.solar_panel_flow * 0.95  # gradual flow reduction
            elif solar_panel_agent.action == 1:  # charge
                # Charge energy but reduce flow efficiency
                energy_gain = min(2.0, self.solar_panel_flow * 1.5)
                self.solar_panel_energy = min(SimulationConfig.SOLAR_PANEL_ENERGY_RANGE, 
                                      self.solar_panel_energy + energy_gain)
                self.solar_panel_flow = max(0, self.solar_panel_flow - 0.3)
            elif solar_panel_agent.action == 2:  # discharge
                # Discharge energy, restore flow capacity
                self.solar_panel_energy = max(0, self.solar_panel_energy - 1.5)
                flow_restoration = min(0.5, (SimulationConfig.SOLAR_PANEL_FLOW_RANGE - self.solar_panel_flow) * 0.4)
                self.solar_panel_flow = min(SimulationConfig.SOLAR_PANEL_FLOW_RANGE, 
                                    self.solar_panel_flow + flow_restoration)
        
        # Natural solar panel dynamics with exponential trends
        if np.random.random() < SimulationConfig.SOLAR_PANEL_ENV_UPDATE_PROB:
            # Environmental energy input with saturation
            energy_input = np.random.exponential(0.5)
            self.solar_panel_energy = min(SimulationConfig.SOLAR_PANEL_ENERGY_RANGE, 
                                   self.solar_panel_energy + energy_input)
        
        # Factory domain: Resource production with wear and maintenance
        if factory_agent and np.random.random() < SimulationConfig.FACTORY_ACTION_EFFECT_PROB:
            if factory_agent.action == 0:  # idle
                # Equipment slowly degrades when idle
                self.factory_quality = max(0, self.factory_quality - 0.05)
            elif factory_agent.action == 1:  # produce
                # Production increases material but wears equipment
                production_rate = max(0.1, self.factory_quality / SimulationConfig.FACTORY_QUALITY_RANGE)
                self.factory_material = min(SimulationConfig.FACTORY_MATERIAL_RANGE, 
                                       self.factory_material + production_rate)
                self.factory_quality = max(0, self.factory_quality - 0.2)
            elif factory_agent.action == 2:  # maintain
                # Maintenance improves quality, may consume some material
                maintenance_cost = min(0.1, self.factory_material * 0.05)
                self.factory_material = max(0, self.factory_material - maintenance_cost)
                quality_gain = min(1.0, maintenance_cost * 10 + 0.3)
                self.factory_quality = min(SimulationConfig.FACTORY_QUALITY_RANGE, 
                                      self.factory_quality + quality_gain)
        
        # Natural factory dynamics with realistic decay
        if np.random.random() < SimulationConfig.FACTORY_ENV_UPDATE_PROB:
            # Material naturally depletes (consumption/spoilage)
            depletion = np.random.exponential(0.3)
            self.factory_material = max(0, self.factory_material - depletion)
        
        if np.random.random() < SimulationConfig.FACTORY_QUALITY_UPDATE_PROB:
            # Quality has some natural restoration
            restoration = np.random.gamma(2, 0.1)  # Small positive restoration
            self.factory_quality = min(SimulationConfig.FACTORY_QUALITY_RANGE, 
                                  self.factory_quality + restoration)
        
    def get(self, key, default=0):
        """Get environment variable"""
        return getattr(self, key, default)
    
    def get_state_dict(self):
        """Return environment states with all values discretized to integers"""
        return {
            'env_solar_panel_energy': int(round(self.solar_panel_energy)),
            'env_solar_panel_flow': int(round(self.solar_panel_flow)),
            'env_factory_material': int(round(self.factory_material)),
            'env_factory_quality': int(round(self.factory_quality))
        }


def generate_decoupled_trace(steps=None):
    """
    Generate trace from independent Solar Panel and Factory agents.
    
    Creates realistic simulation of:
    - Solar Panel agent: Energy collection system with sleep/charge/discharge actions
      • sleep (0): Low power mode with slow energy decay
      • charge (1): Active energy collection with efficiency cost
      • discharge (2): Energy release with capacity restoration
      
    - Factory agent: Resource production system with idle/produce/maintain actions
      • idle (0): Equipment gradually degrades when unused
      • produce (1): Manufacture goods but wear equipment
      • maintain (2): Service equipment to improve quality
    
    Both agents operate in separate domains with realistic dynamics including:
    - Sensor noise and saturation
    - Threshold-based decision making  
    - Exponential decay and momentum effects
    - Resource constraints and trade-offs
    
    Args:
        steps: Number of simulation steps (default from config)
        
    Returns:
        List of dictionaries containing variable states at each timestep
    """
    if steps is None:
        steps = SimulationConfig.SIMULATION_STEPS
        
    solar_panel_agent = IndependentAgent('A', 'solar_panel')
    factory_agent = IndependentAgent('B', 'factory')
    
    agents = [solar_panel_agent, factory_agent]
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