"""
SwarmRL -- Day 5: PettingZoo Environment

Wraps the 3D drone simulation in a standard PettingZoo ParallelEnv interface.
Handles:
  - reset(): Spawns drones and clears state
  - step(): Steps the ActionProcessor, computes rewards, checks terminations
  - Observation and Action space mappings
"""

import numpy as np
from pettingzoo import ParallelEnv
from env.world_config import WORLD_CONFIG, GRID_CELLS_PER_AXIS
from env.drone import Drone
from env.action_processor import ActionProcessor
from env.spaces import get_observation_space, get_action_space

class SwarmEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "swarmrl_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.cfg = WORLD_CONFIG
        self.num_drones = self.cfg["drones"]["num_drones"]
        self.max_steps = self.cfg["episode"]["max_steps"]
        self.target_coverage = self.cfg["coverage"]["target_coverage"]
        
        self.possible_agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.agents = self.possible_agents[:]
        
        self.observation_spaces = {agent: get_observation_space() for agent in self.possible_agents}
        self.action_spaces = {agent: get_action_space() for agent in self.possible_agents}
        
        self.processor = ActionProcessor(smoothing=0.5)
        
        self.drones = []
        self.coverage_grid = np.zeros(tuple(GRID_CELLS_PER_AXIS), dtype=bool)
        self.current_step = 0
        
        # Reward configuration
        self.reward_explore = 0.1     # per new cell explored
        self.reward_collision = -5.0  # penalty per collision
        self.reward_step = -0.01      # small penalty per step to encourage speed
        self.reward_success = 100.0   # huge bonus for reaching target coverage

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.processor.reset()
        self.coverage_grid.fill(False)
        
        # Spawn drones
        spawn_cfg = self.cfg["spawn"]
        center = np.array(spawn_cfg["cluster_center"])
        spread = spawn_cfg["cluster_spread"]
        
        self.drones = []
        for i in range(self.num_drones):
            pos = center + np.random.uniform(-spread, spread, size=3)
            pos[2] = max(pos[2], spawn_cfg["min_altitude"])
            self.drones.append(Drone(drone_id=i, position=pos))
            
        # Get initial observations using step_all with zero actions
        empty_actions = {d.id: np.zeros(3) for d in self.drones}
        result = self.processor.step_all(self.drones, empty_actions, self.coverage_grid, self.current_step)
        
        obs = {f"drone_{did}": o for did, o in result["observations"].items()}
        infos = {a: {} for a in self.agents}
        
        return obs, infos

    def step(self, actions):
        """
        PettingZoo step function.
        actions: dict mapping agent_name -> action_array
        """
        self.current_step += 1
        
        # Map string agent names to integer drone IDs for the processor
        processor_actions = {}
        for agent_name, action in actions.items():
            drone_id = int(agent_name.split("_")[1])
            processor_actions[drone_id] = action
            
        # Step the physics and processing pipeline
        result = self.processor.step_all(self.drones, processor_actions, self.coverage_grid, self.current_step)
        
        # Build PettingZoo returns
        obs = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        # Calculate base shared reward from coverage
        new_cells = result["new_cells"]
        shared_coverage_reward = new_cells * self.reward_explore
        
        global_coverage = result["coverage"]
        success = global_coverage >= self.target_coverage
        
        # Check truncations (max steps reached)
        is_truncated = self.current_step >= self.max_steps
        
        # Calculate collisions per drone
        dd_collisions = result["collisions"]["drone_drone"]
        do_collisions = result["collisions"]["drone_obstacle"]
        
        collision_counts = {d.id: 0 for d in self.drones}
        for a, b in dd_collisions:
            collision_counts[a] += 1
            collision_counts[b] += 1
        for a, _ in do_collisions:
            collision_counts[a] += 1
            
        for drone in self.drones:
            agent_name = f"drone_{drone.id}"
            
            # Map observation
            obs[agent_name] = result["observations"][drone.id]
            
            # Calculate reward
            drone_reward = shared_coverage_reward + self.reward_step
            
            # Collision penalty
            cols = collision_counts[drone.id]
            if cols > 0:
                drone_reward += cols * self.reward_collision
                
            # Success bonus
            if success:
                drone_reward += self.reward_success
                
            rewards[agent_name] = float(drone_reward)
            
            # Terminations / Truncations
            terminations[agent_name] = success
            truncations[agent_name] = is_truncated
            
            # Info
            infos[agent_name] = {
                "coverage": global_coverage,
                "collisions": cols,
                "cells_explored": drone.cells_explored
            }
            
        # Remove dead agents from self.agents if any (we don't kill them yet, but good practice)
        if success or is_truncated:
            self.agents = []
            
        return obs, rewards, terminations, truncations, infos
