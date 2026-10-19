"""
SwarmRL -- Day 18: Curriculum Learning Architecture for Multi-Agent Swarms

Implements progressive 3-stage curriculum learning to accelerate cooperative
exploration and prevent stagnation:

Stage 1: DISPERSION & OBSTACLE AVOIDANCE
  - Drones deploy from base camp cluster [10, 10, 5]
  - Priority: fast dispersion without inter-drone collisions
  - Strict collision penalty (-4.0), boosted exploration reward (+0.8), shorter episodes (100 steps)
  - Promotion: >= 12% coverage and <= 2.0 average collisions

Stage 2: PRIORITY ZONE A & COOPERATIVE FLIGHT
  - Focus: Primary disaster sector Zone A (radius 45m at [50, 50, 0])
  - Medium horizon (250 steps), formation bonus active (5-10m sweet spot)
  - Promotion: >= 35% Zone A coverage and >= 30% global coverage

Stage 3: FULL DISASTER ZONE & LARGE FLEET
  - Full 100x100x50m world, dual sectors (Zone A + Zone B), up to 50 drones
  - Target coverage: 85% success threshold, full episode horizon (500 steps)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CurriculumStage:
    """Specification of a single curriculum stage."""
    stage_id: int
    name: str
    description: str
    max_steps: int
    target_coverage: float
    explore_reward: float
    collision_penalty: float
    formation_bonus: bool
    # Promotion criteria
    min_coverage_threshold: float
    max_collision_threshold: float
    min_zone_a_threshold: float = 0.0
    required_consecutive_successes: int = 3


class CurriculumManager:
    """
    Orchestrates progressive multi-agent curriculum stages.
    Evaluates policy performance, tracks consecutive promotion criteria,
    and dynamically reconfigures environment parameters.
    """

    def __init__(self, stages: Optional[List[CurriculumStage]] = None):
        self.stages = stages or self._default_stages()
        self.current_stage_idx = 0
        self.consecutive_successes = 0
        self.stage_history: List[Dict[str, Any]] = []

    @staticmethod
    def _default_stages() -> List[CurriculumStage]:
        return [
            CurriculumStage(
                stage_id=1,
                name="Dispersion & Safety",
                description="Cluster dispersal from base camp with strict inter-drone collision avoidance.",
                max_steps=100,
                target_coverage=0.15,
                explore_reward=0.8,
                collision_penalty=-4.0,
                formation_bonus=False,
                min_coverage_threshold=0.12,
                max_collision_threshold=2.0,
                required_consecutive_successes=3,
            ),
            CurriculumStage(
                stage_id=2,
                name="Zone A & Formation",
                description="Cooperative exploration of primary disaster sector Zone A with formation flight.",
                max_steps=250,
                target_coverage=0.40,
                explore_reward=0.5,
                collision_penalty=-2.0,
                formation_bonus=True,
                min_coverage_threshold=0.28,
                min_zone_a_threshold=0.35,
                max_collision_threshold=4.0,
                required_consecutive_successes=3,
            ),
            CurriculumStage(
                stage_id=3,
                name="Full Disaster Coverage",
                description="Global 3D coverage of entire disaster zone including Zone B sector.",
                max_steps=500,
                target_coverage=0.85,
                explore_reward=0.5,
                collision_penalty=-2.0,
                formation_bonus=True,
                min_coverage_threshold=0.75,
                min_zone_a_threshold=0.60,
                max_collision_threshold=6.0,
                required_consecutive_successes=5,
            ),
        ]

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    def apply_to_env(self, env: Any):
        """
        Dynamically adapts environment settings to match the current curriculum stage.
        Supports both SwarmEnv and CommSwarmEnv wrappers.
        """
        inner = getattr(env, "_inner", env)
        stage = self.current_stage

        inner.max_steps = stage.max_steps
        inner.target_coverage = stage.target_coverage
        inner.reward_explore = stage.explore_reward
        inner.reward_collision = stage.collision_penalty

        if hasattr(env, "formation_bonus"):
            env.formation_bonus = stage.formation_bonus

    def evaluate_progress(self, metrics: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Evaluates rollout telemetry metrics against the current stage's promotion criteria.

        Args:
            metrics: Dict containing 'final_coverage', 'collisions', and optional 'zone_a_coverage'

        Returns:
            Tuple of (promoted: bool, reason: Optional[str])
        """
        stage = self.current_stage
        coverage = float(metrics.get("final_coverage", 0.0))
        collisions = float(metrics.get("collisions", 0.0))
        zone_a = float(metrics.get("zone_a_coverage", 0.0))

        # Check conditions
        cov_ok = coverage >= stage.min_coverage_threshold
        col_ok = collisions <= stage.max_collision_threshold
        za_ok = (stage.min_zone_a_threshold <= 0.0) or (zone_a >= stage.min_zone_a_threshold)

        record = {
            "stage_id": stage.stage_id,
            "coverage": coverage,
            "collisions": collisions,
            "zone_a": zone_a,
            "passed": bool(cov_ok and col_ok and za_ok),
        }
        self.stage_history.append(record)

        if record["passed"]:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0

        # Check for promotion
        if self.consecutive_successes >= stage.required_consecutive_successes:
            if not self.is_final_stage:
                old_name = stage.name
                self.current_stage_idx += 1
                self.consecutive_successes = 0
                new_stage = self.current_stage
                reason = f"Promoted from '{old_name}' to Stage {new_stage.stage_id}: '{new_stage.name}'"
                return True, reason
            else:
                return False, "Mastered final curriculum stage"

        return False, None

    def get_status_dict(self) -> Dict[str, Any]:
        """Returns structured status dictionary for telemetry logging and UI."""
        stage = self.current_stage
        return {
            "stage_id": stage.stage_id,
            "stage_name": stage.name,
            "description": stage.description,
            "max_steps": stage.max_steps,
            "target_coverage": stage.target_coverage,
            "consecutive_successes": self.consecutive_successes,
            "required_successes": stage.required_consecutive_successes,
            "is_final_stage": self.is_final_stage,
        }
