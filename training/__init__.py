# SwarmRL Training Package

from training.centralized_critic import (
    GlobalStateBuilder,
    EntityAttentionCritic,
    MLPCentralizedCritic,
    ValueNorm,
    MultiAgentGAE,
    CentralizedCriticLoss,
    AGENT_FEATURE_DIM,
    ENV_FEATURE_DIM,
    MAX_FLAT_STATE_DIM,
)
from training.rollout_buffer import MultiAgentRolloutBuffer
from training.mappo_policy import MAPPOActor, MAPPOPolicy
from training.mappo_trainer import MAPPOConfig, MAPPOTrainer
from training.curriculum import CurriculumStage, CurriculumManager
from training.tune_mappo import (
    DEFAULT_PARAM_GRID,
    sample_hyperparameters,
    run_trial,
    compute_pareto_frontier,
)
from training.scale_benchmark import benchmark_fleet_scale

__all__ = [
    "GlobalStateBuilder",
    "EntityAttentionCritic",
    "MLPCentralizedCritic",
    "ValueNorm",
    "MultiAgentGAE",
    "CentralizedCriticLoss",
    "AGENT_FEATURE_DIM",
    "ENV_FEATURE_DIM",
    "MAX_FLAT_STATE_DIM",
    "MultiAgentRolloutBuffer",
    "MAPPOActor",
    "MAPPOPolicy",
    "MAPPOConfig",
    "MAPPOTrainer",
    "CurriculumStage",
    "CurriculumManager",
    "DEFAULT_PARAM_GRID",
    "sample_hyperparameters",
    "run_trial",
    "compute_pareto_frontier",
    "benchmark_fleet_scale",
]
