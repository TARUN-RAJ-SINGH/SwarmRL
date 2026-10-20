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
from training.ablation_study import (
    ABLATION_VARIANTS,
    run_single_variant_training,
    run_full_ablation_study,
    plot_ablation_comparisons,
    generate_ablation_markdown_report,
)
from training.train_long_horizon import train_long_horizon

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
    "ABLATION_VARIANTS",
    "run_single_variant_training",
    "run_full_ablation_study",
    "plot_ablation_comparisons",
    "generate_ablation_markdown_report",
    "train_long_horizon",
]
