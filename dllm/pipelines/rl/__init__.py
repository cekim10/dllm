"""
RL pipeline exports.

Run from repo root:
  python -c "from dllm.pipelines.rl import DiffuGRPOConfig; print(DiffuGRPOConfig)"
"""

from .grpo import (
    SUPPORTED_DATASETS,
    DiffuGRPOConfig,
    DiffuGRPOTrainer,
    get_dataset_and_rewards,
)

__all__ = [
    "DiffuGRPOConfig",
    "DiffuGRPOTrainer",
    "get_dataset_and_rewards",
    "SUPPORTED_DATASETS",
]
