"""Stage-2 dynamics dataset (NEW, isolated).

Subclass of ``RealAlohaDataset`` that exposes the right-arm joint-torque target
for stage-2 training. Use with ``action_mode=right_qpos`` (the base loader then
returns the raw 8-DoF ``q_des`` as the action) on a dataset whose episodes
contain ``obs/joint_torque`` (8,), as produced by the unified converter.

Adds, on top of the base dataset:
  * ``batch["joint_torque"]``            -> (B, T, 8), raw N·m

Torque is intentionally NOT normalized: the model predicts and is supervised in
physical N·m so the predictions are directly usable as actual torque downstream,
with no normalizer statistics required. (Images and actions are still normalized
by the base class as before.)

The base ``RealAlohaDataset`` replay-buffer builder already loads ``joint_torque``
into the buffer when present (small guarded addition); everything else lives
here so the base class behavior is otherwise untouched.
"""
from typing import Dict

import numpy as np
import torch

from interactive_world_sim.datasets.latent_dynamics.real_aloha_dataset import (
    RealAlohaDataset,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer


class WorldFtDynDataset(RealAlohaDataset):
    """RealAlohaDataset + per-frame right-arm joint-torque target (raw N·m)."""

    def get_normalizer(self, mode: str = "none", **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(mode, **kwargs)
        assert "joint_torque" in self.replay_buffer.keys(), (
            "WorldFtDynDataset requires `obs/joint_torque` in the episodes; "
            "regenerate the dataset with the unified converter."
        )
        # NOTE: deliberately no `normalizer["joint_torque"]` -- torque stays in raw
        # N·m end to end (target, loss, and prediction).
        return normalizer

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        # read before super() consumes/deletes its own keys (joint_torque is not
        # one of them, but be defensive)
        torque = np.asarray(sample["joint_torque"], dtype=np.float32)  # (T, 8)
        data = super()._sample_to_data(sample)
        data["joint_torque"] = torch.from_numpy(torque)
        return data
