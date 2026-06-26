"""Stage-2 dynamics + torque experiment (NEW, isolated).

Subclass of ``LatentDynExperiment`` that registers the stage-2 dataset and
algorithm so they can be selected by config name, without touching the base
experiment's registries.
"""
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model_torque import (
    LatentWorldModelTorque,
)
from interactive_world_sim.datasets.latent_dynamics.world_ft_dyn_dataset import (
    WorldFtDynDataset,
)

from .exp_latent_dyn import LatentDynExperiment


class LatentDynTorqueExperiment(LatentDynExperiment):
    """Latent dynamics + joint-torque experiment."""

    compatible_algorithms = dict(
        LatentDynExperiment.compatible_algorithms,
        latent_world_model_torque=LatentWorldModelTorque,
    )

    compatible_datasets = dict(
        LatentDynExperiment.compatible_datasets,
        world_ft_dyn_dataset=WorldFtDynDataset,
    )
