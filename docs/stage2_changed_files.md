# Stage-2 implementation — files to commit

Tracking list so we commit/push only the stage-2 work after testing.

## New files
- `interactive_world_sim/datasets/latent_dynamics/world_ft_dyn_dataset.py` — `WorldFtDynDataset` (q_des action + joint_torque target + torque normalizer). **[done, tested]**
- `interactive_world_sim/algorithms/latent_dynamics/models/torque_predictor.py` — `TorquePredictor` (causal GRU latent+action -> torque). **[done, tested]**
- `interactive_world_sim/algorithms/latent_dynamics/latent_world_model_torque.py` — `LatentWorldModelTorque` (stage-2 dense torque loss). **[done, smoke-tested]**
- `interactive_world_sim/experiments/exp_latent_dyn_torque.py` — experiment subclass registering the new dataset + algorithm. **[done]**
- `configurations/algorithm/latent_world_model_torque.yaml` — **[done]**
- `configurations/dataset/world_ft_dyn_dataset.yaml` — **[done]**
- `configurations/experiment/exp_latent_dyn_torque.yaml` — **[done]**
- `scripts/train/train_world_ft_stage2.sh` — **[done, smoke-tested]**

### A100 note (baked into the stage-2 script)
The latent-dynamics attention forces **flash-only** SDPA on A100 (sm_80), which has
no fp32 kernel; the experiment defaults to `precision: 32-true` (fine for stage 1,
whose decoder doesn't use this attention). The stage-2 script therefore sets
`experiment.training.precision=16-mixed` so the dynamics attention runs in fp16.

## Modified existing files (small, guarded, backward-compatible)
- `scripts/data_collection/convert_world_ft.py` — rewritten as the unified raw->hdf5 converter (images + q_des + joint_torque + wrench/vel/tau). **[done, tested]**
- `interactive_world_sim/datasets/latent_dynamics/real_aloha_dataset.py` — replay-buffer builder: `right_qpos` raw-action branch + load `obs/joint_torque` when present (guarded; existing modes unchanged). **[done, tested]**
- `interactive_world_sim/experiments/__init__.py` — register `exp_latent_dyn_torque` in `exp_registry` (2 lines). **[todo §6]**

## Docs (optional to commit)
- `docs/stage2_dynamics_plan_updated.md` — the agreed plan (dense torque supervision folded in).
- `docs/stage2_dynamics_plan.md` — earlier draft.
- `docs/stage2_changed_files.md` — this tracking file.

## Generated data (NOT committed — gitignored under data/)
- `data/world_ft_v2/{train,val}/episode_*.hdf5` + cache.
