"""Browser-based teleoperation using Gradio.

Drop-in replacement for teleoperate_keyboard.py that runs headless (no X server).
Open the printed URL in your laptop's browser (via VS Code port forwarding).

Install once:
    pip install gradio

Run with the same Hydra args as teleoperate_keyboard.py, e.g.:
    python scripts/inference/teleoperate_gradio.py \
        +output_dir='data/wm_demo' \
        +use_joystick=false +use_dataset=false +act_horizon=1 +scene=real \
        "+ckpt_paths=['outputs/pusht_cam1/checkpoints/best.ckpt']" \
        dataset=real_aloha_dataset \
        dataset.dataset_dir=data/mini/pusht/val \
        "dataset.obs_keys=['camera_1_color']"

Controls (click buttons OR press keys with the browser tab focused):
    W/A/S/D            left-arm xy delta
    I/J/K/L            right-arm xy delta (or z for bimanual_rope)
    Q/E                left-arm z (bimanual_rope only)
    U/O                right-arm z (bimanual_rope only)
    R                  start recording
    Shift+S            save episode and advance
    Shift+Q            discard episode and advance
"""

import math
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
try:
    torch.backends.cuda.enable_cudnn_sdp(True)
except AttributeError:
    pass

SDPA_BACKENDS = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
from omegaconf import DictConfig, OmegaConf
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5, save_dict_to_hdf5
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.algorithms.models.attention import Attention as _Attention
from interactive_world_sim.utils.action_utils import joint_pos_to_action_primitive
from interactive_world_sim.utils.aloha_conts import (
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN,
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN,
)
from interactive_world_sim.utils.draw_utils import concat_img_h, plot_single_3d_pos_traj
from interactive_world_sim.utils.normalizer import LinearNormalizer


def load_model(ckpt_path: str) -> pl.LightningModule:
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.n_frames = 10
    cfg.algorithm.n_frames = 10
    if "diffusion" in cfg.algorithm and "sampling_timesteps" in cfg.algorithm.diffusion:
        cfg.algorithm.diffusion.sampling_timesteps = 10
    if (
        "diffusion" in cfg.algorithm.dynamics
        and "sampling_timesteps" in cfg.algorithm.dynamics.diffusion
    ):
        cfg.algorithm.dynamics.diffusion.sampling_timesteps = 10
    cfg.algorithm.load_ae = None
    algo = LatentWorldModel.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.algorithm,
        map_location="cuda:0",
        dtype=dtype,
        strict=False,
        weights_only=False,
    )
    algo.dynamics = algo.dynamics.to(dtype)
    algo.eval()
    algo.dynamics.eval()
    return algo


def key_to_delta(key: str, scene: str) -> np.ndarray:
    if scene in [
        "real",
        "real_cam_0",
        "sim",
        "bimanual_sweep_cam_0",
        "bimanual_sweep_cam_1",
        "single_grasp_cam_0",
        "single_grasp_cam_1",
    ]:
        delta = np.zeros(4)
        mapping = {"w": (1, 1), "s": (1, -1), "a": (0, -1), "d": (0, 1),
                   "i": (3, 1), "k": (3, -1), "j": (2, -1), "l": (2, 1)}
    elif scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
        delta = np.zeros(6)
        mapping = {"w": (1, 1), "s": (1, -1), "a": (0, -1), "d": (0, 1),
                   "i": (4, 1), "k": (4, -1), "j": (3, -1), "l": (3, 1),
                   "q": (2, 1), "e": (2, -1), "u": (5, 1), "o": (5, -1)}
    else:
        raise NotImplementedError(f"scene '{scene}' not recognized")
    if key in mapping:
        idx, val = mapping[key]
        delta[idx] = val
    return delta / (np.linalg.norm(delta) + 1e-8)


def kybd_action_to_rob_action(delta_action: np.ndarray, scene: str) -> np.ndarray:
    if scene == "real":
        return np.array([-delta_action[3], delta_action[2], -delta_action[1], delta_action[0]])
    if scene == "real_cam_0":
        return np.array([delta_action[1], -delta_action[0], delta_action[3], -delta_action[2]])
    if scene == "bimanual_rope_cam_0":
        return np.array([delta_action[1], -delta_action[0], delta_action[2], 0.0,
                         delta_action[4], -delta_action[3], delta_action[5], 0.0])
    if scene == "bimanual_rope_cam_1":
        return np.array([-delta_action[4], delta_action[3], delta_action[5], 0.0,
                         -delta_action[1], delta_action[0], delta_action[2], 0.0])
    if scene == "sim":
        return delta_action.copy()
    if scene == "bimanual_sweep_cam_1":
        return np.array([-delta_action[3], delta_action[2], -delta_action[1], delta_action[0]])
    if scene == "bimanual_sweep_cam_0":
        return np.array([delta_action[1], -delta_action[0], delta_action[3], -delta_action[2]])
    if scene == "single_grasp_cam_1":
        return np.array([-delta_action[3], delta_action[2], -delta_action[1], delta_action[0]])
    if scene == "single_grasp_cam_0":
        return np.array([delta_action[1], -delta_action[0], delta_action[3], -delta_action[2]])
    raise NotImplementedError(f"scene '{scene}' not recognized")


def dict_list_to_np(episode: dict) -> dict:
    for key in list(episode.keys()):
        if isinstance(episode[key], list):
            episode[key] = np.stack(episode[key], axis=0)
        elif isinstance(episode[key], dict):
            episode[key] = dict_list_to_np(episode[key])
    return episode


def scale_delta(delta_action: np.ndarray, scene: str, action_range_scale: np.ndarray) -> np.ndarray:
    if scene in ["real", "real_cam_0"]:
        return delta_action / (50.0 * action_range_scale)
    if scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
        return delta_action / (30.0 * action_range_scale)
    if scene in ["bimanual_sweep_cam_0", "bimanual_sweep_cam_1"]:
        return delta_action / 20.0
    if scene == "sim":
        return delta_action / (100.0 * action_range_scale)
    if scene in ["single_grasp_cam_0", "single_grasp_cam_1"]:
        out = delta_action.copy()
        out[:3] = out[:3] * action_range_scale[:3].max() / (50.0 * action_range_scale[:3])
        out[3] = out[3] / 10.0
        return out
    raise NotImplementedError(f"scene '{scene}' not recognized")


def render_latent(
    models: list[LatentWorldModel],
    normalizer: LinearNormalizer,
    curr_latent_tensor_list: list[torch.Tensor],
    resolution: int,
    scene: str,
    curr_action: torch.Tensor,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (display_rgb, raw_rgb_for_save)."""
    xs_pred_vis_ls = []
    xs_pred_np_ret = None
    for v_i, model in enumerate(models):
        with sdpa_kernel(SDPA_BACKENDS):
            xs_pred = render_img_cm(
                model,
                curr_latent_tensor_list[v_i][:, -1],
                resolution,
                normalizer=normalizer,
                num_views=len(model.obs_keys),
            )
        xs_pred_np = xs_pred.permute(0, 2, 3, 1).detach().cpu().float().numpy()[0]
        xs_pred_np = np.clip((xs_pred_np * 255).astype(np.uint8), 0, 255)
        if xs_pred_np_ret is None:
            xs_pred_np_ret = xs_pred_np
        xs_pred_vis = cv2.resize(xs_pred_np, (640, 640), interpolation=cv2.INTER_AREA)
        xs_pred_vis = cv2.cvtColor(xs_pred_vis, cv2.COLOR_RGB2BGR)
        xs_pred_vis_ls.append(xs_pred_vis)
    concat_img = concat_img_h(xs_pred_vis_ls)
    if scene == "sim":
        curr_action_unnorm = normalizer["action"].unnormalize(curr_action).detach().cpu().numpy()
        action_3d = curr_action_unnorm.reshape(2, 1, 2)
        action_3d = np.concatenate([action_3d, 0.02 * np.ones((2, 1, 1))], axis=-1)
        concat_img = plot_single_3d_pos_traj(concat_img, intrinsics, extrinsics, action_3d, radius=5)
    return concat_img, xs_pred_np_ret


def annotate(img_bgr: np.ndarray, episode_id: int, recording: bool, step_i: int) -> np.ndarray:
    img = img_bgr.copy()
    text = f"Episode: {episode_id} step={step_i}" + (" [Recording]" if recording else "")
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def init_episode_state(cfg: DictConfig, models: list[LatentWorldModel], normalizer: LinearNormalizer,
                      episode_id: int) -> Optional[dict]:
    """Load HDF5 for `episode_id`, build initial latent + action. Return None if no more episodes."""
    load_epi_path = f"{cfg.dataset.dataset_dir}/episode_{episode_id}.hdf5"
    if not Path(load_epi_path).exists():
        return None
    load_epi_data, _ = load_dict_from_hdf5(load_epi_path)
    device = models[0].device
    t = 0
    if cfg.scene == "sim":
        curr_action = load_epi_data["action"][t]
    else:
        joint_pos = load_epi_data["obs"]["joint_pos"][t]
        num_rob = joint_pos.shape[0] // 7
        for r_i in range(num_rob):
            joint_pos[r_i * 7 + 6] = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(
                PUPPET_GRIPPER_JOINT_NORMALIZE_FN(joint_pos[r_i * 7 + 6])
            )
        kin_helper = KinHelper("trossen_vx300s")
        if cfg.scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
            ctrl_mode = "bimanual_rope"
        elif cfg.scene in ["bimanual_sweep_cam_0", "bimanual_sweep_cam_1"]:
            ctrl_mode = "bimanual_sweep_v2"
        elif cfg.scene in ["single_grasp_cam_0", "single_grasp_cam_1"]:
            ctrl_mode = "single_grasp"
        elif cfg.scene in ["real", "real_cam_0", "sim"]:
            ctrl_mode = "bimanual_push"
        else:
            raise NotImplementedError(f"scene '{cfg.scene}' not recognized")
        robot_bases = (
            load_epi_data["robot_bases"][t]
            if "robot_bases" in load_epi_data
            else load_epi_data["obs"]["world_t_robot_base"][t]
        )
        curr_action = joint_pos_to_action_primitive(
            joint_pos=joint_pos, ctrl_mode=ctrl_mode,
            base_pose_in_world=robot_bases, kin_helper=kin_helper,
        )
    curr_action = torch.from_numpy(curr_action).to(device).float()
    curr_action = normalizer["action"].normalize(curr_action)

    curr_latent_tensor_list = []
    for o_i, obs_key in enumerate(cfg.dataset.obs_keys):
        raw_img = load_epi_data["obs"]["images"][obs_key][t]
        raw_img = center_crop(raw_img, (cfg.dataset.resolution, cfg.dataset.resolution))
        raw_img = cv2.resize(raw_img, (cfg.dataset.resolution, cfg.dataset.resolution),
                             interpolation=cv2.INTER_AREA)
        img = raw_img / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img_tensor = normalizer[obs_key].normalize(img_tensor).to(device)
        with torch.no_grad(), sdpa_kernel(SDPA_BACKENDS):
            curr_latent_tensor_list.append(models[o_i].encoder_forward(img_tensor)[:, None])

    dtype = models[0].dtype
    curr_latent_tensor_list = [t.to(dtype) for t in curr_latent_tensor_list]
    curr_action = curr_action.to(dtype)

    return {
        "curr_latent_tensor_list": curr_latent_tensor_list,
        "curr_action": curr_action,
        "action_hist": [],
        "recording": False,
        "episode_data": {"action": [], "obs": {"images": {cfg.dataset.obs_keys[0]: []}}},
        "step_i": 0,
        "last_xs_pred_np": None,
    }


def save_episode(state: dict, cfg: DictConfig, episode_id: int) -> None:
    obs_key = cfg.dataset.obs_keys[0]
    episode_data = state["episode_data"]
    if not episode_data["action"]:
        return
    episode_data["action"] = np.stack(episode_data["action"], axis=0)
    episode_data["obs"]["images"][obs_key] = np.stack(
        episode_data["obs"]["images"][obs_key], axis=0
    )
    episode_data = dict_list_to_np(episode_data)
    config_dict = {"obs": {"images": {obs_key: {
        "chunks": (1, cfg.dataset.resolution, cfg.dataset.resolution, 3),
        "dtype": "uint8",
    }}}}
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dict_to_hdf5(episode_data, config_dict, f"{out_dir}/episode_{episode_id}.hdf5")


# Key handler JS — listens for keypresses and clicks the matching button.
KEY_JS = """
() => {
    const map = {
        'w': 'btn-w', 'a': 'btn-a', 's': 'btn-s', 'd': 'btn-d',
        'i': 'btn-i', 'j': 'btn-j', 'k': 'btn-k', 'l': 'btn-l',
        'q': 'btn-q', 'e': 'btn-e', 'u': 'btn-u', 'o': 'btn-o',
        'r': 'btn-record'
    };
    const shiftMap = {'S': 'btn-save', 'Q': 'btn-discard'};
    document.addEventListener('keydown', (e) => {
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
        const id = e.shiftKey ? shiftMap[e.key] : map[e.key.toLowerCase()];
        if (id) {
            e.preventDefault();
            const btn = document.getElementById(id);
            if (btn) btn.click();
        }
    });
    return [];
}
"""


def build_app(cfg: DictConfig, models: list[LatentWorldModel], normalizer: LinearNormalizer):
    device = models[0].device
    dtype = models[0].dtype
    hist_context = 10
    act_horizon = cfg.act_horizon

    fovy = 45.0
    H = W = 512
    f = 0.5 * H / math.tan(fovy * math.pi / 360)
    intrinsics = np.array([W / 2, H / 2, f, f])
    extrinsics = np.array([[1, 0, 0, 0], [0, -1, 0, -0.019], [0, 0, -1, 0.685], [0, 0, 0, 1]])

    action_max = models[0].normalizer["action"].state_dict()["params_dict.input_stats.max"]
    action_min = models[0].normalizer["action"].state_dict()["params_dict.input_stats.min"]
    action_range = action_max - action_min
    if cfg.scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
        action_range = torch.cat([action_range[:3], action_range[4:7]]).to(device)
    action_range_scale = (action_range / action_range.max()).detach().cpu().numpy()

    def render_current(state):
        display_bgr, xs_pred_np = render_latent(
            models, normalizer, state["curr_latent_tensor_list"], cfg.dataset.resolution,
            cfg.scene, state["curr_action"], intrinsics, extrinsics,
        )
        state["last_xs_pred_np"] = xs_pred_np
        return annotate(display_bgr, state["episode_id"], state["recording"], state["step_i"])

    def step_with_key(key: str, state):
        if state is None or state.get("done"):
            return state, gr.update(), state.get("status", "All episodes finished.")
        delta_action = key_to_delta(key, cfg.scene)
        delta_action = scale_delta(delta_action, cfg.scene, action_range_scale)
        if np.linalg.norm(delta_action) < 1e-8:
            return state, gr.update(), state["status"]
        delta_action_rob = kybd_action_to_rob_action(delta_action, cfg.scene)
        delta_action_tensor = torch.from_numpy(delta_action_rob).to(device).to(dtype)
        state["curr_action"] = torch.clamp(state["curr_action"] + delta_action_tensor, -1.0, 1.0)
        state.setdefault("action_ls", []).append(state["curr_action"])
        if len(state["action_ls"]) < act_horizon:
            return state, gr.update(), state["status"]

        action_chunk = torch.stack(state["action_ls"]).reshape(1, -1)
        if state["recording"] and state["last_xs_pred_np"] is not None:
            action_chunk_unnorm = normalizer["action"].unnormalize(action_chunk).detach().cpu().numpy()
            state["episode_data"]["action"].append(action_chunk_unnorm)
            state["episode_data"]["obs"]["images"][cfg.dataset.obs_keys[0]].append(state["last_xs_pred_np"])
        state["action_hist"].append(action_chunk)
        action = torch.cat(state["action_hist"], dim=0)[-(hist_context + 1):]
        action = rearrange(action, "t a -> 1 t a").to(device=device, dtype=dtype)
        new_latents = []
        for i in range(len(models)):
            with torch.no_grad(), sdpa_kernel(SDPA_BACKENDS):
                latent_pred = models[i].dynamics_forward(
                    state["curr_latent_tensor_list"][i], action
                )
            new_lat = torch.cat([state["curr_latent_tensor_list"][i], latent_pred], axis=1)[:, -hist_context:]
            new_latents.append(new_lat)
        state["curr_latent_tensor_list"] = new_latents
        state["action_ls"] = []
        state["step_i"] += 1
        img = render_current(state)
        return state, img, state["status"]

    def start_recording(state):
        if state is None or state.get("done"):
            return state, gr.update(), state.get("status", "")
        state["recording"] = True
        state["status"] = f"Episode {state['episode_id']}: recording (step {state['step_i']})"
        img = render_current(state)
        return state, img, state["status"]

    def save_and_advance(state):
        if state is None or state.get("done"):
            return state, gr.update(), state.get("status", "")
        if not state["recording"] and not state["episode_data"]["action"]:
            state["status"] = "Nothing recorded — press R first."
            return state, gr.update(), state["status"]
        save_episode(state, cfg, state["episode_id"])
        next_id = state["episode_id"] + 1
        new_state = init_episode_state(cfg, models, normalizer, next_id)
        if new_state is None:
            state["done"] = True
            state["status"] = f"Saved episode {state['episode_id']}. No more dataset episodes."
            return state, gr.update(), state["status"]
        new_state["episode_id"] = next_id
        new_state["status"] = f"Saved episode {state['episode_id']}. Now on episode {next_id}."
        img = render_current(new_state)
        return new_state, img, new_state["status"]

    def discard_and_advance(state):
        if state is None or state.get("done"):
            return state, gr.update(), state.get("status", "")
        next_id = state["episode_id"]
        if state["episode_data"]["action"]:
            next_id = state["episode_id"] + 1
        new_state = init_episode_state(cfg, models, normalizer, next_id)
        if new_state is None:
            state["done"] = True
            state["status"] = "No more dataset episodes."
            return state, gr.update(), state["status"]
        new_state["episode_id"] = next_id
        new_state["status"] = f"Discarded. Now on episode {next_id}."
        img = render_current(new_state)
        return new_state, img, new_state["status"]

    initial = init_episode_state(cfg, models, normalizer, 0)
    if initial is None:
        raise RuntimeError(f"No episode_0.hdf5 found in {cfg.dataset.dataset_dir}")
    initial["episode_id"] = 0
    initial["status"] = "Episode 0 ready. Press R to start recording."

    with gr.Blocks(title="World-model teleop") as app:
        gr.Markdown(
            "# World-model teleop\n"
            "Click a control button or press its key with this tab focused. "
            "Use **R** to start recording, **Shift+S** to save, **Shift+Q** to discard."
        )
        status = gr.Textbox(value=initial["status"], label="Status", interactive=False)
        image = gr.Image(label="Prediction", height=640, type="numpy")
        state = gr.State(initial)

        with gr.Row():
            for key in ["w", "a", "s", "d"]:
                gr.Button(key.upper(), elem_id=f"btn-{key}").click(
                    fn=lambda s, k=key: step_with_key(k, s),
                    inputs=state, outputs=[state, image, status],
                )
        with gr.Row():
            for key in ["i", "j", "k", "l"]:
                gr.Button(key.upper(), elem_id=f"btn-{key}").click(
                    fn=lambda s, k=key: step_with_key(k, s),
                    inputs=state, outputs=[state, image, status],
                )
        if cfg.scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
            with gr.Row():
                for key in ["q", "e", "u", "o"]:
                    gr.Button(key.upper(), elem_id=f"btn-{key}").click(
                        fn=lambda s, k=key: step_with_key(k, s),
                        inputs=state, outputs=[state, image, status],
                    )
        with gr.Row():
            gr.Button("Start Recording (R)", elem_id="btn-record", variant="primary").click(
                fn=start_recording, inputs=state, outputs=[state, image, status],
            )
            gr.Button("Save Episode (Shift+S)", elem_id="btn-save", variant="secondary").click(
                fn=save_and_advance, inputs=state, outputs=[state, image, status],
            )
            gr.Button("Discard Episode (Shift+Q)", elem_id="btn-discard").click(
                fn=discard_and_advance, inputs=state, outputs=[state, image, status],
            )

        app.load(fn=lambda s: render_current(s), inputs=state, outputs=image, js=KEY_JS)

    return app


def patch_attention_backends(models: list[LatentWorldModel]) -> None:
    """Override the A100-only FLASH-only backend list with a list that has fp32 fallbacks.

    The model hardcodes `cuda_backends = [FLASH_ATTENTION]` on A100, but flash requires
    fp16/bf16. With fp32 inputs, every backend in that single-item list fails → 'No
    available kernel'. We swap in a list that allows math (universal) and mem-efficient
    (fp32-safe) as fallbacks, with flash kept first for the fp16/bf16 case.
    """
    fallback = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    n_patched = 0
    for model in models:
        for module in model.modules():
            if isinstance(module, _Attention):
                module.cuda_backends = fallback
                n_patched += 1
    print(f"Patched {n_patched} Attention modules with fp32-safe backend fallbacks.")


@hydra.main(version_base=None, config_path="../../configurations", config_name="config")
def main(cfg: DictConfig) -> None:
    models = [load_model(p) for p in cfg.ckpt_paths]
    patch_attention_backends(models)
    normalizer = models[0].normalizer
    for model in models:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {total_params / 1e6:.2f}M params, {total_params * 4 / (1024**2):.2f} MB (fp32)")
    app = build_app(cfg, models, normalizer)
    launch_kwargs = dict(server_name="0.0.0.0", server_port=7860, show_error=True)
    if cfg.get("share", False):
        launch_kwargs["share"] = True
    if cfg.get("root_path"):
        launch_kwargs["root_path"] = cfg.root_path
    app.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    main()
