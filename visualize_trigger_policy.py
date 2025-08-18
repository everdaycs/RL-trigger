# visualize_trigger_policy.py
# -------------------------------------------------------------
# Visualize PPO policy (or a masked-random baseline) on the
# US2DPriorEnv environment. Produces several matplotlib figures:
# 1) Coverage vs. steps
# 2) Cumulative time vs. steps
# 3) Info gain per step
# 4) Overlap per step
# 5) Fail indicator per step
# 6) True map (occupancy) and final observed map
#
# Usage:
#   python visualize_trigger_policy.py --model ppo_trigger.pt
#   # or without a model path to use a masked-random baseline
# -------------------------------------------------------------

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from datetime import datetime

from env_us2d_prior import EnvConfig, VecEnvs, US2DPriorEnv, world_to_grid

# --- PPO actor (same arch as training) ---

class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        hid = 256
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
        )
        self.pi = nn.Linear(hid, n_actions)

    def forward(self, obs: torch.Tensor):
        x = self.body(obs)
        return self.pi(x)

def masked_sample(logits: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    # mask: 1 valid, 0 invalid
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    dist = torch.distributions.Categorical(logits=masked_logits)
    a = dist.sample()
    return a.cpu().numpy()

def run_episode(env: US2DPriorEnv, actor: Actor = None, device='cpu', max_steps=400):
    obs = env.reset()
    obs_dim = obs.shape[0]
    n_actions = env.cfg.n_sensors

    if actor is not None:
        actor.eval()

    cov_hist = []
    time_hist = []
    info_hist = []
    overlap_hist = []
    fail_hist = []
    dt_hist = []

    # Keep a copy of prior for later visualization
    prior_obs_grid = env.obs_grid.copy()
    true_grid = env.true_grid.copy()

    # pose history (world coords) for trajectory plotting; include initial pose
    pose_hist = [env.pose.copy()]

    for t in range(max_steps):
        mask_np = env.action_mask().astype(np.float32)
        obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_np[None, :], dtype=torch.float32, device=device)

        if actor is None:
            # masked-random baseline
            valid_idx = np.where(mask_np > 0.5)[0]
            a = np.random.choice(valid_idx)
        else:
            with torch.no_grad():
                logits = actor(obs_t)
                a = masked_sample(logits, mask_t)[0]

        obs, r_raw, done, info = env.step(int(a))

        # record pose after step
        pose_hist.append(env.pose.copy())

        cov_hist.append(info["coverage"])
        time_hist.append(env.t_now)
        info_hist.append(info["info_gain"])
        overlap_hist.append(info["overlap"])
        fail_hist.append(info["fail"])
        dt_hist.append(info["dt"])

        if done:
            break

    final_obs_grid = env.obs_grid.copy()
    return {
        "coverage": np.array(cov_hist, dtype=float),
        "cum_time": np.array(time_hist, dtype=float),
        "info_gain": np.array(info_hist, dtype=float),
        "overlap": np.array(overlap_hist, dtype=float),
        "fail": np.array(fail_hist, dtype=float),
        "dt": np.array(dt_hist, dtype=float),
        "true_grid": true_grid,
        "prior_obs_grid": prior_obs_grid,
        "final_obs_grid": final_obs_grid,
        "steps": len(cov_hist),
    "pose": env.pose.copy(),
    "pose_hist": np.stack(pose_hist, axis=0),
    }

def plot_curves(results):
    steps = np.arange(1, results["steps"] + 1)

    figs = {}

    # 1) Coverage vs steps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(steps, results["coverage"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage vs Steps")
    fig.tight_layout()
    figs["coverage"] = fig

    # 2) Cumulative time vs steps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(steps, results["cum_time"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Time (s)")
    ax.set_title("Cumulative Time vs Steps")
    fig.tight_layout()
    figs["cum_time"] = fig

    # 3) Info gain per step
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(steps, results["info_gain"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Info Gain (fraction new cells)")
    ax.set_title("Info Gain per Step")
    fig.tight_layout()
    figs["info_gain"] = fig

    # 4) Overlap per step
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(steps, results["overlap"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Overlap (fraction known cells)")
    ax.set_title("Overlap per Step")
    fig.tight_layout()
    figs["overlap"] = fig

    # 5) Fail per step
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(steps, results["fail"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Fail (0 or 1)")
    ax.set_title("Fail per Step")
    fig.tight_layout()
    figs["fail"] = fig

    return figs

def plot_maps(results, cfg: EnvConfig):
    H, W = results["true_grid"].shape
    figs = {}

    # 6a) True occupancy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(results["true_grid"], origin="lower", interpolation="nearest")
    ax.set_title("True Occupancy (1=occ, 0=free)")
    ax.set_xlabel("X (cols)")
    ax.set_ylabel("Y (rows)")
    fig.tight_layout()
    figs["true_grid"] = fig

    # 6b) Prior observed map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(results["prior_obs_grid"], origin="lower", interpolation="nearest")
    ax.set_title("Prior Observed Map (-1 unknown, 0 free, 1 occ)")
    ax.set_xlabel("X (cols)")
    ax.set_ylabel("Y (rows)")
    fig.tight_layout()
    figs["prior_obs_grid"] = fig

    # 6c) Final observed map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(results["final_obs_grid"], origin="lower", interpolation="nearest")
    ax.set_title("Final Observed Map (-1 unknown, 0 free, 1 occ)")
    ax.set_xlabel("X (cols)")
    ax.set_ylabel("Y (rows)")
    # overlay trajectory if present
    if "pose_hist" in results:
        poses = results["pose_hist"]
        coords = [world_to_grid(p[0], p[1], H, W, cfg.map_res) for p in poses]
        rs = [c[0] for c in coords]
        cs = [c[1] for c in coords]
        ax.plot(cs, rs, color="cyan", linewidth=1.5, marker=None)
        ax.plot(cs[0], rs[0], marker="o", color="green", markersize=6, label="start")
        ax.plot(cs[-1], rs[-1], marker="x", color="red", markersize=6, label="end")
        ax.legend()
    fig.tight_layout()
    figs["final_obs_grid"] = fig

    return figs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="Path to trained model (ppo_trigger.pt). If empty, use masked-random baseline.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--steps", type=int, default=400, help="Max episode steps")
    parser.add_argument("--outdir", type=str, default="visualizations", help="Directory to save visualizations and CSV")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_cfg = EnvConfig()
    env = US2DPriorEnv(env_cfg)

    # Build an actor if model provided
    actor = None
    if args.model and len(args.model) > 0:
        # Infer obs_dim from a reset
        obs_dim = env.reset().shape[0]
        n_actions = env_cfg.n_sensors
        actor = Actor(obs_dim, n_actions)
        try:
            state = torch.load(args.model, map_location="cpu")
            actor.load_state_dict(state)
            print(f"Loaded model from {args.model}")
        except Exception as e:
            print(f"Failed to load model: {e}. Falling back to masked-random.")

    results = run_episode(env, actor=actor, device="cpu", max_steps=args.steps)

    # Plot curves and maps (get Figure objects)
    curve_figs = plot_curves(results)
    map_figs = plot_maps(results, env_cfg)

    # Create output directory with timestamped subdirectory (YYYY-MM-DD_HH-MM-SS)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir_full = os.path.join(args.outdir, stamp)
    os.makedirs(outdir_full, exist_ok=True)

    # Save CSV summary with timestamp
    csv = np.stack([
        np.arange(1, results["steps"] + 1),
        results["coverage"],
        results["cum_time"],
        results["info_gain"],
        results["overlap"],
        results["fail"],
        results["dt"],
    ], axis=1)
    csv_path = os.path.join(outdir_full, f"episode_summary_{stamp}.csv")
    np.savetxt(csv_path, csv, delimiter=",",
               header="step,coverage,cum_time,info_gain,overlap,fail,dt",
               comments="")
    print(f"Saved per-step summary to {csv_path}")

    # Save figures into timestamped subdirectory
    all_figs = {}
    all_figs.update(curve_figs)
    all_figs.update(map_figs)
    for name, fig in all_figs.items():
        fname = os.path.join(outdir_full, f"{name}_{stamp}.png")
        fig.savefig(fname)
        print(f"Saved figure {fname}")

    plt.show()

if __name__ == "__main__":
    main()
