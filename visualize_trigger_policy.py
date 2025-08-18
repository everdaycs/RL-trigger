
# Examples:
#   python visualize_trigger_policy.py --compare --model ppo_trigger.pt --steps 400 --seed 123 --fps 4 --trail-len 0 --no-show
#   python visualize_trigger_policy.py --steps 400 --no-show   # baseline only
#   python visualize_trigger_policy.py --model ppo_trigger.pt --no-show   # trained only
# -------------------------------------------------------------

import argparse
import os
from datetime import datetime
import numpy as np
import random
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from env_us2d_prior import EnvConfig, US2DPriorEnv, world_to_grid
from models.networks import Actor  # 若你的项目是 networks.Actor，请改成: from networks import Actor


# ----------------- utils -----------------

def masked_sample(logits: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Sample an action from categorical logits with invalid actions masked out."""
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    dist = torch.distributions.Categorical(logits=masked_logits)
    a = dist.sample()
    return a.cpu().numpy()


def snapshot_rng():
    """Snapshot RNG states for Python, NumPy, and PyTorch."""
    return {
        "np": np.random.get_state(),
        "py": random.getstate(),
        "torch": torch.get_rng_state(),
    }


def restore_rng(st):
    """Restore RNG states for Python, NumPy, and PyTorch."""
    np.random.set_state(st["np"])
    random.setstate(st["py"])
    torch.set_rng_state(st["torch"])


# ----------------- core run -----------------

def run_episode(env: US2DPriorEnv, actor: Actor = None, device='cpu', max_steps=400):
    """
    Always start an episode with env.reset(); deterministic start is guaranteed
    by restoring RNG right before calling this function.
    """
    obs = env.reset()

    if actor is not None:
        actor.eval()

    cov_hist, time_hist, info_hist, overlap_hist, fail_hist, dt_hist = [], [], [], [], [], []

    prior_obs_grid = env.obs_grid.copy()
    true_grid = env.true_grid.copy()

    pose_hist = [env.pose.copy()]
    obs_grid_hist = [env.obs_grid.copy()]

    for _ in range(max_steps):
        mask_np = env.action_mask().astype(np.float32)
        obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_np[None, :], dtype=torch.float32, device=device)

        if actor is None:
            # Untrained baseline: masked-random
            valid_idx = np.where(mask_np > 0.5)[0]
            a = np.random.choice(valid_idx)
        else:
            with torch.no_grad():
                logits = actor(obs_t)
                a = masked_sample(logits, mask_t)[0]

        obs, r_raw, done, info = env.step(int(a))

        pose_hist.append(env.pose.copy())
        obs_grid_hist.append(env.obs_grid.copy())

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
        "pose_hist": np.stack(pose_hist, axis=0),
        "obs_grid_hist": np.stack(obs_grid_hist, axis=0),
    }


# ----------------- plotting -----------------

def plot_curves(results):
    steps = np.arange(1, results["steps"] + 1)
    figs = {}

    def mkfig(y, title, ylabel):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(steps, y)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    figs["coverage"]  = mkfig(results["coverage"], "Coverage vs Steps", "Coverage")
    figs["cum_time"]  = mkfig(results["cum_time"], "Cumulative Time vs Steps", "Cumulative Time (s)")
    figs["info_gain"] = mkfig(results["info_gain"], "Info Gain per Step", "Info Gain (fraction new cells)")
    figs["overlap"]   = mkfig(results["overlap"], "Overlap per Step", "Overlap (fraction known cells)")
    figs["fail"]      = mkfig(results["fail"], "Fail per Step", "Fail (0 or 1)")
    return figs


def plot_maps(results, cfg: EnvConfig):
    H, W = results["true_grid"].shape
    figs = {}

    def show_grid(grid, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(grid, origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("X (cols)")
        ax.set_ylabel("Y (rows)")
        fig.tight_layout()
        return fig

    figs["true_grid"]       = show_grid(results["true_grid"], "True Occupancy (1=occ, 0=free)")
    figs["prior_obs_grid"]  = show_grid(results["prior_obs_grid"], "Prior Observed Map (-1 unknown, 0 free, 1 occ)")

    # final grid + trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(results["final_obs_grid"], origin="lower", interpolation="nearest")
    ax.set_title("Final Observed Map (-1 unknown, 0 free, 1 occ)")
    ax.set_xlabel("X (cols)")
    ax.set_ylabel("Y (rows)")
    poses = results["pose_hist"]
    coords = [world_to_grid(float(p[0]), float(p[1]), H, W, cfg.map_res) for p in poses]
    rs = [c[0] for c in coords]
    cs = [c[1] for c in coords]
    ax.plot(cs, rs, linewidth=1.5)
    ax.plot(cs[0], rs[0], marker="o", markersize=6, label="start")
    ax.plot(cs[-1], rs[-1], marker="x", markersize=6, label="end")
    ax.legend()
    fig.tight_layout()
    figs["final_obs_grid"] = fig
    return figs


def save_episode(results, outdir_full, cfg: EnvConfig, stamp, fps=4, trail_len=0):
    # 1) CSV
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
               header="step,coverage,cum_time,info_gain,overlap,fail,dt", comments="")
    print(f"Saved per-step summary to {csv_path}")

    # 2) Figures
    curve_figs = plot_curves(results)
    map_figs = plot_maps(results, cfg)
    figs = {**curve_figs, **map_figs}
    for name, fig in figs.items():
        fname = os.path.join(outdir_full, f"{name}_{stamp}.png")
        fig.savefig(fname); plt.close(fig)
        print(f"Saved figure {fname}")

    # 3) Animation with trajectory
    try:
        obs_hist = results.get('obs_grid_hist', None)
        poses = results.get('pose_hist', None)
        if obs_hist is not None and len(obs_hist) > 0 and poses is not None and len(poses) > 0:
            H, W = obs_hist.shape[1], obs_hist.shape[2]

            coords = [world_to_grid(float(p[0]), float(p[1]), H, W, cfg.map_res) for p in poses]
            rs = np.array([c[0] for c in coords], dtype=float)
            cs = np.array([c[1] for c in coords], dtype=float)

            anim_fig, anim_ax = plt.subplots()
            im = anim_ax.imshow(obs_hist[0], origin='lower', interpolation='nearest', vmin=-1, vmax=1)
            anim_ax.set_title('Observed Map Over Time')
            anim_ax.set_xlabel("X (cols)")
            anim_ax.set_ylabel("Y (rows)")
            anim_ax.set_xlim([-0.5, W - 0.5])
            anim_ax.set_ylim([-0.5, H - 0.5])

            (trail_line,) = anim_ax.plot([], [], linewidth=1.8, animated=True)
            (head_point,) = anim_ax.plot([], [], marker='o', markersize=5, animated=True)
            start_mark, = anim_ax.plot(cs[0], rs[0], marker="o", markersize=6, label="start")
            end_mark,   = anim_ax.plot(cs[-1], rs[-1], marker="x", markersize=6, label="end")
            anim_ax.legend(loc="upper right")

            artists = [im, trail_line, head_point]

            def update_frame(i):
                im.set_data(obs_hist[i])
                lo = max(0, i + 1 - trail_len) if trail_len and trail_len > 0 else 0
                trail_line.set_data(cs[lo:i+1], rs[lo:i+1])
                head_point.set_data([cs[i]], [rs[i]])  # use lists to avoid Matplotlib deprecation warnings
                return tuple(artists)

            mp4_path = os.path.join(outdir_full, f"map_update_{stamp}.mp4")
            try:
                if hasattr(animation.writers, "is_available") and animation.writers.is_available("ffmpeg"):
                    writer = animation.FFMpegWriter(fps=fps)
                    ani = animation.FuncAnimation(anim_fig, update_frame, frames=len(obs_hist), blit=True)
                    ani.save(mp4_path, writer=writer)
                    print(f"Saved animation {mp4_path}")
                else:
                    raise RuntimeError("ffmpeg writer not available")
            except Exception:
                frames_dir = os.path.join(outdir_full, 'frames')
                os.makedirs(frames_dir, exist_ok=True)
                for i in range(len(obs_hist)):
                    p = os.path.join(frames_dir, f"frame_{i:04d}.png")
                    plt.imsave(p, obs_hist[i], origin='lower', vmin=-1, vmax=1)
                print(f"FFmpeg not available — saved frames to {frames_dir}")
            finally:
                plt.close(anim_fig)
    except Exception as e:
        print('Failed to create animation:', e)


def plot_and_save_compare(untrained, trained, compare_dir, cfg: EnvConfig, stamp):
    os.makedirs(compare_dir, exist_ok=True)

    def overlay_plot(yname, title, ylabel):
        u = untrained[yname]; t = trained[yname]
        su = np.arange(1, len(u) + 1); stp = np.arange(1, len(t) + 1)
        L = min(len(u), len(t))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(su[:L], u[:L], label="untrained")
        ax.plot(stp[:L], t[:L], label="trained")
        ax.set_xlabel("Step"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); fig.tight_layout()
        outp = os.path.join(compare_dir, f"{yname}_compare_{stamp}.png")
        fig.savefig(outp); plt.close(fig)
        print(f"Saved compare figure {outp}")

    overlay_plot("coverage",  "Coverage vs Steps (Compare)", "Coverage")
    overlay_plot("cum_time",  "Cumulative Time vs Steps (Compare)", "Cumulative Time (s)")
    overlay_plot("info_gain", "Info Gain per Step (Compare)", "Info Gain (fraction new cells)")
    overlay_plot("overlap",   "Overlap per Step (Compare)", "Overlap (fraction known cells)")
    overlay_plot("fail",      "Fail per Step (Compare)", "Fail (0 or 1)")

    # final observed maps side-by-side
    H, W = untrained["true_grid"].shape
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(untrained["final_obs_grid"], origin="lower", interpolation="nearest")
    axes[0].set_title("Untrained Final Observed"); axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
    poses = untrained["pose_hist"]
    coords = [world_to_grid(float(p[0]), float(p[1]), H, W, cfg.map_res) for p in poses]
    rs = [c[0] for c in coords]; cs = [c[1] for c in coords]
    axes[0].plot(cs, rs, linewidth=1.2)

    axes[1].imshow(trained["final_obs_grid"], origin="lower", interpolation="nearest")
    axes[1].set_title("Trained Final Observed"); axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    poses = trained["pose_hist"]
    coords = [world_to_grid(float(p[0]), float(p[1]), H, W, cfg.map_res) for p in poses]
    rs = [c[0] for c in coords]; cs = [c[1] for c in coords]
    axes[1].plot(cs, rs, linewidth=1.2)

    fig.tight_layout()
    outp = os.path.join(compare_dir, f"final_obs_compare_{stamp}.png")
    fig.savefig(outp); plt.close(fig)
    print(f"Saved compare figure {outp}")


# ----------------- main -----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true",
                        help="Run untrained baseline and trained model on the SAME env and SAME start, and export untrained/, trained/, compare/ under visualizations/")
    parser.add_argument("--model", type=str, default="", help="Path to trained model (ppo_trigger.pt). If empty, only untrained baseline runs (unless --compare is set, which requires a model).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--steps", type=int, default=400, help="Max episode steps")
    # 输出根目录固定使用 visualizations（用户要求）
    parser.add_argument("--outdir", type=str, default="visualizations", help="Root output directory (default: visualizations)")
    parser.add_argument("--no-show", action="store_true", help="Do not open GUI windows; just save results and exit.")
    parser.add_argument("--fps", type=int, default=4, help="FPS for MP4 animation")
    parser.add_argument("--trail-len", type=int, default=0, help="Show only last K steps of the trajectory in animation (0=full)")
    args = parser.parse_args()

    # 强制输出到 visualizations（即便用户传了别的，也仍放在 visualizations 下）
    out_root = "visualizations"

    # Deterministic RNG seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    env_cfg = EnvConfig()
    env = US2DPriorEnv(env_cfg)

    # --- Build actor if model provided (use a temp env to get obs_dim so main env RNG isn't consumed) ---
    actor = None
    if args.model:
        temp_env = US2DPriorEnv(env_cfg)
        obs_dim = temp_env.reset().shape[0]
        n_actions = env_cfg.n_sensors
        del temp_env

        actor = Actor(obs_dim, n_actions)
        try:
            # Safe load: weights_only + strict=False to ignore extra heads (e.g., value head)
            state = torch.load(args.model, map_location="cpu", weights_only=True)
            missing, unexpected = actor.load_state_dict(state, strict=False)
            if unexpected:
                print(f"Warning: unexpected keys ignored: {unexpected}")
            print(f"Loaded model from {args.model}")
        except Exception as e:
            print(f"Failed to load model: {e}. Falling back to baseline only.")
            actor = None

    # Timestamp: YYYYMMDD_HHMMSS for sortable order
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(out_root, stamp)

    if args.compare:
        if actor is None:
            raise ValueError("--compare 需要提供可加载的 --model")
        # 保证 SAME env & SAME start：RNG snapshot + reset
        rng0 = snapshot_rng()

        # 1) Untrained
        restore_rng(rng0)
        res_untrained = run_episode(env, actor=None, device="cpu", max_steps=args.steps)

        # 2) Trained
        restore_rng(rng0)
        res_trained = run_episode(env, actor=actor, device="cpu", max_steps=args.steps)

        # 导出
        untrained_dir = os.path.join(root, "untrained")
        trained_dir   = os.path.join(root, "trained")
        compare_dir   = os.path.join(root, "compare")
        os.makedirs(untrained_dir, exist_ok=True)
        os.makedirs(trained_dir, exist_ok=True)
        os.makedirs(compare_dir, exist_ok=True)

        save_episode(res_untrained, untrained_dir, env_cfg, stamp, fps=args.fps, trail_len=args.trail_len)
        save_episode(res_trained,   trained_dir,   env_cfg, stamp, fps=args.fps, trail_len=args.trail_len)
        plot_and_save_compare(res_untrained, res_trained, compare_dir, env_cfg, stamp)

    else:
        # 单独运行：若有 model 则跑训练模型，否则跑 baseline
        sub = "trained" if actor is not None else "untrained"
        out_dir = os.path.join(root, sub)
        os.makedirs(out_dir, exist_ok=True)

        res = run_episode(env, actor=actor, device="cpu", max_steps=args.steps)
        save_episode(res, out_dir, env_cfg, stamp, fps=args.fps, trail_len=args.trail_len)

    # Optional show (only if interactive backend)
    if not args.no_show:
        backend = matplotlib.get_backend().lower()
        if backend in {"qt5agg", "tkagg", "macosx"}:
            plt.show()
        else:
            print("Non-interactive backend detected; skipping plt.show(). Use --no-show to suppress this message.")


if __name__ == "__main__":
    main()
