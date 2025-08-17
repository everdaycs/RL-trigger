# train_trigger_ppo.py
# -------------------------------------------------------------
# PPO training for ultrasonic trigger order with global prior env
# Env: env_us2d_prior.US2DPriorEnv / VecEnvs
# Reward: info_gain - w_overlap*overlap - w_fail*fail - w_time*dt
# -------------------------------------------------------------

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_us2d_prior import EnvConfig, VecEnvs

# =========================
# ======  CONFIG  =========
# =========================

@dataclass
class TrainConfig:
    # reward weights
    w_info: float = 1.0
    w_fail: float = 3.0
    w_time: float = 0.05
    w_overlap: float = 0.2

    # PPO / training
    device: str = "cpu"         # "cuda" if available
    total_updates: int = 500
    n_envs: int = 64
    rollout_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    minibatch_size: int = 512
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    print_every: int = 5
    save_every: int = 50
    save_path: str = "ppo_trigger.pt"

# =========================
# =====  PPO AGENT  =======
# =========================

class MaskedCategorical(torch.distributions.Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        super().__init__(logits=logits)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        hid = 256
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
        )
        self.pi = nn.Linear(hid, n_actions)
        self.v = nn.Linear(hid, 1)

    def forward(self, obs: torch.Tensor):
        x = self.body(obs)
        return self.pi(x), self.v(x)

    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        logits, v = self.forward(obs)
        dist = MaskedCategorical(logits=logits, mask=mask)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, ent, v.squeeze(-1)

# =========================
# ========  PPO  ==========
# =========================

def compute_gae(rewards, values, dones, gamma, lam):
    T, B = rewards.shape
    adv = np.zeros((T, B), dtype=np.float32)
    lastgaelam = np.zeros(B, dtype=np.float32)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + values[:-1]
    return adv, ret

def train(env_cfg: EnvConfig, cfg: TrainConfig):
    # device
    if torch.cuda.is_available():
        cfg.device = "cuda"

    vec = VecEnvs(env_cfg, cfg.n_envs)
    obs_dim, n_actions = vec.obs_dim, vec.n_actions

    dev = torch.device(cfg.device)
    net = ActorCritic(obs_dim, n_actions).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    batch_size = cfg.n_envs * cfg.rollout_steps
    mb_size = cfg.minibatch_size
    assert batch_size % mb_size == 0, "batch_size must be divisible by minibatch_size"

    global_step = 0
    start_time = time.time()

    for update in range(1, cfg.total_updates + 1):
        # rollout storage
        obs_buf = np.zeros((cfg.rollout_steps, cfg.n_envs, obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps, cfg.n_envs), dtype=np.int64)
        logp_buf = np.zeros((cfg.rollout_steps, cfg.n_envs), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps, cfg.n_envs), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps, cfg.n_envs), dtype=np.bool_)
        val_buf = np.zeros((cfg.rollout_steps + 1, cfg.n_envs), dtype=np.float32)
        ent_buf = np.zeros((cfg.rollout_steps, cfg.n_envs), dtype=np.float32)

        obs_np = vec.obs.copy()
        for t in range(cfg.rollout_steps):
            mask_np = vec.masks().astype(np.float32)

            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=dev)
            mask_t = torch.tensor(mask_np, dtype=torch.float32, device=dev)

            with torch.no_grad():
                a_t, logp_t, ent_t, v_t = net.act(obs_t, mask_t)

            a_np = a_t.cpu().numpy()
            logp_np = logp_t.cpu().numpy()
            v_np = v_t.cpu().numpy()
            ent_np = ent_t.cpu().numpy()

            next_obs_np, r_tuple_np, d_np, infos = vec.step(a_np)
            # r_tuple_np: [B, 4] = (info_gain, overlap, fail, dt)
            info_gain = r_tuple_np[:, 0]
            overlap = r_tuple_np[:, 1]
            fail = r_tuple_np[:, 2]
            dt = r_tuple_np[:, 3]
            r_np = (cfg.w_info * info_gain
                    - cfg.w_fail * fail
                    - cfg.w_time * dt
                    - cfg.w_overlap * overlap).astype(np.float32)

            # store
            obs_buf[t] = obs_np
            act_buf[t] = a_np
            logp_buf[t] = logp_np
            rew_buf[t] = r_np
            done_buf[t] = d_np
            val_buf[t] = v_np
            ent_buf[t] = ent_np

            obs_np = next_obs_np
            global_step += cfg.n_envs

        # bootstrap value
        with torch.no_grad():
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=dev)
            logits, v_last = net.forward(obs_t)
            v_last_np = v_last.squeeze(-1).cpu().numpy()
            val_buf[-1] = v_last_np

        # GAE
        adv_np, ret_np = compute_gae(rew_buf, val_buf, done_buf, cfg.gamma, cfg.gae_lambda)

        # flatten
        obs_b = torch.tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=dev)
        act_b = torch.tensor(act_buf.reshape(-1), dtype=torch.int64, device=dev)
        logp_b_old = torch.tensor(logp_buf.reshape(-1), dtype=torch.float32, device=dev)
        adv_b = torch.tensor(adv_np.reshape(-1), dtype=torch.float32, device=dev)
        ret_b = torch.tensor(ret_np.reshape(-1), dtype=torch.float32, device=dev)

        # normalize advantages
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # PPO epochs
        idxs = np.arange(batch_size)
        for epoch in range(cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, mb_size):
                end = start + mb_size
                mb_idx = idxs[start:end]

                mb_obs = obs_b[mb_idx]
                mb_act = act_b[mb_idx]
                mb_logp_old = logp_b_old[mb_idx]
                mb_adv = adv_b[mb_idx]
                mb_ret = ret_b[mb_idx]

                logits, v = net.forward(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)  # no mask in update phase
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_logp_old)
                pg1 = ratio * mb_adv
                pg2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                pg_loss = -torch.min(pg1, pg2).mean()

                v_loss = 0.5 * F.mse_loss(v.squeeze(-1), mb_ret)
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

        if update % cfg.print_every == 0:
            avg_rew = rew_buf.mean()
            avg_ent = ent_buf.mean()
            fps = int(global_step / (time.time() - start_time + 1e-9))
            cov = vec.coverage()
            print(f"[Upd {update:04d}] reward {avg_rew:+.3f} | coverage {cov:.3f} | ent {avg_ent:.3f} | fps {fps}")

        if update % cfg.save_every == 0:
            torch.save(net.state_dict(), cfg.save_path)

    torch.save(net.state_dict(), cfg.save_path)
    print("Training finished. Saved model to", cfg.save_path)

# =========================
# ========== RUN =========
# =========================

if __name__ == "__main__":
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    env_cfg = EnvConfig()
    cfg = TrainConfig()

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available. Falling back to CPU.")
        cfg.device = "cpu"

    train(env_cfg, cfg)
