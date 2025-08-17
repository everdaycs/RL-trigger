# env_us2d_prior.py
# -------------------------------------------------------------
# 2D ultrasonic trigger scheduling environment with:
# - Global occupancy grid (ground-truth from circle obstacles)
# - Random start pose (x, y, yaw)
# - Partial prior map (a portion of the true map revealed)
# - Cone-style FoV via multiple sub-rays per trigger
#
# Notes:
# - No IMU preintegration modeling here.
# - No entropy reward; info_gain is fraction of newly-known cells.
# - Reward is returned as components: (info_gain, overlap, fail, dt)
#   The trainer combines with weights.
# -------------------------------------------------------------

import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np

# =========================
# ======  CONFIG  =========
# =========================

@dataclass
class EnvConfig:
    # Sensors
    n_sensors: int = 12
    sensor_fov_deg: float = 65.0           # cone FoV per sensor
    max_range: float = 8.0                  # [m]
    speed_of_sound: float = 343.0           # [m/s]
    base_overhead_s: float = 0.003          # [s]
    dropout_prob: float = 0.03              # random dropout
    theta_min_deg: float = 20.0             # hard angular separation
    dt_min_s: float = 0.02                  # hard time separation

    # World (continuous obstacles)
    world_size: float = 12.0                # [m] square side centered at (0,0)
    n_circles: int = 12
    circle_radius_range: Tuple[float, float] = (0.3, 1.0)

    # Grid (discrete map used for sensing updates & coverage)
    map_size_m: float = 12.0                # [m] square map size
    map_res: float = 0.1                    # [m] cell resolution
    prior_known_ratio: float = 0.35         # fraction of cells to reveal as prior
    prior_noise_flip_prob: float = 0.02     # small fraction of prior cells flipped to simulate errors
    episode_max_steps: int = 400
    coverage_stop: float = 0.9

    # FoV rasterization
    subrays_per_fov: int = 9                # number of sub-rays across FoV cone

# =========================
# ======  UTILITIES  ======
# =========================

def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

def world_to_grid(x: float, y: float, H: int, W: int, res: float) -> Tuple[int, int]:
    # Map frame: (0,0) at bottom-left, x right, y up.
    # World frame: centered at (0,0). Convert world -> map indices.
    half = (W * res) / 2.0
    gx = int((x + half) / res)
    gy = int((y + half) / res)
    return gy, gx  # (row, col)

def grid_to_world(r: int, c: int, H: int, W: int, res: float) -> Tuple[float, float]:
    half = (W * res) / 2.0
    x = (c + 0.5) * res - half
    y = (r + 0.5) * res - half
    return x, y

def in_bounds(r: int, c: int, H: int, W: int) -> bool:
    return (0 <= r < H) and (0 <= c < W)

# Bresenham-like ray traversal on grid
def raycast_grid(x0: float, y0: float, theta: float, max_range: float, H: int, W: int, res: float):
    """Yield grid indices (r,c) along a ray from (x0,y0) until max_range."""
    # Step world in small increments of res/2 to be robust
    step = res * 0.5
    dx, dy = math.cos(theta) * step, math.sin(theta) * step
    n_steps = int(max_range / step) + 1
    x, y = x0, y0
    for _ in range(n_steps):
        r, c = world_to_grid(x, y, H, W, res)
        if not in_bounds(r, c, H, W):
            break
        yield (r, c)
        x += dx; y += dy

# =========================
# ======  ENV SIM  ========
# =========================

class CircleObstacle:
    def __init__(self, cx: float, cy: float, r: float):
        self.cx = cx
        self.cy = cy
        self.r = r

    def contains(self, x: float, y: float) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2

class US2DPriorEnv:
    """
    Global grid + random start pose + prior reveal.
    Sensing uses cone FoV via multiple sub-rays.
    """
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        # Sensors directions in robot frame
        self.sensor_dirs = np.linspace(-math.pi, math.pi, cfg.n_sensors, endpoint=False)
        self.theta_min = math.radians(cfg.theta_min_deg)

        # Build grid shape
        self.H = int(round(cfg.map_size_m / cfg.map_res))
        self.W = int(round(cfg.map_size_m / cfg.map_res))

        # State containers
        self.t_now = 0.0
        self.last_fire_t = np.full(cfg.n_sensors, -1e9, dtype=np.float32)
        self.last_dist = np.zeros(cfg.n_sensors, dtype=np.float32)
        self.last_valid = np.zeros(cfg.n_sensors, dtype=np.float32)
        self.last_conf = np.zeros(cfg.n_sensors, dtype=np.float32)
        self.last_action = -1
        self.step_count = 0

        # World obstacles
        self.circles: List[CircleObstacle] = []

        # True occupancy grid: -1 unknown (for coverage), 0 free, 1 occ
        self.true_grid = -np.ones((self.H, self.W), dtype=np.int8)

        # Observed grid (agent's map): -1 unknown, 0 free, 1 occ
        self.obs_grid = -np.ones((self.H, self.W), dtype=np.int8)

        # Prior mask: which cells are revealed initially
        self.prior_mask = np.zeros((self.H, self.W), dtype=bool)

        # Robot pose (x,y,yaw) in world frame (m, m, rad)
        self.pose = np.zeros(3, dtype=np.float32)

    # ---------- World & Prior ----------

    def _random_world(self):
        # Random circles in a continuous world
        self.circles = []
        half = self.cfg.world_size / 2.0
        for _ in range(self.cfg.n_circles):
            r = random.uniform(*self.cfg.circle_radius_range)
            cx = random.uniform(-half + 1.0, half - 1.0)
            cy = random.uniform(-half + 1.0, half - 1.0)
            self.circles.append(CircleObstacle(cx, cy, r))

        # Rasterize to true grid
        self.true_grid.fill(0)  # start as free
        for r in range(self.H):
            for c in range(self.W):
                x, y = grid_to_world(r, c, self.H, self.W, self.cfg.map_res)
                for circ in self.circles:
                    if circ.contains(x, y):
                        self.true_grid[r, c] = 1
                        break

    def _random_pose(self):
        # sample a free cell
        candidates = np.argwhere(self.true_grid == 0)
        if candidates.size == 0:
            # fallback: set to center free
            self.pose[:] = 0.0
            return
        idx = np.random.randint(0, len(candidates))
        r, c = candidates[idx]
        x, y = grid_to_world(r, c, self.H, self.W, self.cfg.map_res)
        yaw = np.random.uniform(-math.pi, math.pi)
        self.pose[:] = (x, y, yaw)

    def _build_prior(self):
        self.obs_grid.fill(-1)
        self.prior_mask[:] = False
        # Reveal a random subset of cells according to prior_known_ratio
        total = self.H * self.W
        n_prior = int(self.cfg.prior_known_ratio * total)
        idxs = np.random.choice(total, size=n_prior, replace=False)
        r_idx = idxs // self.W
        c_idx = idxs % self.W
        self.prior_mask[r_idx, c_idx] = True

        # Copy truth to obs_grid for prior cells, with small flip noise
        for r, c in zip(r_idx, c_idx):
            val = self.true_grid[r, c]
            if np.random.rand() < self.cfg.prior_noise_flip_prob:
                # flip: free<->occ for robustness
                if val == 0:
                    val = 1
                elif val == 1:
                    val = 0
            self.obs_grid[r, c] = val

    # ---------- API ----------

    def reset(self) -> np.ndarray:
        self._random_world()
        self._build_prior()
        self._random_pose()

        self.t_now = 0.0
        self.last_fire_t[:] = -1e9
        self.last_dist[:] = 0.0
        self.last_valid[:] = 0.0
        self.last_conf[:] = 0.0
        self.last_action = -1
        self.step_count = 0

        return self._get_obs()

    def coverage(self) -> float:
        known = (self.obs_grid != -1).sum()
        total = self.obs_grid.size
        return known / (total + 1e-8)

    def _get_obs(self) -> np.ndarray:
        # Per-sensor features
        tnorm = 1.0  # 1s cap normalization
        per = []
        for i in range(self.cfg.n_sensors):
            dt = max(0.0, self.t_now - float(self.last_fire_t[i]))
            dt_n = min(dt / tnorm, 1.0)
            d_n = min(self.last_dist[i] / self.cfg.max_range, 1.0)
            v = self.last_valid[i]
            per.extend([dt_n, d_n, v])
        per = np.array(per, dtype=np.float32)

        # Global features
        cover = np.array([self.coverage()], dtype=np.float32)
        time_feat = np.array([min(self.t_now / 2.0, 1.0)], dtype=np.float32)
        last_a = np.zeros(self.cfg.n_sensors, dtype=np.float32)
        if self.last_action >= 0:
            last_a[self.last_action] = 1.0

        # Pose (x,y,yaw) normalized by map size (~[-1,1])
        norm = (self.cfg.map_size_m / 2.0)
        px, py, pyaw = self.pose
        pose_feat = np.array([px / norm, py / norm, math.sin(pyaw), math.cos(pyaw)], dtype=np.float32)

        obs = np.concatenate([per, cover, time_feat, last_a, pose_feat], axis=0)
        return obs

    def action_mask(self) -> np.ndarray:
        mask = np.ones(self.cfg.n_sensors, dtype=np.int32)
        # time separation
        for i in range(self.cfg.n_sensors):
            if (self.t_now - float(self.last_fire_t[i])) < self.cfg.dt_min_s:
                mask[i] = 0
        # angular separation (only relative to last action for simplicity)
        if self.last_action >= 0:
            ref_dir = self.sensor_dirs[self.last_action]
            for i in range(self.cfg.n_sensors):
                ang = abs(self._ang_diff(self.sensor_dirs[i], ref_dir))
                if ang < self.theta_min:
                    mask[i] = 0
        return mask

    @staticmethod
    def _ang_diff(a: float, b: float) -> float:
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return d

    # ---------- Sensing ----------

    def _cast_cone_and_update(self, yaw_world: float) -> Tuple[float, float, float]:
        """
        Cast multiple sub-rays in [yaw - fov/2, yaw + fov/2].
        Update obs_grid along each ray using true_grid to determine hit.
        Returns (info_gain, overlap, max_dist_touched).
        """
        fov = math.radians(self.cfg.sensor_fov_deg)
        K = max(1, int(self.cfg.subrays_per_fov))
        info_new = 0
        overlap = 0
        touched = 0
        max_dist = 0.0

        # For each sub-ray
        for theta in np.linspace(yaw_world - fov / 2.0, yaw_world + fov / 2.0, K):
            first_hit = False
            last_cell = None
            for (r, c) in raycast_grid(self.pose[0], self.pose[1], theta, self.cfg.max_range, self.H, self.W, self.cfg.map_res):
                touched += 1
                last_cell = (r, c)

                # If we hit an occupied cell in true map: mark it
                if self.true_grid[r, c] == 1:
                    # occupied
                    if self.obs_grid[r, c] == -1:
                        info_new += 1
                        self.obs_grid[r, c] = 1
                    else:
                        overlap += 1
                    first_hit = True
                    break
                else:
                    # free along the way
                    if self.obs_grid[r, c] == -1:
                        info_new += 1
                        self.obs_grid[r, c] = 0
                    else:
                        overlap += 1

            # If no hit and we reached max range: last_cell may be the furthest
            if last_cell is not None:
                # compute distance from robot to last_cell center
                lx, ly = grid_to_world(last_cell[0], last_cell[1], self.H, self.W, self.cfg.map_res)
                dx, dy = lx - self.pose[0], ly - self.pose[1]
                dist = math.hypot(dx, dy)
                if dist > max_dist:
                    max_dist = dist

        if touched == 0:
            return 0.0, 0.0, 0.0
        info_gain = info_new / touched
        overlap_ratio = overlap / touched
        return info_gain, overlap_ratio, max_dist

    def step(self, a: int):
        mask = self.action_mask()
        assert mask[a] == 1, "Illegal action selected. Ensure the policy respects the mask."

        # Sensor direction in world
        yaw_world = self.pose[2] + self.sensor_dirs[a]

        # Simulate measurement over cone FoV
        info_gain, overlap, max_dist = self._cast_cone_and_update(yaw_world)

        # Random dropout fail
        valid = 1.0
        if random.random() < self.cfg.dropout_prob:
            valid = 0.0

        # Time cost: round-trip
        dist_for_time = max_dist if max_dist > 0 else self.cfg.max_range
        dt = self.cfg.base_overhead_s + (2.0 * dist_for_time / self.cfg.speed_of_sound) + 0.001
        self.t_now += dt
        self.last_fire_t[a] = self.t_now

        # Update last measurement memory
        self.last_dist[a] = dist_for_time
        self.last_valid[a] = valid
        self.last_conf[a] = valid
        self.last_action = a
        self.step_count += 1

        # Termination
        done = False
        if self.step_count >= self.cfg.episode_max_steps:
            done = True
        elif self.coverage() >= self.cfg.coverage_stop:
            done = True

        obs = self._get_obs()
        info = {
            "mask": mask,
            "info_gain": info_gain,
            "overlap": overlap,
            "fail": 1.0 - valid,
            "dt": dt,
            "coverage": self.coverage(),
        }
        reward = (info_gain, overlap, 1.0 - valid, dt)
        return obs, reward, done, info

# =========================
# ===  MULTI-ENV WRAP  ====
# =========================

class VecEnvs:
    def __init__(self, cfg: EnvConfig, n_envs: int):
        self.cfg = cfg
        self.envs = [US2DPriorEnv(cfg) for _ in range(n_envs)]
        self.n_envs = n_envs
        obs = [e.reset() for e in self.envs]
        self.obs_dim = obs[0].shape[0]
        self.n_actions = cfg.n_sensors
        self.obs = np.stack(obs, axis=0)

    def reset(self):
        self.obs = np.stack([e.reset() for e in self.envs], axis=0)
        return self.obs

    def step(self, actions: np.ndarray):
        next_obs, rews, dones, infos = [], [], [], []
        for i, e in enumerate(self.envs):
            ob, r_raw, d, info = e.step(int(actions[i]))
            info_gain, overlap, fail, dt = r_raw
            r = (info_gain, overlap, fail, dt)
            if d:
                ob = e.reset()
            next_obs.append(ob)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        self.obs = np.stack(next_obs, axis=0)
        return self.obs, np.array(rews, dtype=np.float32), np.array(dones, dtype=np.bool_), infos

    def masks(self) -> np.ndarray:
        return np.stack([e.action_mask() for e in self.envs], axis=0)

    def coverage(self) -> float:
        return float(np.mean([e.coverage() for e in self.envs]))
#test