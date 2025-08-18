# env_us2d_prior.py
# -------------------------------------------------------------
# 2D ultrasonic trigger scheduling environment with:
# - Global occupancy grid (ground-truth from circle obstacles)
# - Random start pose (x, y, yaw)
# - Partial prior map (a portion of the true map revealed)
# - Cone-style FoV via multiple sub-rays per trigger
# - (NEW) Optional boundary walls enclosing the map
# - (NEW) Optional robot motion during ping round-trip time window dt
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
    prior_known_ratio: float = 0.2        # fraction of cells to reveal as prior
    prior_noise_flip_prob: float = 0.01     # small fraction of prior cells flipped to simulate errors
    episode_max_steps: int = 256
    coverage_stop: float = 0.2

    # FoV rasterization
    subrays_per_fov: int = 9                # number of sub-rays across FoV cone

    # (NEW) Boundary wall
    with_boundary: bool = True              # add 1-cell-thick walls around map

    # (NEW) Robot motion during ping time window dt (simple unicycle)
    motion_enabled: bool = True             # enable pose propagation over dt
    v_lin_mean: float = 0.20                # mean linear velocity [m/s]
    v_lin_std:  float = 0.05                # std  linear velocity [m/s]
    v_ang_mean: float = 0.0                 # mean angular velocity [rad/s]
    v_ang_std:  float = 0.20                # std  angular velocity [rad/s]
    motion_substeps: int = 5                # Euler sub-steps to avoid tunneling
    keep_inside: bool = True                # handle out-of-bounds/collision
    collide_stop: bool = True               # stop remaining motion on collision
    
    # ---- Prior options (NEW) ----
    prior_mode: str = "local_disk"        # "global" | "local_disk"
    prior_local_radius_m: float = 3.0 # 局部先验半径（以机器人为圆心）

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
    step = res * 0.5  # robust small steps
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
    (NEW) Optional boundary walls.
    (NEW) Optional robot motion during ping time window dt.
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

        # Rasterize to true grid (start as free)
        self.true_grid.fill(0)
        for r in range(self.H):
            for c in range(self.W):
                x, y = grid_to_world(r, c, self.H, self.W, self.cfg.map_res)
                for circ in self.circles:
                    if circ.contains(x, y):
                        self.true_grid[r, c] = 1
                        break

        # (NEW) Add boundary walls (1-cell thick)
        if self.cfg.with_boundary:
            self.true_grid[0, :]  = 1
            self.true_grid[-1, :] = 1
            self.true_grid[:, 0]  = 1
            self.true_grid[:, -1] = 1

    def _random_pose(self):
        # sample a free cell, try to avoid immediate boundary adjacency
        candidates = np.argwhere(self.true_grid == 0)
        if candidates.size == 0:
            self.pose[:] = 0.0
            return

        # Prefer cells away from boundary by 1 cell margin
        for _ in range(200):
            idx = np.random.randint(0, len(candidates))
            r, c = candidates[idx]
            if r <= 1 or r >= self.H - 2 or c <= 1 or c >= self.W - 2:
                continue
            x, y = grid_to_world(r, c, self.H, self.W, self.cfg.map_res)
            yaw = np.random.uniform(-math.pi, math.pi)
            self.pose[:] = (x, y, yaw)
            return

        # Fallback if margin not found
        r, c = candidates[np.random.randint(0, len(candidates))]
        x, y = grid_to_world(r, c, self.H, self.W, self.cfg.map_res)
        yaw = np.random.uniform(-math.pi, math.pi)
        self.pose[:] = (x, y, yaw)

    def _build_prior(self):
        """Build prior map into obs_grid & prior_mask.
           - global: 按 prior_known_ratio 在全图随机揭示
           - local_disk: 以机器人为圆心、半径 prior_local_radius_m 的圆盘内揭示
        """
        self.obs_grid.fill(-1)
        self.prior_mask[:] = False

        flip_p = float(self.cfg.prior_noise_flip_prob)

        if self.cfg.prior_mode == "local_disk":
            # --- 局部先验：以当前 self.pose 为中心 ---
            px, py = float(self.pose[0]), float(self.pose[1])
            rad = float(self.cfg.prior_local_radius_m)
            # 先收集圆盘内所有格子
            idxs = []
            for r in range(self.H):
                # 粗裁剪：y 距离超过半径就跳过
                y = grid_to_world(r, 0, self.H, self.W, self.cfg.map_res)[1]
                if abs(y - py) > rad:
                    continue
                for c in range(self.W):
                    x = grid_to_world(0, c, self.H, self.W, self.cfg.map_res)[0]
                    if (x - px) ** 2 + (y - py) ** 2 <= rad ** 2:
                        idxs.append((r, c))

            if len(idxs) > 0:
                # 在圆盘内按比例抽样（prior_local_ratio），1.0 表示圆盘内全揭示
                take = int(round(self.cfg.prior_known_ratio * len(idxs)))
                if take < len(idxs):
                    idxs = random.sample(idxs, take)

                # 标记 prior_mask 并写入 obs_grid（带翻转噪声）
                for (r, c) in idxs:
                    self.prior_mask[r, c] = True
                    val = int(self.true_grid[r, c])
                    if random.random() < flip_p:
                        if val == 0: val = 1
                        elif val == 1: val = 0
                    self.obs_grid[r, c] = val

            # 注意：此模式下不再使用 prior_known_ratio（全图比例），
            # 如需两者并存，可在此处再额外补充少量全局 prior。

        else:
            # --- 旧逻辑：全图随机 prior ---
            total = self.H * self.W
            n_prior = int(self.cfg.prior_known_ratio * total)
            idxs_flat = np.random.choice(total, size=n_prior, replace=False)
            r_idx = idxs_flat // self.W
            c_idx = idxs_flat % self.W
            self.prior_mask[r_idx, c_idx] = True

            for r, c in zip(r_idx, c_idx):
                val = int(self.true_grid[r, c])
                if np.random.rand() < flip_p:
                    if val == 0: val = 1
                    elif val == 1: val = 0
                self.obs_grid[r, c] = val


    # ---------- API ----------

    def reset(self) -> np.ndarray:
        self._random_world()

        if self.cfg.prior_mode == "local_disk":
            # 需要先有机器人位姿，才能围绕它生成先验
            self._random_pose()
            self._build_prior()  # 将看到 robot-centered 局部先验
        else:
            # 旧逻辑：全局先验对位姿无依赖
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
            v = float(self.last_valid[i])
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

    def _cast_cone_and_update(self, yaw_world: float) -> Tuple[float, float, float, float]:
        """
        Cast multiple sub-rays in [yaw - fov/2, yaw + fov/2].
        Update obs_grid along each ray using true_grid to determine hit.
        Returns (info_gain, overlap, max_dist_touched, min_dist_closest_hit).
        """
        fov = math.radians(self.cfg.sensor_fov_deg)
        K = max(1, int(self.cfg.subrays_per_fov))
        info_new = 0
        overlap = 0
        touched = 0
        max_dist = 0.0
        min_dist = float('inf')

        for theta in np.linspace(yaw_world - fov / 2.0, yaw_world + fov / 2.0, K):
            # collect cells visited by this subray and optional hit
            last_cell = None
            cells_this = []
            hit_this = None
            for (r, c) in raycast_grid(self.pose[0], self.pose[1], theta,
                                       self.cfg.max_range, self.H, self.W, self.cfg.map_res):
                cells_this.append((r, c))
                last_cell = (r, c)
                # stop on occupied
                if self.true_grid[r, c] == 1:
                    hit_this = (r, c)
                    break

            # store subray data
            # compute furthest distance touched by this subray (last_cell)
            if last_cell is not None:
                lx, ly = grid_to_world(last_cell[0], last_cell[1], self.H, self.W, self.cfg.map_res)
                dx, dy = lx - self.pose[0], ly - self.pose[1]
                dist = math.hypot(dx, dy)
                if dist > max_dist:
                    max_dist = dist
            else:
                dist = 0.0

            # if hit, consider for min_dist
            if hit_this is not None:
                hx, hy = grid_to_world(hit_this[0], hit_this[1], self.H, self.W, self.cfg.map_res)
                hdist = math.hypot(hx - self.pose[0], hy - self.pose[1])
                if hdist < min_dist:
                    min_dist = hdist

            # append collected data
            # We reuse touched/info_new/overlap later based on selection
            # store as tuple: (cells_this, hit_this, last_cell, dist)
            if 'subrays' not in locals():
                subrays = []
            subrays.append((cells_this, hit_this, last_cell, dist))

        # Decide which subray(s) to use for updating the map
        # If any subray had a hit, choose the one with the closest hit (min_dist)
        info_new = 0
        overlap = 0
        touched = 0
        if min_dist != float('inf') and min_dist > 0.0:
            # find subray index with hit at distance == min_dist (closest)
            chosen = None
            for idx, (cells_this, hit_this, last_cell, dist) in enumerate(subrays):
                if hit_this is not None:
                    hx, hy = grid_to_world(hit_this[0], hit_this[1], self.H, self.W, self.cfg.map_res)
                    hdist = math.hypot(hx - self.pose[0], hy - self.pose[1])
                    if abs(hdist - min_dist) < 1e-6:
                        chosen = idx
                        break
            # fallback: choose first hit subray
            if chosen is None:
                for idx, (cells_this, hit_this, last_cell, dist) in enumerate(subrays):
                    if hit_this is not None:
                        chosen = idx
                        break

            cells_this, hit_this, last_cell, dist = subrays[chosen]
            touched = len(cells_this)
            for (r, c) in cells_this:
                if self.true_grid[r, c] == 1:
                    if self.obs_grid[r, c] == -1:
                        info_new += 1
                        self.obs_grid[r, c] = 1
                    else:
                        overlap += 1
                    break
                else:
                    if self.obs_grid[r, c] == -1:
                        info_new += 1
                        self.obs_grid[r, c] = 0
                    else:
                        overlap += 1
        else:
            # no occupied hits in any subray: keep original behavior and update all subrays
            for (cells_this, hit_this, last_cell, dist) in subrays:
                touched += len(cells_this)
                for (r, c) in cells_this:
                    if self.obs_grid[r, c] == -1:
                        info_new += 1
                        self.obs_grid[r, c] = int(self.true_grid[r, c])
                    else:
                        overlap += 1

        if touched == 0:
            return 0.0, 0.0, 0.0, 0.0
        info_gain = info_new / float(touched)
        overlap_ratio = overlap / float(touched)
        # if no occupied cell was seen in any subray, min_dist will remain inf
        if min_dist == float('inf'):
            min_dist = 0.0
        return info_gain, overlap_ratio, max_dist, min_dist

    # ---------- (NEW) Motion ----------

    def _propagate_pose(self, dt: float):
        """Propagate robot pose over dt using a simple unicycle model with noise.
           x += v*cos(yaw)*h, y += v*sin(yaw)*h, yaw += w*h
           Uses sub-steps to avoid tunneling; checks boundary/collision on true_grid.
        """
        if not self.cfg.motion_enabled or dt <= 0.0:
            return

        # Sample velocities (once per step is sufficient and stable)
        v = max(0.0, float(np.random.normal(self.cfg.v_lin_mean, self.cfg.v_lin_std)))
        w = float(np.random.normal(self.cfg.v_ang_mean, self.cfg.v_ang_std))

        n = max(1, int(self.cfg.motion_substeps))
        h = dt / n

        for _ in range(n):
            x, y, yaw = float(self.pose[0]), float(self.pose[1]), float(self.pose[2])
            nx = x + v * math.cos(yaw) * h
            ny = y + v * math.sin(yaw) * h
            nyaw = (yaw + w * h + math.pi) % (2 * math.pi) - math.pi

            r, c = world_to_grid(nx, ny, self.H, self.W, self.cfg.map_res)
            out = not in_bounds(r, c, self.H, self.W)

            collide = False
            if not out:
                # true_grid: 1=occupied/wall, 0=free
                collide = (self.true_grid[r, c] == 1)

            if (out or collide) and self.cfg.keep_inside:
                if self.cfg.collide_stop:
                    break  # stop remaining motion this step
                else:
                    # Alternative: only update yaw, keep position (glance off)
                    self.pose[2] = nyaw
                    continue
            else:
                self.pose[0] = nx
                self.pose[1] = ny
                self.pose[2] = nyaw

    # ---------- Step ----------

    def step(self, a: int):
        mask = self.action_mask()
        assert mask[a] == 1, "Illegal action selected. Ensure the policy respects the mask."

        # Sensor direction in world
        yaw_world = self.pose[2] + self.sensor_dirs[a]

        # Simulate measurement over cone FoV
        info_gain, overlap, max_dist, min_dist = self._cast_cone_and_update(yaw_world)

        # Random dropout fail (note: current design updates map regardless of fail)
        valid = 1.0
        if random.random() < self.cfg.dropout_prob:
            valid = 0.0

        # Time cost: round-trip
        # Use the shortest hit distance among subrays as the sensor's reported distance
        # If no hit (min_dist == 0.0), fall back to max_range
        dist_for_time = min_dist if min_dist > 0 else self.cfg.max_range
        dt = self.cfg.base_overhead_s + (2.0 * dist_for_time / self.cfg.speed_of_sound) + 0.001
        self.t_now += dt
        self.last_fire_t[a] = self.t_now

        # Update last measurement memory
        self.last_dist[a] = dist_for_time
        self.last_valid[a] = valid
        self.last_conf[a] = valid
        self.last_action = a
        self.step_count += 1

        # (NEW) Propagate pose during the ping time window
        self._propagate_pose(dt)

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
