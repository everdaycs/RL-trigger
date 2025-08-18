from typing import List, Tuple, Optional
import math
import numpy as np
from .utils import raycast_grid, grid_to_world


def cast_cone(env, yaw_world: float, subrays: int) -> Tuple[float, float, float, float, List]:
    """Cast subrays in env and return (info_gain, overlap, max_dist, min_dist, subray_data).

    subray_data: list of tuples (cells_list, hit_cell_or_None, last_cell, dist)
    """
    fov = math.radians(env.cfg.sensor_fov_deg)
    K = max(1, int(subrays))
    info_new = 0
    overlap = 0
    touched = 0
    max_dist = 0.0
    min_dist = float('inf')
    subrays_out = []

    for theta in np.linspace(yaw_world - fov / 2.0, yaw_world + fov / 2.0, K):
        last_cell = None
        cells_this = []
        hit_this = None
        for (r, c) in raycast_grid(env.pose[0], env.pose[1], theta,
                                   env.cfg.max_range, env.H, env.W, env.cfg.map_res):
            cells_this.append((r, c))
            last_cell = (r, c)
            if env.true_grid[r, c] == 1:
                hit_this = (r, c)
                break

        if last_cell is not None:
            lx, ly = grid_to_world(last_cell[0], last_cell[1], env.H, env.W, env.cfg.map_res)
            dx, dy = lx - env.pose[0], ly - env.pose[1]
            dist = math.hypot(dx, dy)
            if dist > max_dist:
                max_dist = dist
        else:
            dist = 0.0

        if hit_this is not None:
            hx, hy = grid_to_world(hit_this[0], hit_this[1], env.H, env.W, env.cfg.map_res)
            hdist = math.hypot(hx - env.pose[0], hy - env.pose[1])
            if hdist < min_dist:
                min_dist = hdist

        subrays_out.append((cells_this, hit_this, last_cell, dist))

    # Decide on update strategy: if any hit, choose closest hit subray; else update all
    info_new = 0
    overlap = 0
    touched = 0
    if min_dist != float('inf') and min_dist > 0.0:
        # choose closest hit subray
        chosen = None
        for idx, (cells_this, hit_this, last_cell, dist) in enumerate(subrays_out):
            if hit_this is not None:
                hx, hy = grid_to_world(hit_this[0], hit_this[1], env.H, env.W, env.cfg.map_res)
                hdist = math.hypot(hx - env.pose[0], hy - env.pose[1])
                if abs(hdist - min_dist) < 1e-6:
                    chosen = idx
                    break
        if chosen is None:
            for idx, (cells_this, hit_this, last_cell, dist) in enumerate(subrays_out):
                if hit_this is not None:
                    chosen = idx
                    break
        cells_this, hit_this, last_cell, dist = subrays_out[chosen]
        touched = len(cells_this)
        for (r, c) in cells_this:
            if env.true_grid[r, c] == 1:
                if env.obs_grid[r, c] == -1:
                    info_new += 1
                    env.obs_grid[r, c] = 1
                else:
                    overlap += 1
                break
            else:
                if env.obs_grid[r, c] == -1:
                    info_new += 1
                    env.obs_grid[r, c] = 0
                else:
                    overlap += 1
    else:
        for (cells_this, hit_this, last_cell, dist) in subrays_out:
            touched += len(cells_this)
            for (r, c) in cells_this:
                if env.obs_grid[r, c] == -1:
                    info_new += 1
                    env.obs_grid[r, c] = int(env.true_grid[r, c])
                else:
                    overlap += 1

    if touched == 0:
        return 0.0, 0.0, 0.0, 0.0, subrays_out
    info_gain = info_new / float(touched)
    overlap_ratio = overlap / float(touched)
    if min_dist == float('inf'):
        min_dist = 0.0
    return info_gain, overlap_ratio, max_dist, min_dist, subrays_out
