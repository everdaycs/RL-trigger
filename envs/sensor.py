from typing import List, Tuple, Optional
import math
import numpy as np
from .utils import raycast_grid, grid_to_world


def cast_cone(env, yaw_world: float, subrays: int) -> Tuple[float, float, float, float, List]:
    """Conservative free-space carving: on any hit, clear entire cone up to nearest hit as free.
    
    Behavior:
    1. Cast all subrays to find hits and distances
    2. If any subray hits obstacle, find minimum hit distance
    3. Clear entire cone up to min_dist as free (0) - do NOT mark hit cells as occupied
    4. If no hits, clear entire cone to max range as free (0)
    
    Returns (info_gain, overlap_ratio, max_dist, min_dist, subray_data).
    """
    fov = math.radians(env.cfg.sensor_fov_deg)
    K = max(1, int(subrays))
    
    # First pass: gather all subrays and find minimum hit distance
    max_dist = 0.0
    min_dist = float('inf')
    subrays_out = []

    for theta in np.linspace(yaw_world - fov / 2.0, yaw_world + fov / 2.0, K):
        last_cell = None
        cells_this = []
        hit_this = None
        
        # Traverse ray until hit or max range
        for (r, c) in raycast_grid(env.pose[0], env.pose[1], theta,
                                   env.cfg.max_range, env.H, env.W, env.cfg.map_res):
            if last_cell is not None and (r, c) == last_cell:
                continue  # skip duplicate cells
            
            cells_this.append((r, c))
            last_cell = (r, c)
            
            # Check for hit in true_grid
            if env.true_grid[r, c] == 1:
                hit_this = (r, c)
                break
        
        # Calculate distances
        if last_cell is not None:
            lx, ly = grid_to_world(last_cell[0], last_cell[1], env.H, env.W, env.cfg.map_res)
            dx, dy = lx - float(env.pose[0]), ly - float(env.pose[1])
            dist = math.hypot(dx, dy)
            if dist > max_dist:
                max_dist = dist
        else:
            dist = 0.0

        if hit_this is not None:
            hx, hy = grid_to_world(hit_this[0], hit_this[1], env.H, env.W, env.cfg.map_res)
            hdist = math.hypot(hx - float(env.pose[0]), hy - float(env.pose[1]))
            if hdist < min_dist:
                min_dist = hdist

        subrays_out.append((cells_this, hit_this, last_cell, dist))

    # Second pass: update observation grid based on conservative free-space policy
    visited = set()
    info_new = 0
    overlap = 0
    touched = 0
    
    # Determine clear distance: use min_dist if any hit, else use max_range
    if min_dist == float('inf'):
        min_dist = 0.0
        clear_dist = env.cfg.max_range
    else:
        clear_dist = min_dist
    
    # Clear entire cone up to clear_dist as free
    for (cells_this, hit_this, last_cell, dist) in subrays_out:
        for (r, c) in cells_this:
            # Calculate distance from robot to this cell
            cx, cy = grid_to_world(r, c, env.H, env.W, env.cfg.map_res)
            cell_dist = math.hypot(cx - float(env.pose[0]), cy - float(env.pose[1]))
            
            # Only update cells that are strictly before the clear distance
            # Do NOT mark hit cells as occupied - conservative free-space only
            if cell_dist < clear_dist:
                if (r, c) not in visited:
                    visited.add((r, c))
                    touched += 1
                    
                    if env.obs_grid[r, c] == -1:
                        # New information: mark as free
                        info_new += 1
                        env.obs_grid[r, c] = 0
                    else:
                        # Already known (overlap)
                        overlap += 1
                        # Still update to free if it was unknown
                        if env.obs_grid[r, c] == -1:
                            env.obs_grid[r, c] = 0

    # Calculate final statistics
    if touched == 0:
        return 0.0, 0.0, max_dist, min_dist, subrays_out
    
    info_gain = info_new / float(touched)
    overlap_ratio = overlap / float(touched)
    
    return info_gain, overlap_ratio, max_dist, min_dist, subrays_out
