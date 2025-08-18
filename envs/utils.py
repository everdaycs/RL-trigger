import math
from typing import Tuple

import numpy as np


def world_to_grid(x: float, y: float, H: int, W: int, res: float) -> Tuple[int, int]:
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
