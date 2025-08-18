import numpy as np
import math


def propagate_pose(env, dt: float):
    if not env.cfg.motion_enabled or dt <= 0.0:
        return

    v = max(0.0, float(np.random.normal(env.cfg.v_lin_mean, env.cfg.v_lin_std)))
    w = float(np.random.normal(env.cfg.v_ang_mean, env.cfg.v_ang_std))

    n = max(1, int(env.cfg.motion_substeps))
    h = dt / n

    for _ in range(n):
        x, y, yaw = float(env.pose[0]), float(env.pose[1]), float(env.pose[2])
        nx = x + v * math.cos(yaw) * h
        ny = y + v * math.sin(yaw) * h
        nyaw = (yaw + w * h + math.pi) % (2 * math.pi) - math.pi

        r, c = env.world_to_grid(nx, ny)
        out = not env.in_bounds(r, c)

        collide = False
        if not out:
            collide = (env.true_grid[r, c] == 1)

        if (out or collide) and env.cfg.keep_inside:
            if env.cfg.collide_stop:
                break
            else:
                env.pose[2] = nyaw
                continue
        else:
            env.pose[0] = nx
            env.pose[1] = ny
            env.pose[2] = nyaw
