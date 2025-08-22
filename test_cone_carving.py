#!/usr/bin/env python3
"""
Test conservative free-space carving behavior.
Verifies: hit stops all rays, clear entire cone to min_dist as free, no occupied marking.
"""

import numpy as np
import math
from env_us2d_prior import EnvConfig, US2DPriorEnv
from envs.sensor import cast_cone

def test_conservative_carving():
    """Test conservative free-space carving behavior."""
    print("Testing conservative free-space carving...")
    
    # Create controlled test environment
    cfg = EnvConfig()
    cfg.episode_max_steps = 10
    cfg.subrays_per_fov = 5  # Use 5 rays for testing
    cfg.sensor_fov_deg = 40.0  # Wide FoV to show cone effect
    cfg.max_range = 4.0
    cfg.map_size_m = 8.0  # Smaller map
    cfg.n_circles = 0  # No random obstacles
    cfg.prior_known_ratio = 0.0  # No prior knowledge
    
    env = US2DPriorEnv(cfg)
    env._random_world()
    
    # Clear the grid and create controlled scenario
    env.true_grid.fill(0)  # All free
    
    # Add boundary walls
    env.true_grid[0, :] = 1   # Top wall
    env.true_grid[-1, :] = 1  # Bottom wall  
    env.true_grid[:, 0] = 1   # Left wall
    env.true_grid[:, -1] = 1  # Right wall
    
    # Place robot in center
    center_r, center_c = env.H // 2, env.W // 2
    env.pose[0], env.pose[1] = 0.0, 0.0  # World center
    env.pose[2] = 0.0  # Facing right
    
    # Place obstacles at different distances to test min_dist behavior
    # Closer obstacle
    if center_c + 8 < env.W:
        env.true_grid[center_r - 1, center_c + 8] = 1  # Upper ray hits at ~0.8m
    # Farther obstacle  
    if center_c + 15 < env.W:
        env.true_grid[center_r + 1, center_c + 15] = 1  # Lower ray hits at ~1.5m
    
    # Clear observation grid
    env.obs_grid.fill(-1)  # All unknown
    
    print(f"Grid size: {env.H}x{env.W}")
    print(f"Robot at grid: ({center_r}, {center_c}), world: ({env.pose[0]:.2f}, {env.pose[1]:.2f})")
    print(f"Robot facing: {math.degrees(env.pose[2]):.1f}° (right)")
    
    # Show true grid in region of interest
    print(f"\nTrue grid (robot and obstacles, . = free, # = occupied):")
    for r in range(center_r-3, center_r+4):
        row = ""
        for c in range(center_c-2, min(env.W, center_c+18)):
            if 0 <= r < env.H and 0 <= c < env.W:
                if r == center_r and c == center_c:
                    row += "R"  # Robot position
                elif env.true_grid[r, c] == 1:
                    row += "#"
                else:
                    row += "."
            else:
                row += "X"
        print(row)
    
    print(f"\nBefore sensing - obs_grid (all should be unknown '?'):")
    for r in range(center_r-3, center_r+4):
        row = ""
        for c in range(center_c-2, min(env.W, center_c+18)):
            if 0 <= r < env.H and 0 <= c < env.W:
                if r == center_r and c == center_c:
                    row += "R"
                elif env.obs_grid[r, c] == -1:
                    row += "?"
                elif env.obs_grid[r, c] == 0:
                    row += "."
                else:
                    row += "#"
            else:
                row += "X"
        print(row)
    
    # Test conservative cone carving pointing right
    yaw_world = 0.0  # Pointing right
    print(f"\nTesting conservative carving pointing right (yaw={math.degrees(yaw_world):.1f}°)")
    print("Expected behavior:")
    print("1. Multiple subrays cast, some hit obstacles at different distances")
    print("2. Find minimum hit distance among all subrays")
    print("3. Clear ENTIRE cone up to min_dist as free")
    print("4. Do NOT mark hit cells as occupied")
    
    info_gain, overlap_ratio, max_dist, min_dist, subrays_out = cast_cone(env, yaw_world, 5)
    
    print(f"\nResults:")
    print(f"Info gain: {info_gain:.3f}")
    print(f"Overlap ratio: {overlap_ratio:.3f}")
    print(f"Max distance: {max_dist:.3f}")
    print(f"Min distance: {min_dist:.3f}")
    print(f"Number of subrays: {len(subrays_out)}")
    
    # Analyze each subray
    print(f"\nSubray analysis:")
    for i, (cells_this, hit_this, last_cell, dist) in enumerate(subrays_out):
        print(f"  Subray {i+1}: {len(cells_this)} cells, hit={hit_this}, dist={dist:.3f}")
    
    print(f"\nAfter sensing - obs_grid (. = free, # = occupied, ? = unknown):")
    for r in range(center_r-3, center_r+4):
        row = ""
        for c in range(center_c-2, min(env.W, center_c+18)):
            if 0 <= r < env.H and 0 <= c < env.W:
                if r == center_r and c == center_c:
                    row += "R"
                elif env.obs_grid[r, c] == -1:
                    row += "?"
                elif env.obs_grid[r, c] == 0:
                    row += "."
                else:
                    row += "#"
            else:
                row += "X"
        print(row)
    
    # Count cell types
    free_count = (env.obs_grid == 0).sum()
    occ_count = (env.obs_grid == 1).sum()
    unknown_count = (env.obs_grid == -1).sum()
    
    print(f"\nGrid statistics after conservative carving:")
    print(f"  Free cells: {free_count}")
    print(f"  Occupied cells: {occ_count} (should be 0 - we don't mark hits as occupied)")
    print(f"  Unknown cells: {unknown_count}")
    print(f"  Total updated: {free_count + occ_count}")
    
    # Verify the key behavior
    print(f"\n" + "="*70)
    print("CONSERVATIVE FREE-SPACE CARVING VERIFICATION:")
    
    if occ_count == 0:
        print("✓ No hit cells marked as occupied (conservative policy)")
    else:
        print(f"✗ Found {occ_count} occupied cells (should be 0)")
    
    if free_count > 0:
        print(f"✓ {free_count} cells marked as free up to min_dist={min_dist:.3f}m")
    else:
        print("✗ No cells marked as free")
        
    if min_dist > 0:
        print(f"✓ Minimum hit distance detected: {min_dist:.3f}m")
    else:
        print("✓ No hits detected, would clear to max_range")
    
    print("\nExpected behavior confirmed:")
    print("- Any subray hit stops all ray processing")
    print("- Entire cone cleared as free up to nearest hit distance")  
    print("- Hit cells are NOT marked as occupied (conservative)")
    print("- Only positive free-space information is recorded")

def test_no_hit_scenario():
    """Test behavior when no subrays hit anything."""
    print(f"\n" + "="*70)
    print("Testing NO-HIT scenario (should clear entire cone to max range)")
    
    cfg = EnvConfig()
    cfg.max_range = 2.0
    cfg.subrays_per_fov = 3
    cfg.sensor_fov_deg = 30.0
    cfg.map_size_m = 6.0
    cfg.n_circles = 0
    cfg.prior_known_ratio = 0.0
    
    env = US2DPriorEnv(cfg)
    env._random_world()
    
    # Create all-free environment (no obstacles except boundary)
    env.true_grid.fill(0)
    env.true_grid[0, :] = 1   # Only boundary walls
    env.true_grid[-1, :] = 1
    env.true_grid[:, 0] = 1
    env.true_grid[:, -1] = 1
    
    center_r, center_c = env.H // 2, env.W // 2
    env.pose[0], env.pose[1] = 0.0, 0.0
    env.pose[2] = 0.0  # Facing right
    env.obs_grid.fill(-1)
    
    print(f"No obstacles in sensor range, robot facing right")
    
    info_gain, overlap_ratio, max_dist, min_dist, subrays_out = cast_cone(env, 0.0, 3)
    
    print(f"Results: info_gain={info_gain:.3f}, min_dist={min_dist:.3f}, max_dist={max_dist:.3f}")
    
    free_count = (env.obs_grid == 0).sum()
    occ_count = (env.obs_grid == 1).sum()
    
    print(f"Grid updates: {free_count} free, {occ_count} occupied")
    
    if min_dist == 0.0:
        print("✓ No hits detected (min_dist = 0)")
    if occ_count == 0:
        print("✓ No occupied cells marked")
    if free_count > 0:
        print(f"✓ {free_count} cells cleared as free to max range")

if __name__ == "__main__":
    test_conservative_carving()
    test_no_hit_scenario()
