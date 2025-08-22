#!/usr/bin/env python3
"""
测试改进的奖励函数设计
适用于保守自由空间清理策略
"""

import numpy as np
from dataclasses import dataclass
from env_us2d_prior import EnvConfig, US2DPriorEnv

@dataclass 
class EnhancedRewardConfig:
    """改进的奖励函数配置"""
    
    # 方案1: 覆盖率导向
    w_coverage_gain: float = 5.0      # 覆盖率增益权重
    w_info: float = 2.0               # info_gain权重（降低）
    w_overlap: float = 1.0            # overlap惩罚权重（降低）
    w_exploration: float = 1.0        # 探索新区域奖励
    w_efficiency: float = 0.5         # 效率奖励（距离/时间）
    
    # 方案2: 自适应重叠惩罚
    overlap_threshold: float = 0.7    # 重叠率阈值
    adaptive_overlap: bool = True     # 自适应重叠惩罚
    
    # 方案3: 距离探索奖励
    distance_reward: bool = True      # 距离探索奖励
    max_distance_bonus: float = 1.0   # 最大距离奖励
    
def enhanced_reward_v1(info_gain, overlap, fail, dt, coverage_gain, total_coverage):
    """
    方案1: 覆盖率导向奖励
    
    核心思想：
    1. 主要奖励覆盖率增长而非信息增益
    2. 降低重叠惩罚，允许合理的重复扫描
    3. 增加探索奖励，鼓励发现新区域
    """
    cfg = EnhancedRewardConfig()
    
    # 基础奖励：覆盖率增益
    coverage_reward = cfg.w_coverage_gain * coverage_gain
    
    # 信息增益奖励（降权）
    info_reward = cfg.w_info * info_gain
    
    # 自适应重叠惩罚
    if cfg.adaptive_overlap and overlap < cfg.overlap_threshold:
        overlap_penalty = 0  # 低重叠率不惩罚
    else:
        overlap_penalty = cfg.w_overlap * max(0, overlap - cfg.overlap_threshold)
    
    # 探索效率奖励
    efficiency_bonus = cfg.w_efficiency * (1.0 / (dt + 0.001))  # 快速探索奖励
    
    # 总奖励
    reward = coverage_reward + info_reward - overlap_penalty + efficiency_bonus
    
    return reward

def enhanced_reward_v2(info_gain, overlap, fail, dt, max_dist, min_dist):
    """
    方案2: 距离探索奖励
    
    核心思想：
    1. 奖励远距离探索（发现新的自由空间）
    2. 平衡近距离精细扫描和远距离粗略探索
    3. 根据传感器性能调整奖励
    """
    cfg = EnhancedRewardConfig()
    
    # 基础信息奖励
    base_reward = cfg.w_info * info_gain
    
    # 距离探索奖励
    if cfg.distance_reward:
        # 奖励探索到远距离的自由空间
        distance_bonus = cfg.max_distance_bonus * (max_dist / 8.0)  # 假设max_range=8.0
        
        # 如果有命中，额外奖励发现障碍物
        if min_dist > 0:
            obstacle_discovery_bonus = 0.5
        else:
            obstacle_discovery_bonus = 0
    else:
        distance_bonus = 0
        obstacle_discovery_bonus = 0
    
    # 降低重叠惩罚
    overlap_penalty = cfg.w_overlap * overlap
    
    # 总奖励
    reward = base_reward + distance_bonus + obstacle_discovery_bonus - overlap_penalty
    
    return reward

def enhanced_reward_v3(info_gain, overlap, fail, dt, coverage, step_count):
    """
    方案3: 阶段性奖励
    
    核心思想：
    1. 早期重视探索范围，后期重视精细度
    2. 根据当前覆盖率调整奖励策略
    3. 避免早期饱和问题
    """
    cfg = EnhancedRewardConfig()
    
    # 阶段判断
    early_stage = coverage < 0.1  # 覆盖率低于10%为早期
    mid_stage = 0.1 <= coverage < 0.3  # 10%-30%为中期
    late_stage = coverage >= 0.3  # 30%以上为后期
    
    if early_stage:
        # 早期：重视快速覆盖，容忍重叠
        reward = 3.0 * info_gain - 0.5 * overlap
    elif mid_stage:
        # 中期：平衡覆盖和精度
        reward = 2.0 * info_gain - 1.5 * overlap
    else:
        # 后期：重视精细扫描，严格控制重叠
        reward = 1.0 * info_gain - 3.0 * overlap
    
    # 时间效率奖励
    reward += 0.1 / (dt + 0.001)
    
    return reward

def test_enhanced_rewards():
    """测试改进的奖励函数"""
    print("测试改进的奖励函数...")
    
    # 创建测试环境
    cfg = EnvConfig()
    cfg.episode_max_steps = 20
    env = US2DPriorEnv(cfg)
    obs = env.reset()
    
    print(f"环境初始化完成，覆盖率: {env.coverage():.3f}")
    
    prev_coverage = env.coverage()
    
    for step in range(10):
        # 获取可用动作
        mask = env.action_mask()
        available_actions = [i for i in range(len(mask)) if mask[i] == 1]
        
        if not available_actions:
            break
            
        # 随机选择动作
        action = np.random.choice(available_actions)
        
        # 执行动作
        obs, reward_tuple, done, info = env.step(action)
        
        # 解析原始奖励组件
        info_gain, overlap, fail, dt = reward_tuple
        current_coverage = env.coverage()
        coverage_gain = current_coverage - prev_coverage
        
        # 计算不同的奖励函数
        original_reward = 3.0 * info_gain - 3.0 * overlap
        
        enhanced_v1 = enhanced_reward_v1(
            info_gain, overlap, fail, dt, coverage_gain, current_coverage
        )
        
        enhanced_v2 = enhanced_reward_v2(
            info_gain, overlap, fail, dt, 
            info.get('max_dist', 0), info.get('min_dist', 0)
        )
        
        enhanced_v3 = enhanced_reward_v3(
            info_gain, overlap, fail, dt, current_coverage, step
        )
        
        print(f"步骤 {step+1}:")
        print(f"  info_gain: {info_gain:.3f}, overlap: {overlap:.3f}")
        print(f"  coverage: {current_coverage:.3f} (+{coverage_gain:.3f})")
        print(f"  原始奖励: {original_reward:+.3f}")
        print(f"  改进v1:   {enhanced_v1:+.3f}")
        print(f"  改进v2:   {enhanced_v2:+.3f}")
        print(f"  改进v3:   {enhanced_v3:+.3f}")
        print()
        
        prev_coverage = current_coverage
        
        if done:
            break
    
    print("="*60)
    print("奖励函数对比总结:")
    print("原始: 3.0*(info_gain - overlap) -> 后期全为负值")
    print("改进v1: 主要奖励覆盖率增长，降低重叠惩罚")
    print("改进v2: 增加距离探索奖励，鼓励远程探索")
    print("改进v3: 阶段性奖励，早期容忍重叠")

if __name__ == "__main__":
    test_enhanced_rewards()
