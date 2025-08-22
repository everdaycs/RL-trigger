# 保守自由空间清理策略的奖励函数调整建议

## 🔍 问题诊断

### 当前奖励函数问题
```python
# 当前: r = 3.0*(info_gain - overlap) 
# 问题: 保守策略下，后期info_gain→0, overlap→100%，奖励恒为-3.0
```

从测试数据看到：
- **第6步后**：info_gain=0, overlap=100% → 奖励=-3.0
- **覆盖率停滞**：6.0%左右无法继续探索  
- **学习信号丢失**：所有动作得到相同负奖励

## 💡 推荐方案：混合覆盖率导向奖励

### 核心设计原则
1. **覆盖率增长为主**：直接奖励地图覆盖率提升
2. **自适应重叠惩罚**：低重叠率时不惩罚，高重叠率时才惩罚
3. **探索效率奖励**：鼓励快速有效的探索
4. **阶段性调整**：根据探索进度调整奖励策略

### 具体实现

```python
@dataclass
class RewardConfig:
    # 推荐配置
    w_coverage: float = 10.0      # 覆盖率增长主要奖励
    w_info: float = 1.0           # 信息增益辅助奖励
    w_overlap_base: float = 1.0   # 基础重叠惩罚
    overlap_threshold: float = 0.6 # 重叠率容忍阈值
    w_efficiency: float = 0.5     # 时间效率奖励
    
def enhanced_reward(info_gain, overlap, fail, dt, coverage_gain, current_coverage):
    """改进的奖励函数"""
    
    # 1. 主要奖励：覆盖率增长
    coverage_reward = 10.0 * coverage_gain
    
    # 2. 辅助奖励：信息增益
    info_reward = 1.0 * info_gain
    
    # 3. 自适应重叠惩罚
    if overlap <= 0.6:
        overlap_penalty = 0  # 容忍合理重叠
    else:
        overlap_penalty = 1.0 * (overlap - 0.6) / 0.4  # 超过阈值才惩罚
    
    # 4. 探索效率奖励
    efficiency_bonus = 0.5 / (dt + 0.001)
    
    # 5. 阶段性调整
    if current_coverage < 0.1:
        # 早期：鼓励快速探索
        reward = coverage_reward * 1.5 + info_reward - overlap_penalty * 0.5
    elif current_coverage < 0.3:
        # 中期：平衡探索和精度
        reward = coverage_reward + info_reward - overlap_penalty
    else:
        # 后期：重视精细扫描
        reward = coverage_reward * 0.8 + info_reward * 1.2 - overlap_penalty * 1.5
    
    # 添加效率奖励
    reward += efficiency_bonus
    
    return reward
```

### 与现有代码集成

修改 `train_trigger_ppo.py` 中的奖励计算：

```python
# 在环境step后，添加覆盖率跟踪
prev_coverage = getattr(env, '_prev_coverage', 0.0)  # 记录上一步覆盖率
current_coverage = env.coverage()
coverage_gain = max(0.0, current_coverage - prev_coverage)
env._prev_coverage = current_coverage

# 使用改进的奖励函数
r_np = enhanced_reward(
    info_gain, overlap, fail, dt, 
    coverage_gain, current_coverage
)
```

## 📊 预期效果

### 奖励信号对比
| 场景 | 原始奖励 | 改进奖励 | 效果 |
|------|----------|----------|------|
| 探索新区域 | +1.5 | +10.5 | 强烈正向激励 |
| 低重叠重扫 | -1.5 | +8.0 | 允许合理重复 |
| 高重叠重扫 | -3.0 | +5.0 | 仍有正向但较低 |
| 纯重复扫描 | -3.0 | +2.0 | 避免负奖励陷阱 |

### 学习动态改善
1. **早期**：快速扩展覆盖范围，建立地图框架
2. **中期**：平衡新区域探索和已知区域细化
3. **后期**：重视扫描质量，提高地图精度

## 🔧 实施建议

### 立即实施
1. **修改TrainConfig**：调整奖励权重
2. **更新奖励计算**：集成覆盖率跟踪
3. **重新训练**：使用新奖励函数训练模型

### 进一步优化
1. **动态权重**：根据训练进度自动调整权重
2. **多目标奖励**：结合覆盖率、精度、效率多个指标
3. **课程学习**：从简单环境逐步过渡到复杂环境

这套改进方案专门针对保守自由空间清理策略设计，可以有效解决当前的奖励饱和和学习停滞问题。
