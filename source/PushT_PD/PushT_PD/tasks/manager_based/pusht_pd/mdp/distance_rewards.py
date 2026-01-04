# source/PushT_PD/PushT_PD/tasks/manager_based/pusht_pd/mdp/rewards.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_goal_distance(
    env: ManagerBasedRLEnv, 
    std: float, 
    minimal_distance: float, 
    target_cfg: SceneEntityCfg, 
    subject_cfg: SceneEntityCfg
) -> torch.Tensor:
    """计算被推物体和目标之间的欧氏距离。
    
    返回的是距离值 (scalar)。如果作为惩罚项(Cost)，Config里的权重 weight 应该是负数。
    """
    # 1. 获取目标 (Goal) 的位置
    # target_cfg 指向我们在 Scene 里定义的 goal_object
    target_pos = env.scene[target_cfg.name].data.root_pos_w

    # 2. 获取物体 (Subject) 的位置
    # subject_cfg 指向 t_object
    subject_pos = env.scene[subject_cfg.name].data.root_pos_w

    # 3. 计算距离 (欧氏距离 L2 Norm)
    # dim=-1 表示沿着坐标(x,y,z)维度计算
    distance = torch.norm(target_pos - subject_pos, dim=-1)

    # (可选) 你可以在这里加上 std 归一化或者 tanh，
    # 但最简单的做法是直接返回距离，由 Config 里的 weight 控制大小
    return distance