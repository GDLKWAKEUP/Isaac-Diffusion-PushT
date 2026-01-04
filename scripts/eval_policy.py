import argparse
import os
import sys
sys.path.append("/home/jiji/workspace/isaac/project/PushT_PD/diffusion_policy-main")
import numpy as np

import torch
import cv2
import collections
import hydra
import dill
import time

# 1. 启动仿真器 (必须最先执行)
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy (8-Dim Action)")
parser.add_argument("--checkpoint", type=str, default="/home/jiji/workspace/isaac/project/PushT_PD/diffusion_policy-main/data/outputs/2026.01.04/11.46.19_train_diffusion_unet_hybrid_isaac_pusht/checkpoints/epoch=150-train_loss=0.012.ckpt", required=True, help="Path to the .ckpt file")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
# 强制开启相机渲染
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入 Isaac Lab 依赖
import gymnasium as gym
import PushT_PD.tasks.manager_based
from PushT_PD.tasks.manager_based.pusht_pd.pusht_pd_env_cfg import PushtPdEnvCfg

# 3. 导入 Diffusion Policy 依赖
# (确保你已经 export PYTHONPATH=$PYTHONPATH:/path/to/diffusion_policy)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# =============================================================================
#  辅助函数
# =============================================================================
def transform_image(img_numpy, target_shape=(96, 96)):
    """
    预处理图像：缩放 -> 归一化 -> CHW
    输入: (H, W, C) uint8
    输出: (C, H, W) float32
    """
    if img_numpy.shape[:2] != target_shape:
        img_numpy = cv2.resize(img_numpy, target_shape, interpolation=cv2.INTER_AREA)
    
    img_tensor = torch.from_numpy(img_numpy).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1) # HWC -> CHW
    return img_tensor

# =============================================================================
#  主逻辑
# =============================================================================
def main():
    # --- A. 加载模型 Checkpoint ---
    print(f"[INFO] Loading model from: {args_cli.checkpoint}")
    
    # 使用 dill 加载 pickle 文件 (处理 lambda 等复杂对象)
    payload = torch.load(open(args_cli.checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # 实例化 Workspace 并加载权重
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 获取 Policy 模型
    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    
    # 获取时间步参数
    n_obs_steps = cfg.n_obs_steps     # 观察窗口 (例如 2)
    n_action_steps = cfg.n_action_steps # 执行步数 (例如 8)
    print(f"[INFO] Model ready. Obs steps: {n_obs_steps}, Action steps: {n_action_steps}")

    # --- B. 创建 Isaac Lab 环境 ---
    env_cfg = PushtPdEnvCfg()
    env_cfg.scene.num_envs = 1
    
    # ★★★ 关键：动作缩放必须为 1.0 ★★★
    # 因为 Diffusion Policy 输出的是我们在录制时记录的真实物理值
    env_cfg.actions.joint_pos.scale = 1.0
    env_cfg.actions.joint_pos.use_default_offset = False
    
    # 如果你的环境配置里夹爪是 Binary (1维)，不需要改
    # 如果是 JointPos (2维)，我们在下面代码里手动处理维度扩展
    
    env = gym.make("Isaac-PushT-Franka-v0", cfg=env_cfg, render_mode="rgb_array")
    
    # 获取句柄
    robot_entity = env.unwrapped.scene["robot"]
    cam_global = env.unwrapped.scene["camera_global"]
    cam_wrist = env.unwrapped.scene["camera_wrist"]

    # --- C. 初始化推理循环 ---
    obs, _ = env.reset()
    
    # 观测历史队列 (用于构建时序输入)
    obs_deque = collections.deque([None] * n_obs_steps, maxlen=n_obs_steps)
    
    print("[INFO] Starting inference loop... Press Ctrl+C to stop.")

    while simulation_app.is_running():
        # 1. --- 获取当前观测 (Observation) ---
        # 图像 (H, W, 3)
        img_g_raw = cam_global.data.output["rgb"][0].cpu().numpy()
        img_w_raw = cam_wrist.data.output["rgb"][0].cpu().numpy()
        
        # 状态 (8维): 7关节 + 1夹爪宽度
        # robot.data.joint_pos 形状是 (Num_Envs, 9)
        current_joints = robot_entity.data.joint_pos[0]
        arm_joints = current_joints[0:7] # 前7个
        gripper_width = current_joints[7:9].sum().unsqueeze(0) # 后2个求和 = 宽度
        
        # 拼成 8 维向量 (必须和训练时的 state 结构一致)
        agent_pos = torch.cat([arm_joints, gripper_width]).float()

        # 2. --- 预处理 ---
        img_g_tensor = transform_image(img_g_raw)
        img_w_tensor = transform_image(img_w_raw)
        
        # 3. --- 放入队列 ---
        # 字典 Key 必须和训练配置 yaml 里的 shape_meta 一致
        obs_dict = {
            'img_global': img_g_tensor,
            'img_wrist': img_w_tensor,
            # ★★★ 修复：改成 agent_pos 以匹配训练配置 ★★★
            'agent_pos': agent_pos 
        }
        
        # 如果队列为空(刚重置)，复制第一帧填满队列
        if obs_deque[0] is None:
            for _ in range(n_obs_steps):
                obs_deque.append(obs_dict)
        else:
            obs_deque.append(obs_dict)
            
        # 4. --- 拼凑 Batch ---
        # 目标形状: {key: (Batch=1, T_obs, ...)}
        batch_obs = collections.defaultdict(list)
        for x in obs_deque:
            for k, v in x.items():
                batch_obs[k].append(v)
        
        for k, v in batch_obs.items():
            batch_obs[k] = torch.stack(v).unsqueeze(0).to(device)

        # 5. --- 模型推理 ---
        with torch.no_grad():
            result = policy.predict_action(batch_obs)
            
        # ★★★ 修复：先通过 key 取出 action，再取 batch 0 ★★★
        action_pred = result['action'][0] # (T_pred, 8)

        # 6. --- 执行动作序列 (Closed-loop Action Execution) ---
        # 我们只执行前 n_action_steps 步
        
        for i in range(n_action_steps):
            # 获取当前步动作 (8维): [Arm_1...Arm_7, Gripper]
            action_step = action_pred[i]
            
            # --- ★★★ 核心：处理 8维 -> 9维 的转换 ★★★ ---
            arm_action = action_step[:7] # 前7个是机械臂
            gripper_action_val = action_step[7] # 第8个是夹爪
            
            # 这里的处理取决于你的 PushtPdEnvCfg 里的 ActionsCfg
            # 情况 A: 如果夹爪配置是 BinaryJointPositionAction (1维)
            full_action = action_step 
            
            # 情况 B: 如果夹爪配置是 JointPositionAction (2维，控制两个手指)
            # 我们需要把这 1 个值复制成 2 个
            # full_action = torch.cat([
            #     arm_action, 
            #     gripper_action_val.unsqueeze(0), 
            #     gripper_action_val.unsqueeze(0)
            # ])
            # 结果 full_action 是 9 维
            
            # 加上 Batch 维度并转到 GPU
            full_action = full_action.unsqueeze(0).to(env.unwrapped.device)
            
            # Step 环境
            env.step(full_action)
            
            # 检查结束条件
            # 注意：在执行动作序列的中间，我们通常不检查 done，或者检查了就直接 break
            # 这里简单处理：如果物体掉落或超时，直接重置
            # 为了获取 done 状态，我们需要 peek 一下 (但 step 并没有返回 done 这里是 Isaac Lab 特性)
            # 在 Isaac Lab env.step 会自动处理 reset，这里我们只做循环
            
            # (可选) 稍微 sleep 一下如果是实机，但在仿真里不需要
    
    env.close()

if __name__ == "__main__":
    main()