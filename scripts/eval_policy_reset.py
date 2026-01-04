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

# 1. 启动仿真器 (开启相机)
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy with Manual Reset (F Key)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .ckpt file")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入依赖
import gymnasium as gym
import carb.input
import omni.appwindow

import PushT_PD.tasks.manager_based
from PushT_PD.tasks.manager_based.pusht_pd.pusht_pd_env_cfg import PushtPdEnvCfg

# 导入 Diffusion Policy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# =============================================================================
#  辅助函数
# =============================================================================
def transform_image(img_numpy, target_shape=(96, 96)):
    """图像预处理: Resize -> Normalize -> CHW"""
    if img_numpy.shape[:2] != target_shape:
        img_numpy = cv2.resize(img_numpy, target_shape, interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy(img_numpy).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    return img_tensor

def is_f_key_pressed():
    """检测 F 键是否被按下"""
    app_window = omni.appwindow.get_default_app_window()
    if not app_window: return False
    keyboard = app_window.get_keyboard()
    if not keyboard: return False
    
    input_interface = carb.input.acquire_input_interface()
    # 检测 F 键 (1.0 表示按下)
    if input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.K) > 0:
        return True
    return False

# =============================================================================
#  主逻辑
# =============================================================================
def main():
    # --- A. 加载模型 ---
    print(f"[INFO] Loading model: {args_cli.checkpoint}")
    payload = torch.load(open(args_cli.checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema: policy = workspace.ema_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    print(f"[INFO] Model Ready. Action Horizon: {n_action_steps}")

    # --- B. 创建环境 ---
    env_cfg = PushtPdEnvCfg()
    env_cfg.scene.num_envs = 1
    # 强制动作缩放为 1.0 (必须与训练数据一致)
    env_cfg.actions.joint_pos.scale = 1.0
    env_cfg.actions.joint_pos.use_default_offset = False
    
    env = gym.make("Isaac-PushT-Franka-v0", cfg=env_cfg, render_mode="rgb_array")
    
    robot_entity = env.unwrapped.scene["robot"]
    cam_global = env.unwrapped.scene["camera_global"]
    cam_wrist = env.unwrapped.scene["camera_wrist"]

    # --- C. 初始化 ---
    obs, _ = env.reset()
    
    # 观测历史队列
    obs_deque = collections.deque([None] * n_obs_steps, maxlen=n_obs_steps)
    
    print("-" * 50)
    print("[INFO] 开始推理。")
    print("       按 'K' 键 -> 重置环境 (机械臂归位 + 方块随机刷新)")
    print("       按 'Ctrl+C' -> 退出")
    print("-" * 50)

    # 标记：是否刚刚重置过 (防止长按F导致疯狂重置)
    just_reset = False

    while simulation_app.is_running():
        # --- 1. 检测重置按键 ---
        if is_f_key_pressed():
            if not just_reset:
                print(">>> 检测到 F 键：正在重置环境...")
                
                # ★★★ 核心：执行重置 ★★★
                # 这会触发 EventCfg 里的 reset_robot (归位) 和 reset_object (随机刷新)
                env.reset()
                
                # ★★★ 关键：清空历史记忆 ★★★
                # 否则模型会看到一张"上一局"的图和"这一局"的图，导致精神分裂
                obs_deque.clear()
                obs_deque.extend([None] * n_obs_steps)
                
                just_reset = True
                
                # 跳过本次循环，等待下一帧获取新图像
                continue 
        else:
            just_reset = False # 松开按键后，允许下一次重置

        # --- 2. 获取观测 ---
        img_g = cam_global.data.output["rgb"][0].cpu().numpy()
        img_w = cam_wrist.data.output["rgb"][0].cpu().numpy()
        
        # 获取状态 (8维: 7臂+1爪)
        curr_joints = robot_entity.data.joint_pos[0]
        arm_joints = curr_joints[0:7]
        gripper_width = curr_joints[7:9].sum().unsqueeze(0)
        agent_pos = torch.cat([arm_joints, gripper_width]).float()

        # 预处理
        img_g_tensor = transform_image(img_g)
        img_w_tensor = transform_image(img_w)
        
        # 存入队列
        obs_dict = {
            'img_global': img_g_tensor,
            'img_wrist': img_w_tensor,
            'agent_pos': agent_pos # 确保 key 和训练时一致
        }
        
        if obs_deque[0] is None:
            for _ in range(n_obs_steps): obs_deque.append(obs_dict)
        else:
            obs_deque.append(obs_dict)

        # --- 3. 模型推理 ---
        batch_obs = collections.defaultdict(list)
        for x in obs_deque:
            for k, v in x.items(): batch_obs[k].append(v)
        
        for k, v in batch_obs.items():
            batch_obs[k] = torch.stack(v).unsqueeze(0).to(device)

        with torch.no_grad():
            action_pred = policy.predict_action(batch_obs)['action'][0]

        # --- 4. 执行动作序列 (Closed-loop) ---
        # 这是一个小循环，执行预测出的前几步
        # 在执行期间，我们也要检查 F 键，以便能随时打断
        
        for i in range(n_action_steps):
            # ★ 随时响应重置 ★
            if is_f_key_pressed() and not just_reset:
                break # 跳出动作循环，回到主循环触发重置
            
            # 获取动作 (8维)
            action = action_pred[i]
            
            # 8维 -> 9维 (复制夹爪信号)
            # arm_action = action[:7]
            # gripper_val = action[7]
            # full_action = torch.cat([arm_action, gripper_val.unsqueeze(0), gripper_val.unsqueeze(0)])
            full_action = action
            # 发送给环境
            env.step(full_action.unsqueeze(0).to(env.unwrapped.device))
            
            # (可选) 检查是否成功/超时
            # if done or truncated: break

    env.close()

if __name__ == "__main__":
    main()