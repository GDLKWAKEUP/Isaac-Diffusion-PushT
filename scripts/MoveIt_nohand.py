import argparse
import torch
import numpy as np
import pickle
import os
import time
from datetime import datetime

# 1. 启动仿真器
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Collect demonstration data using Gamepad")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--save_path", type=str, default="data/demos", help="Path to save data")
parser.add_argument("--teleop_device", type=str, default="gamepad", help="Input device")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入依赖
import gymnasium as gym
import carb.input
import omni.appwindow
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_mul, quat_conjugate

# 导入你的环境
import PushT_PD.tasks.manager_based
from PushT_PD.tasks.manager_based.pusht_pd.pusht_pd_env_cfg import PushtPdEnvCfg
from isaaclab.utils.math import quat_mul, quat_conjugate

# =============================================================================
#  手柄控制器 (修改：B键触发复位)
# =============================================================================
def compute_orientation_error(current_quat, target_quat):
    """
    计算从 current 到 target 的旋转误差 (Axis-Angle 格式)。
    返回: [rot_x, rot_y, rot_z] 表示需要转动的轴和角度
    """
    # 1. 计算差异四元数: q_diff = q_target * q_current^(-1)
    # 这是计算 "还需要转多少才能到目标"
    quat_error = quat_mul(target_quat, quat_conjugate(current_quat))
    
    # 2. 提取实部(w)和虚部(v=xyz)
    w = quat_error[:, 0]
    v = quat_error[:, 1:4]
    
    # 3. 转换为轴角 (Axis-Angle)
    # 公式: angle = 2 * atan2(||v||, w)
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    
    # 避免除以 0
    mask = norm_v > 1e-6
    axis = torch.zeros_like(v)
    axis[mask.squeeze()] = v[mask.squeeze()] / norm_v[mask.squeeze()]
    
    angle = 2.0 * torch.atan2(norm_v, w.unsqueeze(-1))
    
    # 将角度限制在 [-pi, pi] 之间，走最短路径
    angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
    
    # 结果 = 轴 * 角 (这就是 IK 需要的旋转速度指令)
    return axis * angle

class GamepadInterface:
    def __init__(self):
        self._app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._app_window.get_gamepad(0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        
        self.move_speed = 0.015 
        self.gripper_cmd = 0.04 
        # B 键也需要防抖锁
        self._key_locks = {"A": False, "B": False, "START": False, "BACK": False, "Y": False}
        
        if self._gamepad is None:
            print("[WARNING] 未检测到手柄！")

    def get_command(self):
        delta_pos = torch.zeros(3) 
        flags = {"reset": False, "start_rec": False, "stop_rec": False, "go_home": False}
        
        if self._gamepad is None:
            self._gamepad = self._app_window.get_gamepad(0)
            return delta_pos, self.gripper_cmd, flags

        # --- 1. 摇杆控制 ---
        val_x = -self._get_combined_axis(carb.input.GamepadInput.LEFT_STICK_UP, carb.input.GamepadInput.LEFT_STICK_DOWN)
        delta_pos[0] = val_x * self.move_speed

        val_y = -self._get_combined_axis(carb.input.GamepadInput.LEFT_STICK_LEFT, carb.input.GamepadInput.LEFT_STICK_RIGHT)
        delta_pos[1] = val_y * self.move_speed

        val_z = -self._get_combined_axis(carb.input.GamepadInput.RIGHT_STICK_UP, carb.input.GamepadInput.RIGHT_STICK_DOWN)
        delta_pos[2] = val_z * self.move_speed

        # --- 2. 按键控制 ---
        # A: 夹爪切换
        if self._check_button_toggle(carb.input.GamepadInput.A, "A"):
            self.gripper_cmd = -1.0 if self.gripper_cmd > 0.0 else 1.0

        # ★★★ 新增：B 键触发“自动回位” ★★★
        if self._check_button_toggle(carb.input.GamepadInput.B, "B"):
            flags["go_home"] = True

        if self._check_button_toggle(carb.input.GamepadInput.MENU2, "START"): flags["start_rec"] = True
        if self._check_button_toggle(carb.input.GamepadInput.MENU1, "BACK"): flags["stop_rec"] = True
        if self._check_button_toggle(carb.input.GamepadInput.Y, "Y"): flags["reset"] = True

        return delta_pos, self.gripper_cmd, flags

    def _get_combined_axis(self, pos_key, neg_key):
        val = self._input.get_gamepad_value(self._gamepad, pos_key) - self._input.get_gamepad_value(self._gamepad, neg_key)
        return val if abs(val) > 0.1 else 0.0

    def _is_pressed(self, btn): return self._input.get_gamepad_value(self._gamepad, btn) > 0.5
    
    def _check_button_toggle(self, btn, name):
        is_down = self._is_pressed(btn)
        if is_down and not self._key_locks.get(name, False):
            self._key_locks[name] = True
            return True
        elif not is_down:
            self._key_locks[name] = False
        return False

# =============================================================================
#  主程序
# =============================================================================
def main():
    # --- 配置区域 ---
    # ★★★ 接口：在这里调整自动移动的速度 ★★★
    # 0.01 = 很慢, 0.05 = 中等, 0.1 = 快
    homing_speed_scale = 0.02 
    
    # 1. 环境初始化
    env_cfg = PushtPdEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.actions.joint_pos.scale = 1.0 
    env_cfg.actions.joint_pos.use_default_offset = False
    
    env = gym.make("Isaac-PushT-Franka-v0", cfg=env_cfg, render_mode="rgb_array")
    
    # 2. IK 初始化
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls",
        ik_params={"lambda_val": 0.1}, 
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=env.unwrapped.device)
    teleop = GamepadInterface()
    
    # 数据变量
    recorded_episodes = []
    current_episode_buffer = {"actions": [], "rewards": []}
    is_recording = False
    
    obs, _ = env.reset()
    ik_controller.reset()
    robot_entity = env.unwrapped.scene["robot"] 
    
    # 3. ★★★ 预定位置设定 ★★★
    # 我们把脚本启动时的姿态作为“预定位置”
    # 你也可以在这里手写一个固定的姿态 tensor
    preset_joint_target = robot_entity.data.joint_pos[:, 0:7].clone()
    print(f"[INFO] 预定位置已设定。")
    print(f"[INFO] 按 'B' 键让机械臂缓慢回到预定位置。")
    
    # 状态标记
    is_homing = False # 是否正在自动移动中
    target_quat_down = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.unwrapped.device)
    
    # 姿态纠正的力度 (Gain)，越大纠正越快，但也容易震荡
    orientation_gain = 5.0 
    while simulation_app.is_running():
        # A. 获取手柄指令
        delta_pos_cpu, gripper_cmd, flags = teleop.get_command()
        delta_pos = delta_pos_cpu.to(env.unwrapped.device)

        # B. 处理 B 键逻辑 (切换自动移动模式)
        if flags["go_home"]:
            is_homing = not is_homing # 再次按B可以取消
            if is_homing:
                print(">>> 启动自动归位... (按B或动摇杆取消)")
            else:
                print(">>> 取消自动归位，切回手动控制。")

        # 检测用户是否在手动乱动摇杆 (如果动了，就打断自动归位)
        if torch.sum(torch.abs(delta_pos)) > 0:
            if is_homing:
                is_homing = False
                print(">>> 检测到手动输入，自动归位已中断。")

        # ... (录制/重置逻辑保持不变) ...
        if flags["start_rec"] and not is_recording:
            is_recording = True
            current_episode_buffer = {"actions": [], "rewards": []}
            print("\n[REC] 开始录制")
        if flags["stop_rec"] and is_recording:
            is_recording = False
            recorded_episodes.append({"actions": np.stack(current_episode_buffer["actions"]), "rewards": np.stack(current_episode_buffer["rewards"])})
            print("\n[REC] 停止录制")
        if flags["reset"]:
            obs, _ = env.reset()
            ik_controller.reset()
            # 重置后，更新预定位置为新的出生位置 (或者保持不变，看你需求)
            preset_joint_target = robot_entity.data.joint_pos[:, 0:7].clone()
            is_homing = False
            continue

        # =========================================================
        #  C. 核心控制逻辑 (分支：手动 vs 自动)
        # =========================================================
        
        # 获取当前信息
        ee_idx = robot_entity.find_bodies("panda_hand")[0][0]
        ee_current_state = robot_entity.data.body_state_w[:, ee_idx, :7]
        ee_pos_curr = ee_current_state[:, 0:3]
        ee_quat_curr = ee_current_state[:, 3:7]
        all_joint_pos = robot_entity.data.joint_pos
        arm_joint_pos = all_joint_pos[:, 0:7]
        
        # 定义最终发给机器人的 7轴 动作
        final_joint_actions = None

        if is_homing:
            # --- 自动模式 (线性插值移动) ---
            
            # 1. 计算误差: 目标 - 当前
            error = preset_joint_target - arm_joint_pos
            
            # 2. 计算这一帧的步长
            # 限制最大速度为 homing_speed_scale
            # torch.clamp 限制每一步最多走这么多
            step = torch.clamp(error, min=-homing_speed_scale, max=homing_speed_scale)
            
            # 3. 计算下一步的目标
            target_pos_next = arm_joint_pos + step
            
            # 4. 赋值
            final_joint_actions = target_pos_next
            
            # 5. 检查是否到达 (如果误差非常小，就停止 homing)
            if torch.max(torch.abs(error)) < 0.01:
                print(">>> 已到达预定位置。")
                is_homing = False
                # 同步 IK 控制器的内部状态，防止切回手动时跳变
                ik_controller.reset() 
                
        else:
            # --- 手动模式 (IK 控制 + 零空间保持) ---
            
            # 1. 设置 IK 目标
            # 使用简单的姿态纠正 (让夹爪垂直向下)
            # 这里我简化了之前复杂的零空间计算，只保留核心逻辑
            
            # 1. 计算【旋转纠正指令】
            # 我们忽略手柄的旋转输入，强制计算 "回到垂直向下" 的指令
            rot_command = compute_orientation_error(ee_quat_curr, target_quat_down)
            
            # 乘以增益系数，让纠正更有力
            rot_command = rot_command * orientation_gain
            
            # 2. 组合最终 IK 指令
            # delta_pos (来自手柄的位移) + rot_command (强制姿态纠正)
            # delta_pos 需要 unsqueeze 变成 (1, 3)
            ik_command = torch.cat([delta_pos.unsqueeze(0), rot_command], dim=-1)
            
            # 3. 设置 IK 目标
            # 注意：在 pose_rel 模式下，command 前3位是位移增量，后3位是旋转向量增量
            ik_controller.set_command(
                ik_command, 
                ee_pos=ee_pos_curr, 
                ee_quat=ee_quat_curr
            )
            
            # 2. 获取 Jacobian
            full_jacobian = robot_entity.root_physx_view.get_jacobians()
            ee_jacobian = full_jacobian[:, ee_idx, :, 0:7]
            if ee_jacobian.dim() == 2: ee_jacobian = ee_jacobian.unsqueeze(0)

            # 3. IK 计算
            ik_actions = ik_controller.compute(
                ee_pos=ee_pos_curr, ee_quat=ee_quat_curr,
                jacobian=ee_jacobian, joint_pos=arm_joint_pos
            )
            
            # 4. 简单的零空间纠正 (为了让它不动如山)
            # 往 preset_joint_target 拉
            posture_error = preset_joint_target - arm_joint_pos
            
            # 简单投影: actions = ik + gain * error * dt
            # (这里省略了复杂的投影矩阵，因为之前的代码太长，只要增益不大，直接加也没问题)
            # 为了严谨，建议还是用之前那个完整的投影代码。
            # 这里为了简洁演示“缓慢移动”，我只用 IK 输出。
            # 如果你需要那个不动如山，把之前的 Null-space 代码块贴在这里覆盖 ik_actions
            
            final_joint_actions = ik_actions

        # 8. 组合夹爪
        gripper_action = torch.tensor([[gripper_cmd]], device=env.unwrapped.device)
        full_actions = torch.cat([ik_actions, gripper_action], dim=-1)

        # D. 执行
        next_obs, reward, terminated, truncated, info = env.step(full_actions)
        
        # E. 录制
        if is_recording:
            current_episode_buffer["actions"].append(full_actions.cpu().numpy())
            current_episode_buffer["rewards"].append(reward.cpu().numpy())

        if terminated.any() or truncated.any():
            env.reset()
            ik_controller.reset()
            preset_joint_target = robot_entity.data.joint_pos[:, 0:7].clone() # 重置后更新目标

    env.close()
    # (保存代码省略)

if __name__ == "__main__":
    main()