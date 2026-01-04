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
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入依赖
import gymnasium as gym
import carb.input
import omni.appwindow
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_mul, quat_conjugate, quat_from_euler_xyz, euler_xyz_from_quat

# 导入你的环境
import PushT_PD.tasks.manager_based
from PushT_PD.tasks.manager_based.pusht_pd.pusht_pd_env_cfg import PushtPdEnvCfg

# =============================================================================
#  辅助函数：计算姿态误差
# =============================================================================
def compute_orientation_error(current_quat, target_quat):
    """计算从 current 到 target 的旋转误差 (Axis-Angle)"""
    quat_error = quat_mul(target_quat, quat_conjugate(current_quat))
    w = quat_error[:, 0]
    v = quat_error[:, 1:4]
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    mask = norm_v > 1e-6
    axis = torch.zeros_like(v)
    axis[mask.squeeze()] = v[mask.squeeze()] / norm_v[mask.squeeze()]
    angle = 2.0 * torch.atan2(norm_v, w.unsqueeze(-1))
    angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
    return axis * angle

# =============================================================================
#  手柄控制器 (LT/RT控制升降, 右摇杆控制旋转)
# =============================================================================
class GamepadInterface:
    def __init__(self):
        self._app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._app_window.get_gamepad(0)
        
        # 调速参数
        self.move_speed = 0.015  # XY平面速度
        self.z_speed = 0.01      # 升降速度 (LT/RT)
        self.rot_speed = 0.05    # 旋转速度 (弧度/帧)
        
        self.gripper_cmd = 0.04 # 默认张开
        self._key_locks = {"A": False, "START": False, "BACK": False, "Y": False}
        
        if self._gamepad is None:
            print("[WARNING] 未检测到手柄！")

    def get_command(self):
        # 返回: delta_xyz, delta_yaw (旋转增量), gripper_cmd, flags
        delta_pos = torch.zeros(3) 
        delta_yaw = 0.0
        flags = {"reset": False, "start_rec": False, "stop_rec": False}
        
        if self._gamepad is None:
            self._gamepad = self._app_window.get_gamepad(0)
            return delta_pos, delta_yaw, self.gripper_cmd, flags

        # --- 1. 左摇杆: 平面 XY 移动 ---
        val_x = -self._get_combined_axis(carb.input.GamepadInput.LEFT_STICK_UP, carb.input.GamepadInput.LEFT_STICK_DOWN)
        delta_pos[0] = val_x * self.move_speed

        val_y = -self._get_combined_axis(carb.input.GamepadInput.LEFT_STICK_LEFT, carb.input.GamepadInput.LEFT_STICK_RIGHT)
        delta_pos[1] = val_y * self.move_speed

        # --- 2. 扳机键: 高度 Z 移动 (LT升, RT降) ---
        lt_val = self._input.get_gamepad_value(self._gamepad, carb.input.GamepadInput.LEFT_TRIGGER)
        rt_val = self._input.get_gamepad_value(self._gamepad, carb.input.GamepadInput.RIGHT_TRIGGER)
        
        # 扳机值通常是 0.0 到 1.0
        # 如果按下 LT，z 增加；按下 RT，z 减少
        z_input = lt_val - rt_val
        # 死区处理 (防止没按到底也有微小数值)
        if abs(z_input) < 0.05: z_input = 0.0
        
        delta_pos[2] = z_input * self.z_speed

        # --- 3. 右摇杆: 旋转 (Yaw) ---
        # 只读取左右方向 (X轴)
        rot_val = -self._get_combined_axis(carb.input.GamepadInput.RIGHT_STICK_LEFT, carb.input.GamepadInput.RIGHT_STICK_RIGHT)
        delta_yaw = rot_val * self.rot_speed

        # --- 4. 按键 ---
        # A: 夹爪切换 (改回正常的一键切换逻辑)
        if self._check_button_toggle(carb.input.GamepadInput.A, "A"):
            # 切换逻辑: 正数变负数，负数变正数
            self.gripper_cmd = -1.0 if self.gripper_cmd > 0.0 else 1.0

        if self._check_button_toggle(carb.input.GamepadInput.MENU2, "START"): flags["start_rec"] = True
        if self._check_button_toggle(carb.input.GamepadInput.MENU1, "BACK"): flags["stop_rec"] = True
        if self._check_button_toggle(carb.input.GamepadInput.Y, "Y"): flags["reset"] = True

        return delta_pos, delta_yaw, self.gripper_cmd, flags

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
    # 1. 配置环境
    env_cfg = PushtPdEnvCfg()
    env_cfg.scene.num_envs = 1
    # 确保动作缩放为 1.0
    env_cfg.actions.joint_pos.scale = 1.0 
    env_cfg.actions.joint_pos.use_default_offset = False
    
    # 这里的夹爪配置使用的是 BinaryJointPositionActionCfg (1维)
    
    env = gym.make("Isaac-PushT-Franka-v0", cfg=env_cfg, render_mode="rgb_array")
    
    # 2. 初始化 IK
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
        ik_params={"lambda_val": 0.1}, 
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=env.unwrapped.device)
    teleop = GamepadInterface()
    
    recorded_episodes = []
    current_episode_buffer = {"actions": [], "rewards": []}
    is_recording = False
    
    # 3. ★★★ 旋转状态变量 ★★★
    # 我们维护一个当前的 Yaw 角度，初始为 0
    current_yaw = 0.0
    
    print("-" * 60)
    print("[INFO] 手柄控制模式 (Pro)")
    print("       左摇杆: XY 平移")
    print("       右摇杆: 旋转 (Yaw)")
    print("       LT / RT: 上升 / 下降 (Z)")
    print("       A 键   : 夹爪抓取/松开")
    print("       START  : 开始录制 | BACK: 停止保存 | Y: 重置")
    print("-" * 60)
    
    # 4. 初始姿态锁定 (不动如山逻辑)
    obs, _ = env.reset()
    ik_controller.reset()
    robot_entity = env.unwrapped.scene["robot"] 
    # 锁定当前关节角度作为零空间目标
    default_joint_pos = robot_entity.data.joint_pos[:, 0:7].clone()
    
    step_count = 0

    while simulation_app.is_running():
        # A. 获取手柄指令
        delta_pos_cpu, delta_yaw, gripper_cmd, flags = teleop.get_command()
        delta_pos = delta_pos_cpu.to(env.unwrapped.device)

        # B. 录制/重置逻辑
        if flags["start_rec"] and not is_recording:
            print(f"\n[REC] 开始录制 Episode {len(recorded_episodes)}")
            is_recording = True
            current_episode_buffer = {"actions": [], "rewards": []}
        
        if flags["stop_rec"] and is_recording:
            print(f"\n[REC] 停止录制. 帧数: {len(current_episode_buffer['actions'])}")
            is_recording = False
            if len(current_episode_buffer['actions']) > 0:
                ep_data = {"actions": np.stack(current_episode_buffer["actions"]), "rewards": np.stack(current_episode_buffer["rewards"])}
                recorded_episodes.append(ep_data)

        if flags["reset"]:
            obs, _ = env.reset()
            ik_controller.reset()
            # 重置后，更新姿态锁定目标，并重置旋转角度
            default_joint_pos = robot_entity.data.joint_pos[:, 0:7].clone()
            ee_idx = robot_entity.find_bodies("panda_hand")[0][0]
            curr_ee_quat = robot_entity.data.body_state_w[:, ee_idx, 3:7]
            
            # 反解出当前的 Yaw (Z轴旋转)
            # euler_xyz_from_quat 返回的是 (Batch, 3) 也就是 roll, pitch, yaw
            _, _, yaw_measured = euler_xyz_from_quat(curr_ee_quat)
            
            # 更新控制变量
            current_yaw = yaw_measured[0].item()
            if is_recording: is_recording = False
            continue

        # =========================================================
        #  C. 核心控制逻辑: 动态姿态计算
        # =========================================================
        
        ee_idx = robot_entity.find_bodies("panda_hand")[0][0]
        ee_current_state = robot_entity.data.body_state_w[:, ee_idx, :7]
        ee_pos_curr = ee_current_state[:, 0:3]
        ee_quat_curr = ee_current_state[:, 3:7]
        
        # 1. 更新目标 Yaw 角度
        current_yaw += delta_yaw
        
        # 2. 构建目标四元数 (垂直向下 + 动态 Yaw)
        # 垂直向下通常对应 Euler(Roll=180, Pitch=0)
        # 我们使用 Isaac Lab 的 quat_from_euler_xyz (输入必须是 tensor)
        roll = torch.tensor([torch.pi], device=env.unwrapped.device) # 180度
        pitch = torch.tensor([0.0], device=env.unwrapped.device)
        yaw = torch.tensor([current_yaw], device=env.unwrapped.device)
        
        # 这是一个 batch 操作，即使只有一个环境
        target_quat = quat_from_euler_xyz(roll, pitch, yaw)
        # 3. 计算姿态纠正指令
        # 这里的 rot_command 包含了 "保持垂直" 和 "执行旋转" 两个意图
        gain_tilt = 8.0  # 垂直保持力 (XY轴)：设大一点，让它像被钉子钉住一样垂直
        gain_spin = 0.6  # 旋转跟随力 (Z轴) ：可以设小一点，让旋转更平滑，或者根据需要调整
        
        # 2. 构建权重向量 (Batch, 3) -> [x_gain, y_gain, z_gain]
        # 注意：这里假设是在世界坐标系下计算误差
        orientation_gains = torch.tensor(
            [[gain_tilt, gain_tilt, gain_spin]], 
            device=env.unwrapped.device
        )
        
        # 3. 计算原始误差
        raw_error = compute_orientation_error(ee_quat_curr, target_quat)
        
        # 4. 逐元素相乘 (Element-wise multiplication)
        rot_command = raw_error * orientation_gains
        
        # 4. 组合 IK 指令
        # [dx, dy, dz] + [rot_x, rot_y, rot_z]
        ik_command = torch.cat([delta_pos.unsqueeze(0), rot_command], dim=-1)
        
        # 5. 设置 IK
        ik_controller.set_command(ik_command, ee_pos=ee_pos_curr, ee_quat=ee_quat_curr)
        
        # 6. 获取数据 & 计算 IK
        full_jacobian = robot_entity.root_physx_view.get_jacobians()
        ee_jacobian = full_jacobian[:, ee_idx, :, 0:7]
        if ee_jacobian.dim() == 2: ee_jacobian = ee_jacobian.unsqueeze(0)
        
        all_joint_pos = robot_entity.data.joint_pos
        arm_joint_pos = all_joint_pos[:, 0:7]

        joint_actions = ik_controller.compute(
            ee_pos=ee_pos_curr, ee_quat=ee_quat_curr,
            jacobian=ee_jacobian, joint_pos=arm_joint_pos
        )
        
        # 7. ★★★ 零空间姿态保持 (不动如山) ★★★
        # (选配：如果机器人仍然有点扭，请保留这段代码)
        # 计算 Null-space 投影
        j = ee_jacobian
        j_t = j.transpose(1, 2)
        eye_6 = torch.eye(6, device=env.unwrapped.device).unsqueeze(0)
        j_pinv = torch.matmul(j_t, torch.inverse(torch.matmul(j, j_t) + (0.01 ** 2) * eye_6))
        
        eye_7 = torch.eye(7, device=env.unwrapped.device).unsqueeze(0)
        null_projector = eye_7 - torch.matmul(j_pinv, j)
        
        posture_error = default_joint_pos - arm_joint_pos
        null_space_step = torch.matmul(null_projector, posture_error.unsqueeze(-1)).squeeze(-1)
        
        dt = env_cfg.sim.dt
        # 叠加 Null-space 修正
        joint_actions_corrected = joint_actions + null_space_step * 10.0 * dt

        # 8. 组合夹爪
        gripper_action = torch.tensor([[gripper_cmd]], device=env.unwrapped.device)
        full_actions = torch.cat([joint_actions_corrected, gripper_action], dim=-1)

        # D. 执行
        next_obs, reward, terminated, truncated, info = env.step(full_actions)
        
        # E. 录制
        if is_recording:
            current_episode_buffer["actions"].append(full_actions.cpu().numpy())
            current_episode_buffer["rewards"].append(reward.cpu().numpy())
            if step_count % 10 == 0: print("●", end="\r")

        obs = next_obs
        step_count += 1
        
        if terminated.any() or truncated.any():
            env.reset()
            ik_controller.reset()
            # 重置逻辑
            default_joint_pos = robot_entity.data.joint_pos[:, 0:7].clone()
            current_yaw = 0.0

    env.close()
    
    # 保存
    if len(recorded_episodes) > 0:
        if not os.path.exists(args_cli.save_path): os.makedirs(args_cli.save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args_cli.save_path}/demos_xbox_{timestamp}.pkl"
        with open(filename, "wb") as f: pickle.dump(recorded_episodes, f)
        print(f"\n[SUCCESS] Saved to {filename}")

if __name__ == "__main__":
    main()