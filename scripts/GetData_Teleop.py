import argparse
import torch
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# =============================================================================
# 1. 启动仿真器 (Isaac Lab 2.x 核心修改: 包名从 omni.isaac.lab 变为 isaaclab)
# =============================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect demonstration data using Gamepad")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument(
    "--save_path", 
    type=str, 
    default="/home/jiji/workspace/isaac/project/PushT_PD/data/demos", 
    help="Path to save data"
)
parser.add_argument("--teleop_device", type=str, default="gamepad", help="Input device")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
record_interval = 5 
# =============================================================================
# 2. 导入依赖 (注意所有 import 都已更新为 isaaclab)
# =============================================================================
import gymnasium as gym
import carb.input
import omni.appwindow

# 控制器
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
# 数学工具
from isaaclab.utils.math import quat_mul, quat_conjugate, quat_from_euler_xyz, euler_xyz_from_quat
# 配置工具
from isaaclab.utils import configclass

# --- ★★★ 传感器配置 (基于 Isaac Lab 2.2.0) ★★★ ---
# CameraCfg 在 isaaclab.sensors 中
from isaaclab.sensors import CameraCfg, Camera
# PinholeCameraCfg 通常在 isaaclab.sim.spawners.sensors 中
from isaaclab.sim.spawners.sensors import PinholeCameraCfg

# 导入你的环境 (保持不变)
import PushT_PD.tasks.manager_based
from PushT_PD.tasks.manager_based.pusht_pd.pusht_pd_env_cfg import PushtPdEnvCfg

# =============================================================================
#  辅助函数 & 手柄类 (逻辑保持不变)
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

class GamepadInterface:
    def __init__(self):
        self._app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._app_window.get_gamepad(0)
        
        self.move_speed = 0.015
        self.z_speed = 0.01
        self.rot_speed = 0.05
        
        self.gripper_cmd = 0.04
        self._key_locks = {"A": False, "START": False, "BACK": False, "Y": False}
        
        if self._gamepad is None:
            print("[WARNING] 未检测到手柄！")

    def get_command(self):
        delta_pos = torch.zeros(3) 
        delta_yaw = 0.0
        flags = {"reset": False, "start_rec": False, "stop_rec": False}
        
        if self._gamepad is None:
            self._gamepad = self._app_window.get_gamepad(0)
            return delta_pos, delta_yaw, self.gripper_cmd, flags

        val_x = -self._get_combined_axis(carb.input.GamepadInput.LEFT_STICK_UP, carb.input.GamepadInput.LEFT_STICK_DOWN)
        delta_pos[0] = val_x * self.move_speed

        val_y = -self._get_combined_axis(carb.input.GamepadInput.LEFT_STICK_LEFT, carb.input.GamepadInput.LEFT_STICK_RIGHT)
        delta_pos[1] = val_y * self.move_speed

        lt_val = self._input.get_gamepad_value(self._gamepad, carb.input.GamepadInput.LEFT_TRIGGER)
        rt_val = self._input.get_gamepad_value(self._gamepad, carb.input.GamepadInput.RIGHT_TRIGGER)
        z_input = lt_val - rt_val
        if abs(z_input) < 0.05: z_input = 0.0
        delta_pos[2] = z_input * self.z_speed

        rot_val = -self._get_combined_axis(carb.input.GamepadInput.RIGHT_STICK_LEFT, carb.input.GamepadInput.RIGHT_STICK_RIGHT)
        delta_yaw = rot_val * self.rot_speed

        if self._check_button_toggle(carb.input.GamepadInput.A, "A"):
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
    env_cfg.actions.joint_pos.scale = 1.0 
    env_cfg.actions.joint_pos.use_default_offset = False
    
    # --- ★★★ 修改：使用 Isaac Lab 2.2.0 Sensor API ★★★ ---
    CAM_RES = (96, 96) 
    
    # # Global Camera (第三人称)
    # # CameraCfg 需要 spawn 参数，传入 PinholeCameraCfg
    # env_cfg.scene.camera_global = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/CameraGlobal",
    #     update_period=0,
    #     height=CAM_RES[0], width=CAM_RES[1],
    #     data_types=["rgb"],
    #     spawn=PinholeCameraCfg(
    #         focal_length=24.0, 
    #         focus_distance=400.0, 
    #         horizontal_aperture=20.955
    #     ),
    # )

    # # Wrist Camera (手眼)
    # env_cfg.scene.camera_wrist = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_hand/camera_wrist", 
    #     update_period=0,
    #     height=CAM_RES[0], width=CAM_RES[1],
    #     data_types=["rgb"],
    #     spawn=PinholeCameraCfg(
    #         focal_length=18.0, 
    #         focus_distance=400.0, 
    #         horizontal_aperture=20.955
    #     ),
    # )
    
    # 2. 启动环境
    env = gym.make("Isaac-PushT-Franka-v0", cfg=env_cfg, render_mode="rgb_array")
    
    # 3. 初始化控制器
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
        ik_params={"lambda_val": 0.1}, 
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=env.unwrapped.device)
    teleop = GamepadInterface()
    
    recorded_episodes = []
    
    # 数据Buffer
    current_episode_buffer = {
        "actions": [], 
        "rewards": [],
        "obs": {
            "image_global": [],
            "image_wrist": [],
            "joint_pos": [],
            "ee_pos": [],
            "ee_quat": [],
            "gripper": []
        }
    }
    
    is_recording = False
    current_yaw = 0.0
    
# 我们按时间戳创建一个文件夹，里面存放 episode_0.pkl, episode_1.pkl ...
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args_cli.save_path, f"record_{timestamp}")
    
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
        
    print(f"[INFO] 本次数据将实时保存至: {session_dir}")
    
    episode_idx = 0 # 用于给文件编号
    
    obs, _ = env.reset()
    ik_controller.reset()
    robot_entity = env.unwrapped.scene["robot"] 
    default_joint_pos = robot_entity.data.joint_pos[:, 0:7].clone()
    
    step_count = 0

    while simulation_app.is_running():
        # A. 获取指令
        delta_pos_cpu, delta_yaw, gripper_cmd, flags = teleop.get_command()
        delta_pos = delta_pos_cpu.to(env.unwrapped.device)

        # B. 录制状态机
        if flags["start_rec"] and not is_recording:
            print(f"\n[REC] 开始录制 Episode {len(recorded_episodes)}")
            is_recording = True
            
            # ★★★ 修复 2：在这里直接初始化完整的嵌套结构 ★★★
            # 这样就不需要在后面写复杂的 if len(...) check 了
            current_episode_buffer = {
                "actions": [], 
                "rewards": [],
                "obs": {
                    "image_global": [],
                    "image_wrist": [],
                    "joint_pos": [],
                    "ee_pos": [],
                    "ee_quat": [],
                    "gripper": []
                }
            }
        
        if flags["stop_rec"] and is_recording:
# 获取帧数
            frame_count = len(current_episode_buffer["actions"])
            print(f"\n[REC] 停止录制. 共 {frame_count} 帧.")
            
            is_recording = False
            
            # --- 只有数据有效才保存 ---
            if frame_count > 0:
                print(f"[SAVE] 正在写入硬盘...")
                
                # 1. 打包数据 (List -> Numpy)
                ep_data = {
                    "actions": np.stack(current_episode_buffer["actions"]),
                    "rewards": np.stack(current_episode_buffer["rewards"]),
                    "obs": {}
                }
                # 遍历处理 obs 内的所有数据
                for key, val_list in current_episode_buffer["obs"].items():
                    ep_data["obs"][key] = np.stack(val_list)

                # 2. ★★★ 核心修改：生成文件名并立即保存 ★★★
                filename = os.path.join(session_dir, f"episode_{episode_idx}.pkl")
                
                with open(filename, "wb") as f:
                    pickle.dump(ep_data, f)
                
                print(f"[SUCCESS] 第 {episode_idx} 条轨迹已保存: {filename}")
                
                # 3. 计数器 +1
                episode_idx += 1
            else:
                print("[WARN] 数据为空，未保存。")

            # 4. 重置 Buffer (清空内存)
            current_episode_buffer = {
                "actions": [], "rewards": [],
                "obs": {"image_global": [], "image_wrist": [], "joint_pos": [], "ee_pos": [], "ee_quat": [], "gripper": []}
            }

        if flags["reset"]:
            obs, _ = env.reset()
            ik_controller.reset()
            default_joint_pos = robot_entity.data.joint_pos[:, 0:7].clone()
            ee_idx = robot_entity.find_bodies("panda_hand")[0][0]
            curr_ee_quat = robot_entity.data.body_state_w[:, ee_idx, 3:7]
            _, _, yaw_measured = euler_xyz_from_quat(curr_ee_quat)
            current_yaw = yaw_measured[0].item()
            if is_recording: 
                print("\n[REC] 重置环境，录制中断！")
                is_recording = False
            continue

        # C. 机器人控制
        ee_idx = robot_entity.find_bodies("panda_hand")[0][0]
        ee_current_state = robot_entity.data.body_state_w[:, ee_idx, :7]
        ee_pos_curr = ee_current_state[:, 0:3]
        ee_quat_curr = ee_current_state[:, 3:7]
        
        current_yaw += delta_yaw
        roll = torch.tensor([torch.pi], device=env.unwrapped.device)
        pitch = torch.tensor([0.0], device=env.unwrapped.device)
        yaw = torch.tensor([current_yaw], device=env.unwrapped.device)
        
        target_quat = quat_from_euler_xyz(roll, pitch, yaw)
        orientation_gains = torch.tensor([[8.0, 8.0, 0.6]], device=env.unwrapped.device)
        raw_error = compute_orientation_error(ee_quat_curr, target_quat)
        rot_command = raw_error * orientation_gains
        
        ik_command = torch.cat([delta_pos.unsqueeze(0), rot_command], dim=-1)
        ik_controller.set_command(ik_command, ee_pos=ee_pos_curr, ee_quat=ee_quat_curr)
        
        full_jacobian = robot_entity.root_physx_view.get_jacobians()
        ee_jacobian = full_jacobian[:, ee_idx, :, 0:7]
        if ee_jacobian.dim() == 2: ee_jacobian = ee_jacobian.unsqueeze(0)
        
        all_joint_pos = robot_entity.data.joint_pos
        arm_joint_pos = all_joint_pos[:, 0:7]

        joint_actions = ik_controller.compute(
            ee_pos=ee_pos_curr, ee_quat=ee_quat_curr,
            jacobian=ee_jacobian, joint_pos=arm_joint_pos
        )
        
        # Null-space control
        j = ee_jacobian
        j_t = j.transpose(1, 2)
        eye_6 = torch.eye(6, device=env.unwrapped.device).unsqueeze(0)
        j_pinv = torch.matmul(j_t, torch.inverse(torch.matmul(j, j_t) + 1e-4 * eye_6))
        eye_7 = torch.eye(7, device=env.unwrapped.device).unsqueeze(0)
        null_projector = eye_7 - torch.matmul(j_pinv, j)
        posture_error = default_joint_pos - arm_joint_pos
        null_space_step = torch.matmul(null_projector, posture_error.unsqueeze(-1)).squeeze(-1)
        
        joint_actions_corrected = joint_actions + null_space_step * 10.0 * env_cfg.sim.dt
        gripper_action = torch.tensor([[gripper_cmd]], device=env.unwrapped.device)
        full_actions = torch.cat([joint_actions_corrected, gripper_action], dim=-1)

        # D. 环境步进
        next_obs, reward, terminated, truncated, info = env.step(full_actions)
        
        # E. 录制数据
        if is_recording:
            if step_count % record_interval == 0:
                # --- 0. 初始化 Buffer (仅在第一帧执行) ---
                if len(current_episode_buffer["actions"]) == 0:
                    # 重置/初始化结构
                    current_episode_buffer = {
                        "actions": [], 
                        "rewards": [],
                        "obs": {
                            "image_global": [],
                            "image_wrist": [],
                            "joint_pos": [],
                            "ee_pos": [],
                            "ee_quat": [],
                            "gripper": []
                        }
                    }

                # --- 1. 提取图像 ---
                # Global Camera
                cam_global = env.unwrapped.scene["camera_global"]
                # data.output["rgb"] 形状: (Num_Envs, H, W, 3)
                # 我们取 [0] 拿到 (H, W, 3)
                rgb_global = cam_global.data.output["rgb"][0].cpu().numpy().astype(np.uint8)
                # 如果想要 CHW，解除下面这行的注释，但建议存 HWC
                # rgb_global = rgb_global.transpose(2, 0, 1) 
                
                # Wrist Camera
                cam_wrist = env.unwrapped.scene["camera_wrist"]
                rgb_wrist = cam_wrist.data.output["rgb"][0].cpu().numpy().astype(np.uint8)

                # --- 2. 提取本体感知 ---
                curr_joints = arm_joint_pos[0].cpu().numpy() # (7,)
                curr_ee_pos = ee_pos_curr[0].cpu().numpy()   # (3,)
                curr_ee_quat = ee_quat_curr[0].cpu().numpy() # (4,)
                # 夹爪宽度 (2个手指位置之和)
                curr_gripper_width = all_joint_pos[0, 7:9].sum().cpu().numpy()
                # 此时 curr_gripper_width 是一个标量 (scalar)，如果你想要数组形式:
                curr_gripper_width = np.array([curr_gripper_width], dtype=np.float32)

                # --- 3. 存入 Buffer ---
                current_episode_buffer["obs"]["image_global"].append(rgb_global)
                current_episode_buffer["obs"]["image_wrist"].append(rgb_wrist)
                current_episode_buffer["obs"]["joint_pos"].append(curr_joints)
                current_episode_buffer["obs"]["ee_pos"].append(curr_ee_pos)
                current_episode_buffer["obs"]["ee_quat"].append(curr_ee_quat)
                current_episode_buffer["obs"]["gripper"].append(curr_gripper_width)
                
                # --- 4. 存 Action ---
                # full_actions 是 (1, 8)，我们要存 (8,)
                current_episode_buffer["actions"].append(full_actions[0].cpu().numpy())
                current_episode_buffer["rewards"].append(reward.cpu().numpy())
                
                if step_count % 10 == 0: print("●", end="\r")

        obs = next_obs
        step_count += 1
        
        if terminated.any() or truncated.any():
            env.reset()
            ik_controller.reset()
            default_joint_pos = robot_entity.data.joint_pos[:, 0:7].clone()
            current_yaw = 0.0

    env.close()
    
    # # 退出时保存
    # if len(recorded_episodes) > 0:
    #     save_dir = args_cli.save_path
    #     if not os.path.exists(save_dir): 
    #         os.makedirs(save_dir)
    #         print(f"[INFO] Created directory: {save_dir}")
            
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = os.path.join(save_dir, f"demos_diffusion_{timestamp}.pkl")
        
    #     with open(filename, "wb") as f: 
    #         pickle.dump(recorded_episodes, f)
    #     print(f"\n[SUCCESS] Saved {len(recorded_episodes)} episodes to:")
    #     print(f"          {filename}")

if __name__ == "__main__":
    main()