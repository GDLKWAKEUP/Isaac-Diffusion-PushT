

import argparse
from isaaclab.app import AppLauncher

# 1. 启动仿真器
parser = argparse.ArgumentParser(description="Run PushT Environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入依赖
import gymnasium as gym
import torch
import carb # 引入底层库用于监听键盘
import omni.appwindow
import PushT_PD.tasks.manager_based # 注册环境
from PushT_PD.tasks.manager_based.pusht_pd.pusht_pd_env_cfg import PushtPdEnvCfg

def is_f_key_pressed():
    """检测 F 键是否按下"""
    app_window = omni.appwindow.get_default_app_window()
    if not app_window: return False
    keyboard = app_window.get_keyboard()
    if not keyboard: return False
    
    # 检测 F 键状态
    input_interface = carb.input.acquire_input_interface()
    if input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.F) > 0:
        return True
    return False

def main():
    print("[INFO] 正在初始化环境配置...")
    
    # ★★★ 关键修改：手动实例化配置 ★★★
    # 我们在这里创建配置对象，并设置环境数量
    env_cfg = PushtPdEnvCfg()
    env_cfg.scene.num_envs = 1 # 设置为1个环境进行测试
    
    # ★★★ 关键修改：将配置传给 gym.make ★★★
    # 这样 ManagerBasedRLEnv 就能收到 cfg 参数了
    env = gym.make("Isaac-PushT-Franka-v0", cfg=env_cfg, render_mode="rgb_array")
    
    print(f"[INFO] 环境创建成功: {env.spec.id}")
    obs, _ = env.reset()

    # 运行循环
    while simulation_app.is_running():
        # 发送全0动作
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        # 3. ★★★ 在主循环里检测按键 ★★★
        is_pressed = is_f_key_pressed()
        
        if is_pressed and not f_key_was_pressed:
            # 只有在“刚刚按下”的那一帧触发重置
            print(">>> 检测到 F 键：重置环境！")
            env.reset()
            f_key_was_pressed = True # 锁住
        elif not is_pressed:
            # 松开后解锁
            f_key_was_pressed = False

        # 4. 处理常规的重置逻辑 (超时等)
        if terminated.any() or truncated.any():
            env.reset()
    env.close()

if __name__ == "__main__":
    main()