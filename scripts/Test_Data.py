import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def inspect_pickle(file_path):
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 -> {file_path}")
        return

    print(f"\n{'='*60}")
    print(f"📂 正在加载数据: {file_path}")
    print(f"{'='*60}")

    # 2. 加载数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 3. 基础信息检查
    total_frames = len(data["actions"])
    print(f"⏱️  总帧数 (Total Frames): {total_frames}")
    
    # 检查顶层键
    print(f"\n🔑 顶层 Keys: {list(data.keys())}")
    
    # --- 检查动作 (Actions) ---
    actions = data["actions"]
    print(f"\n🎮 [Actions]:")
    print(f"   Shape: {actions.shape} | Type: {actions.dtype}")
    print(f"   Range: Min={actions.min():.4f}, Max={actions.max():.4f}")
    print(f"   前5帧示例:\n{actions[:5]}")

    # --- 检查奖励 (Rewards) ---
    rewards = data["rewards"]
    print(f"\n🎁 [Rewards]:")
    print(f"   Shape: {rewards.shape} | Avg: {rewards.mean():.4f}")

    # --- 检查观测 (Observations) ---
    print(f"\n👁️  [Observations]:")
    obs = data["obs"]
    
    # 遍历 obs 中的每一项
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            print(f"   🔹 {key:<15} Shape: {str(val.shape):<20} Type: {val.dtype}")
        else:
            print(f"   🔹 {key:<15} Type: {type(val)}")

    # 4. 可视化检查 (重点！)
    print(f"\n{'='*60}")
    print("🖼️  正在生成可视化窗口... (请查看弹出的窗口)")
    
    # 创建一个大图：上面是图像，下面是轨迹波形
    fig = plt.figure(figsize=(15, 10))
    
    # --- A. 图像可视化 (抽取 第0帧, 中间帧, 最后帧) ---
    indices = [0, total_frames // 2, total_frames - 1]
    titles = ["Start (Frame 0)", "Middle", "End"]
    
    for i, idx in enumerate(indices):
        # Global Camera
        if "image_global" in obs and len(obs["image_global"]) > idx:
            ax = fig.add_subplot(3, 3, i + 1)
            img = obs["image_global"][idx]
            ax.imshow(img)
            ax.set_title(f"Global - {titles[i]}")
            ax.axis('off')
            
        # Wrist Camera
        if "image_wrist" in obs and len(obs["image_wrist"]) > idx:
            ax = fig.add_subplot(3, 3, i + 4)
            img = obs["image_wrist"][idx]
            ax.imshow(img)
            ax.set_title(f"Wrist - {titles[i]}")
            ax.axis('off')

    # --- B. 数据曲线可视化 (EE Position & Gripper) ---
    # 绘制末端执行器位置变化 (XYZ)
    if "ee_pos" in obs:
        ax_pos = fig.add_subplot(3, 2, 5)
        ee_pos = obs["ee_pos"]
        ax_pos.plot(ee_pos[:, 0], label="X", color='r', alpha=0.7)
        ax_pos.plot(ee_pos[:, 1], label="Y", color='g', alpha=0.7)
        ax_pos.plot(ee_pos[:, 2], label="Z", color='b', alpha=0.7)
        ax_pos.set_title("End-Effector Position (XYZ)")
        ax_pos.legend()
        ax_pos.grid(True)

    # 绘制夹爪开合状态
    if "gripper" in obs:
        ax_grip = fig.add_subplot(3, 2, 6)
        gripper = obs["gripper"]
        ax_grip.plot(gripper, label="Width", color='k')
        ax_grip.set_title("Gripper Width")
        ax_grip.set_ylim(-0.01, 0.1) # Franka 夹爪范围通常是 0~0.08
        ax_grip.legend()
        ax_grip.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Inspect a single episode pickle file.")
    # 这里的 default 可以改成你刚才生成的某一个具体文件路径，方便直接运行
    parser.add_argument("--file", type=str, required=True, help="Path to the .pkl file")
    
    args = parser.parse_args()
    inspect_pickle(args.file)