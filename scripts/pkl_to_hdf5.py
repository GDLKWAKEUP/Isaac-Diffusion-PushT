import os
import h5py
import pickle
import numpy as np
import glob
import argparse
import cv2
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="包含 pkl 文件的文件夹路径")
    parser.add_argument('--output', type=str, default="data/isaac_pusht.hdf5", help="输出的 hdf5 文件路径")
    args = parser.parse_args()

    # 1. 搜集 pkl 文件
    # 递归查找所有子文件夹中的 pkl
    pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "**/*.pkl"), recursive=True))
    print(f"🔍 找到 {len(pkl_files)} 条轨迹数据")

    if len(pkl_files) == 0:
        print("❌ 未找到数据，请检查路径")
        return

    # 2. 创建 HDF5 文件
    # Robomimic 格式要求：根目录下有一个 'data' 组，每个 demo 是 'demo_0', 'demo_1'...
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    f = h5py.File(args.output, "w")
    grp = f.create_group("data")

    total_samples = 0

    print("🚀 开始转换...")
    for i, pkl_path in enumerate(tqdm(pkl_files)):
        with open(pkl_path, 'rb') as pkl_f:
            ep_data = pickle.load(pkl_f)

        # --- A. 数据预处理 ---
        
        # 1. 动作 (Action): (T, 8)
        actions = ep_data['actions'].astype(np.float32)
        
        # 2. 状态 (State): 拼接 Joint(7) + Gripper(1) -> (T, 8)
        joint_pos = ep_data['obs']['joint_pos']
        gripper = ep_data['obs']['gripper']
        # 确保 gripper 是 (T, 1)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        state = np.concatenate([joint_pos, gripper], axis=-1).astype(np.float32)

        # 3. 图像 (Image): 缩放 256 -> 96
        # Global
        img_g_raw = ep_data['obs']['image_global'] # (T, 256, 256, 3)
        img_g_96 = []
        for img in img_g_raw:
            img_g_96.append(cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA))
        img_g_96 = np.array(img_g_96, dtype=np.uint8)
        # HWC -> CHW (Robomimic Dataset 内部通常会自动转，存 HWC 最通用)
        # 这里我们存 HWC: (T, 96, 96, 3)

        # Wrist
        img_w_raw = ep_data['obs']['image_wrist']
        img_w_96 = []
        for img in img_w_raw:
            img_w_96.append(cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA))
        img_w_96 = np.array(img_w_96, dtype=np.uint8)

        # --- B. 写入 HDF5 结构 ---
        # 结构: data/demo_i/obs/key
        
        demo_grp = grp.create_group(f"demo_{i}")
        
        # 写入总步数属性 (重要)
        demo_grp.attrs["num_samples"] = len(actions)
        total_samples += len(actions)

        # 写入 Observation
        obs_grp = demo_grp.create_group("obs")
        obs_grp.create_dataset("img_global", data=img_g_96)
        obs_grp.create_dataset("img_wrist", data=img_w_96)
        obs_grp.create_dataset("state", data=state) # 8维状态

        # 写入 Action
        demo_grp.create_dataset("actions", data=actions)
        
        # 写入 Rewards (可选)
        demo_grp.create_dataset("rewards", data=ep_data['rewards'])

    # --- C. 写入全局元数据 ---
    # Robomimic 需要知道总样本数
    grp.attrs["total"] = total_samples
    
    # 还有一种 metadata 格式，为了兼容性最好也写上
    if "mask" not in f:
        # 创建一个默认的 mask，包含所有 demo
        mask_grp = f.create_group("mask")
        mask_grp.create_dataset("train", data=np.array([f"demo_{i}" for i in range(len(pkl_files))]).astype("S"))
        # (简单起见，不分验证集，或者你可以手动分)

    f.close()
    print(f"\n✅ 转换完成！文件保存至: {args.output}")
    print(f"📊 总轨迹数: {len(pkl_files)}")
    print(f"⏱️ 总帧数: {total_samples}")

if __name__ == "__main__":
    main()