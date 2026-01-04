# scripts/convert_to_zarr.py
import os
import zarr
import numpy as np
import pickle
import glob
import argparse
from tqdm import tqdm
import cv2 

def main():
    parser = argparse.ArgumentParser()
    # 输入你存放 pkl 的那个文件夹路径
    parser.add_argument('--input_dir', type=str, required=True, help="Path to record_xxx folder")
    parser.add_argument('--output', type=str, default="data/isaac_pusht.zarr")
    args = parser.parse_args()

    # 1. 搜集所有 pkl 文件
    pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    print(f"找到 {len(pkl_files)} 条轨迹")

    # 2. 创建 Zarr 结构
    root = zarr.open(args.output, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    # 3. 临时列表
    all_img_global = []
    all_img_wrist = []
    all_state = []
    all_action = []
    episode_ends = []
    cur_end = 0

    for pkl_path in tqdm(pkl_files):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # --- 数据提取与映射 ---
        # 1. 图像提取
        img_g_raw = data['obs']['image_global'] # (T, 256, 256, 3)
        img_w_raw = data['obs']['image_wrist']  # (T, 256, 256, 3)
    
    # ★★★ 新增：缩放到 96x96 ★★★
    # 遍历每一帧进行缩放 (或者用 batch resize 如果显存够)
        img_g_resized = []
        img_w_resized = []
    
        for i in range(len(img_g_raw)):
            # cv2.resize 输入是 (W, H)
            g_s = cv2.resize(img_g_raw[i], (96, 96), interpolation=cv2.INTER_AREA)
            w_s = cv2.resize(img_w_raw[i], (96, 96), interpolation=cv2.INTER_AREA)
            img_g_resized.append(g_s)
            img_w_resized.append(w_s)
            
        img_g = np.array(img_g_resized) # 变回 numpy array (T, 96, 96, 3)
        img_w = np.array(img_w_resized)
        # 2. 状态 (State): 拼接 关节角度(7) + 夹爪宽度(1) = 8维
        # 注意：你的 pkl 里 gripper 可能是 (T, 1) 也可能是 (T,)
        joint = data['obs']['joint_pos']
        gripper = data['obs']['gripper']
        if gripper.ndim == 1: gripper = gripper[:, None]
        state = np.concatenate([joint, gripper], axis=-1) # Shape (T, 8)

        # 3. 动作 (Action): 已经是 8 维了
        action = data['actions']

        # --- 存入列表 ---
        all_img_global.append(img_g)
        all_img_wrist.append(img_w)
        all_state.append(state)
        all_action.append(action)
        
        cur_end += len(action)
        episode_ends.append(cur_end)

    # 4. 写入 Zarr (合并数组)
    # Compressor 设为 None 也就是不压缩，或者 'default'
    print("正在写入 Zarr...")
    data_group.create_dataset('img_global', data=np.concatenate(all_img_global, axis=0), chunks=(100, 256, 256, 3), dtype='uint8')
    data_group.create_dataset('img_wrist', data=np.concatenate(all_img_wrist, axis=0), chunks=(100, 256, 256, 3), dtype='uint8')
    data_group.create_dataset('state', data=np.concatenate(all_state, axis=0), chunks=(1000, 8), dtype='float32')
    data_group.create_dataset('action', data=np.concatenate(all_action, axis=0), chunks=(1000, 8), dtype='float32')
    meta_group.create_dataset('episode_ends', data=np.array(episode_ends), dtype='int64')

    print(f"转换完成！保存至: {args.output}")

if __name__ == "__main__":
    main()