import os
import shutil
import glob
from tqdm import tqdm
import argparse

def merge_datasets(input_root, output_dir):
    # 1. 检查输入目录
    if not os.path.exists(input_root):
        print(f"❌ 错误: 输入目录不存在 -> {input_root}")
        return

    # 2. 创建输出目录 (如果不存在)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    else:
        print(f"⚠️  输出目录已存在: {output_dir} (新文件将追加到这里)")

    # 3. 找到所有录制文件夹 (record_*)
    # 按照文件夹名字排序，确保时间顺序 (record_2025...)
    record_folders = sorted([
        f for f in os.listdir(input_root) 
        if os.path.isdir(os.path.join(input_root, f)) and "record" in f
    ])

    if not record_folders:
        print("❌ 未发现任何 'record_' 开头的文件夹！")
        return

    print(f"📂 发现 {len(record_folders)} 个录制文件夹，准备合并...")

    # 4. 开始合并
    global_idx = 0 # 全局编号计数器
    total_files = 0

    # 为了进度条好看，先统计总文件数
    for folder in record_folders:
        folder_path = os.path.join(input_root, folder)
        total_files += len(glob.glob(os.path.join(folder_path, "*.pkl")))

    pbar = tqdm(total=total_files, desc="Processing")

    for folder in record_folders:
        folder_path = os.path.join(input_root, folder)
        
        # 获取该文件夹下所有 pkl，按文件名排序 (episode_0, episode_1...)
        # 注意：直接 sort 字符串会导致 episode_10 排在 episode_2 前面，需要特殊处理
        pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
        
        # 智能排序: 按文件名中的数字排序
        pkl_files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))

        for src_file in pkl_files:
            # 定义新的文件名: episode_{global_idx}.pkl
            new_filename = f"episode_{global_idx}.pkl"
            dst_file = os.path.join(output_dir, new_filename)

            # 复制文件
            shutil.copy2(src_file, dst_file)

            # 更新计数器
            global_idx += 1
            pbar.update(1)
            pbar.set_postfix({"Last Folder": folder})

    pbar.close()
    print(f"\n🎉 合并完成！")
    print(f"📊 总共处理: {global_idx} 条轨迹")
    print(f"💾 保存位置: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge scattered episode pkl files into one folder.")
    parser.add_argument("--input", type=str, default="data/demos", help="Root directory containing record_xxx folders")
    parser.add_argument("--output", type=str, default="data/demos_merged", help="Directory to save merged files")
    
    args = parser.parse_args()
    
    merge_datasets(args.input, args.output)