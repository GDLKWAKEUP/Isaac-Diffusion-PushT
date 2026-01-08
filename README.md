![output](https://github.com/user-attachments/assets/a24f0615-aaa7-43f0-b73e-f21ef4665346)

# Isaac Sim Push-T with Diffusion Policy 🤖

[吉林大学] [电子科学与工程]

这是一个基于 **NVIDIA Isaac Sim** 和 **Diffusion Policy** 的 Sim2Real 机器人操作复现项目。
实现了从 Blender 场景建模、Xbox 手柄遥操作数据采集、到 Diffusion Policy 模型训练及闭环推理的全流程。


## ✨ 主要功能 (Features)

*   🏗️ **仿真环境**：在 Isaac Sim 中搭建了经典的 Push-T 任务场景 (Franka Emika Panda)。
*   🎮 **遥操作接口**：支持 Xbox 手柄控制，集成了 IK 逆运动学求解与**零空间姿态锁定 (Null-space Control)**，操作手感极佳。
*   🧠 **模仿学习**：适配了 `diffusion_policy` 算法，支持多模态输入 (Global Camera + Wrist Camera + Proprioception)。
*   🔄 **闭环推理**：实现了 Sim 端的推理脚本，支持实时重置与连续测试。

## 🛠️ 安装指南 (Installation)

本项目依赖两个环境：`isaaclab` (用于仿真) 和 `robodiff` (用于训练)。

### 1. Isaac Sim 环境
请确保已安装 NVIDIA Isaac Sim 4.0+ 和 Isaac Lab。
```bash
# 安装本项目包
pip install -e .

数据与模型下载:https://drive.google.com/drive/folders/1q6FfucyZIzDt94ckfBWqtmdUEntLJDGh?usp=drive_link
