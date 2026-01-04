# source/PushT_PD/PushT_PD/__init__.py

import os
import toml

# 1. 获取当前文件的目录 (即 source/PushT_PD/PushT_PD/)
PUSH_T_PD_EXT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 定义 assets 文件夹的路径
# 这样你在 config.py 里就能通过 PUSH_T_ASSETS_DIR 找到 usd 文件
PUSH_T_ASSETS_DIR = os.path.join(PUSH_T_PD_EXT_DIR, "assets")

# 3. 自动注册环境 (这一行让 gym.make 能找到你的环境)
# 它会触发 tasks/manager_based/__init__.py 的运行
from .tasks.manager_based import *

print(f"[INFO] PushT_PD Extension Loaded.")
print(f"[INFO] Assets Path: {PUSH_T_ASSETS_DIR}")
