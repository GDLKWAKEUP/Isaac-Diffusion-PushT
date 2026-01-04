# source/PushT_PD/PushT_PD/tasks/manager_based/__init__.py

import gymnasium as gym

# 导入你的环境配置类
# 注意：这里使用的是相对导入，或者你可以用完整的 PushT_PD.tasks...
from .pusht_pd.pusht_pd_env_cfg import PushtPdEnvCfg

# 注册环境
gym.register(
    id="Isaac-PushT-Franka-v0",  # 这是你在 play.py 里调用的名字
    entry_point="isaaclab.envs:ManagerBasedRLEnv", # 使用 Isaac Lab 标准管理器
    kwargs={
        "env_cfg_entry_point": PushtPdEnvCfg, # 指向你的配置类
    },
    disable_env_checker=True,
)
