# source/PushT_PD/PushT_PD/tasks/manager_based/pusht_pd/pusht_pd_env_cfg.py

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
# 添加到文件顶部的 imports 区域
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg 
# ★ 使用 Isaac Lab 标准 MDP 库，无需自己编写基础函数
import isaaclab.envs.mdp as mdp
from . import mdp as local_mdp 

import carb.input
import omni.appwindow

# =========================================================
#  环境配置 (Env Configuration)
# =========================================================

# 1. 导入 Franka 机器人配置
try:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
except ImportError:
    from isaaclab.assets import FRANKA_PANDA_CFG 

# 2. 导入你的资产路径
# 确保你已经 pip install -e . 并且 __init__.py 配置正确
from PushT_PD import PUSH_T_ASSETS_DIR

# =========================================================
#  场景配置 (Scene Definition)
# =========================================================

@configclass
class PushtPdSceneCfg(InteractiveSceneCfg):
    """Configuration for the Push-T scene."""

    # (A) 静态背景 (加载 background.usd - 含桌子、底座、可视化的Marker)
    background = AssetBaseCfg(
        prim_path="/World/Background",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PUSH_T_ASSETS_DIR}/background.usd",
            scale=(1.0, 1.0, 1.0),

            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
         
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # (B) 机器人 (Franka)
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 机器人安装位置 (根据 Blender 脚本计算: Y=-0.62, Z=0.75)
    robot.init_state.pos = (0.0, -0.50, 0.76)
    robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
    # 初始关节角度 (避免僵硬的直立状态)
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0, 
        "panda_joint2": -0.8,   # 肩膀前倾约 45 度，将手臂送入工作区
        "panda_joint3": 0.0,
        "panda_joint4": -2.5,   # 肘部弯曲约 145 度，保持抬高姿态
        "panda_joint5": 0.0, 
        "panda_joint6": 1.80,   # ★关键★：手腕下压，补偿手臂角度，使夹爪垂直向下
        "panda_joint7": 0.785,  # 夹爪水平旋转 45 度 (符合 T 型抓取习惯)
        "panda_finger_joint1": 0.04, # 张开状态
        "panda_finger_joint2": 0.04,
    }
    # 定义肩部关节 (1-4) 的 PD 参数
    robot.actuators["panda_shoulder"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        stiffness=400000.0,  # P
        damping=60000.0,     # D
    )
    
    # 定义手臂关节 (5-7) 的 PD 参数
    robot.actuators["panda_forearm"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        stiffness=400000.0,
        damping=60000.0,
    )
    # 定义夹爪关节的 PD 参数
    robot.actuators["panda_gripper"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"],
        stiffness=10000.0,
        damping=100.0,
    )
    # (C) T型物体 (被推对象)
    t_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/T_Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PUSH_T_ASSETS_DIR}/t_object.usd",
 
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.751), # 初始放在左下角
            rot=(0.707, 0.707, 0.0, 0.0), # 平放
        ),
    )

    # (D) ★ 新增：目标占位符 (Virtual Goal)
    # 虽然 background.usd 里画了标记，但物理引擎不知道那是目标。
    # 我们加载一个不可见(或静态)的T物体在桌子中心，用于计算奖励距离。
    goal_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal_Target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PUSH_T_ASSETS_DIR}/t_object.usd",
            scale=(1.0, 1.0, 1.0),
            # 设为不可见 (因为背景里已经有绿色的贴图了)
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.0),
            # 或者设为静态刚体，不参与碰撞 (Trigger模式)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True, 
                rigid_body_enabled=True, # 必须为 True，否则我们在 Observation 里读不到它的位置
            ),

            # 2. ★★★ 关键修改：关闭碰撞 ★★★
            # 这样机械臂和物体就能穿过它，而不会撞到它
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False, 
            ),

            # 3. 视觉属性：绿色半透明
            # 注意：如果之前删了这行导致报错，可以先不加，或者确保 sim_utils 里有 PreviewSurfaceCfg
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), 
                opacity=0.3
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.80), # 桌子中心，稍微贴着桌面
            rot=(0.0, 0.0, 0.0, 1.0), # 角度要和 Marker 一致 (Blender里是旋转过的)
        ),
    )

    # (E) 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0), 
            intensity=3000.0,
            angle=0.0
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.707, 0.0, 0.707, 0.0), # 倾斜照射
        ),
    )
 # ... (之前的 robot, t_object 等)

 # (E) 全局相机 (Top-down view)
    camera_global = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_Global", # 挂在环境根节点下
        update_period=0.1, # 10Hz 录制
        height=256, 
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
        # ★★★ 修复：使用 offset 而不是 init_state ★★★
        # 相对于环境原点的偏移
        offset=CameraCfg.OffsetCfg(
            # 移到桌子正上方 (桌子在 Y=-0.6)
            pos=(0.0, -0.008, 2.5), 
            # 标准垂直向下 (Identity 通常是看向 -Z，而在 Isaac Sim Z轴向上，
            # 所以 Identity 是看向下方的，或者试一下绕 X 转 180)
            # 试试这个：(1.0, 0.0, 0.0, 0.0) -> 这是最原始的朝向
            rot=(0.0, 1.0, 0.0, 0.0), 
        ),
    )

    # (F) 手腕相机 (Wrist camera)
    camera_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/Camera_Wrist", # 挂在机械臂手上
        update_period=0.1,
        height=256, 
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, 
            horizontal_aperture=20.955
        ),
        # ★★★ 修复：使用 offset (相对于 panda_hand 的偏移) ★★★
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.05), 
            # 这里的旋转是为了调整相机在手上的朝向，使其看向手指方向
            rot=(0.707, 0.0, 0.0, 0.707) 
        ), 
    )

# =========================================================
#  环境配置 (Env Configuration)
# =========================================================

@configclass
class ActionsCfg:
    """Action specifications."""
    # 关节位置控制
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["panda_joint.*"], 
        scale=1.0, 
        use_default_offset=False
    )

    # 抓手控制
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger.*": 0.04}, 
        close_command_expr={"panda_finger.*": 0.0},
    )

@configclass
class ObservationsCfg:
    """Observation specifications."""
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. 机器人状态
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # 2. 物体状态 (需要推的物体)
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("t_object")})
        object_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("t_object")})
        
        # 3. 目标状态 (关键！告诉Agent要去哪)
        target_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("goal_object")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Reset events."""
    # 重置机器人关节
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 重置被推物体 (随机位置)
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 1. 位置随机范围
            "pose_range": {
                "x": (-0.2, 0.2),    # 在中心点 X 轴左右 10cm 范围内随机
                "y": (-0.2, 0.2),    # 在中心点 Y 轴前后 10cm 范围内随机
                "z": (0.0, 0.0),     # 高度不变
                "pitch": (-3.14, 3.14) # Z轴旋转随机 (全角度覆盖)
            },
            # 2. 速度随机范围 (必须填写的参数)
            # 设置为 0 表示重置后物体是静止的
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 1e-15), # 防止物体旋转
            },
            # 3. 指定资产
            "asset_cfg": SceneEntityCfg("t_object"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms."""
    
    # 1. 距离奖励 (最重要的奖励)
    # 计算 t_object 和 goal_object 之间的距离，距离越近奖励越大
    object_dist = RewTerm(
        func=local_mdp.object_goal_distance,
        weight=-1.0, # 负权重表示这是一个 Cost (距离越远惩罚越大) -> 或者看具体函数实现
        params={
            "std": 0.5, # 归一化参数
            "minimal_distance": 0.02,
            "target_cfg": SceneEntityCfg("goal_object"), # 目标
            "subject_cfg": SceneEntityCfg("t_object"),   # 被推物体
        },
    )
    
    # 2. 存活奖励 (防止太快结束)
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    
    # 3. 动作惩罚 (防止机器人抽搐)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

@configclass
class TerminationsCfg:
    """Termination terms."""

    
    # (可选) 如果物体掉出桌子，结束回合
    object_falling = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("t_object")}
    )

@configclass
class PushtPdEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PushtPdSceneCfg = PushtPdSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4 # 控制频率 50Hz (假设 sim.dt=0.01)
        self.episode_length_s = 5.0
        # 调整查看器视角
        self.viewer.eye = (1.5, 0.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        # 仿真步长
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        # 物理引擎参数调整
        self.sim.physx.solver_type = 1 
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.01

    