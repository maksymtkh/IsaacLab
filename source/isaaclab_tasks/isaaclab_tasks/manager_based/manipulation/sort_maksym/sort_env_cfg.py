# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from numpy import pi

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

"""
Three different environments are represented here:
1.  Small working area with 9 different nuts (3 sizes, 3 colors, see ALL_NUT_CFGS)
2.  Big working area with 9 different nuts (3 sizes, 3 colors, see ALL_NUT_CFGS)
3.  Big working are with 3 different nuts (1 size, 3 colors, see ONEOFAKIND_NUT_CFGS)
"""

# -----------------------------------------------------------------------------
# General nut configuration (centralized + reusable)
# -----------------------------------------------------------------------------

ALL_NUT_CFGS = [
    SceneEntityCfg("nut_m8_red"),
    SceneEntityCfg("nut_m8_green"),
    SceneEntityCfg("nut_m8_blue"),
    SceneEntityCfg("nut_m12_red"),
    SceneEntityCfg("nut_m12_green"),
    SceneEntityCfg("nut_m12_blue"),
    SceneEntityCfg("nut_m16_red"),
    SceneEntityCfg("nut_m16_green"),
    SceneEntityCfg("nut_m16_blue"),
]

ONEOFAKIND_NUT_CFGS = [
    SceneEntityCfg("nut_m8_red"),
    SceneEntityCfg("nut_m8_green"),
    SceneEntityCfg("nut_m8_blue"),
]

TARGET_NUT_CFG = SceneEntityCfg("nut_m8_red")

BOUND_SMALLWA = {"x": (0.29, 0.69), "y": (0.03, 0.44)}

TARGET_BOUND_SMALLWA = {
        "left":   {"min_x": 0.59, "max_x": 0.69, "min_y": 0.04, "max_y": 0.16},
        "middle": {"min_x": 0.59, "max_x": 0.69, "min_y": 0.17, "max_y": 0.29},
        "right":  {"min_x": 0.59, "max_x": 0.69, "min_y": 0.30,  "max_y": 0.42},
    }

TARGET_BOUND_BIGWA = {
        "left":   {"min_x": 0.60, "max_x": 0.72, "min_y": -0.10, "max_y": 0.11},
        "middle": {"min_x": 0.60, "max_x": 0.72, "min_y": 0.13, "max_y": 0.34},
        "right":  {"min_x": 0.60, "max_x": 0.72, "min_y": 0.36,  "max_y": 0.57},
    }

TARGET = "left"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

##
# Scene definition for small working area
##
@configclass
class ObjectTableSmallWASceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(usd_path=f"USD_Files/Table/table_smallwa_final.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.06]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# Scene definition for big working area
##
@configclass
class ObjectTableBigWASceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(usd_path=f"USD_Files/Table/table_bigwa_final.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.06]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(
            func=mdp.object_obs,
            params={
                "nut_names": [TARGET_NUT_CFG],
            })
        nut_position = ObsTerm(
            func=mdp.nut_positions_in_world_frame,
            params={
                "nut_names": ALL_NUT_CFGS,
            })
        nut_orientation = ObsTerm(
            func=mdp.nut_orientations_in_world_frame,
            params={
                "nut_names": ALL_NUT_CFGS,
            })
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": TARGET_NUT_CFG,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class ObservationsOOAKCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(
            func=mdp.object_obs,
            params={
                "nut_names": [TARGET_NUT_CFG],
            })
        nut_position = ObsTerm(
            func=mdp.nut_positions_in_world_frame,
            params={
                "nut_names": ONEOFAKIND_NUT_CFGS,
            })
        nut_orientation = ObsTerm(
            func=mdp.nut_orientations_in_world_frame,
            params={
                "nut_names": ONEOFAKIND_NUT_CFGS,
            })
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": TARGET_NUT_CFG,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsSmallWACfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # === Nuts are under the table ===

    nut_height_below_minimum = DoneTerm(
        func=mdp.height_below_minimum,
        params={
            "pose_range": {"z": -0.025},   # minimum allowed height
            "asset_cfgs": ALL_NUT_CFGS,
        },
    )

    # === Nuts are out of the workig area ===

    nut_out_of_bounds = DoneTerm(
        func=mdp.position_xy_out_of_bounds,
        params={
            "pose_range": BOUND_SMALLWA,
            "asset_cfgs": ALL_NUT_CFGS,
        },
    )

    # === Robot joint manual safety limits ===

    robot_joint_out_of_manual_limits = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "bounds": (-100 /360*2*pi, -5 /360*2*pi),  # radians
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_ids=[1],  # shoulder_lift_joint, UR5e
            ),
        },
    )

    # === Specific term for TARGET_NUT into left zone ===

    success = DoneTerm(
        func=mdp.task_done_place_with_gripper_check,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": TARGET_NUT_CFG,
            "zone_cfg": TARGET_BOUND_SMALLWA[TARGET],
            "max_height": 0.1,
            "vel_threshold": 0.1,
        },
    )



@configclass
class TerminationsBigWACfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # === Nuts are under the table ===

    nut_height_below_minimum = DoneTerm(
        func=mdp.height_below_minimum,
        params={
            "pose_range": {"z": -0.025},   # minimum allowed height
            "asset_cfgs": ALL_NUT_CFGS,
        },
    )

    # === Robot joint manual safety limits ===

    robot_joint_out_of_manual_limits = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "bounds": (-100 /360*2*pi, -5 /360*2*pi),  # radians
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_ids=[1],  # shoulder_lift_joint, UR5e
            ),
        },
    )

    # === Specific term for TARGET_NUT into left zone ===

    success = DoneTerm(
        func=mdp.task_done_place_with_gripper_check,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": TARGET_NUT_CFG,
            "zone_cfg": TARGET_BOUND_BIGWA[TARGET],
            "max_height": 0.1,
            "vel_threshold": 0.1,
        },
    )



@configclass
class TerminationsOOAKCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # === Nuts are under the table ===

    nut_height_below_minimum = DoneTerm(
        func=mdp.height_below_minimum,
        params={
            "pose_range": {"z": -0.025},   # minimum allowed height
            "asset_cfgs": ONEOFAKIND_NUT_CFGS,
        },
    )

    # === Robot joint manual safety limits ===

    robot_joint_out_of_manual_limits = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "bounds": (-100 /360*2*pi, -5 /360*2*pi),  # radians
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_ids=[1],  # shoulder_lift_joint, UR5e
            ),
        },
    )

    # === Specific term for TARGET_NUT into left zone ===

    success = DoneTerm(
        func=mdp.task_done_place_with_gripper_check,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": TARGET_NUT_CFG,
            "zone_cfg": TARGET_BOUND_BIGWA[TARGET],
            "max_height": 0.1,
            "vel_threshold": 0.1,
        },
    )



@configclass
class SortSmallWAEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSmallWASceneCfg = ObjectTableSmallWASceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsSmallWACfg = TerminationsSmallWACfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625



@configclass
class SortBigWAEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableBigWASceneCfg = ObjectTableBigWASceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsBigWACfg = TerminationsBigWACfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class SortOOAKEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableBigWASceneCfg = ObjectTableBigWASceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsOOAKCfg = ObservationsOOAKCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsOOAKCfg = TerminationsOOAKCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
