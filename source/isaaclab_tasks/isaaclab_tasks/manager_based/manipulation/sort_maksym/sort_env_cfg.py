# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

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


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
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
        spawn=UsdFileCfg(usd_path=f"/home/MA_LaToOm/Desktop/USD_ur5e_withgripper/Table/table_complete_1.usd"),
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
        object = ObsTerm(func=mdp.object_obs)
        nut_position = ObsTerm(func=mdp.nut_positions_in_world_frame)
        nut_orientation = ObsTerm(func=mdp.nut_orientations_in_world_frame)
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
                "object_cfg": SceneEntityCfg("nut_m8_red"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


#Has to be changed
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # === Done terms for all 9 nuts (m8, m12, m16 × red, green, blue) ===

    nut_m8_red_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("nut_m8_red")},
    )

    nut_m8_green_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("nut_m8_green")},
    )

    nut_m8_blue_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("nut_m8_blue")},
    )

    nut_m12_red_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("nut_m12_red")},
    )

    nut_m12_green_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("nut_m12_green")},
    )

    nut_m12_blue_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("nut_m12_blue")},
    )

    nut_m16_red_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("nut_m16_red")},
    )

    nut_m16_green_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("nut_m16_green")},
    )

    nut_m16_blue_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("nut_m16_blue")},
    )


    '''success = DoneTerm(
        func=mdp.object_a_is_into_b,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_a_cfg": SceneEntityCfg("nut_m8_red"),
            "object_b_cfg": SceneEntityCfg("blue_sorting_bin"),
            "xy_threshold": 0.10,
            "height_diff": 0.06,
            "height_threshold": 0.04,
        },
    )'''

    # === Specific term for nut_m8_red into left zone ===

    success = DoneTerm(
    func=mdp.task_done_place_with_gripper_check,
    params={
        "robot_cfg": SceneEntityCfg("robot"),
        "object_cfg": SceneEntityCfg("nut_m8_red"),
        "zone": "left",
        "max_height": 0.1,
        "vel_threshold": 0.1,
    },
)



@configclass
class SortEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

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
