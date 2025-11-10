# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.sort_maksym import mdp
from isaaclab_tasks.manager_based.manipulation.sort_maksym.mdp import ur5e_sort_events
from isaaclab_tasks.manager_based.manipulation.sort_maksym.sort_env_cfg import SortEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.universal_robots_maksym import UR5e_wr_gripper_CFG
from numpy import pi


@configclass
class EventCfg:
    """Configuration for events."""

    # Define an event to initialize the UR5e arm joint pose
    init_ur5e_arm_pose = EventTerm(
        func=ur5e_sort_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [
                7 /360*2*pi,            # shoulder_pan_joint
                -50 /360*2*pi,     # shoulder_lift_joint
                25 /360*2*pi,      # elbow_joint
                -70 /360*2*pi,     # wrist_1_joint
                -90 /360*2*pi,     # wrist_2_joint
                95 /360*2*pi,      # wrist_3_joint
                0.0430,                  # gripper_joint_left
                0.0430,                  # gripper_joint_right
            ],
        },
    )

    randomize_ur5e_joint_state = EventTerm(
        func=ur5e_sort_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Reset all nuts - iteration version (9 nuts)
    randomize_nut_positions = EventTerm(
        func=ur5e_sort_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.72), "y": (-0.1, 0.55), "z": (0.0, 0.0), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [
                #SceneEntityCfg("nut_m8_red"),
                SceneEntityCfg("nut_m8_green"),
                SceneEntityCfg("nut_m8_blue"),
                SceneEntityCfg("nut_m12_red"),
                SceneEntityCfg("nut_m12_green"),
                SceneEntityCfg("nut_m12_blue"),
                SceneEntityCfg("nut_m16_red"),
                SceneEntityCfg("nut_m16_green"),
                SceneEntityCfg("nut_m16_blue"),
            ],
        },
    )

    # Reset all nuts - iteration version (9 nuts)
    randomize_nut_positions_m8 = EventTerm(
        func=ur5e_sort_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.52), "y": (-0.1, 0.55), "z": (0.0, 0.0), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [
                SceneEntityCfg("nut_m8_red"),
            ],
        },
    )

@configclass
class UR5eSortEnvCfg(SortEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set UR5e as robot
        self.scene.robot = UR5e_wr_gripper_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (ur5e)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger.*.*"],
            open_command_expr={"finger.*_.*": 0.00},
            close_command_expr={"finger.*_.*": 0.043},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["finger.*_.*"]
        self.gripper_open_val = 0.00
        self.gripper_threshold = 0.02

        # Rigid body properties of each nut
        nut_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=40,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        """In case the iteration version is not used, here is the single nut spawn code:
        # Nut M8
        self.scene.nut_m8 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Nut_M8",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85, 0.25, 0.03), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                #usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_nut_m8_loose/factory_nut_m8_loose.usd",
                #usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
                usd_path=f"/home/MA_LaToOm/Desktop/USD_ur5e_withgripper/Nuts/nut_m8.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=nut_properties,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        )"""

        # Iteration version:
        # Spawn multiple nut sizes and colors (iteration version)
        nut_sizes = ["m8", "m12", "m16"]
        nut_colors = {
            "red":   (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue":  (0.0, 0.0, 1.0),
        }

        for size in nut_sizes:
            for color_name, color_value in nut_colors.items():
                setattr(
                    self.scene,
                    f"nut_{size}_{color_name}",
                    RigidObjectCfg(
                        # Keep literal {ENV_REGEX_NS}, interpolate size/color
                        prim_path=f"{{ENV_REGEX_NS}}/Nut_{size.upper()}_{color_name}",
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(0.85, 0.25, 0.0),
                            rot=(1.0, 0.0, 0.0, 0.0),
                        ),
                        spawn=UsdFileCfg(
                            usd_path=f"/home/MA_LaToOm/Desktop/USD_ur5e_withgripper/Nuts/nut_{size}.usd",
                            scale=(1.0, 1.0, 1.0),
                            rigid_props=nut_properties,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=color_value
                            ),
                        ),
                    ),
                )


        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5e/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5e/Gripper/gripper_wr",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5e/Gripper/wr_finger_right",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5e/Gripper/wr_finger_left",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )