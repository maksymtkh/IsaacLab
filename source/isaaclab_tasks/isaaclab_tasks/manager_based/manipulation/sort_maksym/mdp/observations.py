# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# === Common nut definitions ===
NUT_SIZES = ["m8", "m12", "m16"]
NUT_COLORS = ["red", "green", "blue"]
NUT_NAMES = [f"nut_{size}_{color}" for size in NUT_SIZES for color in NUT_COLORS]

def nut_positions_in_world_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of all nuts (m8, m12, m16 × red, green, blue) in the world frame."""
    nut_positions = []
    for nut_name in NUT_NAMES:
        nut: RigidObject = env.scene[nut_name]
        nut_positions.append(nut.data.root_pos_w)
    return torch.cat(nut_positions, dim=1)

def instance_randomize_nut_positions_in_world_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The position of all nuts (m8, m12, m16 × red, green, blue) in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    nut_positions_all = []

    # Iterate through all defined nuts
    for nut_name in NUT_NAMES:
        nut: RigidObjectCollection = env.scene[nut_name]
        nut_pos_w = []
        for env_id in range(env.num_envs):
            # Same access pattern as your original function
            nut_pos_w.append(
                nut.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3]
            )
        nut_positions_all.append(torch.stack(nut_pos_w))

    # Concatenate all nuts’ positions along dim=1
    return torch.cat(nut_positions_all, dim=1)



def nut_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The orientation of all nuts (m8, m12, m16 × red, green, blue) in the world frame."""
    nut_orientations = []
    for nut_name in NUT_NAMES:
        nut: RigidObject = env.scene[nut_name]
        nut_orientations.append(nut.data.root_quat_w)

    return torch.cat(nut_orientations, dim=1)


def instance_randomize_nut_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The orientation of all nuts (m8, m12, m16 × red, green, blue) in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    nut_orientations_all = []

    # Iterate through all defined nuts
    for nut_name in NUT_NAMES:
        nut: RigidObjectCollection = env.scene[nut_name]
        nut_quat_w = []
        for env_id in range(env.num_envs):
            # Same access pattern as in the original function
            nut_quat_w.append(
                nut.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4]
            )
        nut_orientations_all.append(torch.stack(nut_quat_w))

    # Concatenate all nuts’ orientations along dim=1
    return torch.cat(nut_orientations_all, dim=1)


def object_obs(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Object observations (in world frame):
        - all 9 nuts' positions (world - env_origin)
        - all 9 nuts' quaternions (world)
        - gripper->nut vectors for all 9 nuts
        - per-color pairwise deltas:
            (m8 - m12), (m12 - m16), (m8 - m16) for red/green/blue
    """
    # End-effector world pose
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    # Gather per-nut tensors in canonical order defined by NUT_NAMES
    pos_list = []
    quat_list = []
    ee_rel_list = []
    for nut_name in NUT_NAMES:
        nut: RigidObject = env.scene[nut_name]
        pos_w = nut.data.root_pos_w
        quat_w = nut.data.root_quat_w
        pos_list.append(pos_w - env.scene.env_origins)
        quat_list.append(quat_w)
        ee_rel_list.append(pos_w - ee_pos_w)

    # Per-color pairwise deltas (mirrors original m8↔m12↔m16 logic, done for each color)
    pairwise_list = []
    for color in NUT_COLORS:
        # indices into NUT_NAMES for this color and sizes m8, m12, m16
        i_m8  = NUT_NAMES.index(f"nut_m8_{color}")
        i_m12 = NUT_NAMES.index(f"nut_m12_{color}")
        i_m16 = NUT_NAMES.index(f"nut_m16_{color}")

        # Use world positions (not shifted by env origins) for deltas, as in your original
        pos_m8_w  = (pos_list[i_m8]  + env.scene.env_origins)
        pos_m12_w = (pos_list[i_m12] + env.scene.env_origins)
        pos_m16_w = (pos_list[i_m16] + env.scene.env_origins)

        pairwise_list.extend([
            pos_m8_w  - pos_m12_w,   # m8 -> m12
            pos_m12_w - pos_m16_w,   # m12 -> m16
            pos_m8_w  - pos_m16_w,   # m8 -> m16
        ])

    return torch.cat(
        (
            torch.cat(pos_list, dim=1),   # 9 * 3
            torch.cat(quat_list, dim=1),  # 9 * 4
            torch.cat(ee_rel_list, dim=1),# 9 * 3
            torch.cat(pairwise_list, dim=1),  # 3 colors * 3 pairs * 3 = 27
        ),
        dim=1,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame) for all 9 nuts:
        - per-nut pos (world - env_origin)
        - per-nut quat (world)
        - gripper -> nut vectors
        - per-color pairwise deltas: (m8 - m12), (m12 - m16), (m8 - m16)
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    # End-effector world position
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    # Collect per-nut world positions & orientations using the focus index per nut
    pos_w_list = []
    quat_w_list = []
    for i, nut_name in enumerate(NUT_NAMES):
        nut: RigidObjectCollection = env.scene[nut_name]
        pos_each_env = []
        quat_each_env = []
        for env_id in range(env.num_envs):
            focus_idx = env.rigid_objects_in_focus[env_id][i]
            pos_each_env.append(nut.data.object_pos_w[env_id, focus_idx, :3])
            quat_each_env.append(nut.data.object_quat_w[env_id, focus_idx, :4])
        pos_w_list.append(torch.stack(pos_each_env))   # [N, 3]
        quat_w_list.append(torch.stack(quat_each_env)) # [N, 4]

    # Positions relative to env origins (as in your original)
    pos_rel_env_list = [p - env.scene.env_origins for p in pos_w_list]

    # Gripper -> nut vectors (world)
    ee_rel_list = [p - ee_pos_w for p in pos_w_list]

    # Per-color pairwise deltas (world): (m8 - m12), (m12 - m16), (m8 - m16)
    pairwise_list = []
    for color in ["red", "green", "blue"]:
        i_m8  = NUT_NAMES.index(f"nut_m8_{color}")
        i_m12 = NUT_NAMES.index(f"nut_m12_{color}")
        i_m16 = NUT_NAMES.index(f"nut_m16_{color}")

        p8  = pos_w_list[i_m8]
        p12 = pos_w_list[i_m12]
        p16 = pos_w_list[i_m16]

        pairwise_list.extend([
            p8  - p12,   # m8 -> m12
            p12 - p16,   # m12 -> m16
            p8  - p16,   # m8 -> m16
        ])

    return torch.cat(
        (
            torch.cat(pos_rel_env_list, dim=1),  # 9 * 3
            torch.cat(quat_w_list,     dim=1),   # 9 * 4
            torch.cat(ee_rel_list,     dim=1),   # 9 * 3
            torch.cat(pairwise_list,   dim=1),   # 3 colors * 3 pairs * 3 = 27
        ),
        dim=1,
    )
###


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observation gripper_pos only support parallel gripper for now"
            finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"

            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )

    return grasped


def object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Check if an object is stacked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                stacked,
            )
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                stacked,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def nut_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    nut_m8_cfg: SceneEntityCfg = SceneEntityCfg("nut_m8"),
    nut_m12_cfg: SceneEntityCfg = SceneEntityCfg("nut_m12"),
    nut_m16_cfg: SceneEntityCfg = SceneEntityCfg("nut_m16"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """The position and orientation of the nuts in the robot base frame."""

    nut_m8: RigidObject = env.scene[nut_m8_cfg.name]
    nut_m12: RigidObject = env.scene[nut_m12_cfg.name]
    nut_m16: RigidObject = env.scene[nut_m16_cfg.name]

    pos_nut_m8_world = nut_m8.data.root_pos_w
    pos_nut_m12_world = nut_m12.data.root_pos_w
    pos_nut_m16_world = nut_m16.data.root_pos_w

    quat_nut_m8_world = nut_m8.data.root_quat_w
    quat_nut_m12_world = nut_m12.data.root_quat_w
    quat_nut_m16_world = nut_m16.data.root_quat_w

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    pos_nut_m8_base, quat_nut_m8_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_nut_m8_world, quat_nut_m8_world
    )
    pos_nut_m12_base, quat_nut_m12_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_nut_m12_world, quat_nut_m12_world
    )
    pos_nut_m16_base, quat_nut_m16_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_nut_m16_world, quat_nut_m16_world
    )

    pos_nuts_base = torch.cat((pos_nut_m8_base, pos_nut_m12_base, pos_nut_m16_base), dim=1)
    quat_nuts_base = torch.cat((quat_nut_m8_base, quat_nut_m12_base, quat_nut_m16_base), dim=1)

    if return_key == "pos":
        return pos_nuts_base
    elif return_key == "quat":
        return quat_nuts_base
    elif return_key is None:
        return torch.cat((pos_nuts_base, quat_nuts_base), dim=1)


def object_abs_obs_in_base_frame(
    env: ManagerBasedRLEnv,
    nut_m8_cfg: SceneEntityCfg = SceneEntityCfg("nut_m8"),
    nut_m12_cfg: SceneEntityCfg = SceneEntityCfg("nut_m12"),
    nut_m16_cfg: SceneEntityCfg = SceneEntityCfg("nut_m16"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Object Abs observations (in base frame): remove the relative observations, and add abs gripper pos and quat in robot base frame
        nut_m8 pos,
        nut_m8 quat,
        nut_m12 pos,
        nut_m12 quat,
        nut_m16 pos,
        nut_m16 quat,
        gripper pos,
        gripper quat,
    """
    nut_m8: RigidObject = env.scene[nut_m8_cfg.name]
    nut_m12: RigidObject = env.scene[nut_m12_cfg.name]
    nut_m16: RigidObject = env.scene[nut_m16_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    nut_m8_pos_w = nut_m8.data.root_pos_w
    nut_m8_quat_w = nut_m8.data.root_quat_w

    nut_m12_pos_w = nut_m12.data.root_pos_w
    nut_m12_quat_w = nut_m12.data.root_quat_w

    nut_m16_pos_w = nut_m16.data.root_pos_w
    nut_m16_quat_w = nut_m16.data.root_quat_w

    pos_nut_m8_base, quat_nut_m8_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, nut_m8_pos_w, nut_m8_quat_w
    )
    pos_nut_m12_base, quat_nut_m12_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, nut_m12_pos_w, nut_m12_quat_w
    )
    pos_nut_m16_base, quat_nut_m16_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, nut_m16_pos_w, nut_m16_quat_w
    )

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    return torch.cat(
        (
            pos_nut_m8_base,
            quat_nut_m8_base,
            pos_nut_m12_base,
            quat_nut_m12_base,
            pos_nut_m16_base,
            quat_nut_m16_base,
            ee_pos_base,
            ee_quat_base,
        ),
        dim=1,
    )

### End for nuts environment


def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """
    The end effector pose in the robot base frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_in_base, ee_quat_in_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_frame_pos_w, ee_frame_quat_w
    )

    if return_key == "pos":
        return ee_pos_in_base
    elif return_key == "quat":
        return ee_quat_in_base
    elif return_key is None:
        return torch.cat((ee_pos_in_base, ee_quat_in_base), dim=1)
