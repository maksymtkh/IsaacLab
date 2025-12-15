# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def task_done_place_with_gripper_check(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    zone: str = "middle",  # "left", "middle", or "right"
    max_height: float = 1.10,
    vel_threshold: float = 0.20,
) -> torch.Tensor:
    """
    Determine if the object placement task is complete.
    The function checks if:
      1) the object is within the selected XY placement zone ("left", "middle", or "right")
      2) the object is below the given height
      3) object velocity is below the given threshold
      4) gripper fingers are near open position
    """

    # Define the XY zones for bigger working area
    zones = {
        "left":   {"min_x": 0.6, "max_x": 0.72, "min_y": 0.036, "max_y": 0.16},
        "middle": {"min_x": 0.6, "max_x": 0.72, "min_y": 0.13, "max_y": 0.34},
        "right":  {"min_x": 0.6, "max_x": 0.72, "min_y": 0.36,  "max_y": 0.57},
    }

    """# Define the XY zones for smaller working area
    # !!!Just for left was changed
    zones = {
        "left":   {"min_x": 0.59, "max_x": 0.69, "min_y": -0.1, "max_y": 0.11},
        "middle": {"min_x": 0.6, "max_x": 0.72, "min_y": 0.13, "max_y": 0.34},
        "right":  {"min_x": 0.6, "max_x": 0.72, "min_y": 0.36,  "max_y": 0.57},
    }"""

    if zone not in zones:
        raise ValueError(f"Invalid zone '{zone}'. Must be one of: {list(zones.keys())}")

    zone_cfg = zones[zone]

    # Entities
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Object pose/vel relative to environment origins
    obj_pos_w = obj.data.root_pos_w
    obj_vel_w = obj.data.root_vel_w

    obj_x = obj_pos_w[:, 0] - env.scene.env_origins[:, 0]
    obj_y = obj_pos_w[:, 1] - env.scene.env_origins[:, 1]
    obj_h = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
    obj_vel_abs = torch.abs(obj_vel_w)

    # --- Zone check ---
    inside_x = torch.logical_and(obj_x > zone_cfg["min_x"], obj_x < zone_cfg["max_x"])
    inside_y = torch.logical_and(obj_y > zone_cfg["min_y"], obj_y < zone_cfg["max_y"])
    in_zone = torch.logical_and(inside_x, inside_y)

    # --- Height + velocity conditions ---
    below_height = obj_h < max_height
    slow_movement = torch.logical_and(
        torch.logical_and(obj_vel_abs[:, 0] < vel_threshold, obj_vel_abs[:, 1] < vel_threshold),
        obj_vel_abs[:, 2] < vel_threshold,
    )

    # Combine all criteria
    done = torch.logical_and(in_zone, below_height)
    done = torch.logical_and(done, slow_movement)

    # --- Gripper open check ---
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "This function only supports parallel grippers"

        j0 = robot.data.joint_pos[:, gripper_joint_ids[0]]
        j1 = robot.data.joint_pos[:, gripper_joint_ids[1]]
        open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
        tol = torch.tensor(env.cfg.gripper_threshold, dtype=torch.float32).to(env.device)

        gripper_open = torch.logical_and(
            torch.abs(torch.abs(j0) - open_val) < tol,
            torch.abs(torch.abs(j1) - open_val) < tol,
        )

        done = torch.logical_and(done, gripper_open)
    else:
        raise ValueError("No gripper_joint_names found in environment config")

    return done

def height_below_minimum(
    env: ManagerBasedRLEnv,
    pose_range: dict[str, float],
    asset_cfgs: list[SceneEntityCfg],
) -> torch.Tensor:
    """Terminate if ANY rigid object from asset_cfgs has height (z) below the minimum threshold."""
    minimum_height = pose_range["z"]

    terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for asset_cfg in asset_cfgs:
        asset: RigidObject = env.scene[asset_cfg.name]
        height = asset.data.root_pos_w[:, 2]
        terminate |= height < minimum_height

    return terminate

def position_xy_out_of_bounds(
    env: ManagerBasedRLEnv,
    pose_range: dict[str, tuple[float, float]],
    asset_cfgs: list[SceneEntityCfg],
) -> torch.Tensor:
    """Terminate if ANY rigid object from asset_cfgs leaves the allowed XY bounds."""
    min_x, max_x = pose_range["x"]
    min_y, max_y = pose_range["y"]

    terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for asset_cfg in asset_cfgs:
        asset: RigidObject = env.scene[asset_cfg.name]
        root_pos = asset.data.root_pos_w[:, :2]  # (x, y)

        out_x = (root_pos[:, 0] < min_x) | (root_pos[:, 0] > max_x)
        out_y = (root_pos[:, 1] < min_y) | (root_pos[:, 1] > max_y)

        terminate |= out_x | out_y

    return terminate

