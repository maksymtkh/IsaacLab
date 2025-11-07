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


def object_a_is_into_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    xy_threshold: float = 0.03,  # xy_distance_threshold
    height_threshold: float = 0.04,  # height_distance_threshold
    height_diff: float = 0.0,  # expected height_diff
) -> torch.Tensor:
    """Check if an object a is put into another object b by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]

    # check object a is into object b
    pos_diff = object_a.data.root_pos_w - object_b.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    success = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    # Check gripper positions
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

        success = torch.logical_and(
            success,
            torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
            < env.cfg.gripper_threshold,
        )
        success = torch.logical_and(
            success,
            torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
            < env.cfg.gripper_threshold,
        )
    else:
        raise ValueError("No gripper_joint_names found in environment config")

    return success

#In case function def object_a_is_into_b doesnt work!!! Copy pasted from another environment in isac lab
def task_done_pick_place(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = 0.35,
    max_y: float = 0.60,
    max_height: float = 1.10,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Determine if the object placement task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x pos threshold)

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        min_x: Minimum x position of the object for task completion.
        max_x: Maximum x position of the object for task completion.
        min_y: Minimum y position of the object for task completion.
        max_y: Maximum y position of the object for task completion.
        max_height: Maximum height (z position) of the object for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]

    # Extract wheel position relative to environment origin
    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    object_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

    # Check all success conditions and combine with logical AND
    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    return done

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

    # Define the XY zones
    zones = {
        "left":   {"min_x": 0.6, "max_x": 0.72, "min_y": -0.1, "max_y": 0.11},
        "middle": {"min_x": 0.6, "max_x": 0.72, "min_y": 0.13, "max_y": 0.34},
        "right":  {"min_x": 0.6, "max_x": 0.72, "min_y": 0.36,  "max_y": 0.57},
    }

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


