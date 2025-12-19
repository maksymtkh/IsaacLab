# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,

    sort_ik_rel_env_bigwa_cfg,
    sort_ik_rel_env_ooak_cfg,
    sort_ik_rel_env_smallwa_cfg,

    sort_joint_pos_env_bigwa_cfg,
    sort_joint_pos_env_ooak_cfg,
    sort_joint_pos_env_smallwa_cfg,

    sort_ik_rel_visuomotor_env_bigwa_cfg,
    sort_ik_rel_visuomotor_env_ooak_cfg,
    sort_ik_rel_visuomotor_env_smallwa_cfg

)

# -----------------------------------------------------------------------------
# Register Gym environments
# -----------------------------------------------------------------------------

"""
Environments IDs
| Control Mode     | Parameter | BigWA                                                      | SmallWA                                                     | OOAK                                                       |
|------------------|-----------|------------------------------------------------------------|-------------------------------------------------------------|------------------------------------------------------------|
| Joint Position   | -         | Isaac-Sort-BigWA-UR5e-v0                                   | Isaac-Sort-SmallWA-UR5e-v0                                  | Isaac-Sort-OOAK-UR5e-v0                                    |
| IK (No Camera)   | -         | Isaac-Sort-BigWA-UR5e-IK-Rel-v0                            | Isaac-Sort-SmallWA-UR5e-IK-Rel-v0                           | Isaac-Sort-OOAK-UR5e-IK-Rel-v0                             |
| IK Visuomotor    | Param 1   | Isaac-Sort-BigWA-UR5e-IK-Rel-Visuomotor-Param1-v0          | Isaac-Sort-SmallWA-UR5e-IK-Rel-Visuomotor-Param1-v0         | Isaac-Sort-OOAK-UR5e-IK-Rel-Visuomotor-Param1-v0           |
| IK Visuomotor    | Param 2   | Isaac-Sort-BigWA-UR5e-IK-Rel-Visuomotor-Param2-v0          | -                                                           | -                                                          |
| IK Visuomotor    | Param 3   | Isaac-Sort-BigWA-UR5e-IK-Rel-Visuomotor-Param3-v0          | Isaac-Sort-SmallWA-UR5e-IK-Rel-Visuomotor-Param3-v0         | -                                                          |
| IK Visuomotor    | Param 4   | Isaac-Sort-BigWA-UR5e-IK-Rel-Visuomotor-Param4-v0          | Isaac-Sort-SmallWA-UR5e-IK-Rel-Visuomotor-Param4-v0         | -                                                          |

Parameters for Visuomotions are as followed:
1.  Data type:  RGB
    Focus:      100m
2.  Data type:  RGB, Depth
    Focus:      100m
3.  Data type:  RGB
    Focus:      0.5m
4.  Data type:  RGB
    Focus:      0.1m

Environments interpretation:
1.  biwa -      big working area
2.  smallwa -   small working area
3.  ooak -      one of a kind; 1 nut size, 3 colors on big working area

IK Visuomotor sum up
| Parameter     | bigwa | smallwa | ooak |
|---------------|-------|---------|------|
| Parameter 1   |   x   |     x   |  x   |
| Parameter 2   |   x   |         |      |
| Parameter 3   |   x   |     x   |      |
| Parameter 4   |   x   |     x   |      |

"""

# -----------------------------------------------------------------------------
# Joint Position Control
# -----------------------------------------------------------------------------

gym.register(
    id="Isaac-Sort-BigWA-UR5e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_joint_pos_env_bigwa_cfg.UR5eSortEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-OOAK-UR5e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_joint_pos_env_ooak_cfg.UR5eSortEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-SmallWA-UR5e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_joint_pos_env_smallwa_cfg.UR5eSortEnvCfg,
    },
    disable_env_checker=True,
)

# -----------------------------------------------------------------------------
# Inverse Kinematics Policy without Cameras
# -----------------------------------------------------------------------------

gym.register(
    id="Isaac-Sort-BigWA-UR5e-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_env_bigwa_cfg.UR5eSortEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-OOAK-UR5e-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_env_ooak_cfg.UR5eSortEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-SmallWA-UR5e-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_env_smallwa_cfg.UR5eSortEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

"""# -----------------------------------------------------------------------------
# Inverse Kinematics Policy with Camera Parameter 1
# -----------------------------------------------------------------------------

gym.register(
    id="Isaac-Sort-BigWA-UR5e-IK-Rel-Visuomotor-Param1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_bigwa_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_param1.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-SmallWA-UR5e-IK-Rel-Visuomotor-Param1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_smallwa_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_200.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-OOAK-UR5e-IK-Rel-Visuomotor-Param1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_ooak_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_param1.json"),
    },
    disable_env_checker=True,
)

# -----------------------------------------------------------------------------
# Inverse Kinematics Policy with Camera Parameter 2
# -----------------------------------------------------------------------------

gym.register(
    id="Isaac-Sort-BigWA-UR5e-IK-Rel-Visuomotor-Param2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_bigwa_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_param2.json"),
    },
    disable_env_checker=True,
)

# -----------------------------------------------------------------------------
# Inverse Kinematics Policy with Camera Parameter 3
# -----------------------------------------------------------------------------

gym.register(
    id="Isaac-Sort-BigWA-UR5e-IK-Rel-Visuomotor-Param3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_bigwa_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_param3.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-SmallWA-UR5e-IK-Rel-Visuomotor-Param3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_smallwa_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_param3.json"),
    },
    disable_env_checker=True,
)

# -----------------------------------------------------------------------------
# Inverse Kinematics Policy with Camera Parameter 4
# -----------------------------------------------------------------------------

gym.register(
    id="Isaac-Sort-BigWA-UR5e-IK-Rel-Visuomotor-Param4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_bigwa_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_param4.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-SmallWA-UR5e-IK-Rel-Visuomotor-Param4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_smallwa_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_param4.json"),
    },
    disable_env_checker=True,
)"""

# -----------------------------------------------------------------------------
# Inverse Kinematics Policy with Camera Parameter 1
# -----------------------------------------------------------------------------

gym.register(
    id="Isaac-Sort-BigWA-UR5e-IK-Rel-Visuomotor-Param1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_bigwa_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_480x640.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-SmallWA-UR5e-IK-Rel-Visuomotor-Param1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_smallwa_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_480x640.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Sort-OOAK-UR5e-IK-Rel-Visuomotor-Param1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sort_ik_rel_visuomotor_env_ooak_cfg.UR5eSortVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_480x640.json"),
    },
    disable_env_checker=True,
)