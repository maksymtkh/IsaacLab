#!/usr/bin/env python3
"""
Plot XY trajectories and Z-vs-time for end-effector and object
for:
  - ALL demos in DATASET_PATH_ORIGINAL
  - 10 RANDOM demos in DATASET_PATH (generated)

Visualization:
    - XY trajectory (top view) with workspace bounds
    - Z over time (height) for EEF and object

Output filenames clearly distinguish:
    <demo>_orig_xy_z.png      -> from original dataset
    <demo>_gen_xy_z.png       -> from generated dataset
"""

import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG (edit these to your needs)
# ---------------------------------------------------------------------

# Workspace boundaries (choose SMALL or BIG)
X_BOUNDS_SMALL = (0.24, 0.74)
Y_BOUNDS_SMALL = (-0.02, 0.49)

X_BOUNDS_BIG = (0.25, 0.77)
Y_BOUNDS_BIG = (-0.15, 0.60)

"""DATASET_PATH_ORIGINAL = "./datasets/11_dataset_Sort_SmallWA.hdf5"
DATASET_PATH_GENERATED = "./datasets/39_generated_dataset_Sort_SmallWA.hdf5"
OUTPUT_DIR = "./visuals/plot_path/plot_smallwa"

X_BOUNDS = X_BOUNDS_SMALL
Y_BOUNDS = Y_BOUNDS_SMALL"""

DATASET_PATH_ORIGINAL = "./datasets/10_dataset_Sort_BigWA.hdf5"
DATASET_PATH_GENERATED = "./visuals/plot_policy_path/datasets/Isaac-Sort-BigWA-UR5e-IK-Rel-v0.hdf5"
OUTPUT_DIR = "./visuals/plot_policy_path"

X_BOUNDS = X_BOUNDS_BIG
Y_BOUNDS = Y_BOUNDS_BIG

EEF_KEY = "eef_pos"   # example, change to your key
OBJ_KEY = "object"    # observation key for object pose
OBS_GROUP = "obs"     # "obs" or "next_obs"

NUM_RANDOM_GEN = 10   # plot this many random demos from generated dataset
RANDOM_SEED = 0       # for reproducibility

FIGSIZE = (8.0, 10.0)   # (width, height)
DPI = 150
SHOW_PLOT = False

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def ensure_xyz(arr: np.ndarray, key: str) -> np.ndarray:
    """
    Ensure array is (T, >=3). Return arr[:, :3].
    """
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{key} must have shape (T, >=3), got {arr.shape}")
    return arr[:, :3]


def plot_episode_xy_z(
    eef_xyz: np.ndarray,
    obj_xyz: np.ndarray,
    demo: str,
    dataset_label: str,
    output_dir: str,
    figsize: Tuple[float, float],
    dpi: int,
    show: bool,
):
    """
    Plot XY trajectories (top subplot) and Z vs time (bottom subplot)
    for a single episode.

    dataset_label: short string to distinguish origin, e.g. "orig" or "gen"
    """
    T = eef_xyz.shape[0]
    t = np.arange(T)

    fig, (ax_xy, ax_z) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # -------------------------
    # XY trajectory (top-down)
    # -------------------------
    # --- trajectory lines (below) ---
    ax_xy.plot(
        eef_xyz[:, 0], eef_xyz[:, 1],
        color="green",
        label="EEF Trajektorie",
        linewidth=2,
        zorder=2,
    )
    ax_xy.plot(
        obj_xyz[:, 0], obj_xyz[:, 1],
        color="blue",
        linestyle="--",
        label="Objekt Trajektorie",
        linewidth=2,
        zorder=2,
    )

    # --- start/end markers (above) ---
    # EEF start / end
    ax_xy.scatter(
        *eef_xyz[0, :2],
        marker="o",
        s=60,
        color="green",
        linewidths=1.5,
        label="EEF Start",
        zorder=5,
    )
    ax_xy.scatter(
        *eef_xyz[-1, :2],
        marker="x",
        s=60,
        color="green",
        linewidths=2,
        label="EEF Ende",
        zorder=5,
    )

    # Object pick / place
    ax_xy.scatter(
        *obj_xyz[0, :2],
        marker="o",
        s=70,
        color="blue",
        linewidths=1.5,
        label="Pick Position",
        zorder=6,
    )
    ax_xy.scatter(
        *obj_xyz[-1, :2],
        marker="x",
        s=60,
        color="blue",
        linewidths=1.5,
        label="Place Position",
        zorder=6,
    )

    # --- workspace bounds ---
    a = 0.05
    xmin, xmax = X_BOUNDS
    ymin, ymax = Y_BOUNDS
    workspace_line = ax_xy.plot(
        [xmin + a, xmax - a, xmax - a, xmin + a, xmin + a],
        [ymin + a, ymin + a, ymax - a, ymax - a, ymin + a],
        "k--",
        linewidth=1,
        label="Großer Arbeitsbereich",
    )[0]

    ax_xy.set_xlim(X_BOUNDS[0], X_BOUNDS[1]+0.05)
    ax_xy.set_ylim(Y_BOUNDS)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    #ax_xy.set_title(f"XY Trajectory — {demo} ({dataset_label})")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

    # -------------------------
    # Z vs time
    # -------------------------
    ax_z.plot(
        t,
        eef_xyz[:, 2],
        color="green",
        linewidth=2,
        label="EEF Trajektorie",
    )
    ax_z.plot(
        t,
        obj_xyz[:, 2],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Objekt Trajektorie",
    )

    ax_z.set_xlabel("Timestep [-]")
    ax_z.set_ylabel("Z [m]")
    #ax_z.set_title("Höhe über Zeit")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc="best")

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{demo}_{dataset_label}_xy_z.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")

    if show:
        plt.show()
    plt.close(fig)


def list_demos(f: h5py.File, obs_group: str) -> List[str]:
    demos = sorted(f["data"].keys())
    # ensure group exists, and consistent naming
    valid_demos = []
    for d in demos:
        grp = f"data/{d}/{obs_group}"
        if grp in f:
            valid_demos.append(d)
    return valid_demos


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------
    # ORIGINAL DATASET: all demos
    # -----------------------------
    """if not os.path.exists(DATASET_PATH_ORIGINAL):
        raise FileNotFoundError(f"Original dataset not found: {DATASET_PATH_ORIGINAL}")

    with h5py.File(DATASET_PATH_ORIGINAL, "r") as f_orig:
        demos_orig = list_demos(f_orig, OBS_GROUP)
        print(f"Original dataset: {DATASET_PATH_ORIGINAL}")
        print(f"  -> Found {len(demos_orig)} demo(s). Plotting ALL of them.")

        for demo in demos_orig:
            obs_group = f_orig[f"data/{demo}/{OBS_GROUP}"]

            if EEF_KEY not in obs_group:
                raise KeyError(
                    f"EEF key '{EEF_KEY}' not found in data/{demo}/{OBS_GROUP}. "
                    f"Available keys: {list(obs_group.keys())}"
                )
            if OBJ_KEY not in obs_group:
                raise KeyError(
                    f"Object key '{OBJ_KEY}' not found in data/{demo}/{OBS_GROUP}. "
                    f"Available keys: {list(obs_group.keys())}"
                )

            eef_xyz = ensure_xyz(obs_group[EEF_KEY][()], EEF_KEY)
            obj_xyz = ensure_xyz(obs_group[OBJ_KEY][()], OBJ_KEY)

            T = min(len(eef_xyz), len(obj_xyz))

            plot_episode_xy_z(
                eef_xyz[:T],
                obj_xyz[:T],
                demo,
                dataset_label="orig",
                output_dir=OUTPUT_DIR,
                figsize=FIGSIZE,
                dpi=DPI,
                show=SHOW_PLOT,
            )"""

    # ------------------------------------
    # GENERATED DATASET: 10 random demos
    # ------------------------------------
    if not os.path.exists(DATASET_PATH_GENERATED):
        raise FileNotFoundError(f"Generated dataset not found: {DATASET_PATH_GENERATED}")

    with h5py.File(DATASET_PATH_GENERATED, "r") as f_gen:
        demos_gen_all = list_demos(f_gen, OBS_GROUP)
        print(f"\nGenerated dataset: {DATASET_PATH_GENERATED}")
        print(f"  -> Found {len(demos_gen_all)} demo(s) in total.")

        rng = np.random.default_rng(RANDOM_SEED)
        num_to_sample = min(NUM_RANDOM_GEN, len(demos_gen_all))
        demos_gen_sampled = list(rng.choice(demos_gen_all, size=num_to_sample, replace=False))

        print(f"  -> Sampling {num_to_sample} random demo(s): {demos_gen_sampled}")

        for demo in demos_gen_sampled:
            obs_group = f_gen[f"data/{demo}/{OBS_GROUP}"]

            if EEF_KEY not in obs_group:
                raise KeyError(
                    f"EEF key '{EEF_KEY}' not found in data/{demo}/{OBS_GROUP}. "
                    f"Available keys: {list(obs_group.keys())}"
                )
            if OBJ_KEY not in obs_group:
                raise KeyError(
                    f"Object key '{OBJ_KEY}' not found in data/{demo}/{OBS_GROUP}. "
                    f"Available keys: {list(obs_group.keys())}"
                )

            eef_xyz = ensure_xyz(obs_group[EEF_KEY][()], EEF_KEY)
            obj_xyz = ensure_xyz(obs_group[OBJ_KEY][()], OBJ_KEY)

            T = min(len(eef_xyz), len(obj_xyz))

            plot_episode_xy_z(
                eef_xyz[:T],
                obj_xyz[:T],
                demo,
                dataset_label="gen",
                output_dir=OUTPUT_DIR,
                figsize=FIGSIZE,
                dpi=DPI,
                show=SHOW_PLOT,
            )


if __name__ == "__main__":
    main()
