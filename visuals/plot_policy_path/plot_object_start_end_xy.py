#!/usr/bin/env python3
"""
Plot start and end XY positions of an object for ALL demos in two robomimic datasets.

- DATASET_PATH_ORIGINAL: expert demos (first 10 shown as blue points)
- DATASET_PATH: generated demos (shown as smooth spatial densities)

Visualization:
    - Generated start positions: green filled density
    - Generated end positions: red   filled density
    - 0 (and very low) density = white
    - Original (first 10) demos: blue markers on top

Suitable for thesis / publication figures.
"""

import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Workspace boundaries (choose SMALL or BIG)
X_BOUNDS_SMALL = (0.24, 0.74)
Y_BOUNDS_SMALL = (-0.02, 0.49)

X_BOUNDS_BIG = (0.25, 0.77)
Y_BOUNDS_BIG = (-0.15, 0.60)

#DATASET_PATH_ORIGINAL = "./datasets/11_dataset_Sort_SmallWA.hdf5"
#DATASET_PATH = "./datasets/39_generated_dataset_Sort_SmallWA.hdf5"
#OUTPUT_DIR = "./visuals/plot_start_end"
#OUTPUT_FILE = "object_start_end_smallwa.png"

X_BOUNDS = X_BOUNDS_BIG
Y_BOUNDS = Y_BOUNDS_BIG

DATASET_PATH_ORIGINAL = "./datasets/10_dataset_Sort_BigWA.hdf5"
DATASET_PATH_SUCCESS = "./visuals/plot_policy_path/datasets/Isaac-Sort-BigWA-UR5e-IK-Rel-v0_200Data.hdf5"
DATASET_PATH_FAILED = "./visuals/plot_policy_path/datasets/Isaac-Sort-BigWA-UR5e-IK-Rel-v0_200Data_failed.hdf5"

OUTPUT_DIR = "./visuals/plot_policy_path"
OUTPUT_FILE = "object_start_end.png"

#X_BOUNDS = X_BOUNDS_BIG
#Y_BOUNDS = Y_BOUNDS_BIG

OBJ_KEY = "object"
OBS_GROUP = "obs"
FILTER_KEY: Optional[str] = None

FIGSIZE = (6.0, 6.0)
DPI = 300
SHOW_PLOT = False

# Matplotlib defaults
plt.rcParams.update(
    {
        "figure.figsize": FIGSIZE,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_demo_list(f: h5py.File, filter_key: Optional[str]) -> List[str]:
    if filter_key is not None:
        print(f"Using filter key: {filter_key}")
        if "mask" not in f or filter_key not in f["mask"]:
            raise KeyError(f"Filter key '{filter_key}' not found under /mask in dataset.")
        demos = [elem.decode("utf-8") for elem in np.array(f[f"mask/{filter_key}"])]
        demos = sorted(demos)
    else:
        demos = sorted(list(f["data"].keys()))

    # Sort by integer index if keys like "demo_0", "demo_1", ...
    try:
        inds = np.argsort([int(d[5:]) for d in demos])
        demos = [demos[i] for i in inds]
    except Exception:
        pass

    return demos


def ensure_xy(arr: np.ndarray, key_name: str) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"Observation '{key_name}' must have shape (T, D) with D >= 2, got {arr.shape}."
        )
    return arr[:, :2]


def compute_density(
    points: np.ndarray,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    bins: int = 150,
    sigma: float = 2.0,
):
    """Histogram2d + optional Gaussian blur; returns H.T, xedges, yedges."""
    if points.size == 0:
        H = np.zeros((bins, bins), dtype=np.float32)
        xedges = np.linspace(x_bounds[0], x_bounds[1], bins + 1)
        yedges = np.linspace(y_bounds[0], y_bounds[1], bins + 1)
        return H, xedges, yedges

    H, xedges, yedges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=[[x_bounds[0], x_bounds[1]],
               [y_bounds[0], y_bounds[1]]],
    )

    if sigma is not None:
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=sigma)
        except ImportError:
            pass

    return H.T, xedges, yedges


def collect_start_end_positions(
    dataset_path: str,
    obj_key: str,
    obs_group: str,
    filter_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with h5py.File(dataset_path, "r") as f:
        demos = get_demo_list(f, filter_key)
        start_positions = []
        end_positions = []

        print(f"\nDataset: {dataset_path}")
        print(f"  Found {len(demos)} demo(s). Collecting object start/end positions...")

        for i, demo in enumerate(demos):
            print(f"  [{i + 1}/{len(demos)}] {demo}")

            base_path = f"data/{demo}/{obs_group}"
            if base_path not in f:
                raise KeyError(
                    f"Group '{base_path}' not found. "
                    f"Available groups under data/{demo}: {list(f[f'data/{demo}'].keys())}"
                )

            obs_grp = f[base_path]
            if obj_key not in obs_grp:
                raise KeyError(
                    f"Object key '{obj_key}' not found under '{base_path}'. "
                    f"Available obs keys: {list(obs_grp.keys())}"
                )

            obj_arr = obs_grp[obj_key][()]  # (T, D)
            obj_xy = ensure_xy(obj_arr, obj_key)

            start_positions.append(obj_xy[0])
            end_positions.append(obj_xy[-1])

    return np.array(start_positions), np.array(end_positions)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Original (reference) dataset
    start_orig, end_orig = collect_start_end_positions(
        DATASET_PATH_ORIGINAL, OBJ_KEY, OBS_GROUP, FILTER_KEY
    )

    start_success, end_success = collect_start_end_positions(
        DATASET_PATH_SUCCESS, OBJ_KEY, OBS_GROUP, FILTER_KEY
    )

    start_failed, end_failed = collect_start_end_positions(
        DATASET_PATH_FAILED, OBJ_KEY, OBS_GROUP, FILTER_KEY
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)


    # --- original demos on top (blue) ---
    start_orig_scatter = None
    if start_orig.shape[0] > 0:
        start_orig_scatter = ax.scatter(
            start_orig[:, 0],
            start_orig[:, 1],
            color="blue",
            marker="o",
            linewidths=1.5,
            s=35,
            label="Original Pick Positionen",
            zorder=5,
        )

    # --- success demos (green) ---
    start_success_scatter = None
    if start_success.shape[0] > 0:
        start_success_scatter = ax.scatter(
            start_success[:, 0],
            start_success[:, 1],
            color="lightgrey",
            alpha=0.8,
            marker="o",
            linewidths=1.5,
            s=35,
            label="Policy Pick Positionen (Success)",
            zorder=3,
        )

    # --- success demos (green) ---
    start_failed_scatter = None
    if start_failed.shape[0] > 0:
        start_failed_scatter = ax.scatter(
            start_failed[:, 0],
            start_failed[:, 1],
            color="red",
            marker="o",
            linewidths=1.5,
            s=35,
            label="Policy Pick Positionen (Failed)",
            zorder=4,
        )



    

    # --- workspace bounds ---
    a = 0.05
    xmin, xmax = X_BOUNDS
    ymin, ymax = Y_BOUNDS
    workspace_line = ax.plot(
        [xmin + a, xmax - a, xmax - a, xmin + a, xmin + a],
        [ymin + a, ymin + a, ymax - a, ymax - a, ymin + a],
        "k--",
        linewidth=1,
        label="Arbeitsbereich",
        zorder=4,
    )[0]

    # axes formatting
    ax.set_xlim(X_BOUNDS[0], X_BOUNDS[1]+0.3)
    ax.set_ylim(Y_BOUNDS[0], Y_BOUNDS[1])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # legend: use the actual scatter artists
    handles, labels = [], []

    if start_orig_scatter is not None:
        handles.append(start_orig_scatter)
        labels.append("Original Pick Positionen")

    if start_success_scatter is not None:
        handles.append(start_success_scatter)
        labels.append("Policy Pick Positionen (erfolgreich)")

    if start_failed_scatter is not None:
        handles.append(start_failed_scatter)
        labels.append("Policy Pick Positionen (fehlgeschlagen)")

    handles.append(workspace_line)
    labels.append("Großer Arbeitsbereich")

    ax.legend(handles, labels, loc="upper right", frameon=True)


    fig.tight_layout()

    out_path = os.path.join(
        OUTPUT_DIR, OUTPUT_FILE,
    )
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")

    if SHOW_PLOT:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
