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

DATASET_PATH_ORIGINAL = "./datasets/11_dataset_Sort_SmallWA.hdf5"
DATASET_PATH = "./datasets/39_generated_dataset_Sort_SmallWA.hdf5"
OUTPUT_DIR = "./visuals/plot_start_end"
OUTPUT_FILE = "object_start_end_smallwa.png"

X_BOUNDS = X_BOUNDS_SMALL
Y_BOUNDS = Y_BOUNDS_SMALL

"""DATASET_PATH_ORIGINAL = "./datasets/10_dataset_Sort_BigWA.hdf5"
DATASET_PATH = "./datasets/38_generated_dataset_Sort_BigWA.hdf5"
OUTPUT_DIR = "./visuals/plot_start_end"
OUTPUT_FILE = "object_start_end_bigwa.png"

X_BOUNDS = X_BOUNDS_BIG
Y_BOUNDS = Y_BOUNDS_BIG"""

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

    # Generated dataset (for densities)
    start_gen, end_gen = collect_start_end_positions(
        DATASET_PATH, OBJ_KEY, OBS_GROUP, FILTER_KEY
    )

    # First 10 original demos to highlight
    N_orig = start_orig.shape[0]
    K = min(10, N_orig)
    start_highlight = start_orig[:K]
    end_highlight = end_orig[:K]

    # Densities over generated data
    start_rest = start_gen
    end_rest = end_gen

    H_start, xedges_s, yedges_s = compute_density(
        start_rest, X_BOUNDS, Y_BOUNDS, bins=150, sigma=2.0
    )
    H_end, xedges_e, yedges_e = compute_density(
        end_rest, X_BOUNDS, Y_BOUNDS, bins=150, sigma=2.0
    )

    # Normalize
    Hs_norm = H_start / H_start.max() if H_start.max() > 0 else H_start
    He_norm = H_end / H_end.max() if H_end.max() > 0 else H_end

    # Mask very low densities → appear white (no tint)
    threshold = 1e-1
    Hs_plot = np.ma.masked_less(Hs_norm, threshold)
    He_plot = np.ma.masked_less(He_norm, threshold)

    xs = 0.5 * (xedges_s[:-1] + xedges_s[1:])
    ys = 0.5 * (yedges_s[:-1] + yedges_s[1:])
    Xg, Yg = np.meshgrid(xs, ys)

    # Colormaps
    green_cmap = cm.get_cmap("Greens")
    red_cmap = cm.get_cmap("Reds")

    levels = np.linspace(threshold, 1.0, 10)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # --- start density (green) ---
    cs_start = ax.contourf(
        Xg,
        Yg,
        Hs_plot,
        levels=levels,
        cmap=green_cmap,
        alpha=0.8,
    )

    # --- end density (red) ---
    cs_end = None
    if He_plot.count() > 0:
        cs_end = ax.contourf(
            Xg,
            Yg,
            He_plot,
            levels=levels,
            cmap=red_cmap,
            alpha=0.8,
        )

    # ---------- separate colorbar axes ----------
    divider = make_axes_locatable(ax)

    # one narrow axis for each colorbar, side by side
    cax_start = divider.append_axes("right", size="3%", pad=0.10)
    cax_end   = divider.append_axes("right", size="3%", pad=0.55)

    # attach colorbars to those axes
    cbar_start = fig.colorbar(cs_start, cax=cax_start)
    cbar_start.set_label("Generierte Pick Positionen (normalisiert)")

    if cs_end is not None:
        cbar_end = fig.colorbar(cs_end, cax=cax_end)
        cbar_end.set_label("Generierte Place Positionen (normalisiert)")


    # --- original demos on top (blue) ---
    start_scatter = end_scatter = None
    if K > 0:
        start_scatter = ax.scatter(
            start_highlight[:, 0],
            start_highlight[:, 1],
            color="blue",
            marker="o",
            linewidths=1.5,
            s=35,
            label="Original Pick Positionen",
            zorder=5,
        )
        end_scatter = ax.scatter(
            end_highlight[:, 0],
            end_highlight[:, 1],
            color="blue",
            marker="x",
            linewidths=1.5,
            s=45,
            label="Original Place Positionen",
            zorder=5,
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
        label="Kleiner Arbeitsbereich",
        zorder=4,
    )[0]

    # axes formatting
    ax.set_xlim(*X_BOUNDS)
    ax.set_ylim(Y_BOUNDS[0], Y_BOUNDS[1]+0.1)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # legend: use real artists, no proxies -> no overlap issues
    handles, labels = [], []
    if start_scatter is not None:
        handles.append(start_scatter)
        labels.append("Original Pick Positionen")
    if end_scatter is not None:
        handles.append(end_scatter)
        labels.append("Original Place Positionen")
    handles.append(workspace_line)
    labels.append("Kleiner Arbeitsbereich")

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
