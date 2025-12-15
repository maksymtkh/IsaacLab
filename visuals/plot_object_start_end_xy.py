#!/usr/bin/env python3
"""
Plot start and end XY positions of an object for ALL demos in a robomimic dataset.

For each demo in /data, this script:
    - Reads data/<demo>/<obs_group>/<obj_key>
    - Takes the first and last timestep
    - Projects to (x, y)
    - Adds them to a single scatter plot

Workspace bounds are fixed to:
    x in [0.29, 0.69]
    y in [0.03, 0.44]
"""

import argparse
import os
from typing import List, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt


# Workspace boundaries
X_BOUNDS = (0.29, 0.69)
Y_BOUNDS = (0.03, 0.44)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot start and end XY positions of an object for all demos (robomimic dataset)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the HDF5 dataset file (robomimic format).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store the generated plot image.",
    )
    parser.add_argument(
        "--obj_key",
        type=str,
        required=True,
        help="Observation key for object pose (e.g., 'object', 'object-state'). "
             "Must exist under data/<demo>/<obs_group>/<obj_key>.",
    )
    parser.add_argument(
        "--obs_group",
        type=str,
        default="obs",
        choices=["obs", "next_obs"],
        help="Observation group to use under each demo. Default: 'obs'.",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="Optional filter key under /mask/<filter_key> to select a subset of demos "
             "(e.g., 'train'). If not provided, all demos under /data are used.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8.0, 8.0],
        help="Figure size as: WIDTH HEIGHT. Default: 8 8",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure. Default: 150.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, display the plot interactively in addition to saving it.",
    )
    return parser.parse_args()


def get_demo_list(f: h5py.File, filter_key: Optional[str]) -> List[str]:
    """
    Return an ordered list of demo keys from the HDF5 file.

    If filter_key is provided, use demos listed in /mask/<filter_key>.
    Otherwise, use all demos under /data.
    """
    if filter_key is not None:
        print(f"Using filter key: {filter_key}")
        if "mask" not in f or filter_key not in f["mask"]:
            raise KeyError(f"Filter key '{filter_key}' not found under /mask in dataset.")
        demos = [elem.decode("utf-8") for elem in np.array(f[f"mask/{filter_key}"])]
        demos = sorted(demos)
    else:
        demos = sorted(list(f["data"].keys()))

    # Try to sort by integer index if keys are like "demo_0", "demo_1", ...
    try:
        inds = np.argsort([int(d[5:]) for d in demos])  # assumes "demo_" prefix
        demos = [demos[i] for i in inds]
    except Exception:
        pass

    return demos


def ensure_xy(arr: np.ndarray, key_name: str) -> np.ndarray:
    """
    Ensure array is at least (T, 2). Return arr[:, :2] as (x, y).

    Raises a ValueError if shape is incompatible.
    """
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"Observation '{key_name}' must have shape (T, D) with D >= 2 for XY plotting, "
            f"but got {arr.shape}."
        )
    return arr[:, :2]


def main():
    args = parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.dataset, "r") as f:
        demos = get_demo_list(f, args.filter_key)

        start_positions = []
        end_positions = []

        print(f"Found {len(demos)} demo(s). Collecting object start/end positions...")

        for i, demo in enumerate(demos):
            print(f"[{i + 1}/{len(demos)}] {demo}")

            base_path = f"data/{demo}/{args.obs_group}"
            if base_path not in f:
                raise KeyError(
                    f"Group '{base_path}' not found in dataset. "
                    f"Available groups under data/{demo}: {list(f[f'data/{demo}'].keys())}"
                )

            obs_group = f[base_path]
            if args.obj_key not in obs_group:
                raise KeyError(
                    f"Object observation key '{args.obj_key}' not found under '{base_path}'. "
                    f"Available obs keys: {list(obs_group.keys())}"
                )

            obj_arr = obs_group[args.obj_key][()]  # (T, D)
            obj_xy = ensure_xy(obj_arr, args.obj_key)

            start_positions.append(obj_xy[0])
            end_positions.append(obj_xy[-1])

        start_positions = np.array(start_positions)  # (N, 2)
        end_positions = np.array(end_positions)      # (N, 2)

    # -------------
    # Plotting
    # -------------
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    ax.scatter(
        start_positions[:, 0],
        start_positions[:, 1],
        marker="o",
        alpha=0.7,
        label="Object start",
    )
    ax.scatter(
        end_positions[:, 0],
        end_positions[:, 1],
        marker="x",
        alpha=0.7,
        label="Object end",
    )

    # Workspace bounds and axis settings
    ax.set_xlim(*X_BOUNDS)
    ax.set_ylim(*Y_BOUNDS)
    ax.set_aspect("equal", adjustable="box")

    # Draw boundary box
    ax.plot(
        [X_BOUNDS[0], X_BOUNDS[1], X_BOUNDS[1], X_BOUNDS[0], X_BOUNDS[0]],
        [Y_BOUNDS[0], Y_BOUNDS[0], Y_BOUNDS[1], Y_BOUNDS[1], Y_BOUNDS[0]],
        "k--",
        linewidth=1,
        label="Workspace bounds",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Object start and end XY positions\nobj_key = {args.obj_key}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_path = os.path.join(args.output_dir, f"object_start_end_xy_{args.obj_key}.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
