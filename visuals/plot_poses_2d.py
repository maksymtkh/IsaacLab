#!/usr/bin/env python3

import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Workspace boundaries
# -------------------------
X_BOUNDS = (0.29, 0.69)
Y_BOUNDS = (0.03, 0.44)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot XY trajectories and Z-vs-time for end-effector and object (no 3D plotting)."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eef_key", type=str, required=True)
    parser.add_argument("--obj_key", type=str, required=True)
    parser.add_argument("--obs_group", type=str, default="obs", choices=["obs", "next_obs"])
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--figsize", type=float, nargs=2, default=[8, 10])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def ensure_xyz(arr: np.ndarray, key: str) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{key} must have shape (T, >=3), got {arr.shape}")
    return arr[:, :3]


def plot_episode_xy_z(
    eef_xyz,
    obj_xyz,
    demo,
    eef_key,
    obj_key,
    output_dir,
    figsize,
    dpi,
    show,
):
    t = np.arange(eef_xyz.shape[0])

    fig, (ax_xy, ax_z) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]}
    )

    # -------------------------
    # XY trajectory (top-down)
    # -------------------------
    ax_xy.plot(eef_xyz[:, 0], eef_xyz[:, 1], label="EEF", linewidth=2)
    ax_xy.plot(obj_xyz[:, 0], obj_xyz[:, 1], "--", label="Object", linewidth=2)

    ax_xy.scatter(*eef_xyz[0, :2], marker="o", label="EEF start")
    ax_xy.scatter(*eef_xyz[-1, :2], marker="^", label="EEF end")
    ax_xy.scatter(*obj_xyz[0, :2], marker="x", label="Object start")
    ax_xy.scatter(*obj_xyz[-1, :2], marker="s", label="Object end")

    # Workspace bounds
    ax_xy.set_xlim(*X_BOUNDS)
    ax_xy.set_ylim(*Y_BOUNDS)
    ax_xy.set_aspect("equal", adjustable="box")

    # Draw boundary box
    ax_xy.plot(
        [X_BOUNDS[0], X_BOUNDS[1], X_BOUNDS[1], X_BOUNDS[0], X_BOUNDS[0]],
        [Y_BOUNDS[0], Y_BOUNDS[0], Y_BOUNDS[1], Y_BOUNDS[1], Y_BOUNDS[0]],
        "k--",
        linewidth=1,
        label="Workspace bounds",
    )

    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_title(f"XY Trajectory — {demo}")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

    # -------------------------
    # Z vs time
    # -------------------------
    ax_z.plot(t, eef_xyz[:, 2], label="EEF Z", linewidth=2)
    ax_z.plot(t, obj_xyz[:, 2], "--", label="Object Z", linewidth=2)

    ax_z.set_xlabel("Timestep")
    ax_z.set_ylabel("Z [m]")
    ax_z.set_title("Height over time")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc="best")

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{demo}_xy_z.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    with h5py.File(args.dataset, "r") as f:
        demos = sorted(f["data"].keys())
        if args.max_episodes:
            demos = demos[: args.max_episodes]

        for demo in demos:
            obs = f[f"data/{demo}/{args.obs_group}"]
            print(obs)

            eef_xyz = ensure_xyz(obs[args.eef_key][()], args.eef_key)
            obj_xyz = ensure_xyz(obs[args.obj_key][()], args.obj_key)
            print(eef_xyz)
            print(obj_xyz)

            T = min(len(eef_xyz), len(obj_xyz))
            print(np.size(obj_xyz))
            plot_episode_xy_z(
                eef_xyz[:T],
                obj_xyz[:],
                demo,
                args.eef_key,
                args.obj_key,
                args.output_dir,
                tuple(args.figsize),
                args.dpi,
                args.show,
            )


if __name__ == "__main__":
    main()
