#!/usr/bin/env python3
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# If a log filename is passed as command-line argument, use that.
# Otherwise, set a default name here.
if len(sys.argv) > 1:
    LOG_FILENAME = sys.argv[1]
else:
    # Change this to your actual log file name in the same folder as this script
    LOG_FILENAME = "Isaac-Sort-BigWA-UR5e-IK-Rel-v0_seed_3560"

SMOOTH_WINDOW = 7
PLOT_FILENAME = LOG_FILENAME + ".pdf"

# ---------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(script_dir, 'data', LOG_FILENAME)
plot_path = os.path.join(script_dir, 'plots', PLOT_FILENAME)

print(log_path)

if not os.path.isfile(log_path):
    raise FileNotFoundError(f"Log file not found: {log_path}")

# ---------------------------------------------------------------------
# Parse log file
#
# We look for:
#   - "model_epoch_XXX.pth"  -> model + epoch
#   - "Success rate: Y"      -> numeric value
#
# IMPORTANT:
#   Do NOT 'continue' after matching the model, because the success rate
#   can be on the same line: "[Model: ...] Success rate: 0.2"
# ---------------------------------------------------------------------

models = []
epochs = []
success_rates = []

current_model = None
current_epoch = None

with open(log_path, "r") as f:
    for raw_line in f:
        line = raw_line.strip()

        # 1) Detect model + epoch (loose pattern)
        #    Examples:
        #      "[Model: model_epoch_480.pth] Trial 0: False"
        #      "[Model: model_epoch_480.pth] Success rate: 0.2"
        m = re.search(r"model_epoch_(\d+)\.pth", line)
        if m:
            current_epoch = int(m.group(1))
            current_model = f"model_epoch_{current_epoch}.pth"
            # IMPORTANT: NO 'continue' here, we still want to check same line for success rate

        # 2) Detect success rate (case-insensitive)
        #    Examples:
        #      "[Model: ...] Success rate: 0.2"
        r = re.search(r"Success rate:\s*([0-9]*\.?[0-9]+)", line, flags=re.IGNORECASE)
        if r and current_model is not None and current_epoch is not None:
            rate = float(r.group(1))
            models.append(current_model)
            epochs.append(current_epoch)
            success_rates.append(rate)

            # Reset so each success rate line gets exactly one model
            current_model = None
            current_epoch = None

if not models:
    raise RuntimeError(
        f"No model / success rate entries parsed from {log_path}.\n"
        f"Check that the file actually contains lines with 'model_epoch_XXX.pth' "
        f"and 'Success rate: ...'."
    )

# ---------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------

df = pd.DataFrame(
    {
        "model": models,
        "epoch": epochs,
        "success_rate": success_rates,
    }
)

# Sort by epoch
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df = df.sort_values("epoch")

# Convert to percent + smoothing
df["erfolg_pct"] = df["success_rate"] * 100.0
df["smooth"] = df["erfolg_pct"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()

print("Parsed data:")
print(df)

# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

mpl.rcParams.update(
    {
        "figure.figsize": (4.0, 2.8),
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

fig, ax = plt.subplots(figsize=(4.0, 2.8))

# Rohdaten (grau)
ax.plot(
    df["epoch"],
    df["erfolg_pct"],
    color="lightgrey",
    alpha=0.8,
    linewidth=1.5,
    label="Rohdaten",
)

# Geglättete Kurve (blau)
ax.plot(
    df["epoch"],
    df["smooth"],
    color="blue",
    linewidth=2,
    label=f"Glättung (Fenster={SMOOTH_WINDOW})",
)

# ---------------------------------------------------------------------
# Mark and annotate best RAW success rate
# ---------------------------------------------------------------------

best_idx = df["erfolg_pct"].idxmax()
best_epoch = df.loc[best_idx, "epoch"]
best_value = df.loc[best_idx, "erfolg_pct"]

# red point at best raw success rate
ax.scatter(
    best_epoch,
    best_value,
    color="red",
    s=40,
    zorder=5,
    label="Maximum",
)

# text above the point
ax.text(
    best_epoch,
    best_value + 2,  # small vertical offset in percent units
    f"{best_value:.1f}%",
    color="red",
    ha="center",
    va="bottom",
    fontsize=9,
)

# ---------------------------------------------------------------------
# Axes formatting
# ---------------------------------------------------------------------

ax.set_xlabel("Epoche [-]")
ax.set_ylabel("Erfolgsrate [%]")
ax.grid(True)
ax.set_ylim(0, 100)

# To avoid duplicate legend entries (if you want to keep it small):
handles, labels = ax.get_legend_handles_labels()
# Optionally deduplicate labels
seen = {}
unique_handles = []
unique_labels = []
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = True
        unique_handles.append(h)
        unique_labels.append(l)
ax.legend(unique_handles, unique_labels)

fig.tight_layout()

fig.savefig(plot_path, dpi=300, bbox_inches="tight")

print(f"\nPlot gespeichert unter: {plot_path}")
