"""Plot measured iteration trajectories, fixed-topology link poses, convergence and nonzero residuals."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_COLOR = "#b84428"
CURRENT_COLOR = "#087e65"
INITIAL_COLOR = "#8a969b"
COORDINATES = 2
FIGURE_SIZE = (15, 9)
RESOLUTION = 150
MIDPOINT_FRACTION = 0.5


def curve(axis, points, **options):
    axis.plot(*zip(*(points + points[:1])), **options)


def pose(axis, row, snapshot, color, label):
    crank, coupler, _, ground = snapshot["lengths"]
    angle = snapshot["couplerAngles"][0]
    import math
    joint = [crank + coupler * math.cos(angle), coupler * math.sin(angle)]
    axis.plot([0, crank, joint[0], ground], [0, 0, joint[1], 0], "o-", color=color, label=label)
    marker = snapshot["curve"][0]
    axis.plot([crank, marker[0], joint[0]], [0, marker[1], joint[1]], "--", color=color, linewidth=1)
    axis.plot(*marker, "o", color=color)
    axis.plot([0, ground], [0, 0], color=INITIAL_COLOR, linewidth=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", default="benchmark-results/linkage-trajectory-example.json")
    parser.add_argument("--output", default="benchmark-results/linkage-trajectory-example.png")
    arguments = parser.parse_args()
    report = json.loads(Path(arguments.report).read_text(encoding="utf-8"))
    reachable, distorted = report["rows"]
    figure, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE, constrained_layout=True)
    snapshots = reachable["snapshots"]
    selected = [snapshots[0], snapshots[int((len(snapshots) - 1) * MIDPOINT_FRACTION)], snapshots[-1]]
    all_points = reachable["target"] + [point for snapshot in selected for point in snapshot["curve"]]
    minima = [min(point[axis] for point in all_points) for axis in range(COORDINATES)]
    maxima = [max(point[axis] for point in all_points) for axis in range(COORDINATES)]
    for axis, snapshot in zip(axes[0], selected):
        curve(axis, reachable["target"], color=TARGET_COLOR, linewidth=3, alpha=0.65, label="Fixed target")
        curve(axis, snapshot["curve"], color=CURRENT_COLOR, linewidth=1.8, label="Current trajectory")
        rms = (COORDINATES * snapshot["cost"]) ** MIDPOINT_FRACTION
        axis.set(title=f"Iteration {snapshot['iteration']} | RMS {rms:.2e}", xlabel="x", ylabel="y", aspect="equal",
                 xlim=(minima[0] - 0.04, maxima[0] + 0.04), ylim=(minima[1] - 0.04, maxima[1] + 0.04))
    axes[0, 0].legend(fontsize=9)
    pose(axes[1, 0], reachable, snapshots[0], INITIAL_COLOR, "Initial links")
    pose(axes[1, 0], reachable, snapshots[-1], CURRENT_COLOR, "Final links")
    curve(axes[1, 0], reachable["target"], color=TARGET_COLOR, linewidth=1, label="Target marker path")
    axes[1, 0].set(title="Same topology / fixed pivots and marker", xlabel="x", ylabel="y", aspect="equal")
    axes[1, 0].legend(fontsize=9)
    for row, color in [(reachable, CURRENT_COLOR), (distorted, TARGET_COLOR)]:
        axes[1, 1].semilogy([item["iteration"] for item in row["snapshots"]],
                           [(COORDINATES * item["cost"]) ** MIDPOINT_FRACTION for item in row["snapshots"]],
                           color=color, label=row["targetCase"])
    axes[1, 1].set(title="Measured objective decreases at recorded steps", xlabel="BFGS iteration", ylabel="Training positional RMS")
    axes[1, 1].legend(fontsize=9)
    curve(axes[1, 2], distorted["target"], color=TARGET_COLOR, linewidth=2, label="Distorted target")
    curve(axes[1, 2], distorted["snapshots"][-1]["curve"], color=CURRENT_COLOR, label="Final fit")
    axes[1, 2].set(title=f"Distorted target | held-out RMS {distorted['verification']['heldOutError']:.3f}",
                   xlabel="x", ylabel="y", aspect="equal")
    axes[1, 2].legend(fontsize=9)
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Fixed four-bar: fit 3 moving link lengths using numopt-js BFGS + public IFT\n"
                    "32 design phases / independent 257-phase verification / dimensionless units", fontsize=14)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=RESOLUTION)
    print(output)


if __name__ == "__main__":
    main()
