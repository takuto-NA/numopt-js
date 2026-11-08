"""Plot paper-guided geared-linkage fits, chain-rule controls and gradient-only speedup measurements."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_COLOR = "#b84428"
CURRENT_COLOR = "#087e65"
INITIAL_COLOR = "#89969e"
CONTROL_COLOR = "#8b5ba8"
FIGURE_SIZE = (13, 13)
RESOLUTION = 150
GEAR_RATIOS = [-2, -1, 1, 2]
PAPER_SPEEDUP = (20, 30)


def plot_curve(axis, points, **options):
    axis.plot(*zip(*(points + points[:1])), **options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", default="benchmark-results/mechanical-character-paper-validation.json")
    parser.add_argument("--output", default="benchmark-results/mechanical-character-paper-validation.png")
    arguments = parser.parse_args()
    report = json.loads(Path(arguments.report).read_text(encoding="utf-8"))
    figure, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    for ratio, axis in zip(GEAR_RATIOS, axes.flat):
        row = next(row for row in report["fits"] if row["name"] == f"geared-five-bar-8:ratio={ratio}"
                   and not row["distorted"] and row["initialIndex"] == 0 and row["method"] == "paper-ift")
        verification = row["verification"]
        plot_curve(axis, verification["initialCurve"], color=INITIAL_COLOR, linestyle="--", label="Initial")
        plot_curve(axis, verification["target"], color=TARGET_COLOR, linewidth=3, alpha=0.65, label="Fixed target")
        plot_curve(axis, verification["heldOutCurve"], color=CURRENT_COLOR, linewidth=1.5, label="BFGS + public IFT")
        axis.set(title=f"Geared five-bar | ratio {ratio:+d} | 8 parameters\nHeld-out RMS {verification['heldOutError']:.2e}",
                 xlabel="x", ylabel="y", aspect="equal")
    axes[0, 0].legend(fontsize=9)
    equations = next(row for row in report["equations"] if row["name"] == "geared-five-bar-8:ratio=2")
    colors = [CURRENT_COLOR, INITIAL_COLOR, TARGET_COLOR, CONTROL_COLOR]
    for row, color in zip(equations["gradients"], colors):
        rows = row["taylor"]["rows"]
        axes[2, 0].loglog([item["step"] for item in rows], [item["remainder"] for item in rows],
                          "o-", color=color, label=row["method"])
    axes[2, 0].set(title="Eqs. (3)-(4): correct vs omitted gradient terms", xlabel="Parameter displacement h",
                   ylabel="|F(p + h d) - F(p) - h g.d|")
    axes[2, 0].legend(fontsize=9)
    for model, color in [("four-bar-6", INITIAL_COLOR), ("geared-five-bar-8:ratio=2", CURRENT_COLOR)]:
        for preparation, style in [("included", "o-"), ("cached", "s--")]:
            rows = [row for row in report["gradients"] if row["name"] == model and row["statePreparation"] == preparation]
            axes[2, 1].plot([row["sampleCount"] for row in rows], [row["speedup"] for row in rows],
                            style, color=color, label=f"{model} / {preparation}")
    axes[2, 1].axhspan(*PAPER_SPEEDUP, color=TARGET_COLOR, alpha=0.1, label="Paper: 20-30x on its mechanisms")
    axes[2, 1].set(title="Gradient timing depends on current-state availability", xlabel="Phase samples", ylabel="FD / IFT median time",
                   ylim=(0, max(PAPER_SPEEDUP[1], max(row["speedup"] for row in report["gradients"])) + 2))
    axes[2, 1].legend(fontsize=9)
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Paper-guided validation of continuous mechanism optimization\n"
                    "Synthetic fixtures / 32 design phases / 257 independent verification phases / dimensionless units", fontsize=14)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=RESOLUTION)
    print(output)


if __name__ == "__main__":
    main()
