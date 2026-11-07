"""Plot synthetic four-bar trajectories, convergence histories, and gradient benchmark timings."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", default="benchmark-results/mechanical-character.json")
    parser.add_argument("--output", default="benchmark-results/mechanical-character.png")
    arguments = parser.parse_args()
    report = json.loads(Path(arguments.report).read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    reachable = [row for row in report["rows"] if row["problem"] == "reachable"]
    colors = dict(zip([row["method"] for row in reachable], plt.get_cmap("tab10").colors))
    target = reachable[0]["target"]
    initial = reachable[0]["initialCurve"]
    axes[0, 0].plot(*zip(*(target + target[:1])), "k-", linewidth=3, alpha=0.4, label="Target")
    axes[0, 0].plot(*zip(*(initial + initial[:1])), "--", color="#888888", label="Initial")
    for row in reachable:
        trial = row["trials"][0]
        if "error" in trial:
            continue
        curve = trial["curve"]
        axes[0, 0].plot(*zip(*(curve + curve[:1])), color=colors[row["method"]], label=row["method"])
        for case, axis in [("reachable", axes[0, 1]), ("distorted", axes[1, 0])]:
            case_row = next(item for item in report["rows"] if item["problem"] == case and item["method"] == row["method"])
            case_trial = case_row["trials"][0]
            if "error" not in case_trial:
                history = case_trial["history"] + [case_trial["verification"]["finalCost"]]
                axis.semilogy(range(len(history)), history, color=colors[row["method"]], label=row["method"])
    axes[0, 0].set(title="Reachable trajectory (32 design phases)", xlabel="x", ylabel="y", aspect="equal")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Reachable: objective vs iteration", xlabel="Iteration", ylabel="Phase-averaged cost")
    axes[1, 0].set(title="Distorted: nonzero residual", xlabel="Iteration", ylabel="Phase-averaged cost")
    for mode in ["analytic", "paper", "public-ift", "redundant-ift", "finite-difference"]:
        values = [row for row in report["gradients"] if row["mode"] == mode]
        axes[1, 1].plot([row["sampleCount"] for row in values], [row["timeMs"]["median"] for row in values],
                        marker="o", label=mode)
    axes[1, 1].set(title="Gradient + current-state assembly (median)", xlabel="Phase samples", ylabel="Milliseconds",
                   xscale="log", yscale="log")
    axes[1, 1].legend(fontsize=9)
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Coros et al. (2013), section 4.2: synthetic validation, not original character data", fontsize=13)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
