"""Plot measured adjoint L-BFGS timings, process peak RSS, and conditioning tradeoffs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "benchmark-results"
METHODS = ["bfgs", "lbfgs", "approximate", "full-dense", "full-hvp"]
LABELS = {"bfgs": "BFGS", "lbfgs": "L-BFGS (q=10)", "lbfgs-5": "L-BFGS (q=5)",
          "lbfgs-20": "L-BFGS (q=20)", "approximate": "Approx. Newton",
          "full-dense": "Full Newton (dense)", "full-hvp": "Full Newton (HVP)"}
COLORS = {"bfgs": "#5f6368", "lbfgs": "#b53389", "approximate": "#da8122",
          "full-dense": "#2676b8", "full-hvp": "#289966", "lbfgs-5": "#d779b6", "lbfgs-20": "#762263"}


def load(name):
    return json.loads((OUTPUT / f"adjoint-lbfgs-{name}.json").read_text(encoding="utf-8"))["rows"]


def draw_series(axis, rows, metric):
    for method in METHODS:
        points = sorted((row for row in rows if row["method"] == method), key=lambda row: row["parameters"])
        xs = [row["parameters"] for row in points]
        ys = [row[metric]["median"] for row in points]
        axis.plot(xs, ys, "o-", color=COLORS[method], label=LABELS[method], markersize=4)
        axis.fill_between(xs, [row[metric]["lowerQuartile"] for row in points],
                          [row[metric]["upperQuartile"] for row in points], color=COLORS[method], alpha=0.15)
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlabel("Design variables m (state count n=128)")
    axis.grid(alpha=0.2)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    curved = [row for row in load("curved") if row["states"] == 128]
    conditioned = [row for row in load("conditioned") if row["parameters"] == 32 and row["states"] == 64]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    draw_series(axes[0, 0], curved, "timeMs")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Solve time (ms, logarithmic)")
    axes[0, 0].set_title("Curved problem: equal accuracy target")
    axes[0, 0].legend(fontsize=9)
    draw_series(axes[0, 1], curved, "peakRssMiB")
    axes[0, 1].set_ylabel("Whole-process peak RSS (MiB)")
    axes[0, 1].set_title("3 independent cold processes; median and quartiles")

    refinement_path = OUTPUT / "adjoint-lbfgs-refined.json"
    rows = json.loads(refinement_path.read_text(encoding="utf-8"))["rows"] if refinement_path.exists() else curved
    for method in ["full-dense", "full-hvp"]:
        points = sorted((row for row in rows if row["method"] == method), key=lambda row: row["parameters"])
        ratios, lower, upper = [], [], []
        for row in points:
            reference = next(other for other in rows if other["parameters"] == row["parameters"] and other["method"] == "lbfgs")
            ratios.append(row["timeMs"]["median"] / reference["timeMs"]["median"])
            lower.append(row["timeMs"]["lowerQuartile"] / reference["timeMs"]["upperQuartile"])
            upper.append(row["timeMs"]["upperQuartile"] / reference["timeMs"]["lowerQuartile"])
        xs = [row["parameters"] for row in points]
        axes[1, 0].plot(xs, ratios, "o-", color=COLORS[method], label=LABELS[method])
        axes[1, 0].fill_between(xs, lower, upper, color=COLORS[method], alpha=0.15)
    axes[1, 0].axhline(1, color="#777777", linestyle="--")
    axes[1, 0].set_xlabel("Design variables m (state count n=128)")
    axes[1, 0].set_ylabel("Time / L-BFGS (q=10)")
    axes[1, 0].set_title("Crossover: above 1 means L-BFGS is faster")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.2)

    order = ["bfgs", "lbfgs-5", "lbfgs", "lbfgs-20", "approximate", "full-dense", "full-hvp"]
    values = [next(row for row in conditioned if row["method"] == method)["iterations"]["median"] for method in order]
    bars = axes[1, 1].bar(range(len(order)), values, color=[COLORS[method] for method in order])
    axes[1, 1].bar_label(bars, padding=3, fontsize=9)
    axes[1, 1].set_xticks(range(len(order)), ["BFGS", "L-BFGS\nq=5", "L-BFGS\nq=10", "L-BFGS\nq=20", "Approx.\nNewton", "Full\ndense", "Full\nHVP"])
    axes[1, 1].set_ylabel("Iterations to the same accuracy target")
    axes[1, 1].set_ylim(0, max(values) * 1.16)
    axes[1, 1].set_title("Condition ratio 1000: m=32, n=64")
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.suptitle("Adjoint L-BFGS: scaling gains and convergence tradeoffs", fontsize=15)
    figure.text(0.5, 0.015, "Analytical derivatives. Timing excludes startup/setup; RSS includes runtime and inputs. Bands show quartiles, not confidence intervals.",
                ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=[0, 0.04, 1, 0.96])
    for suffix in ["png", "svg"]:
        figure.savefig(OUTPUT / f"adjoint-lbfgs-comparison.{suffix}", dpi=170)


if __name__ == "__main__":
    main()
