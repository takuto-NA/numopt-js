"""Plot measured scaling time, process peak RSS, and crossover ratios from saved reports."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

METHODS = ["bfgs", "approximate", "full-dense", "full-hvp"]
LABELS = {"bfgs": "BFGS", "approximate": "Approximate Newton", "full-dense": "Full Newton (dense)",
          "full-hvp": "Full Newton (HVP)", "full-fd": "Full Newton (FD curvature)",
          "approximate-fd": "Approx. Newton (FD curvature)"}
COLORS = {"bfgs": "#444444", "approximate": "#dd842f", "full-dense": "#2676b8", "full-hvp": "#289966",
          "full-fd": "#9855b5", "approximate-fd": "#b65149"}


def load_rows(paths):
    rows = {}
    for source in paths:
        report = json.loads(Path(source).read_text(encoding="utf-8"))
        for row in report["rows"]:
            if row["status"] == "success":
                key = (row["family"], row["parameters"], row["states"], row["method"])
                previous = rows.get(key)
                if previous and not row.get("peakRssMiB"):
                    row = {**row, "peakRssMiB": previous.get("peakRssMiB")}
                rows[key] = row
    return list(rows.values())


def series(ax, rows, x_key, metric, methods=METHODS):
    for method in methods:
        points = sorted([row for row in rows if row["method"] == method and row.get(metric)], key=lambda row: row[x_key])
        if not points:
            continue
        xs = [row[x_key] for row in points]
        ys = [row[metric]["median"] for row in points]
        lower = [row[metric]["lowerQuartile"] for row in points]
        upper = [row[metric]["upperQuartile"] for row in points]
        ax.plot(xs, ys, marker="o", markersize=4, label=LABELS[method], color=COLORS[method])
        ax.fill_between(xs, lower, upper, color=COLORS[method], alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
    ax.grid(True, alpha=0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", default="benchmark-results/adjoint-scaling")
    parser.add_argument("--fd-report")
    args = parser.parse_args()
    rows = load_rows(args.reports)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    slices = [([row for row in rows if row["family"] == "curved" and row["states"] == 128], "parameters", "Curved: 128 states"),
              ([row for row in rows if row["family"] == "curved" and row["states"] == 512], "parameters", "Curved: 512 states"),
              ([row for row in rows if row["family"] == "curved" and row["parameters"] == 16], "states", "Curved: 16 parameters")]
    for column, (subset, x_key, title) in enumerate(slices):
        for index, metric in enumerate(["timeMs", "peakRssMiB"]):
            ax = axes[index, column]
            series(ax, subset, x_key, metric)
            ax.set_title(title)
            ax.set_xlabel("Design parameters m" if x_key == "parameters" else "State variables n")
            ax.set_ylabel("Time to common accuracy (ms)" if index == 0 else "Cold process peak RSS (MiB)")
            if index == 0:
                ax.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4)
    figure.suptitle("Current dense implementation: median with interquartile bands\nTime and memory use separate processes; memory includes Node and model inputs")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(f"{output}.png", dpi=170)
    figure.savefig(f"{output}.svg")
    plt.close(figure)

    panels = [("easy", "Easy", rows), ("curved", "Curved / analytical curvature", rows)]
    if args.fd_report:
        panels.append(("curved", "Curved / FD curvature", load_rows([args.fd_report])))
    figure, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), constrained_layout=True)
    legend_entries = {}
    for ax, (family, title, dataset) in zip(axes, panels):
        subset = [row for row in dataset if row["family"] == family and row["states"] == 128]
        for method in ["full-dense", "full-hvp", "full-fd"]:
            points = sorted([row for row in subset if row["method"] == method], key=lambda row: row["parameters"])
            if not points:
                continue
            xs, ratios, lower, upper = [], [], [], []
            for row in points:
                reference = next((other for other in subset if other["method"] == "bfgs" and other["parameters"] == row["parameters"]), None)
                if not reference:
                    continue
                xs.append(row["parameters"])
                ratios.append(row["timeMs"]["median"] / reference["timeMs"]["median"])
                lower.append(row["timeMs"]["lowerQuartile"] / reference["timeMs"]["upperQuartile"])
                upper.append(row["timeMs"]["upperQuartile"] / reference["timeMs"]["lowerQuartile"])
            ax.plot(xs, ratios, marker="o", markersize=4, label=LABELS[method], color=COLORS[method])
            ax.fill_between(xs, lower, upper, color=COLORS[method], alpha=0.15)
        ax.axhline(1, color="#555555", linestyle="--")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
        if "FD" in title:
            ax.set_yscale("log")
        ax.set_xlabel("Design parameters m (state count n=128)")
        ax.set_ylabel("Time / BFGS (above 1 is slower)")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        legend_entries.update(zip(labels, handles))
    figure.legend(list(legend_entries.values()), list(legend_entries.keys()), loc="outside lower center", ncol=3)
    figure.suptitle("Observed crossover depends on objective and Hessian representation\nBands combine timing quartiles; they are not confidence intervals")
    figure.savefig(f"{output}-crossover.png", dpi=170)
    figure.savefig(f"{output}-crossover.svg")
    print(f"Plots: {output}.png and {output}-crossover.png")


if __name__ == "__main__":
    main()
