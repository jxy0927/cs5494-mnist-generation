"""Render figures from the saved experiment metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"
BLUE = "#1E5799"
GREEN = "#0E796E"
ORANGE = "#A3501F"
MUTED = "#535E71"


def load(name):
    return json.loads((RUNS / name / "history.json").read_text(encoding="utf-8"))


def plot():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 170})

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.45), layout="constrained")
    for ax, run, title in zip(axes, ("fvsbn", "pixelcnn"), ("FVSBN", "PixelCNN")):
        rows = load(run)
        for key, label, color in (("train_nll", "Train", BLUE), ("val_nll", "Validation", GREEN)):
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], "o-",
                    color=color, ms=3, label=label)
        ax.set(title=title, xlabel="Epoch", ylabel="NLL (nats/image)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(FIGURES / "ar_learning.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.45), layout="constrained")
    for ax, run, title in zip(axes, ("gan", "gan_features"), ("Baseline", "Feature matching")):
        rows = load(run)
        ax.plot([r["epoch"] for r in rows], [r["d_loss"] for r in rows], color=BLUE, label="D")
        ax.plot([r["epoch"] for r in rows], [r["g_loss"] for r in rows], color=ORANGE, label="G")
        ax.set(title=title, xlabel="Epoch", ylabel="Loss")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(FIGURES / "gan_learning.png")
    plt.close(fig)

    data = json.loads((RUNS / "gan_probe.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 2.35), layout="constrained")
    for j, (key, label, color) in enumerate((
        ("real", "Real test", MUTED), ("gan", "Baseline", ORANGE),
        ("gan_features", "Feature matching", GREEN),
    )):
        counts = data[key]["counts"]
        ax.bar([i + (j - 1) * .26 for i in range(10)],
               [100 * c / sum(counts) for c in counts], width=.25, label=label, color=color)
    ax.set(xticks=range(10), xlabel="Predicted digit", ylabel="Share (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "gan_coverage.png")
    plt.close(fig)


if __name__ == "__main__":
    plot()


