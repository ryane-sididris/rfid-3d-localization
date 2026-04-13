from pathlib import Path
import os

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


RESULTS_DIR = Path("experiments") / "results"
PNG_PATH = RESULTS_DIR / "keras_freq_conv_xyz_architecture.png"
SVG_PATH = RESULTS_DIR / "keras_freq_conv_xyz_architecture.svg"


def draw_block(ax, x, y, w, h, label, shape, color, depth=0.22, rise=0.16):
    front = Rectangle((x, y), w, h, facecolor=color, edgecolor="#1f2937", linewidth=1.2)
    top = Polygon(
        [(x, y + h), (x + depth, y + h + rise), (x + w + depth, y + h + rise), (x + w, y + h)],
        closed=True,
        facecolor="#ffffff",
        edgecolor="#1f2937",
        linewidth=1.0,
        alpha=0.55,
    )
    side = Polygon(
        [(x + w, y), (x + w + depth, y + rise), (x + w + depth, y + h + rise), (x + w, y + h)],
        closed=True,
        facecolor="#0f172a",
        edgecolor="#1f2937",
        linewidth=1.0,
        alpha=0.16,
    )
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(side)
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x + w / 2, y + h * 0.33, shape, ha="center", va="center", fontsize=9, family="monospace")


def draw_stack(ax, x, y, w, h, layers, label, shape, color):
    for idx in range(layers):
        offset = idx * 0.12
        draw_block(ax, x + offset, y + offset, w, h, "" if idx < layers - 1 else label, "" if idx < layers - 1 else shape, color)


def arrow(ax, x0, y0, x1, y1, text=None):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#334155"),
    )
    if text:
        ax.text((x0 + x1) / 2, y0 + 0.28, text, ha="center", va="bottom", fontsize=9, color="#334155")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 6), dpi=180)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")

    draw_stack(ax, 0.6, 1.0, 1.2, 3.7, 4, "Input sample", "(96, 50, 2)", "#cde7ff")
    ax.text(1.2, 0.5, "96 paths, each path = 50 freq bins x 2 channels", ha="center", fontsize=9, color="#334155")

    arrow(ax, 2.2, 2.85, 3.0, 2.85, "slice one path")
    draw_block(ax, 3.1, 1.35, 0.62, 3.0, "Path", "(50, 2)", "#d9f99d")

    ax.plot([4.1, 9.6], [5.2, 5.2], color="#64748b", linewidth=1.3, linestyle="--")
    ax.plot([4.1, 9.6], [0.9, 0.9], color="#64748b", linewidth=1.3, linestyle="--")
    ax.plot([4.1, 4.1], [0.9, 5.2], color="#64748b", linewidth=1.3, linestyle="--")
    ax.plot([9.6, 9.6], [0.9, 5.2], color="#64748b", linewidth=1.3, linestyle="--")
    ax.text(6.85, 5.45, "Shared path encoder", ha="center", fontsize=11, weight="bold", color="#334155")

    arrow(ax, 3.85, 2.85, 4.35, 2.85)
    draw_block(ax, 4.45, 1.25, 0.92, 3.15, "Conv1D", "k=5, 32\n(50, 32)", "#fef08a")
    arrow(ax, 5.55, 2.85, 6.05, 2.85)
    draw_block(ax, 6.15, 1.75, 0.92, 2.35, "Conv1D", "k=3, s=2\n(25, 32)", "#fde68a")
    arrow(ax, 7.25, 2.85, 7.75, 2.85)
    draw_block(ax, 7.85, 2.0, 1.05, 1.85, "Conv1D", "k=3, s=2\n(13, 64)", "#fdba74")
    arrow(ax, 9.1, 2.85, 9.55, 2.85)
    draw_block(ax, 9.65, 2.15, 0.9, 1.55, "Flatten", "(832,)", "#fca5a5")
    arrow(ax, 10.75, 2.85, 11.25, 2.85)
    draw_block(ax, 11.35, 2.25, 0.8, 1.35, "Dense", "(32,)", "#f9a8d4")

    arrow(ax, 12.35, 2.85, 13.2, 2.85, "apply to all 96 paths")
    draw_stack(ax, 13.35, 1.55, 1.15, 2.5, 4, "Embeddings", "(96, 32)", "#c4b5fd")

    arrow(ax, 14.95, 2.85, 15.7, 2.85)
    draw_block(ax, 15.8, 2.0, 1.05, 1.7, "Flatten", "(3072,)", "#fda4af")
    arrow(ax, 17.05, 2.85, 17.65, 2.85)
    draw_block(ax, 17.75, 1.9, 0.95, 1.9, "Dense", "(256,)", "#93c5fd")
    arrow(ax, 18.9, 2.85, 19.45, 2.85)
    draw_block(ax, 19.55, 2.1, 0.9, 1.5, "Dense", "(128,)", "#86efac")
    arrow(ax, 20.65, 2.85, 21.15, 2.85)
    draw_block(ax, 21.25, 2.35, 0.72, 1.0, "Output", "(3,)", "#f5d0fe")

    ax.text(17.9, 4.85, "Dense fusion head", ha="center", fontsize=11, weight="bold", color="#334155")
    ax.text(21.6, 1.25, "xyz = (x, y, z)", ha="center", fontsize=9, color="#334155")

    ax.text(9.7, 0.35, "Conv1D sees one path at a time: (50, 2) -> (50, 32) -> (25, 32) -> (13, 64)", ha="center", fontsize=9, color="#475569")
    ax.text(17.8, 0.35, "Then path embeddings are fused globally: (96, 32) -> 3072 -> 256 -> 128 -> 3", ha="center", fontsize=9, color="#475569")

    ax.set_xlim(0, 22.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight", dpi=220)
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved architecture diagram to {PNG_PATH}")
    print(f"Saved architecture diagram to {SVG_PATH}")


if __name__ == "__main__":
    main()
