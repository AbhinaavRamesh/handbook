"""
Generate a Big-O Complexity Comparison Chart.

Produces a publication-quality visualization comparing common time complexity
growth rates, styled with a dark theme and Google brand colors.

Output: ../public/viz/complexity_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR.parent / "public" / "viz" / "complexity_comparison.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Theme / colours
# ---------------------------------------------------------------------------
BG_COLOR = "#1e1e2e"          # dark background
SURFACE_COLOR = "#2a2a3d"     # slightly lighter for plot area
TEXT_COLOR = "#cdd6f4"         # soft white text
GRID_COLOR = "#45475a"        # subtle grid

# Google brand colours + two extras for 6 lines
COLORS = {
    "O(1)":        "#34a853",   # Google green
    "O(log n)":    "#4285f4",   # Google blue
    "O(n)":        "#fbbc04",   # Google yellow
    "O(n log n)":  "#ea4335",   # Google red
    "O(n²)":      "#ff6d00",   # orange (danger)
    "O(2ⁿ)":      "#d500f9",   # vivid purple (extreme danger)
}

LINE_WIDTHS = {
    "O(1)":        2.4,
    "O(log n)":    2.4,
    "O(n)":        2.6,
    "O(n log n)":  2.6,
    "O(n²)":      2.4,
    "O(2ⁿ)":      2.4,
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
n = np.arange(1, 51)  # input size 1..50

complexities = {
    "O(1)":        np.ones_like(n, dtype=float),
    "O(log n)":    np.log2(n),
    "O(n)":        n.astype(float),
    "O(n log n)":  n * np.log2(n),
    "O(n²)":      (n ** 2).astype(float),
    "O(2ⁿ)":      np.power(2.0, n),
}

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(SURFACE_COLOR)

# ---------------------------------------------------------------------------
# "Danger Zone" shaded region  (from n² upward)
# ---------------------------------------------------------------------------
ax.fill_between(
    n,
    n ** 2,
    np.power(2.0, n),
    alpha=0.10,
    color="#ff6d00",
    label="_nolegend_",
)

# Add a subtle "DANGER ZONE" watermark inside the shaded area
ax.text(
    35, 3e8,
    "DANGER\nZONE",
    fontsize=18,
    fontweight="bold",
    color="#ff6d00",
    alpha=0.25,
    ha="center",
    va="center",
    rotation=0,
    fontfamily="sans-serif",
)

# ---------------------------------------------------------------------------
# Plot each complexity class
# ---------------------------------------------------------------------------
for label, y_vals in complexities.items():
    ax.plot(
        n,
        y_vals,
        label=label,
        color=COLORS[label],
        linewidth=LINE_WIDTHS[label],
        solid_capstyle="round",
    )

# ---------------------------------------------------------------------------
# End-of-line labels (cleaner than a crowded legend)
# ---------------------------------------------------------------------------
label_offsets = {
    "O(1)":        (51, 1),
    "O(log n)":    (51, np.log2(50)),
    "O(n)":        (51, 50),
    "O(n log n)":  (51, 50 * np.log2(50)),
    "O(n²)":      (51, 2500),
    "O(2ⁿ)":      (43, 2**42),
}

for label, (x, y) in label_offsets.items():
    ax.text(
        x, y, f"  {label}",
        color=COLORS[label],
        fontsize=10,
        fontweight="bold",
        va="center",
        fontfamily="sans-serif",
    )

# ---------------------------------------------------------------------------
# Annotation: "Most interview solutions should be here"
# ---------------------------------------------------------------------------
ax.annotate(
    "Most interview solutions\nshould be here",
    xy=(30, 30 * np.log2(30)),          # point near O(n log n)
    xytext=(8, 1e8),                     # text position
    fontsize=11,
    fontweight="semibold",
    color="#cdd6f4",
    arrowprops=dict(
        arrowstyle="fancy,head_width=0.4,head_length=0.3",
        connectionstyle="arc3,rad=-0.3",
        color="#4285f4",
        lw=1.8,
    ),
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#4285f4",
        edgecolor="#4285f4",
        alpha=0.18,
    ),
    fontfamily="sans-serif",
)

# ---------------------------------------------------------------------------
# Axes, grid, labels
# ---------------------------------------------------------------------------
ax.set_yscale("log")
ax.set_xlim(0, 55)
ax.set_ylim(0.8, 2e15)

ax.set_xlabel("Input Size  n", fontsize=13, color=TEXT_COLOR, fontfamily="sans-serif",
              labelpad=10)
ax.set_ylabel("Operations  (log scale)", fontsize=13, color=TEXT_COLOR,
              fontfamily="sans-serif", labelpad=10)
ax.set_title(
    "Time Complexity Growth Rates",
    fontsize=20,
    fontweight="bold",
    color=TEXT_COLOR,
    fontfamily="sans-serif",
    pad=18,
)

ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=10)
ax.grid(True, which="major", linestyle="--", linewidth=0.5, color=GRID_COLOR, alpha=0.6)
ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color=GRID_COLOR, alpha=0.3)

for spine in ax.spines.values():
    spine.set_color(GRID_COLOR)

# ---------------------------------------------------------------------------
# Legend (compact, semi-transparent)
# ---------------------------------------------------------------------------
legend = ax.legend(
    loc="upper left",
    fontsize=10,
    frameon=True,
    facecolor=SURFACE_COLOR,
    edgecolor=GRID_COLOR,
    labelcolor=TEXT_COLOR,
    framealpha=0.85,
    borderpad=0.8,
)
legend.get_frame().set_linewidth(0.8)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout()
fig.savefig(OUTPUT_PATH, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)

print(f"Chart saved to {OUTPUT_PATH}")
