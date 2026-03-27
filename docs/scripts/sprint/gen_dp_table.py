#!/usr/bin/env python3
"""
Generate Dynamic Programming table-fill visualizations.

Outputs:
  - public/viz/dp_coin_change.png   (Coin Change 1D DP array fill)
  - public/viz/dp_unique_paths.png  (Unique Paths 2D grid fill)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib import patheffects

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "public", "viz")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Color palette (Catppuccin-ish dark theme)
# ---------------------------------------------------------------------------
BG        = "#1e1e2e"
SURFACE   = "#313244"
TEXT      = "#cdd6f4"
SUBTEXT   = "#a6adc8"
GREEN     = "#a6e3a1"
YELLOW    = "#f9e2af"
RED       = "#f38ba8"
BLUE      = "#89b4fa"
MAUVE     = "#cba6f7"
PEACH     = "#fab387"
TEAL      = "#94e2d5"
GRAY      = "#585b70"
DARK_GRAY = "#45475a"

INF = float("inf")


# ===================================================================
# IMAGE 1 -- Coin Change DP Table
# ===================================================================
def generate_coin_change():
    coins = [1, 3, 4]
    amount = 6

    # ---- Run DP and record snapshots --------------------------------
    dp = [0] + [INF] * amount
    # track which coin was optimal for each cell (for arrows)
    best_coin = [None] * (amount + 1)

    snapshots = []           # (label, dp_copy, highlight_idx, source_idx)
    snapshots.append(("Initial", list(dp), None, None))

    for coin in coins:
        for i in range(coin, amount + 1):
            if dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                best_coin[i] = coin
        snapshots.append((f"After coin = {coin}", list(dp), None, None))

    n_rows = len(snapshots)
    n_cols = amount + 1

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(BG)

    cell_w, cell_h = 1.2, 0.85
    margin_top = 1.8
    margin_left = 2.8
    row_gap = 0.35

    # Title
    ax.text(
        margin_left + n_cols * cell_w / 2, margin_top + 0.9,
        "Coin Change DP Table Fill",
        fontsize=20, fontweight="bold", color=TEXT,
        ha="center", va="center",
        path_effects=[patheffects.withStroke(linewidth=3, foreground=BG)],
    )

    # Recurrence
    ax.text(
        margin_left + n_cols * cell_w / 2, margin_top + 0.25,
        "dp[i] = min( dp[i],  dp[i \u2212 coin] + 1 )        coins = [1, 3, 4],  amount = 6",
        fontsize=11, color=SUBTEXT, ha="center", va="center",
        family="monospace",
    )

    # Column headers (amount indices)
    for j in range(n_cols):
        x = margin_left + j * cell_w + cell_w / 2
        y = margin_top - 0.25
        ax.text(x, y, str(j), fontsize=11, color=BLUE,
                ha="center", va="center", fontweight="bold")

    # Header label
    ax.text(margin_left - 0.3, margin_top - 0.25, "i =",
            fontsize=11, color=BLUE, ha="right", va="center", fontweight="bold")

    # Draw rows
    for r, (label, snap, _, _) in enumerate(snapshots):
        y_top = margin_top - (r + 1) * (cell_h + row_gap)

        # Row label
        ax.text(
            margin_left - 0.3, y_top + cell_h / 2,
            label, fontsize=10, color=PEACH,
            ha="right", va="center", fontweight="bold",
        )

        for j in range(n_cols):
            val = snap[j]
            x = margin_left + j * cell_w

            # Determine cell colour
            if r == n_rows - 1 and j == amount:
                # Final answer cell
                color = GREEN
                text_color = "#1e1e2e"
            elif val == INF:
                color = RED
                text_color = "#1e1e2e"
            elif r == 0 and j == 0:
                color = GREEN
                text_color = "#1e1e2e"
            elif val != INF and r > 0 and (r == 0 or snapshots[r - 1][1][j] != val):
                # This cell changed in this row -> yellow
                color = YELLOW
                text_color = "#1e1e2e"
            elif val != INF:
                color = SURFACE
                text_color = TEXT
            else:
                color = GRAY
                text_color = TEXT

            rect = mpatches.FancyBboxPatch(
                (x + 0.05, y_top + 0.05), cell_w - 0.1, cell_h - 0.1,
                boxstyle="round,pad=0.06", facecolor=color,
                edgecolor=DARK_GRAY, linewidth=1.2,
            )
            ax.add_patch(rect)

            display = "\u221e" if val == INF else str(val)
            ax.text(
                x + cell_w / 2, y_top + cell_h / 2,
                display, fontsize=13, fontweight="bold",
                color=text_color, ha="center", va="center",
            )

    # ---- Draw arrows for optimal choices on the FINAL row -----------
    final_row_idx = n_rows - 1
    y_arrow_row = margin_top - final_row_idx * (cell_h + row_gap)

    for j in range(1, n_cols):
        coin = best_coin[j]
        if coin is not None:
            src = j - coin
            x_src = margin_left + src * cell_w + cell_w / 2
            x_dst = margin_left + j * cell_w + cell_w / 2
            y_base = y_arrow_row - 0.15

            arrow = FancyArrowPatch(
                (x_src, y_base), (x_dst, y_base),
                arrowstyle="->,head_width=5,head_length=4",
                color=TEAL, linewidth=1.5,
                connectionstyle="arc3,rad=-0.35",
            )
            ax.add_patch(arrow)
            # Label the arrow with "+1"
            mid_x = (x_src + x_dst) / 2
            ax.text(mid_x, y_base - 0.32, "+1",
                    fontsize=8, color=TEAL, ha="center", va="center",
                    fontweight="bold")

    # ---- Legend -------------------------------------------------------
    legend_items = [
        (GREEN, "Base / Answer"),
        (YELLOW, "Updated this step"),
        (RED, "Infinity (\u221e)"),
        (SURFACE, "Unchanged"),
        (TEAL, "Optimal transition"),
    ]
    lx = margin_left + n_cols * cell_w + 0.3
    ly = margin_top - 0.4
    ax.text(lx, ly + 0.3, "Legend", fontsize=11, color=TEXT, fontweight="bold")
    for idx, (c, lbl) in enumerate(legend_items):
        y = ly - idx * 0.45
        rect = mpatches.FancyBboxPatch(
            (lx, y - 0.12), 0.3, 0.24,
            boxstyle="round,pad=0.03", facecolor=c, edgecolor=DARK_GRAY,
        )
        ax.add_patch(rect)
        ax.text(lx + 0.45, y, lbl, fontsize=9, color=SUBTEXT, va="center")

    # ---- Axes housekeeping -------------------------------------------
    total_h = margin_top + 1.5 + n_rows * (cell_h + row_gap)
    total_w = margin_left + n_cols * cell_w + 3.5
    ax.set_xlim(0, total_w)
    ax.set_ylim(-total_h + margin_top + 1.5, margin_top + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    out = os.path.join(OUT_DIR, "dp_coin_change.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, pad_inches=0.4)
    plt.close(fig)
    print(f"Saved: {out}")


# ===================================================================
# IMAGE 2 -- Unique Paths Grid
# ===================================================================
def generate_unique_paths():
    rows, cols = 5, 4

    # Build the DP table
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = 1
    for j in range(cols):
        dp[0][j] = 1
    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    max_val = dp[-1][-1]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    ax.set_facecolor(BG)

    cell = 1.4
    margin_left = 2.0
    margin_top = 2.0

    # Title
    ax.text(
        margin_left + cols * cell / 2, margin_top + 0.85,
        "Unique Paths \u2014 DP Grid Fill",
        fontsize=20, fontweight="bold", color=TEXT, ha="center",
        path_effects=[patheffects.withStroke(linewidth=3, foreground=BG)],
    )

    # Recurrence
    ax.text(
        margin_left + cols * cell / 2, margin_top + 0.2,
        "dp[i][j] = dp[i\u22121][j] + dp[i][j\u22121]        grid = 5 \u00d7 4",
        fontsize=11, color=SUBTEXT, ha="center", family="monospace",
    )

    # Column headers
    for j in range(cols):
        x = margin_left + j * cell + cell / 2
        ax.text(x, margin_top - 0.2, f"j={j}", fontsize=10, color=BLUE,
                ha="center", fontweight="bold")

    # Row headers
    for i in range(rows):
        y = margin_top - (i + 1) * cell + cell / 2
        ax.text(margin_left - 0.35, y, f"i={i}", fontsize=10, color=BLUE,
                ha="right", va="center", fontweight="bold")

    # --- Helper: intensity colour from value --------------------------
    def cell_color(val, i, j):
        if i == 0 and j == 0:
            return GREEN               # start
        if i == rows - 1 and j == cols - 1:
            return GREEN               # end
        # Gradient: low -> SURFACE, high -> MAUVE
        t = val / max_val
        # Lerp between SURFACE (#313244) and MAUVE (#cba6f7)
        r0, g0, b0 = 0x31 / 255, 0x32 / 255, 0x44 / 255
        r1, g1, b1 = 0xcb / 255, 0xa6 / 255, 0xf7 / 255
        r = r0 + t * (r1 - r0)
        g = g0 + t * (g1 - g0)
        b = b0 + t * (b1 - b0)
        return (r, g, b)

    # --- Draw cells ---------------------------------------------------
    for i in range(rows):
        for j in range(cols):
            x = margin_left + j * cell
            y = margin_top - (i + 1) * cell
            val = dp[i][j]

            fc = cell_color(val, i, j)
            rect = mpatches.FancyBboxPatch(
                (x + 0.06, y + 0.06), cell - 0.12, cell - 0.12,
                boxstyle="round,pad=0.08", facecolor=fc,
                edgecolor=DARK_GRAY, linewidth=1.5,
            )
            ax.add_patch(rect)

            # Determine text colour for contrast
            if isinstance(fc, str):
                tc = "#1e1e2e"
            else:
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                tc = "#1e1e2e" if lum > 0.45 else TEXT

            ax.text(
                x + cell / 2, y + cell / 2,
                str(val), fontsize=16, fontweight="bold",
                color=tc, ha="center", va="center",
            )

            # --- Arrows from top (i-1, j) and left (i, j-1) -----------
            arrow_kw = dict(
                arrowstyle="->,head_width=4,head_length=3",
                linewidth=1.3, alpha=0.7,
            )
            offset_in = 0.13   # offset inside the cell edge
            offset_out = 0.13  # offset outside the source cell edge

            if i > 0:
                # Arrow from cell above
                src_x = x + cell / 2
                src_y = y + cell - offset_out + 0.06  # bottom of cell above
                dst_y = y + cell - offset_in - 0.15
                arrow = FancyArrowPatch(
                    (src_x, src_y + 0.12), (src_x, dst_y + 0.12),
                    color=TEAL, **arrow_kw,
                )
                ax.add_patch(arrow)

            if j > 0:
                # Arrow from cell to the left
                src_y = y + cell / 2
                src_x = x + offset_out - 0.06
                dst_x = x + offset_in + 0.15
                arrow = FancyArrowPatch(
                    (src_x - 0.12, src_y), (dst_x - 0.12, src_y),
                    color=PEACH, **arrow_kw,
                )
                ax.add_patch(arrow)

    # --- Start / End labels ------------------------------------------
    sx = margin_left + cell / 2
    sy = margin_top - cell + cell / 2
    ax.text(sx, sy - cell * 0.42, "START", fontsize=8, color=GREEN,
            ha="center", fontweight="bold")

    ex = margin_left + (cols - 1) * cell + cell / 2
    ey = margin_top - rows * cell + cell / 2
    ax.text(ex, ey - cell * 0.42, "END", fontsize=8, color=GREEN,
            ha="center", fontweight="bold")

    # --- Legend --------------------------------------------------------
    legend_items = [
        (GREEN, "Start / End cell"),
        (MAUVE, "High path count"),
        (SURFACE, "Low path count"),
        (TEAL, "From top  dp[i\u22121][j]"),
        (PEACH, "From left dp[i][j\u22121]"),
    ]
    lx = margin_left + cols * cell + 0.5
    ly = margin_top - 0.5
    ax.text(lx, ly + 0.35, "Legend", fontsize=11, color=TEXT, fontweight="bold")
    for idx, (c, lbl) in enumerate(legend_items):
        y = ly - idx * 0.5
        rect = mpatches.FancyBboxPatch(
            (lx, y - 0.15), 0.32, 0.30,
            boxstyle="round,pad=0.03", facecolor=c, edgecolor=DARK_GRAY,
        )
        ax.add_patch(rect)
        ax.text(lx + 0.50, y, lbl, fontsize=9, color=SUBTEXT, va="center")

    # --- Axes ---------------------------------------------------------
    total_w = margin_left + cols * cell + 4.5
    total_h = margin_top + 1.5 + rows * cell
    ax.set_xlim(0, total_w)
    ax.set_ylim(margin_top - rows * cell - 1.0, margin_top + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    out = os.path.join(OUT_DIR, "dp_unique_paths.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, pad_inches=0.4)
    plt.close(fig)
    print(f"Saved: {out}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating DP visualizations...")
    generate_coin_change()
    generate_unique_paths()
    print("Done.")
