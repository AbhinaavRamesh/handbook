"""
Generate two interview-prep visualizations:

1. ds_complexity_table.png  – Data Structure operations time-complexity table
2. interview_timeline.png   – 45-minute Google interview round game plan

Output: ../public/viz/ds_complexity_table.png
        ../public/viz/interview_timeline.png
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR.parent / "public" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

TABLE_OUTPUT = VIZ_DIR / "ds_complexity_table.png"
TIMELINE_OUTPUT = VIZ_DIR / "interview_timeline.png"

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG_COLOR = "#1e1e2e"
SURFACE_COLOR = "#2a2a3d"
TEXT_COLOR = "#cdd6f4"
GRID_COLOR = "#45475a"
HEADER_BG = "#3b3b58"
HEADER_TEXT = "#ffffff"

# Complexity-based cell colours
COMPLEXITY_COLORS = {
    "O(1)":     "#2ecc71",   # bright green
    "O(1)*":    "#2ecc71",
    "O(1)\u2020":   "#2ecc71",   # dagger
    "O(log n)": "#82e0aa",   # light green
    "O(m)":     "#f9e79f",   # yellow-ish (key-length dependent)
    "O(n)":     "#f4d03f",   # yellow
    "O(n*m)":   "#e74c3c",   # red
    "N/A":      "#636e7e",   # grey
}

# Fallback: anything not matched is a neutral tone
DEFAULT_CELL_COLOR = "#3b3b58"


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE 1 — Data Structure Complexity Table
# ═══════════════════════════════════════════════════════════════════════════
def _cell_bg(text: str) -> str:
    """Return background colour for a complexity cell."""
    return COMPLEXITY_COLORS.get(text, DEFAULT_CELL_COLOR)


def _text_color_for_bg(hex_bg: str) -> str:
    """Return black or white text based on luminance of background."""
    r, g, b = mcolors.to_rgb(hex_bg)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1e1e2e" if luminance > 0.45 else "#ffffff"


def generate_complexity_table():
    columns = ["Data Structure", "Access", "Search", "Insert", "Delete", "Space"]

    rows = [
        ["Array",       "O(1)",     "O(n)",     "O(n)",     "O(n)",     "O(n)"],
        ["Linked List", "O(n)",     "O(n)",     "O(1)",     "O(1)",     "O(n)"],
        ["Hash Table",  "N/A",      "O(1)*",    "O(1)*",    "O(1)*",    "O(n)"],
        ["BST",         "O(log n)", "O(log n)", "O(log n)", "O(log n)", "O(n)"],
        ["Heap",        "O(1)\u2020",   "O(n)",     "O(log n)", "O(log n)", "O(n)"],
        ["Stack",       "O(n)",     "O(n)",     "O(1)",     "O(1)",     "O(n)"],
        ["Queue",       "O(n)",     "O(n)",     "O(1)",     "O(1)",     "O(n)"],
        ["Trie",        "O(m)",     "O(m)",     "O(m)",     "O(m)",     "O(n*m)"],
    ]

    n_rows = len(rows)
    n_cols = len(columns)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 1.5)  # extra room at top for header + title
    ax.axis("off")

    # ---- Title ----
    fig.text(
        0.5, 0.94,
        "Data Structure Operations \u2014 Time Complexity",
        ha="center", va="center",
        fontsize=22, fontweight="bold",
        color=TEXT_COLOR, fontfamily="sans-serif",
    )

    # ---- Geometry ----
    row_height = 0.85
    col_widths = [2.2, 1.0, 1.0, 1.0, 1.0, 1.0]  # first column wider
    # Normalise so they sum to n_cols
    total_w = sum(col_widths)
    col_widths = [w * n_cols / total_w for w in col_widths]

    def col_left(c):
        return sum(col_widths[:c])

    def draw_cell(row_idx, col_idx, text, is_header=False):
        """Draw a rounded-rect cell with text."""
        x = col_left(col_idx)
        # row_idx 0 = bottom row; header is at n_rows
        y = row_idx * row_height
        w = col_widths[col_idx]
        h = row_height

        if is_header:
            bg = HEADER_BG
            fc = HEADER_TEXT
            fw = "bold"
            fs = 13
        elif col_idx == 0:
            # Data-structure name column
            bg = SURFACE_COLOR
            fc = TEXT_COLOR
            fw = "bold"
            fs = 13
        else:
            bg = _cell_bg(text)
            fc = _text_color_for_bg(bg)
            fw = "semibold"
            fs = 13

        pad = 0.04
        rect = FancyBboxPatch(
            (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
            boxstyle="round,pad=0.05",
            facecolor=bg,
            edgecolor=GRID_COLOR,
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fs, fontweight=fw,
            color=fc, fontfamily="sans-serif",
        )

    # ---- Draw header row ----
    header_row_idx = n_rows  # top
    for c, col_name in enumerate(columns):
        draw_cell(header_row_idx, c, col_name, is_header=True)

    # ---- Draw data rows (top-to-bottom, so first data row is just below header) ----
    for r_i, row_data in enumerate(rows):
        visual_row = n_rows - 1 - r_i  # flip so first row is at top
        for c_i, cell_text in enumerate(row_data):
            draw_cell(visual_row, c_i, cell_text)

    # ---- Footnotes ----
    fig.text(
        0.5, 0.04,
        "* = amortized average    \u2020 = min/max only    m = key length",
        ha="center", va="center",
        fontsize=11, color="#8a8fa6", fontfamily="sans-serif",
        style="italic",
    )

    # ---- Legend strip ----
    legend_items = [
        ("O(1)", COMPLEXITY_COLORS["O(1)"]),
        ("O(log n)", COMPLEXITY_COLORS["O(log n)"]),
        ("O(n) / O(m)", COMPLEXITY_COLORS["O(n)"]),
        ("O(n*m)", COMPLEXITY_COLORS["O(n*m)"]),
    ]
    legend_y = 0.075
    start_x = 0.22
    spacing = 0.16
    for i, (label, color) in enumerate(legend_items):
        x = start_x + i * spacing
        fig.patches.append(
            FancyBboxPatch(
                (x - 0.015, legend_y - 0.012), 0.03, 0.024,
                boxstyle="round,pad=0.003",
                facecolor=color,
                edgecolor="none",
                transform=fig.transFigure,
                figure=fig,
            )
        )
        fig.text(
            x + 0.025, legend_y,
            label,
            ha="left", va="center",
            fontsize=10, color=TEXT_COLOR, fontfamily="sans-serif",
        )

    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.12)
    fig.savefig(TABLE_OUTPUT, facecolor=fig.get_facecolor(), edgecolor="none",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[1/2] Saved: {TABLE_OUTPUT}")


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE 2 — 45-Minute Interview Timeline
# ═══════════════════════════════════════════════════════════════════════════
def generate_interview_timeline():
    # tip_dy: vertical offset for the tip row (to stagger crowded tips)
    phases = [
        {"name": "Clarify",         "start": 0,  "end": 3,  "color": "#4285f4",
         "tip": "Ask about inputs,\noutputs, edge cases", "tip_dy": 0},
        {"name": "Brute\nForce",    "start": 3,  "end": 4,  "color": "#ff9800",
         "tip": "State O(n\u00b2)\napproach", "tip_dy": -0.65},
        {"name": "Explain\nOptimal", "start": 4,  "end": 8,  "color": "#fbbc04",
         "tip": "Get buy-in\nbefore coding", "tip_dy": 0},
        {"name": "Code",            "start": 8,  "end": 28, "color": "#34a853",
         "tip": "Clean, production-ready code", "tip_dy": 0},
        {"name": "Dry Run",         "start": 28, "end": 33, "color": "#9b59b6",
         "tip": "Walk through\nan example", "tip_dy": 0},
        {"name": "Complexity",      "start": 33, "end": 35, "color": "#1abc9c",
         "tip": "State time\n& space", "tip_dy": -0.65},
        {"name": "Follow-up",      "start": 35, "end": 45, "color": "#7f8c8d",
         "tip": "Handle\nextensions", "tip_dy": 0},
    ]

    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Layout constants
    bar_y = 5.5          # centre of main timeline bar
    bar_height = 1.8
    bar_top = bar_y + bar_height / 2
    bar_bot = bar_y - bar_height / 2

    # ---- Draw phase segments ----
    for phase in phases:
        duration = phase["end"] - phase["start"]
        rect = FancyBboxPatch(
            (phase["start"], bar_bot),
            duration, bar_height,
            boxstyle="round,pad=0.15",
            facecolor=phase["color"],
            edgecolor="#1e1e2e",
            linewidth=2.5,
            alpha=0.92,
        )
        ax.add_patch(rect)

        mid_x = (phase["start"] + phase["end"]) / 2

        # ---------- Labels inside the bar ----------
        if duration >= 5:
            # Wide segments: name + time inside
            ax.text(mid_x, bar_y + 0.20, phase["name"],
                    ha="center", va="center", fontsize=15, fontweight="bold",
                    color="#ffffff", fontfamily="sans-serif")
            ax.text(mid_x, bar_y - 0.38,
                    f"{phase['start']}\u2013{phase['end']} min",
                    ha="center", va="center", fontsize=10, fontweight="medium",
                    color="#ffffffcc", fontfamily="sans-serif")
        elif duration >= 3:
            ax.text(mid_x, bar_y + 0.18, phase["name"],
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="#ffffff", fontfamily="sans-serif",
                    linespacing=0.9)
            ax.text(mid_x, bar_y - 0.35,
                    f"{phase['start']}\u2013{phase['end']}m",
                    ha="center", va="center", fontsize=9, fontweight="medium",
                    color="#ffffffcc", fontfamily="sans-serif")
        elif duration >= 2:
            ax.text(mid_x, bar_y + 0.12, phase["name"],
                    ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="#ffffff", fontfamily="sans-serif",
                    linespacing=0.85)
            ax.text(mid_x, bar_y - 0.35,
                    f"{phase['start']}\u2013{phase['end']}m",
                    ha="center", va="center", fontsize=7.5, fontweight="medium",
                    color="#ffffffcc", fontfamily="sans-serif")
        else:
            # Very narrow (1 min): label floats above
            ax.text(mid_x, bar_top + 0.65, phase["name"],
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=phase["color"], fontfamily="sans-serif",
                    linespacing=0.85)
            ax.plot([mid_x, mid_x], [bar_top + 0.08, bar_top + 0.60],
                    color=phase["color"], lw=1.0, alpha=0.7)
            ax.text(mid_x, bar_y,
                    f"{phase['start']}\u2013{phase['end']}m",
                    ha="center", va="center", fontsize=6.5, fontweight="medium",
                    color="#ffffffcc", fontfamily="sans-serif")

    # ---- Minute-marker scale (separate row below bar) ----
    scale_y = bar_bot - 0.35
    # thin horizontal line
    ax.plot([0, 45], [scale_y, scale_y], color=GRID_COLOR, lw=0.8, alpha=0.5)
    for m in range(0, 46, 5):
        ax.plot([m, m], [scale_y - 0.12, scale_y + 0.12],
                color=GRID_COLOR, lw=0.8)
        ax.text(m, scale_y - 0.28, str(m),
                ha="center", va="top", fontsize=8,
                color="#8a8fa6", fontfamily="sans-serif")

    # ---- Tips row (well below the scale), staggered via tip_dy ----
    tip_row_y = scale_y - 0.75
    for phase in phases:
        mid_x = (phase["start"] + phase["end"]) / 2
        dy = phase.get("tip_dy", 0)
        this_tip_y = tip_row_y + dy
        ax.text(mid_x, this_tip_y, phase["tip"],
                ha="center", va="top", fontsize=7.5, fontweight="normal",
                color="#a0a8c4", fontfamily="sans-serif",
                style="italic", linespacing=1.15)
        # subtle connector line from scale to tip
        ax.plot([mid_x, mid_x],
                [scale_y - 0.40, this_tip_y + 0.08],
                color=phase["color"], lw=0.6, alpha=0.35)

    # ---- Title ----
    ax.text(22.5, bar_top + 2.0,
            "Your 45-Minute Game Plan",
            ha="center", va="center",
            fontsize=24, fontweight="bold",
            color=TEXT_COLOR, fontfamily="sans-serif")

    # ---- Subtitle ----
    ax.text(22.5, bar_top + 1.45,
            "Google SDE On-Site  \u2014  Coding Round",
            ha="center", va="center",
            fontsize=13, fontweight="normal",
            color="#8a8fa6", fontfamily="sans-serif")

    # ---- "~45% of your time" callout ----
    ax.annotate(
        "~45% of your time",
        xy=(18, bar_top + 0.05),
        xytext=(18, bar_top + 0.75),
        ha="center",
        fontsize=10, fontweight="semibold",
        color="#34a853", fontfamily="sans-serif",
        arrowprops=dict(arrowstyle="-|>", color="#34a853", lw=1.5),
    )

    # ---- Axes config ----
    ax.set_xlim(-2, 48)
    ax.set_ylim(tip_row_y - 2.0, bar_top + 2.6)
    ax.axis("off")

    fig.tight_layout(pad=1.0)
    fig.savefig(TIMELINE_OUTPUT, facecolor=fig.get_facecolor(), edgecolor="none",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[2/2] Saved: {TIMELINE_OUTPUT}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    generate_complexity_table()
    generate_interview_timeline()
    print("\nDone — both visualizations generated.")
