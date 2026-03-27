#!/usr/bin/env python3
"""
Generate two visualizations:
  1. trie_structure.png  – Trie (prefix tree) for a small word set
  2. binary_search.png   – Step-by-step binary search on a sorted array
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── colour palette (Catppuccin-ish on dark bg) ──────────────────────────
BG       = "#1e1e2e"
TEXT     = "#cdd6f4"
ACCENT   = "#89b4fa"   # blue
GREEN    = "#a6e3a1"
YELLOW   = "#f9e2af"
GRAY     = "#585b70"
PINK     = "#f5c2e7"
PEACH    = "#fab387"
RED      = "#f38ba8"
LAVENDER = "#b4befe"
NODE_DEFAULT = "#45475a"
EDGE_COL     = "#7f849c"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "viz")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# 1.  TRIE STRUCTURE
# =====================================================================
def build_trie(words):
    """Return trie as nested dict; '$' key marks end-of-word."""
    root = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = True
    return root


def _layout_trie(node, x, y, x_span, positions, edges, parent_id, prefix):
    """Recursive DFS layout – assign (x, y) to every node."""
    children = sorted(k for k in node if k != "$")
    n = len(children)
    if n == 0:
        return

    # Distribute children evenly across the x_span centred on x
    if n == 1:
        child_xs = [x]
    else:
        child_xs = np.linspace(x - x_span / 2, x + x_span / 2, n)

    dy = -1.4  # vertical spacing

    for i, ch in enumerate(children):
        child_node = node[ch]
        child_id = prefix + ch
        is_end = "$" in child_node
        positions[child_id] = (child_xs[i], y + dy, is_end)
        edges.append((parent_id, child_id, ch))
        _layout_trie(child_node, child_xs[i], y + dy,
                     x_span / max(n, 1.8), positions, edges, child_id, child_id)


def draw_trie():
    words = ["app", "apple", "api", "bat", "ball"]
    trie = build_trie(words)

    # Root node
    positions = {"root": (0, 0, False)}
    edges = []
    _layout_trie(trie, 0, 0, 10, positions, edges, "root", "")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.set_title("Trie — Prefix Tree Structure", fontsize=22,
                  fontweight="bold", color=TEXT, pad=20)

    # Draw edges first (under nodes)
    for pid, cid, ch in edges:
        px, py, _ = positions[pid]
        cx, cy, _ = positions[cid]
        ax.annotate(
            "", xy=(cx, cy), xytext=(px, py),
            arrowprops=dict(arrowstyle="-", color=EDGE_COL, lw=2))
        # Edge label – placed at midpoint, offset slightly
        mx, my = (px + cx) / 2, (py + cy) / 2
        # offset label to the left of the line slightly
        dx = cx - px
        offset_x = -0.18 if dx >= 0 else 0.18
        ax.text(mx + offset_x, my + 0.12, ch, fontsize=14,
                fontweight="bold", color=PEACH, ha="center", va="center",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.8))

    # Draw nodes
    radius = 0.35
    for nid, (nx, ny, is_end) in positions.items():
        face = GREEN if is_end else NODE_DEFAULT
        edge_c = ACCENT if is_end else EDGE_COL
        circle = plt.Circle((nx, ny), radius, fc=face, ec=edge_c, lw=2.5, zorder=5)
        ax.add_patch(circle)
        # Label inside node – show last char or "∅" for root
        label = nid[-1] if nid != "root" else "∅"
        ax.text(nx, ny, label, fontsize=13, fontweight="bold",
                color=BG if is_end else TEXT, ha="center", va="center",
                fontfamily="monospace", zorder=6)

    # Legend
    legend_items = [
        mpatches.Patch(fc=GREEN, ec=ACCENT, label="End-of-word node"),
        mpatches.Patch(fc=NODE_DEFAULT, ec=EDGE_COL, label="Internal node"),
    ]
    leg = ax.legend(handles=legend_items, loc="lower right", fontsize=11,
                    facecolor="#313244", edgecolor=GRAY, labelcolor=TEXT,
                    framealpha=0.95)
    leg.get_frame().set_linewidth(1.5)

    # Subtitle
    ax.text(0, -7.6,
            f'Words: {words}\nPrefix sharing: "app" ⊂ "apple",  "ap" ⊂ "api",  "ba" ⊂ "bat"/"ball"',
            fontsize=11, color=LAVENDER, ha="center", va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#313244", ec=GRAY, lw=1))

    # Fit view
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    margin = 1.2
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin * 2, max(ys) + margin)

    path = os.path.join(OUT_DIR, "trie_structure.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved  {path}")


# =====================================================================
# 2.  BINARY SEARCH STEP-BY-STEP
# =====================================================================
def draw_binary_search():
    arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
    target = 23

    # Steps: (lo, hi, mid, comparison_result)
    steps = [
        (0, 9, 4, "16 < 23  →  go right"),
        (5, 9, 7, "56 > 23  →  go left"),
        (5, 6, 5, "23 == 23  →  found!"),
    ]

    n = len(arr)
    num_steps = len(steps)
    row_h = 1.8  # vertical space per row

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_title("Binary Search — Halving the Search Space", fontsize=22,
                  fontweight="bold", color=TEXT, pad=24)

    cell_w = 1.0
    cell_h = 0.7
    x_start = 0.5  # left margin for cells

    # Draw each step as a row
    for step_idx, (lo, hi, mid, desc) in enumerate(steps):
        y_base = -(step_idx * row_h)

        # Step label on the left
        ax.text(x_start - 0.6, y_base, f"Step {step_idx + 1}",
                fontsize=13, fontweight="bold", color=ACCENT, ha="right", va="center",
                fontfamily="monospace")

        found = (step_idx == num_steps - 1 and "found" in desc)

        for i in range(n):
            cx = x_start + i * cell_w
            cy = y_base

            # Determine cell colour
            if found and i == mid:
                fc = GREEN
                tc = BG
            elif i == mid:
                fc = YELLOW
                tc = BG
            elif lo <= i <= hi:
                fc = "#45475a"
                tc = TEXT
            else:
                fc = "#2a2a3a"
                tc = GRAY

            rect = plt.Rectangle((cx - cell_w / 2, cy - cell_h / 2),
                                 cell_w, cell_h, fc=fc, ec=EDGE_COL, lw=1.5,
                                 zorder=3, clip_on=False)
            ax.add_patch(rect)
            ax.text(cx, cy, str(arr[i]), fontsize=13, fontweight="bold",
                    color=tc, ha="center", va="center", fontfamily="monospace",
                    zorder=4)

            # Index labels (only on first row)
            if step_idx == 0:
                ax.text(cx, y_base + cell_h / 2 + 0.55, str(i),
                        fontsize=9, color=GRAY, ha="center", va="center",
                        fontfamily="monospace")
                if i == 0:
                    ax.text(cx, y_base + cell_h / 2 + 0.80, "index",
                            fontsize=9, color=GRAY, ha="center", va="center",
                            fontfamily="monospace", fontstyle="italic")

        # Pointer arrows below cells
        arrow_y = y_base - cell_h / 2 - 0.15
        label_y = arrow_y - 0.32

        def _draw_ptr(idx, label, color):
            cx = x_start + idx * cell_w
            ax.annotate("", xy=(cx, y_base - cell_h / 2),
                        xytext=(cx, arrow_y - 0.18),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
                        zorder=5)
            ax.text(cx, label_y - 0.20, label, fontsize=10, fontweight="bold",
                    color=color, ha="center", va="top", fontfamily="monospace")

        _draw_ptr(lo, f"lo={lo}", ACCENT)
        if hi != lo:
            _draw_ptr(hi, f"hi={hi}", PINK)
        else:
            # Shift hi label a bit if overlapping lo
            cx = x_start + hi * cell_w
            ax.text(cx + 0.42, label_y - 0.20, f"hi={hi}", fontsize=10,
                    fontweight="bold", color=PINK, ha="center", va="top",
                    fontfamily="monospace")

        mid_color = GREEN if found else YELLOW
        if mid != lo and mid != hi:
            _draw_ptr(mid, f"mid={mid}", mid_color)
        else:
            cx_m = x_start + mid * cell_w
            ax.text(cx_m, label_y - 0.55, f"mid={mid}", fontsize=10,
                    fontweight="bold", color=mid_color, ha="center", va="top",
                    fontfamily="monospace")

        # Description on the right
        desc_x = x_start + n * cell_w + 0.25
        ax.text(desc_x, y_base, desc, fontsize=11, color=PEACH,
                ha="left", va="center", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="#313244", ec=GRAY, lw=1))

    # Legend
    legend_items = [
        mpatches.Patch(fc="#45475a", ec=EDGE_COL, label="Active range"),
        mpatches.Patch(fc="#2a2a3a", ec=EDGE_COL, label="Eliminated"),
        mpatches.Patch(fc=YELLOW, ec=EDGE_COL, label="mid element"),
        mpatches.Patch(fc=GREEN, ec=EDGE_COL, label="Found!"),
    ]
    leg = ax.legend(handles=legend_items, loc="upper right", fontsize=11,
                    facecolor="#313244", edgecolor=GRAY, labelcolor=TEXT,
                    framealpha=0.95)
    leg.get_frame().set_linewidth(1.5)

    # Subtitle
    sub_y = -(num_steps * row_h) - 0.3
    ax.text(x_start + (n * cell_w) / 2 - 0.5, sub_y,
            f"Array: {arr}    Target: {target}    →  O(log n) = O(log {n}) ≈ {np.log2(n):.1f} comparisons max",
            fontsize=11, color=LAVENDER, ha="center", va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#313244", ec=GRAY, lw=1))

    # Fit view
    ax.set_xlim(-1.5, x_start + n * cell_w + 5.5)
    ax.set_ylim(sub_y - 0.9, 1.8)

    path = os.path.join(OUT_DIR, "binary_search.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved  {path}")


# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    draw_trie()
    draw_binary_search()
    print("\nDone – both visualizations generated.")
