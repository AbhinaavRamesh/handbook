#!/usr/bin/env python3
"""
Generate Space Complexity and Backtracking Decision Tree visualizations.

Outputs:
  - public/viz/space_complexity.png   (Memory usage comparison across Big-O classes)
  - public/viz/backtracking_tree.png  (Subset generation decision tree for [1,2,3])
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
FLAMINGO  = "#f2cdcd"
SAPPHIRE  = "#74c7ec"


# ===================================================================
# IMAGE 1 -- Space Complexity Visualization
# ===================================================================
def generate_space_complexity():
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG,
                             gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.35})

    # ---- Top panel: bar chart of memory usage at n=1000 ----
    ax = axes[0]
    ax.set_facecolor(BG)

    categories = ["O(1)\nIn-place", "O(n)\nLinear", "O(n²)\nMatrix", "O(2ⁿ)\nRecursive"]
    # Memory values in bytes for n=1000 (using log scale for display)
    # O(1): 8 bytes, O(n): 8000 bytes (8 KB), O(n^2): 8_000_000 (8 MB), O(2^n): impossible
    memory_bytes = [8, 8_000, 8_000_000, None]  # None = impossible
    colors = [GREEN, BLUE, YELLOW, RED]
    display_vals = [8, 8_000, 8_000_000, 8_000_000 * 50]  # last is placeholder for bar height
    labels_text = ["8 B", "8 KB", "8 MB", "∞  (impossible)"]

    bars = ax.bar(range(4), [np.log10(v) if v else 10 for v in display_vals],
                  color=colors, width=0.55, edgecolor=DARK_GRAY, linewidth=1.5,
                  alpha=0.9, zorder=3)

    # Draw a pattern on the last bar to indicate "impossible"
    bars[3].set_hatch("///")
    bars[3].set_alpha(0.5)

    # Add memory labels on each bar
    for i, (bar_obj, label) in enumerate(zip(bars, labels_text)):
        y_pos = bar_obj.get_height() + 0.15
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, y_pos,
                label, ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=colors[i],
                path_effects=[patheffects.withStroke(linewidth=2, foreground=BG)])

    # Draw stacked blocks inside bars to represent memory composition
    block_data = [
        # (category_idx, blocks: list of (height_frac, color, label))
        (0, [(1.0, GREEN, "Input\n(in-place)")]),
        (1, [(0.4, SURFACE, "Input"), (0.6, BLUE, "Aux Array")]),
        (2, [(0.15, SURFACE, "Input"), (0.85, YELLOW, "2D DP\nTable")]),
        (3, [(0.1, SURFACE, "Input"), (0.9, RED, "Call Stack\n(2ⁿ frames)")]),
    ]

    for cat_idx, blocks in block_data:
        bar_obj = bars[cat_idx]
        bar_x = bar_obj.get_x()
        bar_w = bar_obj.get_width()
        bar_h = bar_obj.get_height()

        y_cursor = 0
        for height_frac, color, blabel in blocks:
            block_h = bar_h * height_frac
            inner_rect = mpatches.FancyBboxPatch(
                (bar_x + 0.03, y_cursor + 0.03),
                bar_w - 0.06, block_h - 0.06,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=DARK_GRAY,
                linewidth=0.8, alpha=0.35, zorder=4,
            )
            ax.add_patch(inner_rect)
            # Block label
            if block_h > 0.8:
                ax.text(bar_x + bar_w / 2, y_cursor + block_h / 2,
                        blabel, ha="center", va="center", fontsize=7.5,
                        color=TEXT, fontweight="bold", zorder=5, alpha=0.85)
            y_cursor += block_h

    # Annotations
    ax.annotate(
        "In-place algorithms\nsave memory!",
        xy=(0, bars[0].get_height()),
        xytext=(-0.1, bars[0].get_height() + 2.5),
        fontsize=9.5, color=GREEN, fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
        zorder=6,
    )

    ax.annotate(
        "DP space optimization:\n2D → 1D (rolling array)",
        xy=(2, bars[2].get_height()),
        xytext=(2.7, bars[2].get_height() + 1.8),
        fontsize=9.5, color=YELLOW, fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.5),
        zorder=6,
    )

    # Styling
    ax.set_xticks(range(4))
    ax.set_xticklabels(categories, fontsize=11, color=TEXT, fontweight="bold")
    ax.set_ylabel("Memory  (log₁₀ bytes)", fontsize=11, color=SUBTEXT, fontweight="bold")
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_yticklabels(["1 B", "10 B", "100 B", "1 KB", "10 KB", "100 KB",
                         "1 MB", "10 MB", "100 MB", "1 GB", "10 GB"],
                        fontsize=8, color=SUBTEXT)
    ax.set_ylim(0, 11.5)
    ax.tick_params(axis="x", colors=TEXT, length=0)
    ax.tick_params(axis="y", colors=SUBTEXT, length=3)
    for spine in ax.spines.values():
        spine.set_color(DARK_GRAY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=DARK_GRAY, alpha=0.4, linestyle="--", zorder=1)

    # Title
    ax.text(
        1.5, 11.0,
        "Space Complexity — Memory Usage at n = 1,000",
        fontsize=18, fontweight="bold", color=TEXT, ha="center",
        path_effects=[patheffects.withStroke(linewidth=3, foreground=BG)],
    )

    # ---- Bottom panel: practical example (Unique Paths rolling array) ----
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.axis("off")

    # Draw a "before and after" box
    box_y = 0.5
    # Before box
    before_rect = mpatches.FancyBboxPatch(
        (0.03, 0.1), 0.44, 0.8,
        boxstyle="round,pad=0.03", facecolor=SURFACE,
        edgecolor=YELLOW, linewidth=2, transform=ax2.transAxes,
    )
    ax2.add_patch(before_rect)
    ax2.text(0.25, 0.82, "Before: Unique Paths", fontsize=12, fontweight="bold",
             color=YELLOW, ha="center", va="center", transform=ax2.transAxes)
    ax2.text(0.25, 0.60, "dp[m][n] — full 2D table", fontsize=10, color=TEXT,
             ha="center", va="center", transform=ax2.transAxes, family="monospace")
    ax2.text(0.25, 0.42, "Space: O(m × n)", fontsize=11, color=RED,
             ha="center", va="center", transform=ax2.transAxes, fontweight="bold")
    # Small grid illustration
    grid_x0, grid_y0 = 0.10, 0.12
    cell_s = 0.028
    for r in range(3):
        for c in range(4):
            rect = mpatches.Rectangle(
                (grid_x0 + c * cell_s, grid_y0 + r * cell_s),
                cell_s, cell_s, facecolor=YELLOW, edgecolor=BG,
                linewidth=0.5, alpha=0.5, transform=ax2.transAxes, zorder=5,
            )
            ax2.add_patch(rect)

    # Arrow between boxes
    ax2.annotate(
        "", xy=(0.53, 0.5), xytext=(0.47, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=3),
    )
    ax2.text(0.50, 0.58, "optimize", fontsize=9, color=TEAL, ha="center",
             va="center", transform=ax2.transAxes, fontweight="bold")

    # After box
    after_rect = mpatches.FancyBboxPatch(
        (0.53, 0.1), 0.44, 0.8,
        boxstyle="round,pad=0.03", facecolor=SURFACE,
        edgecolor=GREEN, linewidth=2, transform=ax2.transAxes,
    )
    ax2.add_patch(after_rect)
    ax2.text(0.75, 0.82, "After: Rolling Array", fontsize=12, fontweight="bold",
             color=GREEN, ha="center", va="center", transform=ax2.transAxes)
    ax2.text(0.75, 0.60, "dp[n] — single row reused", fontsize=10, color=TEXT,
             ha="center", va="center", transform=ax2.transAxes, family="monospace")
    ax2.text(0.75, 0.42, "Space: O(n)", fontsize=11, color=GREEN,
             ha="center", va="center", transform=ax2.transAxes, fontweight="bold")
    # Single row illustration
    row_x0 = 0.63
    row_y0 = 0.18
    for c in range(4):
        rect = mpatches.Rectangle(
            (row_x0 + c * cell_s, row_y0),
            cell_s, cell_s, facecolor=GREEN, edgecolor=BG,
            linewidth=0.5, alpha=0.5, transform=ax2.transAxes, zorder=5,
        )
        ax2.add_patch(rect)

    # Savings callout
    ax2.text(0.50, 0.02, "Unique Paths: O(m·n) → O(n)  with rolling array  —  saves (m−1)×n memory!",
             fontsize=10, color=TEAL, ha="center", va="center", transform=ax2.transAxes,
             fontweight="bold", fontstyle="italic")

    out = os.path.join(OUT_DIR, "space_complexity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, pad_inches=0.4)
    plt.close(fig)
    print(f"Saved: {out}")


# ===================================================================
# IMAGE 2 -- Backtracking Decision Tree (Subset Generation)
# ===================================================================
def generate_backtracking_tree():
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=BG)
    ax.set_facecolor(BG)

    # -------------------------------------------------------------------
    # Build the binary tree for subsets of [1, 2, 3]
    # Each node: (subset_so_far, element_being_decided, children)
    # Tree structure (left=include, right=exclude):
    #
    # Level 0 (decide 1):             []
    #                           /            \
    # Level 1 (decide 2):    [1]              []
    #                       /    \          /     \
    # Level 2 (decide 3): [1,2]  [1]     [2]      []
    #                     / \    / \     / \      / \
    # Level 3 (leaves): [1,2,3][1,2][1,3][1] [2,3][2] [3] []
    # -------------------------------------------------------------------

    # Node positions: (x, y) for each node
    # We'll lay them out manually for precise control

    # Y levels
    y_levels = [8.5, 6.2, 3.9, 1.6]

    # Define every node: (label, x, y, is_leaf, is_pruned)
    # Level 0
    nodes = {}
    nodes["root"] = {"label": "[ ]", "x": 8.0, "y": y_levels[0],
                      "leaf": False, "pruned": False, "decision": "include 1?"}

    # Level 1
    nodes["L1_inc"] = {"label": "[1]", "x": 4.0, "y": y_levels[1],
                        "leaf": False, "pruned": False, "decision": "include 2?"}
    nodes["L1_exc"] = {"label": "[ ]", "x": 12.0, "y": y_levels[1],
                        "leaf": False, "pruned": False, "decision": "include 2?"}

    # Level 2
    nodes["L2_11"] = {"label": "[1,2]", "x": 2.0, "y": y_levels[2],
                       "leaf": False, "pruned": False, "decision": "include 3?"}
    nodes["L2_10"] = {"label": "[1]", "x": 6.0, "y": y_levels[2],
                       "leaf": False, "pruned": False, "decision": "include 3?"}
    nodes["L2_01"] = {"label": "[2]", "x": 10.0, "y": y_levels[2],
                       "leaf": False, "pruned": False, "decision": "include 3?"}
    nodes["L2_00"] = {"label": "[ ]", "x": 14.0, "y": y_levels[2],
                       "leaf": False, "pruned": False, "decision": "include 3?"}

    # Level 3 (leaves)
    leaf_x = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    leaf_labels = ["{1,2,3}", "{1,2}", "{1,3}", "{1}", "{2,3}", "{2}", "{3}", "{ }"]
    for i, (lx, ll) in enumerate(zip(leaf_x, leaf_labels)):
        nodes[f"leaf_{i}"] = {"label": ll, "x": lx, "y": y_levels[3],
                               "leaf": True, "pruned": False}

    # Edges: (parent, child, direction)
    # direction: "include" (left/green) or "exclude" (right/red)
    edges = [
        ("root", "L1_inc", "include"),
        ("root", "L1_exc", "exclude"),
        ("L1_inc", "L2_11", "include"),
        ("L1_inc", "L2_10", "exclude"),
        ("L1_exc", "L2_01", "include"),
        ("L1_exc", "L2_00", "exclude"),
        ("L2_11", "leaf_0", "include"),
        ("L2_11", "leaf_1", "exclude"),
        ("L2_10", "leaf_2", "include"),
        ("L2_10", "leaf_3", "exclude"),
        ("L2_01", "leaf_4", "include"),
        ("L2_01", "leaf_5", "exclude"),
        ("L2_00", "leaf_6", "include"),
        ("L2_00", "leaf_7", "exclude"),
    ]

    # --- Add a pruned branch to illustrate pruning concept ---
    # We'll add a "pruned" ghost branch from L2_00 showing what pruning looks like
    # Imagine we had a constraint "subset sum <= 3" - the branch [2,3] wouldn't exist
    # But for subsets of [1,2,3] there's no natural pruning, so we'll show an
    # illustrative pruned branch off to the side.
    # We'll mark the {2,3} and {3} subtree as pruned for illustration.
    # Actually, let's add a separate pruned subtree from a hypothetical node.

    # Instead, mark a specific branch as pruned for demonstration
    # Let's mark the branch from L2_00 -> leaf_6 as "pruned example"
    pruned_node_key = "pruned_0"
    nodes[pruned_node_key] = {"label": "...", "x": 12.5, "y": y_levels[3] - 0.1,
                               "leaf": True, "pruned": True}

    # --- Draw edges first (below nodes) ---
    for parent_key, child_key, direction in edges:
        p = nodes[parent_key]
        c = nodes[child_key]

        edge_color = GREEN if direction == "include" else RED
        edge_alpha = 0.85
        edge_lw = 2.0

        # Draw the edge line
        ax.plot([p["x"], c["x"]], [p["y"] - 0.35, c["y"] + 0.45],
                color=edge_color, linewidth=edge_lw, alpha=edge_alpha,
                solid_capstyle="round", zorder=2)

        # Edge label at midpoint
        mid_x = (p["x"] + c["x"]) / 2
        mid_y = (p["y"] - 0.35 + c["y"] + 0.45) / 2
        # Offset labels slightly to the side
        offset_x = -0.25 if direction == "include" else 0.25
        edge_label = "✓" if direction == "include" else "✗"
        ax.text(mid_x + offset_x, mid_y, edge_label,
                fontsize=10, color=edge_color, ha="center", va="center",
                fontweight="bold", alpha=0.9, zorder=6)

    # Draw the pruned branch
    prune_parent = nodes["L2_00"]
    prune_child = nodes[pruned_node_key]
    ax.plot([prune_parent["x"] - 0.3, prune_child["x"]],
            [prune_parent["y"] - 0.35, prune_child["y"] + 0.35],
            color=GRAY, linewidth=2.0, alpha=0.4, linestyle="--", zorder=2)

    # --- Draw nodes ---
    for key, node in nodes.items():
        x, y = node["x"], node["y"]
        label = node["label"]
        is_leaf = node["leaf"]
        is_pruned = node.get("pruned", False)

        if is_pruned:
            # Pruned node: grayed out
            node_color = GRAY
            text_color = DARK_GRAY
            edge_col = GRAY
            node_alpha = 0.4
            box_w, box_h = 0.8, 0.5
        elif is_leaf:
            # Leaf node: bright green highlight
            node_color = GREEN
            text_color = BG
            edge_col = "#7bc88a"
            node_alpha = 1.0
            box_w, box_h = 1.1, 0.6
        else:
            # Internal node
            node_color = SURFACE
            text_color = TEXT
            edge_col = BLUE
            node_alpha = 1.0
            box_w, box_h = 1.0, 0.6

        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1", facecolor=node_color,
            edgecolor=edge_col, linewidth=1.8, alpha=node_alpha, zorder=4,
        )
        ax.add_patch(rect)

        ax.text(x, y, label, fontsize=10 if not is_leaf else 9.5,
                fontweight="bold", color=text_color, ha="center", va="center",
                zorder=5, alpha=node_alpha, family="monospace")

        # Decision label above internal nodes
        if not is_leaf and not is_pruned and "decision" in node:
            ax.text(x, y + 0.50, node["decision"],
                    fontsize=8, color=SAPPHIRE, ha="center", va="center",
                    fontstyle="italic", zorder=5)

    # --- Pruned annotation ---
    ax.annotate(
        "Pruned!\n(constraint\n violated)",
        xy=(prune_child["x"], prune_child["y"] + 0.2),
        xytext=(prune_child["x"] + 1.6, prune_child["y"] + 0.8),
        fontsize=9, color=GRAY, fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5, alpha=0.5),
        zorder=6,
    )

    # --- Level labels on the left ---
    level_labels = [
        (y_levels[0], "Root"),
        (y_levels[1], "Level 1\ndecide: 1"),
        (y_levels[2], "Level 2\ndecide: 2"),
        (y_levels[3], "Level 3\ndecide: 3"),
    ]
    for y_pos, llabel in level_labels:
        ax.text(-0.3, y_pos, llabel, fontsize=9, color=MAUVE,
                ha="right", va="center", fontweight="bold")
        # Dashed horizontal line
        ax.axhline(y=y_pos, xmin=0.02, xmax=0.98, color=DARK_GRAY,
                    linewidth=0.6, linestyle=":", alpha=0.4, zorder=1)

    # --- "Leaves = All 2ⁿ subsets" annotation ---
    ax.text(8.0, 0.5, "Leaves = All 2³ = 8 subsets of {1, 2, 3}",
            fontsize=12, color=GREEN, ha="center", va="center",
            fontweight="bold",
            path_effects=[patheffects.withStroke(linewidth=2, foreground=BG)])

    # --- Choose / Explore / Unchoose pattern box (bottom-left, below tree) ---
    pattern_x, pattern_y = 0.2, -1.6
    pattern_rect = mpatches.FancyBboxPatch(
        (pattern_x, pattern_y), 3.5, 2.0,
        boxstyle="round,pad=0.15", facecolor=SURFACE,
        edgecolor=MAUVE, linewidth=2, zorder=6,
    )
    ax.add_patch(pattern_rect)
    ax.text(pattern_x + 1.75, pattern_y + 1.7, "Backtracking Pattern",
            fontsize=10, fontweight="bold", color=MAUVE, ha="center", zorder=7)
    pattern_lines = [
        ("1. Choose", GREEN, "  (include element)"),
        ("2. Explore", BLUE, "  (recurse deeper)"),
        ("3. Unchoose", RED, "  (exclude / backtrack)"),
    ]
    for i, (step, color, desc) in enumerate(pattern_lines):
        yy = pattern_y + 1.25 - i * 0.45
        ax.text(pattern_x + 0.25, yy, step, fontsize=9.5, color=color,
                fontweight="bold", va="center", zorder=7)
        ax.text(pattern_x + 1.55, yy, desc, fontsize=8.5, color=SUBTEXT,
                va="center", zorder=7)

    # --- Legend for edges (bottom-right, below tree) ---
    legend_x, legend_y = 12.5, -1.6
    legend_rect = mpatches.FancyBboxPatch(
        (legend_x, legend_y), 3.2, 2.2,
        boxstyle="round,pad=0.15", facecolor=SURFACE,
        edgecolor=DARK_GRAY, linewidth=1.5, zorder=6,
    )
    ax.add_patch(legend_rect)
    ax.text(legend_x + 1.6, legend_y + 1.9, "Legend", fontsize=11,
            fontweight="bold", color=TEXT, ha="center", zorder=7)

    legend_items = [
        (GREEN, "Include element  (choose)"),
        (RED, "Exclude element  (unchoose)"),
        (GREEN, "Leaf: final subset"),
        (GRAY, "Pruned branch"),
    ]
    for i, (color, desc) in enumerate(legend_items):
        yy = legend_y + 1.45 - i * 0.38
        if i < 2:
            # Edge line
            ax.plot([legend_x + 0.15, legend_x + 0.55], [yy, yy],
                    color=color, linewidth=2.5, zorder=7)
        elif i == 2:
            # Leaf box
            sm_rect = mpatches.FancyBboxPatch(
                (legend_x + 0.2, yy - 0.12), 0.3, 0.24,
                boxstyle="round,pad=0.03", facecolor=color,
                edgecolor="#7bc88a", linewidth=1, zorder=7,
            )
            ax.add_patch(sm_rect)
        else:
            # Pruned dashed line
            ax.plot([legend_x + 0.15, legend_x + 0.55], [yy, yy],
                    color=color, linewidth=2, linestyle="--", alpha=0.5, zorder=7)
        ax.text(legend_x + 0.7, yy, desc, fontsize=8.5, color=SUBTEXT,
                va="center", zorder=7)

    # --- Title ---
    ax.text(8.0, 9.5, "Backtracking — Subset Generation Decision Tree",
            fontsize=20, fontweight="bold", color=TEXT, ha="center",
            path_effects=[patheffects.withStroke(linewidth=3, foreground=BG)],
            zorder=7)
    ax.text(8.0, 9.05, "Generate all subsets of [1, 2, 3]  •  Left = include  •  Right = exclude",
            fontsize=11, color=SUBTEXT, ha="center", zorder=7)

    # --- Axes ---
    ax.set_xlim(-1.5, 17.0)
    ax.set_ylim(-2.2, 10.0)
    ax.set_aspect("equal")
    ax.axis("off")

    out = os.path.join(OUT_DIR, "backtracking_tree.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, pad_inches=0.4)
    plt.close(fig)
    print(f"Saved: {out}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating Space Complexity & Backtracking visualizations...")
    generate_space_complexity()
    generate_backtracking_tree()
    print("Done.")
