#!/usr/bin/env python3
"""
Generate a pattern recognition decision flowchart for algorithm problems.
Outputs a visually appealing PNG that serves as a quick reference during study.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Color palette (dark background theme)
# ---------------------------------------------------------------------------
BG_COLOR = "#1e1e2e"
TEXT_COLOR = "#cdd6f4"
ARROW_COLOR = "#7f849c"
LABEL_COLOR = "#a6adc8"
TITLE_COLOR = "#cdd6f4"

# Tier colours  (fill, border)
COLORS = {
    "root":    ("#585b70", "#7f849c"),
    "graph":   ("#45283c", "#f38ba8"),   # red family
    "tree":    ("#45352e", "#fab387"),    # orange family
    "window":  ("#3e3a2e", "#f9e2af"),   # yellow family
    "dp":      ("#2e3e33", "#a6e3a1"),   # green family
    "ds":      ("#2e3545", "#89b4fa"),   # blue family
    "string":  ("#3a2e45", "#cba6f7"),   # purple family
    "search":  ("#2e3545", "#89dceb"),   # cyan family
    "bt":      ("#3a2e45", "#f5c2e7"),   # pink family
}

# Answer‑node accent colours (brighter fills for leaf/answer nodes)
ANSWER_COLORS = {
    "graph":   ("#6c2f4a", "#f38ba8"),
    "tree":    ("#6c4a2e", "#fab387"),
    "window":  ("#5c5530", "#f9e2af"),
    "dp":      ("#3a5c42", "#a6e3a1"),
    "ds":      ("#2e4a6c", "#89b4fa"),
    "string":  ("#4a2e6c", "#cba6f7"),
    "search":  ("#2e4a5c", "#89dceb"),
    "bt":      ("#5c2e5c", "#f5c2e7"),
}

# ---------------------------------------------------------------------------
# Node definitions:  (id, text, x, y, kind, tier, width, height)
# kind = "decision" | "answer"
# ---------------------------------------------------------------------------

# Layout constants
LEFT = 1.5
MID = 10.0
RIGHT = 16.5
FAR_RIGHT = 21.5

# We lay out rows from top to bottom (y decreasing)
ROW_TOP = 13.0
ROW_STEP = 1.18

def row_y(i):
    return ROW_TOP - i * ROW_STEP

nodes = [
    # Root
    ("root", "What type of\nproblem?", MID, row_y(0), "decision", "root", 3.0, 0.9),

    # --- Row 1: Shortest path ---
    ("q_short",   "Shortest path /\nminimum steps?",     LEFT, row_y(1), "decision", "graph", 2.8, 0.85),
    ("q_weight",  "Weighted\ngraph?",                    LEFT+4.8, row_y(1), "decision", "graph", 2.0, 0.85),
    ("a_dijkstra","Dijkstra's\nAlgorithm",               LEFT+8.6, row_y(1)+0.45, "answer", "graph", 2.2, 0.65),
    ("a_bfs1",    "BFS",                                 LEFT+8.6, row_y(1)-0.45, "answer", "graph", 2.2, 0.65),

    # --- Row 2: Dependencies ---
    ("q_dep",     "Dependencies /\nordering?",           LEFT, row_y(2), "decision", "graph", 2.8, 0.85),
    ("a_topo",    "Topological\nSort",                   LEFT+4.8, row_y(2), "answer", "graph", 2.2, 0.65),

    # --- Row 3: Connected components ---
    ("q_comp",    "Connected components\n/ islands?",     LEFT, row_y(3), "decision", "graph", 2.8, 0.85),
    ("a_dfs_uf",  "DFS /\nUnion-Find",                  LEFT+4.8, row_y(3), "answer", "graph", 2.2, 0.65),

    # --- Row 4: Subarray / substring ---
    ("q_sub",     "Subarray / substring\nwith condition?", LEFT, row_y(4), "decision", "window", 2.8, 0.85),
    ("q_fixed",   "Fixed\nsize?",                        LEFT+4.8, row_y(4), "decision", "window", 2.0, 0.85),
    ("a_fixwin",  "Fixed Sliding\nWindow",               LEFT+8.6, row_y(4)+0.45, "answer", "window", 2.2, 0.65),
    ("a_varwin",  "Variable Sliding\nWindow",            LEFT+8.6, row_y(4)-0.45, "answer", "window", 2.2, 0.65),

    # --- Row 5: Binary search ---
    ("q_sorted",  "Find target in\nsorted data?",        LEFT, row_y(5), "decision", "search", 2.8, 0.85),
    ("a_bsearch", "Binary\nSearch",                      LEFT+4.8, row_y(5), "answer", "search", 2.2, 0.65),

    # --- Row 6: Top K ---
    ("q_topk",    "Top K / Kth\nelement?",               LEFT, row_y(6), "decision", "ds", 2.8, 0.85),
    ("a_heap",    "Heap\n(heapq)",                       LEFT+4.8, row_y(6), "answer", "ds", 2.2, 0.65),

    # --- Row 7: Grouping ---
    ("q_group",   "Grouping / counting\n/ lookup?",      LEFT, row_y(7), "decision", "ds", 2.8, 0.85),
    ("a_hashmap", "HashMap",                             LEFT+4.8, row_y(7), "answer", "ds", 2.2, 0.65),

    # --- Row 8: Tree ---
    ("q_tree",    "Tree\nproblem?",                      LEFT, row_y(8), "decision", "tree", 2.8, 0.85),
    ("q_level",   "Need level\ninfo?",                   LEFT+4.8, row_y(8), "decision", "tree", 2.0, 0.85),
    ("a_tbfs",    "BFS\n(level-order)",                  LEFT+8.6, row_y(8)+0.45, "answer", "tree", 2.2, 0.65),
    ("a_tdfs",    "DFS\n(recursion)",                    LEFT+8.6, row_y(8)-0.45, "answer", "tree", 2.2, 0.65),

    # --- Row 9: Backtracking ---
    ("q_combo",   "All combinations /\npermutations?",   LEFT, row_y(9), "decision", "bt", 2.8, 0.85),
    ("a_bt",      "Backtracking",                        LEFT+4.8, row_y(9), "answer", "bt", 2.2, 0.65),

    # --- Row 10: DP ---
    ("q_dp",      "Optimal substructure\n+ overlapping?", LEFT, row_y(10), "decision", "dp", 2.8, 0.85),
    ("a_dp",      "Dynamic\nProgramming",                LEFT+4.8, row_y(10), "answer", "dp", 2.2, 0.65),

    # --- Row 11: Trie ---
    ("q_prefix",  "Prefix\nmatching?",                   LEFT, row_y(11), "decision", "string", 2.8, 0.85),
    ("a_trie",    "Trie",                                LEFT+4.8, row_y(11), "answer", "string", 2.2, 0.65),
]

# Build lookup
node_map = {n[0]: n for n in nodes}

# ---------------------------------------------------------------------------
# Edges: (from_id, to_id, label, exit_side)
#   exit_side: "right" = exit from right of source
#              "down"  = exit from bottom of source (root fan-out)
#              None    = auto
# ---------------------------------------------------------------------------
edges = [
    # Root -> questions (all go down from root then route left)
    ("root", "q_short",  None, "down"),
    ("root", "q_dep",    None, "down"),
    ("root", "q_comp",   None, "down"),
    ("root", "q_sub",    None, "down"),
    ("root", "q_sorted", None, "down"),
    ("root", "q_topk",   None, "down"),
    ("root", "q_group",  None, "down"),
    ("root", "q_tree",   None, "down"),
    ("root", "q_combo",  None, "down"),
    ("root", "q_dp",     None, "down"),
    ("root", "q_prefix", None, "down"),

    # Question -> sub-question or answer
    ("q_short",  "q_weight",  "Yes", "right"),
    ("q_weight", "a_dijkstra","Yes", "right"),
    ("q_weight", "a_bfs1",   "No",  "right"),

    ("q_dep",    "a_topo",    "Yes", "right"),
    ("q_comp",   "a_dfs_uf",  "Yes", "right"),

    ("q_sub",    "q_fixed",   "Yes", "right"),
    ("q_fixed",  "a_fixwin",  "Yes", "right"),
    ("q_fixed",  "a_varwin",  "No",  "right"),

    ("q_sorted", "a_bsearch", "Yes", "right"),
    ("q_topk",   "a_heap",    "Yes", "right"),
    ("q_group",  "a_hashmap", "Yes", "right"),

    ("q_tree",   "q_level",   "Yes", "right"),
    ("q_level",  "a_tbfs",    "Yes", "right"),
    ("q_level",  "a_tdfs",    "No",  "right"),

    ("q_combo",  "a_bt",      "Yes", "right"),
    ("q_dp",     "a_dp",      "Yes", "right"),
    ("q_prefix", "a_trie",    "Yes", "right"),
]

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 14), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-1, 23)
ax.set_ylim(-1.0, 14.5)
ax.set_aspect("equal")
ax.axis("off")

# Title
ax.text(MID, 14.1, "Which Pattern Should I Use?",
        fontsize=26, fontweight="bold", color=TITLE_COLOR,
        ha="center", va="center",
        fontfamily="sans-serif")

# Subtitle
ax.text(MID, 13.65, "Algorithm Pattern Recognition Decision Flowchart",
        fontsize=13, color=LABEL_COLOR, ha="center", va="center",
        fontstyle="italic", fontfamily="sans-serif")


def draw_node(ax, nid, text, cx, cy, kind, tier, w, h):
    """Draw a single node (decision = rounded rect, answer = rounded rect with accent)."""
    if kind == "answer":
        fill, border = ANSWER_COLORS.get(tier, COLORS.get(tier, COLORS["root"]))
        fontsize = 11
        fontweight = "bold"
        boxstyle = f"round,pad=0.15,rounding_size=0.25"
        border_lw = 2.5
    else:
        fill, border = COLORS.get(tier, COLORS["root"])
        fontsize = 10.5
        fontweight = "normal"
        boxstyle = f"round,pad=0.15,rounding_size=0.3"
        border_lw = 1.8

    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=boxstyle,
        facecolor=fill,
        edgecolor=border,
        linewidth=border_lw,
        zorder=3,
    )
    ax.add_patch(box)

    ax.text(cx, cy, text,
            fontsize=fontsize, fontweight=fontweight,
            color=TEXT_COLOR, ha="center", va="center",
            fontfamily="sans-serif",
            zorder=4,
            linespacing=1.25)


# Draw all nodes
for n in nodes:
    nid, text, cx, cy, kind, tier, w, h = n
    draw_node(ax, nid, text, cx, cy, kind, tier, w, h)


def get_port(nid, side):
    """Return (x, y) for a connection point on the given side of a node."""
    _, _, cx, cy, _, _, w, h = node_map[nid]
    if side == "right":
        return cx + w / 2, cy
    elif side == "left":
        return cx - w / 2, cy
    elif side == "top":
        return cx, cy + h / 2
    elif side == "bottom":
        return cx, cy - h / 2
    return cx, cy


def draw_arrow(ax, x0, y0, x1, y1, label=None, color=ARROW_COLOR, lw=1.3):
    """Draw a styled arrow with optional label."""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            shrinkA=2,
            shrinkB=2,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        # Offset label slightly
        offset_x = 0.0
        offset_y = 0.18
        if abs(x1 - x0) > abs(y1 - y0):
            offset_y = 0.22
        if label == "No":
            offset_y = -0.22

        ax.text(mx + offset_x, my + offset_y, label,
                fontsize=8.5, fontweight="bold",
                color="#f9e2af" if label == "Yes" else "#f38ba8",
                ha="center", va="center",
                fontfamily="sans-serif",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", facecolor=BG_COLOR,
                          edgecolor="none", alpha=0.85))


# Draw edges
for src_id, dst_id, label, exit_side in edges:
    src = node_map[src_id]
    dst = node_map[dst_id]

    if exit_side == "down":
        # Root -> question node: route with an elbow
        # Go down from root bottom, then left to the question's right side
        sx, sy = get_port(src_id, "bottom")
        dx, dy = get_port(dst_id, "right")

        # Intermediate point: go down to question's y, then across
        elbow_x = sx  # stay at root's x going down
        elbow_y = dy  # at the question's y level

        # Draw two segments
        # Vertical segment
        ax.plot([sx, elbow_x], [sy, elbow_y], color=ARROW_COLOR, lw=1.0,
                solid_capstyle="round", zorder=2)
        # Horizontal segment with arrow
        draw_arrow(ax, elbow_x, elbow_y, dx, dy, label=label, lw=1.0)

    elif exit_side == "right":
        sx, sy = get_port(src_id, "right")
        dx, dy = get_port(dst_id, "left")

        # If the target is offset vertically (Yes/No branches), use an elbow
        if abs(sy - dy) > 0.1:
            mid_x = (sx + dx) / 2
            ax.plot([sx, mid_x], [sy, sy], color=ARROW_COLOR, lw=1.3,
                    solid_capstyle="round", zorder=2)
            ax.plot([mid_x, mid_x], [sy, dy], color=ARROW_COLOR, lw=1.3,
                    solid_capstyle="round", zorder=2)
            draw_arrow(ax, mid_x, dy, dx, dy, label=label, lw=1.3)
        else:
            draw_arrow(ax, sx, sy, dx, dy, label=label, lw=1.3)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_x = 16.0
legend_y = ROW_TOP - 0.3
legend_items = [
    ("Graph Algorithms", "graph"),
    ("Tree Traversals", "tree"),
    ("Sliding Window", "window"),
    ("Dynamic Programming", "dp"),
    ("Data Structures", "ds"),
    ("String / Trie", "string"),
    ("Search", "search"),
    ("Backtracking", "bt"),
]

# Legend background
legend_bg = FancyBboxPatch(
    (legend_x - 0.4, legend_y - len(legend_items) * 0.55 - 0.1),
    4.8, len(legend_items) * 0.55 + 0.8,
    boxstyle="round,pad=0.2,rounding_size=0.3",
    facecolor="#313244",
    edgecolor="#585b70",
    linewidth=1.5,
    zorder=3,
)
ax.add_patch(legend_bg)

ax.text(legend_x + 2.0, legend_y + 0.2, "Pattern Categories",
        fontsize=12, fontweight="bold", color=TITLE_COLOR,
        ha="center", va="center", fontfamily="sans-serif", zorder=4)

for i, (lbl, tier) in enumerate(legend_items):
    ly = legend_y - 0.45 - i * 0.55
    _, border = ANSWER_COLORS.get(tier, COLORS.get(tier, COLORS["root"]))
    fill, _ = ANSWER_COLORS.get(tier, COLORS.get(tier, COLORS["root"]))

    swatch = FancyBboxPatch(
        (legend_x, ly - 0.15), 0.5, 0.3,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=fill,
        edgecolor=border,
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(swatch)

    ax.text(legend_x + 0.75, ly, lbl,
            fontsize=10, color=TEXT_COLOR, va="center",
            fontfamily="sans-serif", zorder=4)

# ---------------------------------------------------------------------------
# Tips box at bottom-right
# ---------------------------------------------------------------------------
tips_x = 16.0
tips_y = row_y(9)
tips = [
    "1. Identify the problem category first",
    "2. Follow the decision branches",
    "3. Many problems combine patterns",
    "4. When stuck, try BFS/DFS + HashMap",
]

tips_bg = FancyBboxPatch(
    (tips_x - 0.4, tips_y - 1.5),
    5.2, 2.6,
    boxstyle="round,pad=0.2,rounding_size=0.3",
    facecolor="#313244",
    edgecolor="#585b70",
    linewidth=1.5,
    zorder=3,
)
ax.add_patch(tips_bg)

ax.text(tips_x + 2.2, tips_y + 0.75, "Quick Tips",
        fontsize=12, fontweight="bold", color="#f9e2af",
        ha="center", va="center", fontfamily="sans-serif", zorder=4)

for i, tip in enumerate(tips):
    ax.text(tips_x, tips_y + 0.3 - i * 0.45, tip,
            fontsize=9.5, color=LABEL_COLOR, va="center",
            fontfamily="sans-serif", zorder=4)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_path = "/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/crashcourse/public/viz/pattern_flowchart.png"
plt.tight_layout(pad=0.5)
plt.savefig(output_path, facecolor=BG_COLOR, edgecolor="none",
            bbox_inches="tight", dpi=150)
plt.close()
print(f"Flowchart saved to: {output_path}")
