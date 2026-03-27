#!/usr/bin/env python3
"""
Generate two visualizations:
  1. lca_visualization.png     – Lowest Common Ancestor on a binary tree
  2. graph_representations.png – Same graph shown as visual, adjacency list, adjacency matrix
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
CARD_BG  = "#313244"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "viz")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# 1.  LCA VISUALIZATION
# =====================================================================

# Tree structure: val -> (left_val, right_val), None means no child
TREE = {
    3: (5, 1),
    5: (6, 2),
    1: (0, 8),
    2: (7, 4),
    6: (None, None),
    7: (None, None),
    4: (None, None),
    0: (None, None),
    8: (None, None),
}

# Pre-computed positions for nice layout  (x, y)
# The tree:
#              3
#            /   \
#           5     1
#          / \   / \
#         6   2 0   8
#            / \
#           7   4

NODE_POS = {
    3: (0.0, 0.0),
    5: (-2.0, -1.5),
    1: (2.0, -1.5),
    6: (-3.0, -3.0),
    2: (-1.0, -3.0),
    0: (1.0, -3.0),
    8: (3.0, -3.0),
    7: (-1.6, -4.5),
    4: (-0.4, -4.5),
}


def _draw_tree(ax, highlight_nodes=None, highlight_edges=None, lca_node=None,
               path_nodes_left=None, path_nodes_right=None):
    """Draw the binary tree on the given axes with optional highlights."""
    if highlight_nodes is None:
        highlight_nodes = set()
    if highlight_edges is None:
        highlight_edges = set()
    if path_nodes_left is None:
        path_nodes_left = set()
    if path_nodes_right is None:
        path_nodes_right = set()

    radius = 0.38

    # Draw edges first
    for parent, (left, right) in TREE.items():
        px, py = NODE_POS[parent]
        for child in [left, right]:
            if child is not None:
                cx, cy = NODE_POS[child]
                edge_key = (parent, child)
                if edge_key in highlight_edges:
                    ec = YELLOW
                    lw = 3.5
                    alpha = 1.0
                else:
                    ec = EDGE_COL
                    lw = 2.0
                    alpha = 0.5
                ax.plot([px, cx], [py, cy], color=ec, lw=lw, alpha=alpha, zorder=1)

    # Draw nodes
    for val, (nx, ny) in NODE_POS.items():
        if val == lca_node:
            fc = GREEN
            ec = "#40d67a"
            tc = BG
            lw = 3.5
        elif val in path_nodes_left:
            fc = YELLOW
            ec = PEACH
            tc = BG
            lw = 3.0
        elif val in path_nodes_right:
            fc = PINK
            ec = RED
            tc = BG
            lw = 3.0
        elif val in highlight_nodes:
            fc = ACCENT
            ec = LAVENDER
            tc = BG
            lw = 3.0
        else:
            fc = NODE_DEFAULT
            ec = EDGE_COL
            tc = TEXT
            lw = 2.0

        circle = plt.Circle((nx, ny), radius, fc=fc, ec=ec, lw=lw, zorder=5)
        ax.add_patch(circle)
        ax.text(nx, ny, str(val), fontsize=16, fontweight="bold",
                color=tc, ha="center", va="center",
                fontfamily="monospace", zorder=6)


def _get_path_to_root(target, current=3, path=None):
    """Get path from root down to target node. Returns list of node values."""
    if path is None:
        path = []
    path.append(current)
    if current == target:
        return path
    left, right = TREE.get(current, (None, None))
    if left is not None:
        result = _get_path_to_root(target, left, path[:])
        if result is not None:
            return result
    if right is not None:
        result = _get_path_to_root(target, right, path[:])
        if result is not None:
            return result
    return None


def _get_edges_on_path(path):
    """Convert a node path to a set of edge tuples."""
    edges = set()
    for i in range(len(path) - 1):
        edges.add((path[i], path[i + 1]))
    return edges


def draw_lca():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150, facecolor=BG)

    fig.suptitle("Lowest Common Ancestor \u2014 The Key Insight",
                 fontsize=24, fontweight="bold", color=TEXT, y=0.96)

    # ── Panel 1: LCA(5, 1) = 3 ─────────────────────────────────────
    ax1.set_facecolor(BG)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("LCA(5, 1) = 3", fontsize=18, fontweight="bold",
                   color=GREEN, pad=16)

    path_5 = _get_path_to_root(5)   # [3, 5]
    path_1 = _get_path_to_root(1)   # [3, 1]
    edges_5 = _get_edges_on_path(path_5)
    edges_1 = _get_edges_on_path(path_1)

    _draw_tree(ax1,
               highlight_edges=edges_5 | edges_1,
               lca_node=3,
               path_nodes_left={5},
               path_nodes_right={1})

    # Annotation arrows
    ax1.annotate("Both paths\nmeet here!",
                 xy=(0.0, 0.0), xytext=(2.5, 1.2),
                 fontsize=11, fontweight="bold", color=GREEN,
                 fontfamily="monospace",
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2),
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.4", fc=CARD_BG, ec=GREEN, lw=1.5))

    # Set limits
    ax1.set_xlim(-4.0, 4.5)
    ax1.set_ylim(-5.5, 2.0)

    # ── Panel 2: LCA(5, 4) = 5 ─────────────────────────────────────
    ax2.set_facecolor(BG)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("LCA(5, 4) = 5", fontsize=18, fontweight="bold",
                   color=GREEN, pad=16)

    path_5_to_4 = _get_path_to_root(4)  # [3, 5, 2, 4]
    edges_5_4 = _get_edges_on_path(path_5_to_4)
    # path from root to 5
    path_to_5 = _get_path_to_root(5)    # [3, 5]
    edges_to_5 = _get_edges_on_path(path_to_5)

    _draw_tree(ax2,
               highlight_edges=edges_5_4,
               lca_node=5,
               path_nodes_left=set(),
               path_nodes_right={4, 2})

    # Annotation
    ax2.annotate("5 is ancestor\nof 4 \u2014 LCA is 5!",
                 xy=(-2.0, -1.5), xytext=(-4.0, 0.6),
                 fontsize=11, fontweight="bold", color=GREEN,
                 fontfamily="monospace",
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2),
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.4", fc=CARD_BG, ec=GREEN, lw=1.5))

    ax2.annotate("4 is below 5\nin the subtree",
                 xy=(-0.4, -4.5), xytext=(2.2, -4.5),
                 fontsize=11, fontweight="bold", color=PINK,
                 fontfamily="monospace",
                 arrowprops=dict(arrowstyle="-|>", color=PINK, lw=2),
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.4", fc=CARD_BG, ec=PINK, lw=1.5))

    ax2.set_xlim(-5.0, 4.5)
    ax2.set_ylim(-5.5, 2.0)

    # ── Bottom text: recursive logic ────────────────────────────────
    fig.text(0.5, 0.03,
             'Recursive logic:  At each node, search left and right.\n'
             'If both return non-null, current node is LCA.  '
             'If only one side returns non-null, propagate it up.',
             fontsize=12, color=LAVENDER, ha="center", va="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc=CARD_BG, ec=GRAY, lw=1.5))

    # ── Legend ───────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(fc=GREEN, ec="#40d67a", label="LCA node"),
        mpatches.Patch(fc=YELLOW, ec=PEACH, label="Query node (left path)"),
        mpatches.Patch(fc=PINK, ec=RED, label="Query node (right path)"),
        mpatches.Patch(fc=NODE_DEFAULT, ec=EDGE_COL, label="Other node"),
    ]
    fig.legend(handles=legend_items, loc="lower right", fontsize=10,
               facecolor=CARD_BG, edgecolor=GRAY, labelcolor=TEXT,
               framealpha=0.95, bbox_to_anchor=(0.98, 0.09))

    plt.tight_layout(rect=[0, 0.10, 1, 0.93])

    path = os.path.join(OUT_DIR, "lca_visualization.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved  {path}")


# =====================================================================
# 2.  GRAPH REPRESENTATIONS
# =====================================================================

# Graph: A--B, A--C, B--D, C--D, D--E  (undirected)
GRAPH_NODES = ["A", "B", "C", "D", "E"]
GRAPH_EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]

ADJ_LIST = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "D"],
    "D": ["B", "C", "E"],
    "E": ["D"],
}

# Node positions for visual graph
GRAPH_POS = {
    "A": (0.0, 1.0),
    "B": (-1.2, 0.0),
    "C": (1.2, 0.0),
    "D": (0.0, -1.0),
    "E": (0.0, -2.2),
}

NODE_COLORS = {
    "A": ACCENT,
    "B": GREEN,
    "C": YELLOW,
    "D": PINK,
    "E": PEACH,
}


def draw_graph_representations():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7), dpi=150, facecolor=BG,
                                         gridspec_kw={"width_ratios": [1, 1, 1.2]})

    fig.suptitle("Graph Representations \u2014 Know All Three",
                 fontsize=24, fontweight="bold", color=TEXT, y=0.96)

    # ── Panel 1: Visual Graph ───────────────────────────────────────
    ax1.set_facecolor(BG)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Visual Graph", fontsize=16, fontweight="bold", color=ACCENT, pad=14)

    # Draw edges
    for u, v in GRAPH_EDGES:
        ux, uy = GRAPH_POS[u]
        vx, vy = GRAPH_POS[v]
        ax1.plot([ux, vx], [uy, vy], color=EDGE_COL, lw=2.5, zorder=1)

    # Draw nodes
    radius = 0.3
    for node in GRAPH_NODES:
        nx, ny = GRAPH_POS[node]
        fc = NODE_COLORS[node]
        circle = plt.Circle((nx, ny), radius, fc=fc, ec=TEXT, lw=2.5, zorder=5)
        ax1.add_patch(circle)
        ax1.text(nx, ny, node, fontsize=18, fontweight="bold",
                 color=BG, ha="center", va="center",
                 fontfamily="monospace", zorder=6)

    # Edge labels
    for u, v in GRAPH_EDGES:
        ux, uy = GRAPH_POS[u]
        vx, vy = GRAPH_POS[v]
        mx, my = (ux + vx) / 2, (uy + vy) / 2
        # slight offset perpendicular to edge
        dx, dy = vx - ux, vy - uy
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            ox, oy = -dy / length * 0.18, dx / length * 0.18
        else:
            ox, oy = 0, 0
        ax1.text(mx + ox, my + oy, f"{u}-{v}", fontsize=8, color=GRAY,
                 ha="center", va="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.1", fc=BG, ec="none", alpha=0.8))

    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-3.0, 2.0)

    # Subtitle for Panel 1
    ax1.text(0.0, -2.9, "5 nodes, 5 edges\nUndirected", fontsize=10, color=LAVENDER,
             ha="center", va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", fc=CARD_BG, ec=GRAY, lw=1))

    # ── Panel 2: Adjacency List ─────────────────────────────────────
    ax2.set_facecolor(BG)
    ax2.axis("off")
    ax2.set_title("Adjacency List", fontsize=16, fontweight="bold", color=GREEN, pad=14)

    # Draw as a styled dict representation
    y_start = 0.85
    line_spacing = 0.16
    x_key = 0.08
    x_arrow = 0.22
    x_vals = 0.30

    # Header
    ax2.text(0.5, 0.98, "adjacency_list = {", fontsize=13, fontweight="bold",
             color=TEXT, ha="center", va="top", fontfamily="monospace",
             transform=ax2.transAxes)

    for i, node in enumerate(GRAPH_NODES):
        y = y_start - i * line_spacing
        neighbors = ADJ_LIST[node]

        # Key (node label) with its color
        ax2.text(x_key, y, f'  "{node}"', fontsize=14, fontweight="bold",
                 color=NODE_COLORS[node], ha="left", va="center",
                 fontfamily="monospace", transform=ax2.transAxes)

        # Arrow
        ax2.text(x_arrow, y, ":", fontsize=14, color=GRAY,
                 ha="left", va="center", fontfamily="monospace",
                 transform=ax2.transAxes)

        # Values
        neighbor_str = "[" + ", ".join(f'"{n}"' for n in neighbors) + "]"

        # Color each neighbor reference
        ax2.text(x_vals, y, neighbor_str, fontsize=13, fontweight="bold",
                 color=LAVENDER, ha="left", va="center",
                 fontfamily="monospace", transform=ax2.transAxes)

    # Closing brace
    y_close = y_start - len(GRAPH_NODES) * line_spacing
    ax2.text(0.5, y_close, "}", fontsize=13, fontweight="bold",
             color=TEXT, ha="center", va="center", fontfamily="monospace",
             transform=ax2.transAxes)

    # Complexity note
    ax2.text(0.5, 0.08, "Space: O(V + E)\nDefault for interviews!",
             fontsize=11, fontweight="bold", color=GREEN, ha="center", va="center",
             fontfamily="monospace", transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", fc=CARD_BG, ec=GREEN, lw=1.5))

    # ── Panel 3: Adjacency Matrix ───────────────────────────────────
    ax3.set_facecolor(BG)
    ax3.axis("off")
    ax3.set_title("Adjacency Matrix", fontsize=16, fontweight="bold", color=YELLOW, pad=14)

    n = len(GRAPH_NODES)
    # Build the matrix
    matrix = np.zeros((n, n), dtype=int)
    node_idx = {node: i for i, node in enumerate(GRAPH_NODES)}
    for u, v in GRAPH_EDGES:
        matrix[node_idx[u], node_idx[v]] = 1
        matrix[node_idx[v], node_idx[u]] = 1

    cell_size = 0.12
    x_origin = 0.30
    y_origin = 0.80

    # Column headers
    for j, node in enumerate(GRAPH_NODES):
        ax3.text(x_origin + (j + 0.5) * cell_size, y_origin + 0.06, node,
                 fontsize=13, fontweight="bold", color=NODE_COLORS[node],
                 ha="center", va="center", fontfamily="monospace",
                 transform=ax3.transAxes)

    # Rows
    for i, row_node in enumerate(GRAPH_NODES):
        # Row header
        y = y_origin - (i + 0.5) * cell_size
        ax3.text(x_origin - 0.04, y, row_node, fontsize=13, fontweight="bold",
                 color=NODE_COLORS[row_node], ha="center", va="center",
                 fontfamily="monospace", transform=ax3.transAxes)

        for j, col_node in enumerate(GRAPH_NODES):
            x = x_origin + j * cell_size
            val = matrix[i, j]

            if val == 1:
                fc = ACCENT
                tc = BG
                alpha = 0.85
            elif i == j:
                fc = "#2a2a3a"
                tc = GRAY
                alpha = 0.6
            else:
                fc = NODE_DEFAULT
                tc = GRAY
                alpha = 0.4

            rect = mpatches.FancyBboxPatch(
                (x, y - cell_size / 2), cell_size, cell_size,
                boxstyle="round,pad=0.005",
                fc=fc, ec=EDGE_COL, lw=1.0, alpha=alpha,
                transform=ax3.transAxes, zorder=3)
            ax3.add_patch(rect)

            ax3.text(x + cell_size / 2, y, str(val),
                     fontsize=12, fontweight="bold", color=tc,
                     ha="center", va="center", fontfamily="monospace",
                     transform=ax3.transAxes, zorder=4)

    # Complexity note
    ax3.text(0.5, 0.08, "Space: O(V\u00b2)\nBest for dense graphs",
             fontsize=11, fontweight="bold", color=YELLOW, ha="center", va="center",
             fontfamily="monospace", transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", fc=CARD_BG, ec=YELLOW, lw=1.5))

    # ── Bottom notes ────────────────────────────────────────────────
    fig.text(0.28, 0.02,
             "Interview default: Adjacency List  (O(V+E) space)",
             fontsize=11, fontweight="bold", color=GREEN, ha="center", va="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc=CARD_BG, ec=GREEN, lw=1.2))

    fig.text(0.72, 0.02,
             "Matrix: O(V\u00b2) space, useful for dense graphs & quick edge lookup",
             fontsize=11, fontweight="bold", color=YELLOW, ha="center", va="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc=CARD_BG, ec=YELLOW, lw=1.2))

    plt.tight_layout(rect=[0, 0.07, 1, 0.93])

    path = os.path.join(OUT_DIR, "graph_representations.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved  {path}")


# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    draw_lca()
    draw_graph_representations()
    print("\nDone — both visualizations generated.")
