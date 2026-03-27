"""
Generate Tree Traversal and Min-Heap Operations visualizations.

Produces two publication-quality images styled with a dark theme:
  1. tree_traversals.png  -- Preorder, Inorder, Postorder side-by-side
  2. heap_operations.png  -- Min-heap tree + array + push/pop operations

Output directory: ../public/viz/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR.parent / "public" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

TREE_OUT = VIZ_DIR / "tree_traversals.png"
HEAP_OUT = VIZ_DIR / "heap_operations.png"

# ---------------------------------------------------------------------------
# Theme / colours  (matching existing dark theme)
# ---------------------------------------------------------------------------
BG_COLOR = "#1e1e2e"
SURFACE_COLOR = "#2a2a3d"
TEXT_COLOR = "#cdd6f4"
GRID_COLOR = "#45475a"
MUTED_TEXT = "#a6adc8"

# Accent palette
GREEN = "#34a853"
BLUE = "#4285f4"
RED = "#ea4335"
YELLOW = "#fbbc04"
ORANGE = "#ff6d00"
PURPLE = "#d500f9"
TEAL = "#00bcd4"
PINK = "#f06292"

NODE_EDGE_COLOR = "#585b70"
ARROW_COLOR = "#7f849c"

# ---------------------------------------------------------------------------
# Helper: draw a binary tree
# ---------------------------------------------------------------------------

def _tree_positions():
    """Return {value: (x, y)} positions for the tree:
              1
            /   \\
           2     3
          / \\   / \\
         4   5 6   7
    Coordinate space: x in [0, 6], y in [0, 2].
    """
    return {
        1: (3.0, 2.0),
        2: (1.5, 1.0),
        3: (4.5, 1.0),
        4: (0.75, 0.0),
        5: (2.25, 0.0),
        6: (3.75, 0.0),
        7: (5.25, 0.0),
    }


EDGES = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]


def draw_tree(ax, positions, visit_order, order_colors, title, subtitle,
              node_radius=0.28, show_arrows=True):
    """Draw a single tree panel with nodes coloured by visit order."""
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.9, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(3.0, 2.75, title, fontsize=15, fontweight="bold",
            color=TEXT_COLOR, ha="center", va="bottom", fontfamily="sans-serif")

    # Draw edges
    for parent, child in EDGES:
        px, py = positions[parent]
        cx, cy = positions[child]
        ax.plot([px, cx], [py, cy], color=NODE_EDGE_COLOR, linewidth=2,
                zorder=1)

    # Draw visit-order arrows
    if show_arrows:
        for i in range(len(visit_order) - 1):
            src = visit_order[i]
            dst = visit_order[i + 1]
            sx, sy = positions[src]
            dx, dy = positions[dst]
            ax.annotate(
                "", xy=(dx, dy), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.18,head_length=0.12",
                    color=order_colors[i + 1],
                    lw=1.5,
                    connectionstyle="arc3,rad=0.25",
                    shrinkA=node_radius * 72 * 0.38,
                    shrinkB=node_radius * 72 * 0.38,
                ),
                zorder=2,
            )

    # Draw nodes
    for idx, val in enumerate(visit_order):
        x, y = positions[val]
        color = order_colors[idx]
        circle = plt.Circle((x, y), node_radius, facecolor=color,
                             edgecolor="#cdd6f4", linewidth=1.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(val), fontsize=14, fontweight="bold",
                color="#1e1e2e" if color in [YELLOW, GREEN, TEAL] else "#ffffff",
                ha="center", va="center", zorder=4, fontfamily="sans-serif")

        # Visit-order number badge (small, top-right)
        bx = x + node_radius * 0.7
        by = y + node_radius * 0.7
        badge = plt.Circle((bx, by), 0.14, facecolor=BG_COLOR,
                            edgecolor=color, linewidth=1.2, zorder=5)
        ax.add_patch(badge)
        ax.text(bx, by, str(idx + 1), fontsize=7, fontweight="bold",
                color=color, ha="center", va="center", zorder=6,
                fontfamily="sans-serif")

    # Sequence text
    seq_str = " -> ".join(str(v) for v in visit_order)
    ax.text(3.0, -0.45, seq_str, fontsize=10, fontweight="bold",
            color=TEXT_COLOR, ha="center", va="top", fontfamily="monospace")

    # Subtitle (use cases)
    ax.text(3.0, -0.72, subtitle, fontsize=8.5,
            color=MUTED_TEXT, ha="center", va="top", fontfamily="sans-serif",
            style="italic")


# ===================================================================
# IMAGE 1 : Tree Traversals
# ===================================================================

def generate_tree_traversals():
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(BG_COLOR)

    fig.suptitle("Binary Tree Traversals", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, fontfamily="sans-serif", y=0.97)

    pos = _tree_positions()

    # --- Color gradients for visit order (7 steps) ---
    preorder_colors = [RED, ORANGE, YELLOW, GREEN, TEAL, BLUE, PURPLE]
    inorder_colors = [RED, ORANGE, YELLOW, GREEN, TEAL, BLUE, PURPLE]
    postorder_colors = [RED, ORANGE, YELLOW, GREEN, TEAL, BLUE, PURPLE]

    # Preorder: root, left, right
    draw_tree(axes[0], pos,
              visit_order=[1, 2, 4, 5, 3, 6, 7],
              order_colors=preorder_colors,
              title="Preorder  (Root, L, R)",
              subtitle="Use: serialize tree, copy tree,\nprefix expression")

    # Inorder: left, root, right
    draw_tree(axes[1], pos,
              visit_order=[4, 2, 5, 1, 6, 3, 7],
              order_colors=inorder_colors,
              title="Inorder  (L, Root, R)",
              subtitle="Use: BST sorted order,\nvalidate BST, k-th smallest")

    # Postorder: left, right, root
    draw_tree(axes[2], pos,
              visit_order=[4, 5, 2, 6, 7, 3, 1],
              order_colors=postorder_colors,
              title="Postorder  (L, R, Root)",
              subtitle="Use: delete tree, calc size,\nbottom-up DP, postfix expr")

    # Legend for visit order
    fig.text(0.5, 0.05,
             "Numbers in badges show visit order (1 = first visited).  "
             "Arrows trace the traversal path.",
             fontsize=10, color=MUTED_TEXT, ha="center", fontfamily="sans-serif")

    # Tree structure reminder
    fig.text(0.5, 0.02,
             "Tree:  1 -> [2, 3]   2 -> [4, 5]   3 -> [6, 7]",
             fontsize=9, color=GRID_COLOR, ha="center", fontfamily="monospace")

    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(TREE_OUT, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {TREE_OUT}")


# ===================================================================
# IMAGE 2 : Min-Heap Operations
# ===================================================================

def _heap_tree_positions(n):
    """Return list of (x, y) for a complete binary tree of n nodes.
    Level-order indexed. Root at top-center."""
    positions = []
    import math
    depth = math.floor(math.log2(n)) if n > 0 else 0
    for i in range(n):
        level = math.floor(math.log2(i + 1))
        pos_in_level = i - (2 ** level - 1)
        num_in_level = 2 ** level
        total_width = 2 ** depth
        x = (pos_in_level + 0.5) * (total_width / num_in_level)
        y = depth - level
        positions.append((x, y))
    return positions


def draw_heap_tree(ax, heap, title="Min-Heap Tree", highlight_indices=None,
                   highlight_color=None, node_radius=0.32):
    """Draw a heap as a tree. highlight_indices get a special ring color."""
    if highlight_indices is None:
        highlight_indices = set()
    if highlight_color is None:
        highlight_color = YELLOW

    n = len(heap)
    pos = _heap_tree_positions(n)

    max_x = max(p[0] for p in pos) if pos else 1
    max_y = max(p[1] for p in pos) if pos else 1
    ax.set_xlim(-0.8, max_x + 0.8)
    ax.set_ylim(-0.8, max_y + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text((max_x) / 2, max_y + 0.7, title, fontsize=14, fontweight="bold",
            color=TEXT_COLOR, ha="center", va="bottom", fontfamily="sans-serif")

    # Edges
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        for child in [left, right]:
            if child < n:
                px, py = pos[i]
                cx, cy = pos[child]
                ax.plot([px, cx], [py, cy], color=NODE_EDGE_COLOR,
                        linewidth=1.8, zorder=1)

    # Nodes
    for i in range(n):
        x, y = pos[i]
        is_hl = i in highlight_indices
        fc = BLUE if not is_hl else highlight_color
        ec = "#cdd6f4" if not is_hl else highlight_color
        lw = 1.8 if not is_hl else 2.8
        circle = plt.Circle((x, y), node_radius, facecolor=fc,
                             edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(circle)
        text_color = "#ffffff" if not is_hl else "#1e1e2e"
        ax.text(x, y, str(heap[i]), fontsize=13, fontweight="bold",
                color=text_color, ha="center", va="center", zorder=4,
                fontfamily="sans-serif")

        # Parent <= children label for root
        if i == 0 and n > 1:
            ax.text(x, y + node_radius + 0.15, "min", fontsize=8,
                    color=GREEN, ha="center", va="bottom",
                    fontfamily="sans-serif", fontweight="bold")

    # Show parent <= children property
    if n >= 3:
        rx, ry = pos[0]
        ax.annotate(
            f"{heap[0]} <= {heap[1]}",
            xy=(pos[1][0], pos[1][1] + node_radius),
            xytext=(pos[1][0] - 0.8, pos[1][1] + 0.6),
            fontsize=7.5, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0),
            fontfamily="monospace", fontweight="bold",
            zorder=5,
        )
        ax.annotate(
            f"{heap[0]} <= {heap[2]}",
            xy=(pos[2][0], pos[2][1] + node_radius),
            xytext=(pos[2][0] + 0.8, pos[2][1] + 0.6),
            fontsize=7.5, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0),
            fontfamily="monospace", fontweight="bold",
            zorder=5,
        )


def draw_heap_array(ax, heap, title="Array Representation", highlight_indices=None,
                    highlight_color=None):
    """Draw the heap as an array with index labels and parent/child pointers."""
    if highlight_indices is None:
        highlight_indices = set()
    if highlight_color is None:
        highlight_color = YELLOW

    n = len(heap)
    cell_w = 1.0
    cell_h = 0.7
    start_x = 0.5
    start_y = 0.5

    total_w = n * cell_w
    ax.set_xlim(-0.5, max(total_w + 2.5, 10))
    ax.set_ylim(-1.0, 2.5)
    ax.axis("off")

    ax.text(start_x + total_w / 2, 2.1, title, fontsize=13, fontweight="bold",
            color=TEXT_COLOR, ha="center", fontfamily="sans-serif")

    for i in range(n):
        x = start_x + i * cell_w
        is_hl = i in highlight_indices
        fc = SURFACE_COLOR if not is_hl else highlight_color
        ec = GRID_COLOR if not is_hl else highlight_color

        rect = FancyBboxPatch(
            (x, start_y), cell_w, cell_h,
            boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor=ec, linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(rect)

        text_color = TEXT_COLOR if not is_hl else "#1e1e2e"
        ax.text(x + cell_w / 2, start_y + cell_h / 2, str(heap[i]),
                fontsize=13, fontweight="bold", color=text_color,
                ha="center", va="center", zorder=3, fontfamily="sans-serif")

        # Index label below
        ax.text(x + cell_w / 2, start_y - 0.15, f"[{i}]",
                fontsize=8, color=MUTED_TEXT, ha="center", va="top",
                fontfamily="monospace")

    # Show parent-child index formulas
    formula_y = -0.55
    ax.text(start_x + total_w / 2, formula_y,
            "parent(i) = (i-1)//2      left(i) = 2i+1      right(i) = 2i+2",
            fontsize=9.5, color=TEAL, ha="center", fontfamily="monospace",
            fontweight="bold")

    # Draw curved arrows showing i=0 -> children at 1, 2
    # Arrow from index 0 to index 1
    ax.annotate(
        "", xy=(start_x + 1.5 * cell_w, start_y + cell_h + 0.02),
        xytext=(start_x + 0.5 * cell_w, start_y + cell_h + 0.02),
        arrowprops=dict(arrowstyle="->,head_width=0.15", color=GREEN,
                        lw=1.3, connectionstyle="arc3,rad=-0.4"),
        zorder=5,
    )
    ax.text(start_x + 1.0 * cell_w, start_y + cell_h + 0.35, "2(0)+1",
            fontsize=7, color=GREEN, ha="center", fontfamily="monospace")

    # Arrow from index 0 to index 2
    ax.annotate(
        "", xy=(start_x + 2.5 * cell_w, start_y + cell_h + 0.02),
        xytext=(start_x + 0.5 * cell_w, start_y + cell_h + 0.02),
        arrowprops=dict(arrowstyle="->,head_width=0.15", color=ORANGE,
                        lw=1.3, connectionstyle="arc3,rad=-0.5"),
        zorder=5,
    )
    ax.text(start_x + 1.5 * cell_w, start_y + cell_h + 0.65, "2(0)+2",
            fontsize=7, color=ORANGE, ha="center", fontfamily="monospace")


def draw_operation_panel(ax, before_heap, after_heap, op_name, op_detail,
                         highlight_before=None, highlight_after=None):
    """Draw a before/after pair for a heap operation."""
    ax.set_xlim(-0.3, 10)
    ax.set_ylim(-0.5, 5.2)
    ax.axis("off")

    ax.text(5.0, 5.0, op_name, fontsize=14, fontweight="bold",
            color=YELLOW, ha="center", fontfamily="sans-serif")
    ax.text(5.0, 4.6, op_detail, fontsize=9, color=MUTED_TEXT,
            ha="center", fontfamily="monospace")

    if highlight_before is None:
        highlight_before = set()
    if highlight_after is None:
        highlight_after = set()

    # Before array
    cell_w = 0.85
    cell_h = 0.55

    def draw_mini_array(y_base, heap, label, hl_set, hl_color):
        ax.text(0.2, y_base + cell_h / 2, label, fontsize=9, fontweight="bold",
                color=MUTED_TEXT, ha="right", va="center", fontfamily="sans-serif")
        for i, val in enumerate(heap):
            x = 0.5 + i * cell_w
            is_hl = i in hl_set
            fc = SURFACE_COLOR if not is_hl else hl_color
            ec = GRID_COLOR if not is_hl else hl_color
            rect = FancyBboxPatch(
                (x, y_base), cell_w, cell_h,
                boxstyle="round,pad=0.02",
                facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2,
            )
            ax.add_patch(rect)
            tc = TEXT_COLOR if not is_hl else "#1e1e2e"
            ax.text(x + cell_w / 2, y_base + cell_h / 2, str(val),
                    fontsize=11, fontweight="bold", color=tc,
                    ha="center", va="center", zorder=3, fontfamily="sans-serif")

    draw_mini_array(3.0, before_heap, "Before:", highlight_before, RED)

    # Arrow between before and after
    ax.annotate(
        "", xy=(4.5, 2.3), xytext=(4.5, 2.85),
        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.15",
                        color=YELLOW, lw=2.0),
    )
    ax.text(5.5, 2.55, "sift", fontsize=8, color=YELLOW,
            ha="center", va="center", fontfamily="sans-serif", fontstyle="italic")

    draw_mini_array(1.5, after_heap, "After:", highlight_after, GREEN)

    # Show the heap property check
    if len(after_heap) >= 3:
        ax.text(5.0, 0.9,
                f"Heap OK: {after_heap[0]} <= {after_heap[1]}, {after_heap[0]} <= {after_heap[2]}",
                fontsize=8, color=GREEN, ha="center", fontfamily="monospace",
                fontweight="bold")


def generate_heap_operations():
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    fig.suptitle("Min-Heap: Structure & Operations", fontsize=22,
                 fontweight="bold", color=TEXT_COLOR, fontfamily="sans-serif",
                 y=0.98)

    # Layout:  top-left = tree, top-right = push op
    #          bottom-left = array, bottom-right = pop op

    # -- Top-left: heap tree --
    ax_tree = fig.add_axes([0.02, 0.45, 0.50, 0.48])
    ax_tree.set_facecolor(BG_COLOR)
    heap = [1, 3, 5, 7, 9, 8, 6]
    draw_heap_tree(ax_tree, heap, title="Min-Heap: [1, 3, 5, 7, 9, 8, 6]",
                   highlight_indices={0}, highlight_color=GREEN)

    # -- Bottom-left: array representation --
    ax_array = fig.add_axes([0.02, 0.15, 0.50, 0.28])
    ax_array.set_facecolor(BG_COLOR)
    draw_heap_array(ax_array, heap, title="Array Representation")

    # -- Top-right: heappush --
    ax_push = fig.add_axes([0.54, 0.50, 0.44, 0.42])
    ax_push.set_facecolor(BG_COLOR)
    draw_operation_panel(
        ax_push,
        before_heap=[1, 3, 5, 7, 9, 8, 6, 2],
        after_heap=[1, 2, 5, 3, 9, 8, 6, 7],
        op_name="heapq.heappush(h, 2)",
        op_detail="append 2, then sift UP to restore order",
        highlight_before={7},       # new element at end
        highlight_after={1, 3, 7},  # swapped positions
    )

    # -- Bottom-right: heappop --
    ax_pop = fig.add_axes([0.54, 0.08, 0.44, 0.42])
    ax_pop.set_facecolor(BG_COLOR)
    draw_operation_panel(
        ax_pop,
        before_heap=[1, 3, 5, 7, 9, 8, 6],
        after_heap=[3, 6, 5, 7, 9, 8],
        op_name="heapq.heappop(h)  -> 1",
        op_detail="remove root, move last to root, sift DOWN",
        highlight_before={0},       # root being removed
        highlight_after={0, 1},     # new root position
    )

    # Python heapq reminder at bottom
    reminder_box = FancyBboxPatch(
        (0.15, 0.005), 0.70, 0.05,
        boxstyle="round,pad=0.01",
        transform=fig.transFigure,
        facecolor=SURFACE_COLOR,
        edgecolor=YELLOW,
        linewidth=1.5,
        zorder=10,
    )
    fig.patches.append(reminder_box)
    fig.text(
        0.50, 0.03,
        "Python heapq is MIN-HEAP only!   For max-heap: negate values  "
        "(-val)  or use  heapq.nlargest()",
        fontsize=10, fontweight="bold", color=YELLOW,
        ha="center", va="center", fontfamily="sans-serif",
    )

    # Complexity note
    fig.text(
        0.50, 0.075,
        "push: O(log n)    pop: O(log n)    peek min: O(1)    "
        "heapify: O(n)",
        fontsize=9, color=TEAL, ha="center", fontfamily="monospace",
        fontweight="bold",
    )

    fig.savefig(HEAP_OUT, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {HEAP_OUT}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating tree traversals visualization...")
    generate_tree_traversals()

    print("Generating heap operations visualization...")
    generate_heap_operations()

    print("Done!")
