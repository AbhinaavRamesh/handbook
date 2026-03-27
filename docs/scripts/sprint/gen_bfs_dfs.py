"""
Generate a static BFS vs DFS grid traversal comparison visualization.

Output: ../public/viz/bfs_vs_dfs.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Grid dimensions & traversal helpers
# ---------------------------------------------------------------------------
ROWS, COLS = 5, 5


def bfs_order(rows: int, cols: int) -> list[tuple[int, int]]:
    """Return cells in BFS visit order starting from (0, 0)."""
    visited = [[False] * cols for _ in range(rows)]
    queue = deque([(0, 0)])
    visited[0][0] = True
    order = []
    while queue:
        r, c = queue.popleft()
        order.append((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                visited[nr][nc] = True
                queue.append((nr, nc))
    return order


def dfs_order(rows: int, cols: int) -> list[tuple[int, int]]:
    """Return cells in DFS visit order starting from (0, 0).
    Uses iterative DFS with right-down-left-up priority so the path
    visually dives deep before backtracking."""
    visited = [[False] * cols for _ in range(rows)]
    stack = [(0, 0)]
    order = []
    while stack:
        r, c = stack.pop()
        if visited[r][c]:
            continue
        visited[r][c] = True
        order.append((r, c))
        # Push in reverse priority so preferred direction is popped first
        for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                stack.append((nr, nc))
    return order


bfs = bfs_order(ROWS, COLS)
dfs = dfs_order(ROWS, COLS)

# Build order lookup: (r,c) -> visit index
bfs_idx = {pos: i for i, pos in enumerate(bfs)}
dfs_idx = {pos: i for i, pos in enumerate(dfs)}

# ---------------------------------------------------------------------------
# 2. Colour maps
# ---------------------------------------------------------------------------
bfs_cmap = mcolors.LinearSegmentedColormap.from_list("bfs", ["#3b82f6", "#22d3ee", "#10b981"])
dfs_cmap = mcolors.LinearSegmentedColormap.from_list("dfs", ["#ef4444", "#f97316", "#facc15"])

# ---------------------------------------------------------------------------
# 3. Figure setup
# ---------------------------------------------------------------------------
BG = "#1e1e2e"
CELL = 1.0          # cell size in data units
PAD = 0.06          # gap between cells
CORNER_R = 0.12     # rounded-corner radius
FONT_CLR = "#ffffff"
GRID_LABEL_CLR = "#a0a0b8"

fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
fig.patch.set_facecolor(BG)

# Main title
fig.suptitle("BFS vs DFS \u2014 Grid Traversal Order",
             fontsize=22, fontweight="bold", color="#e2e2f0",
             y=0.96, fontfamily="monospace")

# ---------------------------------------------------------------------------
# 4. Drawing helper
# ---------------------------------------------------------------------------

def draw_grid(ax, order_map, order_list, cmap, title, subtitle, subtitle_color):
    """Draw a single traversal grid on the given axes."""
    ax.set_facecolor(BG)
    total = ROWS * COLS

    for r in range(ROWS):
        for c in range(COLS):
            idx = order_map[(r, c)]
            norm = idx / (total - 1)
            color = cmap(norm)

            # Cell rectangle (rounded)
            rect = mpatches.FancyBboxPatch(
                (c * (CELL + PAD) + PAD / 2, (ROWS - 1 - r) * (CELL + PAD) + PAD / 2),
                CELL - PAD, CELL - PAD,
                boxstyle=f"round,pad={CORNER_R}",
                facecolor=color,
                edgecolor="#ffffff22",
                linewidth=1.2,
            )
            ax.add_patch(rect)

            # Visit-order number
            cx = c * (CELL + PAD) + CELL / 2
            cy = (ROWS - 1 - r) * (CELL + PAD) + CELL / 2
            ax.text(cx, cy, str(idx + 1),
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="#000000" if norm > 0.6 else FONT_CLR,
                    fontfamily="monospace")

    # Draw arrows along the traversal path
    for i in range(len(order_list) - 1):
        r1, c1 = order_list[i]
        r2, c2 = order_list[i + 1]
        x1 = c1 * (CELL + PAD) + CELL / 2
        y1 = (ROWS - 1 - r1) * (CELL + PAD) + CELL / 2
        x2 = c2 * (CELL + PAD) + CELL / 2
        y2 = (ROWS - 1 - r2) * (CELL + PAD) + CELL / 2

        dx = x2 - x1
        dy = y2 - y1
        dist = np.hypot(dx, dy)

        # Only draw arrows for adjacent cells (skip long backtrack jumps for clarity)
        if dist < (CELL + PAD) * 1.5:
            # Shorten arrows so they don't overlap cell numbers
            shrink = 0.30
            sx = x1 + dx * shrink
            sy = y1 + dy * shrink
            sdx = dx * (1 - 2 * shrink)
            sdy = dy * (1 - 2 * shrink)

            ax.annotate("",
                        xy=(sx + sdx, sy + sdy),
                        xytext=(sx, sy),
                        arrowprops=dict(
                            arrowstyle="->,head_width=0.18,head_length=0.12",
                            color="#ffffff55",
                            lw=1.3,
                            connectionstyle="arc3,rad=0.0",
                        ))

    # Draw faint dashed arrows for backtrack jumps
    for i in range(len(order_list) - 1):
        r1, c1 = order_list[i]
        r2, c2 = order_list[i + 1]
        x1 = c1 * (CELL + PAD) + CELL / 2
        y1 = (ROWS - 1 - r1) * (CELL + PAD) + CELL / 2
        x2 = c2 * (CELL + PAD) + CELL / 2
        y2 = (ROWS - 1 - r2) * (CELL + PAD) + CELL / 2

        dx = x2 - x1
        dy = y2 - y1
        dist = np.hypot(dx, dy)

        if dist >= (CELL + PAD) * 1.5:
            ax.annotate("",
                        xy=(x2, y2),
                        xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="->,head_width=0.14,head_length=0.10",
                            color="#ffffff22",
                            lw=0.8,
                            linestyle="dashed",
                            connectionstyle="arc3,rad=0.15",
                        ))

    # Axis cosmetics
    grid_w = COLS * (CELL + PAD)
    grid_h = ROWS * (CELL + PAD)
    ax.set_xlim(-0.3, grid_w + 0.3)
    ax.set_ylim(-1.8, grid_h + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Panel title
    ax.text(grid_w / 2, grid_h + 0.45, title,
            ha="center", va="bottom",
            fontsize=16, fontweight="bold",
            color=subtitle_color, fontfamily="monospace")

    # "Use when" subtitle
    ax.text(grid_w / 2, -0.6, subtitle,
            ha="center", va="top",
            fontsize=10, color=GRID_LABEL_CLR,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a3e",
                      edgecolor="#3a3a50", linewidth=0.8))

    # Start indicator
    sx = 0 * (CELL + PAD) + CELL / 2
    sy = (ROWS - 1) * (CELL + PAD) + CELL + 0.15
    ax.annotate("START", xy=(sx, sy),
                fontsize=8, color="#888898", ha="center", va="bottom",
                fontweight="bold", fontfamily="monospace")

    # Colour-bar legend (gradient strip)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    bar_y = -1.35
    bar_h = 0.18
    bar_x0 = 0.3
    bar_x1 = grid_w - 0.3
    ax.imshow(gradient, aspect="auto", cmap=cmap,
              extent=[bar_x0, bar_x1, bar_y, bar_y + bar_h])
    ax.text(bar_x0, bar_y - 0.12, "Early", fontsize=8,
            color=GRID_LABEL_CLR, ha="left", va="top", fontfamily="monospace")
    ax.text(bar_x1, bar_y - 0.12, "Late", fontsize=8,
            color=GRID_LABEL_CLR, ha="right", va="top", fontfamily="monospace")


# ---------------------------------------------------------------------------
# 5. Render both panels
# ---------------------------------------------------------------------------
draw_grid(axes[0], bfs_idx, bfs, bfs_cmap,
          "BFS (Breadth-First)",
          "Use when:  shortest path  |  level-order",
          "#3b82f6")

draw_grid(axes[1], dfs_idx, dfs, dfs_cmap,
          "DFS (Depth-First)",
          "Use when:  path finding  |  exhaustive search  |  topological sort",
          "#ef4444")

plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.93])

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
out = Path(__file__).resolve().parent.parent / "public" / "viz" / "bfs_vs_dfs.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.4)
plt.close(fig)
print(f"Saved -> {out}")
