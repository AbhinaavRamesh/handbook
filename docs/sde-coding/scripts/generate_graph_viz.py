#!/usr/bin/env python3
"""Generate BFS/DFS graph traversal visualizations for the tutorial site."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import deque
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_bfs_traversal():
    """Generate BFS level-by-level traversal visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title('BFS: Level-Order Traversal (Wave Pattern)', fontsize=16, fontweight='bold')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    # Graph structure - tree-like for clarity
    positions = {
        0: (5, 7),    # Level 0
        1: (2, 5), 2: (8, 5),    # Level 1
        3: (0.5, 3), 4: (3.5, 3), 5: (6.5, 3), 6: (9.5, 3),  # Level 2
    }

    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

    # Colors by level
    level_colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB']
    node_levels = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2}

    # Draw edges
    for u, v in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

    # Draw nodes with level colors
    for node, (x, y) in positions.items():
        level = node_levels[node]
        color = level_colors[level]
        circle = plt.Circle((x, y), 0.5, color=color, ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha='center', va='center', fontsize=16, fontweight='bold', color='white')

    # Add level labels
    for level, (y_pos, label) in enumerate([(7, 'Level 0'), (5, 'Level 1'), (3, 'Level 2')]):
        ax.text(-0.8, y_pos, label, fontsize=11, fontweight='bold', va='center',
               color=level_colors[level])

    # BFS order
    ax.text(5, 1.5, 'BFS Visit Order: 0 → 1 → 2 → 3 → 4 → 5 → 6', ha='center', fontsize=13,
           bbox=dict(boxstyle='round', facecolor='#AED6F1', alpha=0.8))

    # Queue visualization
    ax.text(5, 0.5, 'Queue (FIFO): Processes nodes level by level', ha='center', fontsize=11, style='italic')

    # Legend
    for i, (level, color) in enumerate(zip(['Level 0', 'Level 1', 'Level 2'], level_colors[:3])):
        ax.add_patch(patches.Rectangle((8.5, 6.5 - i*0.6), 0.4, 0.4, facecolor=color, edgecolor='black'))
        ax.text(9.1, 6.7 - i*0.6, level, fontsize=10, va='center')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/bfs_traversal.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/bfs_traversal.png")


def generate_dfs_traversal():
    """Generate DFS depth-first traversal visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title('DFS: Depth-First Traversal (Deep Dive Pattern)', fontsize=16, fontweight='bold')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    # Same graph structure
    positions = {
        0: (5, 7),
        1: (2, 5), 2: (8, 5),
        3: (0.5, 3), 4: (3.5, 3), 5: (6.5, 3), 6: (9.5, 3),
    }

    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

    # DFS visit order
    dfs_order = [0, 1, 3, 4, 2, 5, 6]
    order_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(dfs_order)))

    # Draw edges
    for u, v in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

    # Draw nodes with visit order colors
    for order_idx, node in enumerate(dfs_order):
        x, y = positions[node]
        circle = plt.Circle((x, y), 0.5, color=order_colors[order_idx], ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y - 0.05, str(node), ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(x, y + 0.8, f'#{order_idx + 1}', ha='center', fontsize=10, color='#E74C3C', fontweight='bold')

    # Draw DFS path
    path = [(0, 1), (1, 3), (3, 1), (1, 4), (4, 1), (1, 0), (0, 2), (2, 5), (5, 2), (2, 6)]
    ax.text(5, 1.5, 'DFS Visit Order: 0 → 1 → 3 → 4 → 2 → 5 → 6', ha='center', fontsize=13,
           bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.8))

    ax.text(5, 0.5, 'Stack (LIFO): Goes deep, then backtracks', ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dfs_traversal.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/dfs_traversal.png")


def generate_bfs_vs_dfs():
    """Generate side-by-side BFS vs DFS comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('BFS vs DFS: Same Graph, Different Visit Orders', fontsize=16, fontweight='bold')

    positions = {
        0: (2, 3), 1: (1, 2), 2: (3, 2), 3: (0, 1), 4: (2, 1), 5: (4, 1)
    }
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5)]

    bfs_order = [0, 1, 2, 3, 4, 5]
    dfs_order = [0, 1, 3, 4, 2, 5]

    for ax_idx, (ax, title, order, color_base) in enumerate([
        (axes[0], 'BFS (Level-Order)', bfs_order, plt.cm.Blues),
        (axes[1], 'DFS (Depth-First)', dfs_order, plt.cm.Oranges)
    ]):
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0, 4)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Draw edges
        for u, v in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

        # Draw nodes
        colors = color_base(np.linspace(0.3, 0.9, len(order)))
        order_map = {node: idx for idx, node in enumerate(order)}

        for node, (x, y) in positions.items():
            idx = order_map[node]
            circle = plt.Circle((x, y), 0.35, color=colors[idx], ec='black', linewidth=2, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, f'{idx + 1}', ha='center', va='center', fontsize=14, fontweight='bold')

        order_str = ' → '.join(str(n) for n in order)
        ax.text(2, 0.3, f'Order: {order_str}', ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/bfs_vs_dfs.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/bfs_vs_dfs.png")


def generate_grid_bfs():
    """Generate grid BFS shortest path visualization."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Grid BFS: Shortest Path (Flood Fill)', fontsize=14, fontweight='bold')

    grid = [
        [0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 'E']
    ]

    # BFS distances from (0,0)
    distances = [
        [0, 1, 2, 3, -1],
        [1, -1, -1, 4, 5],
        [2, 3, 4, -1, 6],
        [-1, -1, 5, 6, 7],
        [7, 6, 7, -1, 8]
    ]

    rows, cols = 5, 5

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                color = '#2C3E50'  # Wall
                text = ''
            elif grid[r][c] == 'E':
                color = '#E74C3C'  # End
                text = 'END'
            elif r == 0 and c == 0:
                color = '#27AE60'  # Start
                text = 'START'
            else:
                dist = distances[r][c]
                if dist == -1:
                    color = '#2C3E50'
                    text = ''
                else:
                    # Color by distance
                    color = plt.cm.Blues(0.2 + 0.6 * dist / 8)
                    text = str(dist)

            rect = patches.Rectangle((c, rows - 1 - r), 1, 1, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            if text:
                ax.text(c + 0.5, rows - 0.5 - r, text, ha='center', va='center',
                       fontsize=12 if text.isdigit() else 9, fontweight='bold',
                       color='white' if grid[r][c] in [1, 'E'] or (r == 0 and c == 0) else 'black')

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    ax.text(5.5, 4, 'Legend:', fontsize=11, fontweight='bold')
    ax.add_patch(patches.Rectangle((5.3, 3.3), 0.4, 0.4, facecolor='#27AE60', edgecolor='black'))
    ax.text(5.9, 3.5, 'Start', fontsize=10, va='center')
    ax.add_patch(patches.Rectangle((5.3, 2.7), 0.4, 0.4, facecolor='#E74C3C', edgecolor='black'))
    ax.text(5.9, 2.9, 'End', fontsize=10, va='center')
    ax.add_patch(patches.Rectangle((5.3, 2.1), 0.4, 0.4, facecolor='#2C3E50', edgecolor='black'))
    ax.text(5.9, 2.3, 'Wall', fontsize=10, va='center')
    ax.text(5.5, 1.5, 'Numbers = distance', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/grid_bfs.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/grid_bfs.png")


if __name__ == '__main__':
    generate_bfs_traversal()
    generate_dfs_traversal()
    generate_bfs_vs_dfs()
    generate_grid_bfs()
    print("\n✅ All graph visualizations generated!")
