#!/usr/bin/env python3
"""Generate N-Queens backtracking tree visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_nqueens_tree(n=4):
    """Generate a visualization of N-Queens backtracking."""
    fig, ax = plt.subplots(figsize=(18, 12))

    nodes = []
    edges = []
    solutions = []

    def is_safe(queens, row, col):
        """Check if placing queen at (row, col) is safe."""
        for r, c in enumerate(queens):
            if c == col:  # Same column
                return False
            if abs(r - row) == abs(c - col):  # Diagonal
                return False
        return True

    def build_tree(queens, row, x, y, x_spread, parent_pos=None, choice=None, max_nodes=50):
        """Recursively build the tree structure."""
        if len(nodes) > max_nodes:
            return

        if row == n:
            nodes.append((queens[:], row, x, y, 'success'))
            if parent_pos:
                edges.append((parent_pos, (x, y), choice, 'success'))
            solutions.append(queens[:])
            return

        is_root = parent_pos is None
        nodes.append((queens[:], row, x, y, 'exploring' if is_root else 'normal'))
        if parent_pos:
            edges.append((parent_pos, (x, y), choice, 'normal'))

        # Try each column
        valid_cols = [c for c in range(n) if is_safe(queens, row, c)]
        num_children = len(valid_cols)

        if num_children == 0:
            return

        new_spread = x_spread / max(num_children, 1)
        total_width = new_spread * (num_children - 1) if num_children > 1 else 0
        start_x = x - total_width / 2

        for i, col in enumerate(valid_cols):
            child_x = start_x + i * new_spread if num_children > 1 else x
            child_y = y - 2.0
            build_tree(queens + [col], row + 1, child_x, child_y,
                      new_spread * 0.8, (x, y), f'col {col}')

    # Build tree
    build_tree([], 0, 9, 10, 8)

    # Find bounds
    if nodes:
        min_x = min(node[2] for node in nodes) - 2
        max_x = max(node[2] for node in nodes) + 2
        min_y = min(node[3] for node in nodes) - 2
        max_y = max(node[3] for node in nodes) + 1

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    ax.axis('off')
    ax.set_title(f'{n}-Queens Backtracking Tree', fontsize=16, fontweight='bold', pad=20)

    # Draw edges first
    for start, end, choice, edge_type in edges:
        if edge_type == 'success':
            color = '#4CAF50'
            width = 2
        else:
            color = 'gray'
            width = 1.5
        ax.plot([start[0], end[0]], [start[1] - 0.35, end[1] + 0.35],
               color=color, linewidth=width, zorder=1)
        # Add choice label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.15, choice, fontsize=6, ha='center', va='bottom',
               color='#1976D2', fontweight='bold')

    # Draw mini boards for each node
    def draw_mini_board(queens, x, y, size=0.6, is_solution=False):
        """Draw a mini chess board with queens placed."""
        cell_size = size / n

        # Draw board background
        board_color = '#FFFDE7' if is_solution else '#FAFAFA'
        border_color = '#4CAF50' if is_solution else '#BDBDBD'
        border_width = 3 if is_solution else 1

        for row in range(n):
            for col in range(n):
                cell_color = '#D7CCC8' if (row + col) % 2 == 0 else '#EFEBE9'
                rect = patches.Rectangle(
                    (x - size/2 + col * cell_size, y - size/2 + (n-1-row) * cell_size),
                    cell_size, cell_size,
                    facecolor=cell_color, edgecolor='gray', linewidth=0.3, zorder=2
                )
                ax.add_patch(rect)

        # Draw queens
        for row, col in enumerate(queens):
            qx = x - size/2 + col * cell_size + cell_size/2
            qy = y - size/2 + (n-1-row) * cell_size + cell_size/2
            ax.text(qx, qy, 'Q', fontsize=7, ha='center', va='center',
                   fontweight='bold', color='#D32F2F', zorder=3)

        # Draw border
        border = patches.Rectangle(
            (x - size/2, y - size/2), size, size,
            fill=False, edgecolor=border_color, linewidth=border_width, zorder=4
        )
        ax.add_patch(border)

        # Label
        row_label = f'Row {len(queens)}' if len(queens) < n else 'Solution!'
        ax.text(x, y + size/2 + 0.15, row_label, fontsize=6, ha='center',
               va='bottom', color='#666')

    # Draw nodes
    for queens, row, x, y, node_type in nodes:
        is_solution = node_type == 'success'
        draw_mini_board(queens, x, y, is_solution=is_solution)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FAFAFA', edgecolor='#BDBDBD',
                      label='Partial placement'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFFDE7', edgecolor='#4CAF50',
                      linewidth=3, label='Valid solution'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Add solution count
    ax.text(0.98, 0.98, f'Solutions found: {len(solutions)}\n\n'
            f'Queen positions shown\nas "Q" on board\n\n'
            f'Invalid branches\npruned (not shown)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Add complexity note
    ax.text(0.02, 0.02, f'Constraint checking:\n'
            f'- Same column: O(1) with set\n'
            f'- Same diagonal: row+col or row-col\n'
            f'Time: O(n!), Space: O(n)',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_nqueens_tree(4)
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/nqueens_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated nqueens_tree.png")
