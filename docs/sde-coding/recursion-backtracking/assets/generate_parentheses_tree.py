#!/usr/bin/env python3
"""Generate parentheses generation tree visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_parentheses_tree(n=3):
    """Generate a visualization of generate parentheses backtracking tree."""
    fig, ax = plt.subplots(figsize=(18, 12))

    nodes = []
    edges = []

    def build_tree(s, left, right, x, y, x_spread, parent_pos=None, choice=None):
        """Recursively build the tree structure."""
        is_complete = len(s) == 2 * n
        is_valid = left <= n and right <= left

        if not is_valid:
            nodes.append((s, left, right, x, y, 'invalid'))
            if parent_pos:
                edges.append((parent_pos, (x, y), choice, 'invalid'))
            return

        if is_complete:
            nodes.append((s, left, right, x, y, 'success'))
            if parent_pos:
                edges.append((parent_pos, (x, y), choice, 'success'))
            return

        nodes.append((s, left, right, x, y, 'normal'))
        if parent_pos:
            edges.append((parent_pos, (x, y), choice, 'normal'))

        new_spread = x_spread * 0.55

        # Add '(' if we can
        if left < n:
            build_tree(s + '(', left + 1, right, x - x_spread / 2, y - 1.6,
                      new_spread, (x, y), '(')

        # Add ')' if we can
        if right < left:
            build_tree(s + ')', left, right + 1, x + x_spread / 2, y - 1.6,
                      new_spread, (x, y), ')')

    # Build tree
    build_tree('', 0, 0, 9, 8, 6)

    # Find bounds
    min_x = min(node[3] for node in nodes) - 1.5
    max_x = max(node[3] for node in nodes) + 1.5
    min_y = min(node[4] for node in nodes) - 1
    max_y = max(node[4] for node in nodes) + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.axis('off')
    ax.set_title(f'Generate Parentheses (n={n}) Backtracking Tree', fontsize=16, fontweight='bold', pad=20)

    # Draw edges first
    for start, end, choice, edge_type in edges:
        if edge_type == 'success':
            color = '#4CAF50'
            style = '-'
            width = 2
        elif edge_type == 'invalid':
            color = '#FF5722'
            style = '--'
            width = 1
        else:
            color = 'gray'
            style = '-'
            width = 1.5
        ax.plot([start[0], end[0]], [start[1] - 0.25, end[1] + 0.25],
               color=color, linewidth=width, linestyle=style, zorder=1)
        # Add choice label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.1, choice, fontsize=10, ha='center', va='bottom',
               color='#1976D2', fontweight='bold', fontfamily='monospace')

    # Draw nodes
    for s, left, right, x, y, node_type in nodes:
        if node_type == 'success':
            color = '#C8E6C9'
            edge_color = '#388E3C'
        elif node_type == 'invalid':
            color = '#FFCDD2'
            edge_color = '#D32F2F'
        else:
            color = '#E3F2FD'
            edge_color = '#1976D2'

        display_s = f'"{s}"' if s else '""'
        width = max(1.4, len(display_s) * 0.1 + 0.5)

        box = FancyBboxPatch(
            (x - width / 2, y - 0.3), width, 0.6,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)

        # Main text
        main_text = display_s
        ax.text(x, y + 0.05, main_text, ha='center', va='center', fontsize=9,
               fontweight='bold', fontfamily='monospace', zorder=3)

        # Counts below
        count_text = f'L:{left} R:{right}'
        ax.text(x, y - 0.15, count_text, ha='center', va='center', fontsize=6,
               color='#666', zorder=3)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#E3F2FD', edgecolor='#1976D2',
                      label='Building string'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#C8E6C9', edgecolor='#388E3C',
                      label='Valid combination'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFCDD2', edgecolor='#D32F2F',
                      label='Invalid (would be pruned)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Add solutions
    solutions = [s for s, _, _, _, _, t in nodes if t == 'success']
    sol_text = f"Valid combinations (n={n}):\n" + "\n".join(solutions)

    ax.text(0.98, 0.98, sol_text,
            transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Add constraints explanation
    ax.text(0.02, 0.02, f'Constraints:\n'
            f'- Add "(" if left_count < n\n'
            f'- Add ")" if right_count < left_count\n'
            f'Time: O(4^n / sqrt(n)) - Catalan number\n'
            f'Space: O(n)',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_parentheses_tree(3)
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/parentheses_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated parentheses_tree.png")
