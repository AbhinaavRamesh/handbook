#!/usr/bin/env python3
"""Generate combination sum backtracking tree visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_combination_sum_tree(candidates=[2, 3, 6, 7], target=7):
    """Generate a visualization of combination sum backtracking tree."""
    fig, ax = plt.subplots(figsize=(18, 12))

    nodes = []
    edges = []

    def build_tree(start, path, remaining, x, y, x_spread, parent_pos=None, choice=None, max_depth=4):
        """Recursively build the tree structure with limited depth."""
        if remaining < 0:
            nodes.append((path[:], remaining, x, y, 'invalid'))
            if parent_pos:
                edges.append((parent_pos, (x, y), choice, 'invalid'))
            return

        if remaining == 0:
            nodes.append((path[:], remaining, x, y, 'success'))
            if parent_pos:
                edges.append((parent_pos, (x, y), choice, 'success'))
            return

        # Limit depth for visualization
        if len(path) >= max_depth:
            nodes.append((path[:], remaining, x, y, 'truncated'))
            if parent_pos:
                edges.append((parent_pos, (x, y), choice, 'normal'))
            return

        nodes.append((path[:], remaining, x, y, 'normal'))
        if parent_pos:
            edges.append((parent_pos, (x, y), choice, 'normal'))

        # Only show subset of branches for cleaner visualization
        valid_candidates = [c for c in candidates[start:] if c <= remaining][:3]
        num_children = len(valid_candidates)

        if num_children == 0:
            return

        new_spread = x_spread / max(num_children, 1)
        total_width = new_spread * (num_children - 1) if num_children > 1 else 0
        start_x = x - total_width / 2

        for i, candidate in enumerate(valid_candidates):
            child_x = start_x + i * new_spread if num_children > 1 else x
            child_y = y - 1.8
            idx = candidates.index(candidate)
            build_tree(idx, path + [candidate], remaining - candidate,
                      child_x, child_y, new_spread * 0.7, (x, y), str(candidate))

    # Build tree
    build_tree(0, [], target, 9, 8, 7)

    # Find bounds
    min_x = min(node[2] for node in nodes) - 1.5
    max_x = max(node[2] for node in nodes) + 1.5
    min_y = min(node[3] for node in nodes) - 1
    max_y = max(node[3] for node in nodes) + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.axis('off')
    ax.set_title(f'Combination Sum Backtracking: candidates={candidates}, target={target}',
                fontsize=14, fontweight='bold', pad=20)

    # Draw edges first
    for start, end, choice, edge_type in edges:
        if edge_type == 'invalid':
            color = '#FF5722'
            style = '--'
        elif edge_type == 'success':
            color = '#4CAF50'
            style = '-'
        else:
            color = 'gray'
            style = '-'
        ax.plot([start[0], end[0]], [start[1] - 0.25, end[1] + 0.25],
               color=color, linewidth=1.5, linestyle=style, zorder=1)
        # Add choice label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.1, f'+{choice}', fontsize=7, ha='center', va='bottom',
               color='#1976D2', fontweight='bold')

    # Draw nodes
    for path, remaining, x, y, node_type in nodes:
        if node_type == 'success':
            color = '#90EE90'
            edge_color = '#228B22'
        elif node_type == 'invalid':
            color = '#FFCDD2'
            edge_color = '#D32F2F'
        elif node_type == 'truncated':
            color = '#E1BEE7'
            edge_color = '#7B1FA2'
        else:
            color = '#E3F2FD'
            edge_color = '#1976D2'

        path_str = str(path) if path else '[]'
        width = max(1.2, len(path_str) * 0.1 + 0.5)

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
        if node_type == 'success':
            text = f'{path_str}\nSum={target}!'
        elif node_type == 'invalid':
            text = f'{path_str}\nSum>{target}'
        elif node_type == 'truncated':
            text = f'{path_str}\n...'
        else:
            current_sum = sum(path) if path else 0
            text = f'{path_str}\nrem={remaining}'

        ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold', zorder=3)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#E3F2FD', edgecolor='#1976D2',
                      label='Exploring'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='#228B22',
                      label='Valid Solution'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFCDD2', edgecolor='#D32F2F',
                      label='Sum Exceeded (Pruned)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#E1BEE7', edgecolor='#7B1FA2',
                      label='Truncated for display'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # Add solutions found
    solutions = [p for p, _, _, _, t in nodes if t == 'success']
    sol_text = "Solutions found:\n" + "\n".join(str(s) for s in solutions)

    ax.text(0.98, 0.98, sol_text,
            transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Add complexity note
    ax.text(0.02, 0.02, f'Key: Same element can be used multiple times\n'
            f'Time Complexity: O(n^(target/min))\n'
            f'Pruning: Stop when sum > target',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_combination_sum_tree([2, 3, 6, 7], 7)
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/combination_sum_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated combination_sum_tree.png")
