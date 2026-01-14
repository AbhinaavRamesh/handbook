#!/usr/bin/env python3
"""Generate palindrome partitioning backtracking visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_palindrome_partitioning_tree(s="aab"):
    """Generate a visualization of palindrome partitioning backtracking tree."""
    fig, ax = plt.subplots(figsize=(16, 12))

    nodes = []
    edges = []

    def is_palindrome(substr):
        return substr == substr[::-1]

    def build_tree(start, partition, x, y, x_spread, parent_pos=None, choice=None, depth=0):
        """Recursively build the tree structure."""
        if depth > 4:  # Limit depth for visualization
            return

        is_complete = start == len(s)

        if is_complete:
            nodes.append((partition[:], start, x, y, 'success'))
            if parent_pos:
                edges.append((parent_pos, (x, y), choice, 'success'))
            return

        nodes.append((partition[:], start, x, y, 'normal'))
        if parent_pos:
            edges.append((parent_pos, (x, y), choice, 'normal'))

        # Try all possible substrings from current position
        valid_endings = []
        for end in range(start + 1, len(s) + 1):
            substr = s[start:end]
            if is_palindrome(substr):
                valid_endings.append((end, substr))

        num_children = len(valid_endings)
        if num_children == 0:
            return

        new_spread = x_spread / max(num_children, 1)
        total_width = new_spread * (num_children - 1) if num_children > 1 else 0
        start_x = x - total_width / 2

        for i, (end, substr) in enumerate(valid_endings):
            child_x = start_x + i * new_spread if num_children > 1 else x
            child_y = y - 2.0
            build_tree(end, partition + [substr], child_x, child_y,
                      new_spread * 0.7, (x, y), substr, depth + 1)

    # Build tree
    build_tree(0, [], 8, 9, 7)

    # Find bounds
    min_x = min(node[2] for node in nodes) - 1.5
    max_x = max(node[2] for node in nodes) + 1.5
    min_y = min(node[3] for node in nodes) - 1
    max_y = max(node[3] for node in nodes) + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.axis('off')
    ax.set_title(f'Palindrome Partitioning: "{s}"', fontsize=16, fontweight='bold', pad=20)

    # Draw edges first
    for start_pos, end_pos, choice, edge_type in edges:
        if edge_type == 'success':
            color = '#4CAF50'
            width = 2
        else:
            color = 'gray'
            width = 1.5
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1] - 0.3, end_pos[1] + 0.3],
               color=color, linewidth=width, zorder=1)
        # Add choice label
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        ax.text(mid_x, mid_y + 0.15, f'"{choice}"', fontsize=9, ha='center', va='bottom',
               color='#E65100', fontweight='bold', fontfamily='monospace')

    # Draw nodes
    for partition, idx, x, y, node_type in nodes:
        if node_type == 'success':
            color = '#C8E6C9'
            edge_color = '#388E3C'
        else:
            color = '#E3F2FD'
            edge_color = '#1976D2'

        partition_str = str(partition) if partition else '[]'
        width = max(1.4, len(partition_str) * 0.08 + 0.5)

        box = FancyBboxPatch(
            (x - width / 2, y - 0.35), width, 0.7,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)

        ax.text(x, y + 0.05, partition_str, ha='center', va='center', fontsize=8,
               fontweight='bold', fontfamily='monospace', zorder=3)

        # Show remaining string
        remaining = s[idx:]
        if remaining:
            ax.text(x, y - 0.2, f'rem: "{remaining}"', ha='center', va='center',
                   fontsize=6, color='#666', zorder=3)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#E3F2FD', edgecolor='#1976D2',
                      label='Partial partition'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#C8E6C9', edgecolor='#388E3C',
                      label='Complete partition'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Add solutions
    solutions = [p for p, _, _, _, t in nodes if t == 'success']
    sol_text = f"Valid partitions of \"{s}\":\n"
    for sol in solutions:
        sol_text += f"  {sol}\n"

    ax.text(0.98, 0.98, sol_text,
            transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Add algorithm explanation
    ax.text(0.02, 0.02, f'Algorithm:\n'
            f'1. At each position, try all palindromic substrings\n'
            f'2. Only proceed if substring is palindrome\n'
            f'3. Recurse with remaining string\n'
            f'Time: O(n * 2^n), Space: O(n)',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_palindrome_partitioning_tree("aab")
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/palindrome_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated palindrome_tree.png")
