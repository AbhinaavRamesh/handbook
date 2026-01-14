#!/usr/bin/env python3
"""Generate letter combinations phone tree visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_letter_combinations_tree(digits="23"):
    """Generate a visualization of letter combinations backtracking tree."""
    fig, ax = plt.subplots(figsize=(16, 10))

    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    nodes = []
    edges = []

    def build_tree(path, idx, x, y, x_spread, parent_pos=None, choice=None):
        """Recursively build the tree structure."""
        is_complete = idx == len(digits)

        if is_complete:
            nodes.append((path, idx, x, y, 'success'))
            if parent_pos:
                edges.append((parent_pos, (x, y), choice, 'success'))
            return

        nodes.append((path, idx, x, y, 'normal'))
        if parent_pos:
            edges.append((parent_pos, (x, y), choice, 'normal'))

        digit = digits[idx]
        letters = phone_map.get(digit, '')
        num_children = len(letters)

        new_spread = x_spread / max(num_children, 1)
        total_width = new_spread * (num_children - 1) if num_children > 1 else 0
        start_x = x - total_width / 2

        for i, letter in enumerate(letters):
            child_x = start_x + i * new_spread if num_children > 1 else x
            child_y = y - 2.0
            build_tree(path + letter, idx + 1, child_x, child_y,
                      new_spread * 0.8, (x, y), letter)

    # Build tree
    n = len(digits)
    initial_spread = 3 ** (n - 1) * 1.5 if n > 1 else 2
    build_tree('', 0, 8, n * 2 + 1, initial_spread)

    # Find bounds
    min_x = min(node[2] for node in nodes) - 1.5
    max_x = max(node[2] for node in nodes) + 1.5
    min_y = min(node[3] for node in nodes) - 1
    max_y = max(node[3] for node in nodes) + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.axis('off')
    ax.set_title(f'Letter Combinations of Phone Number: "{digits}"', fontsize=16, fontweight='bold', pad=20)

    # Draw phone pad reference
    ax.text(0.02, 0.98, 'Phone Keypad:\n'
            '2: abc  3: def\n'
            '4: ghi  5: jkl\n'
            '6: mno  7: pqrs\n'
            '8: tuv  9: wxyz',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Draw edges first
    for start, end, choice, edge_type in edges:
        if edge_type == 'success':
            color = '#4CAF50'
            width = 2
        else:
            color = 'gray'
            width = 1.5
        ax.plot([start[0], end[0]], [start[1] - 0.25, end[1] + 0.25],
               color=color, linewidth=width, zorder=1)
        # Add choice label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.15, f'"{choice}"', fontsize=9, ha='center', va='bottom',
               color='#E65100', fontweight='bold', fontfamily='monospace')

    # Draw nodes
    for path, idx, x, y, node_type in nodes:
        if node_type == 'success':
            color = '#C8E6C9'
            edge_color = '#388E3C'
        else:
            color = '#E3F2FD'
            edge_color = '#1976D2'

        display_path = f'"{path}"' if path else '""'
        width = max(1.0, len(display_path) * 0.12 + 0.4)

        box = FancyBboxPatch(
            (x - width / 2, y - 0.3), width, 0.6,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)

        ax.text(x, y, display_path, ha='center', va='center', fontsize=10,
               fontweight='bold', fontfamily='monospace', zorder=3)

        # Show which digit we're processing
        if idx < len(digits):
            digit = digits[idx]
            letters = phone_map.get(digit, '')
            ax.text(x, y + 0.45, f'digit "{digit}" -> {letters}', fontsize=6,
                   ha='center', va='bottom', color='#666', style='italic')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#E3F2FD', edgecolor='#1976D2',
                      label='Building combination'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#C8E6C9', edgecolor='#388E3C',
                      label='Complete combination'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Add solutions
    solutions = [p for p, _, _, _, t in nodes if t == 'success']
    ax.text(0.5, 0.02, f'Total combinations: {len(solutions)}\n'
            f'Results: {solutions}',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_letter_combinations_tree("23")
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/letter_combinations_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated letter_combinations_tree.png")
