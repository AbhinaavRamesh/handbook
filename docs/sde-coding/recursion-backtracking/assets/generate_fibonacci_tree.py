#!/usr/bin/env python3
"""Generate fibonacci recursion tree visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_fibonacci_tree(n=5):
    """Generate a visualization of fibonacci recursion tree."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Store all nodes to draw
    nodes = []
    edges = []

    def build_tree(val, x, y, level, x_spread, parent_pos=None):
        """Recursively build the tree structure."""
        nodes.append((val, x, y, level))
        if parent_pos:
            edges.append((parent_pos, (x, y)))

        if val <= 1:
            return

        # Calculate spread for children
        new_spread = x_spread * 0.5

        # Left child (val - 1)
        left_x = x - x_spread
        left_y = y - 1.3
        build_tree(val - 1, left_x, left_y, level + 1, new_spread, (x, y))

        # Right child (val - 2)
        right_x = x + x_spread
        right_y = y - 1.3
        build_tree(val - 2, right_x, right_y, level + 1, new_spread, (x, y))

    # Build tree structure
    initial_spread = 2 ** (n - 2) * 0.8
    build_tree(n, 8, n + 0.5, 0, initial_spread)

    # Find bounds
    min_x = min(node[1] for node in nodes) - 1.5
    max_x = max(node[1] for node in nodes) + 1.5
    min_y = min(node[2] for node in nodes) - 1
    max_y = max(node[2] for node in nodes) + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.axis('off')
    ax.set_title(f'Fibonacci Recursion Tree: fib({n})', fontsize=16, fontweight='bold', pad=20)

    # Draw edges first
    for start, end in edges:
        ax.plot([start[0], end[0]], [start[1] - 0.25, end[1] + 0.25],
               'gray', linewidth=1.5, zorder=1)

    # Draw nodes
    for val, x, y, level in nodes:
        is_base = (val <= 1)
        color = '#90EE90' if is_base else '#FFE4B5'
        edge_color = '#228B22' if is_base else '#DAA520'

        box = FancyBboxPatch(
            (x - 0.5, y - 0.25), 1.0, 0.5,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)

        if is_base:
            text = f'fib({val})\n= {val}'
        else:
            text = f'fib({val})'
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFE4B5', edgecolor='#DAA520',
                      label='Recursive Call'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='#228B22',
                      label='Base Case'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Count duplicates
    val_counts = {}
    for val, _, _, _ in nodes:
        val_counts[val] = val_counts.get(val, 0) + 1

    duplicate_text = "Duplicate calculations:\n"
    for val in sorted(val_counts.keys(), reverse=True):
        if val_counts[val] > 1:
            duplicate_text += f"fib({val}) called {val_counts[val]} times\n"

    # Add complexity note
    ax.text(0.02, 0.02, f'Total function calls: {len(nodes)}\n'
            f'Time Complexity: O(2^n) without memoization\n'
            f'{duplicate_text}',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_fibonacci_tree(5)
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/fibonacci_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated fibonacci_tree.png")
