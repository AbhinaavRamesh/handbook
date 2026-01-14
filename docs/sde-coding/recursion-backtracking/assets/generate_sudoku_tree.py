#!/usr/bin/env python3
"""Generate sudoku solver backtracking visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_sudoku_visualization():
    """Generate a visualization of sudoku backtracking."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle('Sudoku Solver Backtracking (Simplified 4x4)', fontsize=16, fontweight='bold')

    # Simplified 4x4 sudoku for visualization
    initial_board = [
        [1, 0, 0, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [4, 0, 0, 1]
    ]

    stages = [
        {
            'title': 'Step 1: Initial Board',
            'board': [[1, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 1]],
            'trying': None,
            'note': 'Find first empty cell (0,1)'
        },
        {
            'title': 'Step 2: Try 2 at (0,1)',
            'board': [[1, 2, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 1]],
            'trying': (0, 1),
            'note': '2 is valid - continue'
        },
        {
            'title': 'Step 3: Try 3 at (0,2)',
            'board': [[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 1]],
            'trying': (0, 2),
            'note': '3 is valid - row 0 complete!'
        },
        {
            'title': 'Step 4: Fill row 1',
            'board': [[1, 2, 3, 4], [3, 4, 1, 2], [0, 0, 0, 0], [4, 0, 0, 1]],
            'trying': (1, 3),
            'note': 'Checking constraints...'
        },
        {
            'title': 'Step 5: Backtrack needed!',
            'board': [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 0, 0, 1]],
            'trying': (3, 1),
            'trying_fail': True,
            'note': 'Row 2 done, filling row 3'
        },
        {
            'title': 'Step 6: Solution Found!',
            'board': [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]],
            'trying': None,
            'solved': True,
            'note': 'All cells filled - Valid solution!'
        },
    ]

    for idx, (ax, stage) in enumerate(zip(axes.flat, stages)):
        board = stage['board']
        n = len(board)
        box_size = 2  # 2x2 boxes for 4x4 sudoku

        ax.set_xlim(-0.1, n + 0.1)
        ax.set_ylim(-0.1, n + 0.1)
        ax.set_aspect('equal')
        ax.set_title(stage['title'], fontsize=10, fontweight='bold')
        ax.axis('off')

        # Draw the grid
        for r in range(n):
            for c in range(n):
                val = board[r][c]
                is_initial = initial_board[r][c] != 0

                # Determine cell color
                if stage.get('solved'):
                    color = '#C8E6C9'  # All green for solved
                elif stage['trying'] and (r, c) == stage['trying']:
                    if stage.get('trying_fail'):
                        color = '#FFCDD2'  # Red for failed attempt
                    else:
                        color = '#FFF9C4'  # Yellow for current try
                elif is_initial:
                    color = '#BBDEFB'  # Blue for initial values
                elif val != 0:
                    color = '#E8F5E9'  # Light green for filled
                else:
                    color = '#FAFAFA'  # White for empty

                rect = patches.Rectangle(
                    (c, n - 1 - r), 1, 1,
                    facecolor=color,
                    edgecolor='gray',
                    linewidth=0.5
                )
                ax.add_patch(rect)

                # Add number
                if val != 0:
                    fontweight = 'bold' if is_initial else 'normal'
                    fontcolor = '#1565C0' if is_initial else '#333'
                    ax.text(c + 0.5, n - 0.5 - r, str(val),
                           ha='center', va='center',
                           fontsize=12, fontweight=fontweight, color=fontcolor)

        # Draw thick lines for box boundaries
        for i in range(0, n + 1, box_size):
            ax.axhline(y=i, color='black', linewidth=2)
            ax.axvline(x=i, color='black', linewidth=2)

        # Add note
        ax.text(n / 2, -0.3, stage['note'],
               ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add algorithm explanation at the bottom
    fig.text(0.5, 0.02,
             'Algorithm: Find empty cell -> Try 1-n -> Check row/col/box constraints -> '
             'If valid, recurse -> If stuck, backtrack\n'
             'Blue: Initial given values | Yellow: Currently trying | Green: Filled values | Red: Invalid attempt',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig

if __name__ == "__main__":
    fig = generate_sudoku_visualization()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/sudoku_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated sudoku_tree.png")
