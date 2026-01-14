#!/usr/bin/env python3
"""Generate word search backtracking visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_word_search_visualization():
    """Generate a visualization of word search backtracking."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle('Word Search Backtracking: Finding "ABCCED"', fontsize=16, fontweight='bold')

    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    word = "ABCCED"
    rows, cols = len(board), len(board[0])

    # Different stages of the search
    stages = [
        {
            'title': 'Step 1: Start at A(0,0)',
            'path': [(0, 0)],
            'matched': 'A',
            'trying': None,
            'note': 'Found "A" - matches word[0]'
        },
        {
            'title': 'Step 2: Move to B(0,1)',
            'path': [(0, 0), (0, 1)],
            'matched': 'AB',
            'trying': None,
            'note': 'Found "B" - matches word[1]'
        },
        {
            'title': 'Step 3: Move to C(0,2)',
            'path': [(0, 0), (0, 1), (0, 2)],
            'matched': 'ABC',
            'trying': None,
            'note': 'Found "C" - matches word[2]'
        },
        {
            'title': 'Step 4: Move to C(1,2)',
            'path': [(0, 0), (0, 1), (0, 2), (1, 2)],
            'matched': 'ABCC',
            'trying': None,
            'note': 'Found "C" - matches word[3]'
        },
        {
            'title': 'Step 5: Move to E(2,2)',
            'path': [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
            'matched': 'ABCCE',
            'trying': None,
            'note': 'Found "E" - matches word[4]'
        },
        {
            'title': 'Step 6: Move to D(2,1) - SUCCESS!',
            'path': [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)],
            'matched': 'ABCCED',
            'trying': None,
            'note': 'Found "D" - WORD COMPLETE!'
        },
    ]

    for idx, (ax, stage) in enumerate(zip(axes.flat, stages)):
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.set_title(stage['title'], fontsize=10, fontweight='bold')
        ax.axis('off')

        # Draw the grid
        for r in range(rows):
            for c in range(cols):
                # Determine cell color
                if (r, c) in stage['path']:
                    path_idx = stage['path'].index((r, c))
                    if path_idx == len(stage['path']) - 1:
                        color = '#4CAF50'  # Current position (green)
                    else:
                        color = '#81C784'  # Path (light green)
                elif stage['trying'] and (r, c) == stage['trying']:
                    color = '#FFF176'  # Trying (yellow)
                else:
                    color = '#E3F2FD'  # Unvisited

                rect = patches.FancyBboxPatch(
                    (c - 0.4, rows - 1 - r - 0.4), 0.8, 0.8,
                    boxstyle="round,pad=0.02,rounding_size=0.1",
                    facecolor=color,
                    edgecolor='gray',
                    linewidth=1
                )
                ax.add_patch(rect)

                # Add letter
                ax.text(c, rows - 1 - r, board[r][c], ha='center', va='center',
                       fontsize=14, fontweight='bold')

        # Draw arrows for path
        for i in range(len(stage['path']) - 1):
            r1, c1 = stage['path'][i]
            r2, c2 = stage['path'][i + 1]
            y1, y2 = rows - 1 - r1, rows - 1 - r2
            ax.annotate('', xy=(c2, y2), xytext=(c1, y1),
                       arrowprops=dict(arrowstyle='->', color='#1976D2',
                                      lw=2, shrinkA=15, shrinkB=15))

        # Add note
        ax.text(cols / 2, -0.8, f'Matched: "{stage["matched"]}"',
               ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add legend at the bottom
    fig.text(0.5, 0.02,
             'Green: Current path | Light Green: Previous path | Blue: Unvisited\n'
             'Backtracking: Mark visited -> Explore 4 directions -> Unmark (restore)',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

if __name__ == "__main__":
    fig = generate_word_search_visualization()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/word_search_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated word_search_tree.png")
