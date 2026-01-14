#!/usr/bin/env python3
"""Generate Recursion and Backtracking visualizations."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/recursion'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_fibonacci_tree():
    """Generate Fibonacci recursion tree visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title('Fibonacci Recursion Tree: fib(5)', fontsize=14, fontweight='bold')

    # Tree structure for fib(5)
    nodes = [
        (7, 9, 'fib(5)', '#E74C3C'),
        (4, 7, 'fib(4)', '#F39C12'),
        (10, 7, 'fib(3)', '#27AE60'),
        (2, 5, 'fib(3)', '#27AE60'),
        (6, 5, 'fib(2)', '#3498DB'),
        (8.5, 5, 'fib(2)', '#3498DB'),
        (11.5, 5, 'fib(1)', '#9B59B6'),
        (1, 3, 'fib(2)', '#3498DB'),
        (3, 3, 'fib(1)', '#9B59B6'),
        (5, 3, 'fib(1)', '#9B59B6'),
        (7, 3, 'fib(0)', '#BDC3C7'),
        (8, 3, 'fib(1)', '#9B59B6'),
        (9.5, 3, 'fib(0)', '#BDC3C7'),
    ]

    edges = [
        (7, 9, 4, 7), (7, 9, 10, 7),
        (4, 7, 2, 5), (4, 7, 6, 5),
        (10, 7, 8.5, 5), (10, 7, 11.5, 5),
        (2, 5, 1, 3), (2, 5, 3, 3),
        (6, 5, 5, 3), (6, 5, 7, 3),
        (8.5, 5, 8, 3), (8.5, 5, 9.5, 3),
    ]

    # Draw edges
    for x1, y1, x2, y2 in edges:
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1.5, alpha=0.6)

    # Draw nodes
    for x, y, label, color in nodes:
        circle = plt.Circle((x, y), 0.6, color=color, ec='black', lw=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

    # Add annotations
    ax.text(7, 1, 'Repeated subproblems: fib(3) computed 2 times, fib(2) computed 3 times!',
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.text(7, 0.2, 'Solution: Memoization or Bottom-Up DP', ha='center', fontsize=10)

    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fibonacci_tree.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/fibonacci_tree.png")


def generate_permutations_tree():
    """Generate permutations backtracking tree."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Permutations of [1, 2, 3]: Backtracking Tree', fontsize=14, fontweight='bold')

    # Tree levels
    ax.text(8, 9.5, '[]', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    # Level 1
    level1 = [(2.5, 7.5, '[1]'), (8, 7.5, '[2]'), (13.5, 7.5, '[3]')]
    for x, y, label in level1:
        ax.text(x, y, label, ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='#AED6F1', edgecolor='black'))
        ax.plot([8, x], [9.2, y + 0.4], 'k-', lw=1.5)

    # Level 2
    level2 = [
        (1, 5, '[1,2]'), (4, 5, '[1,3]'),
        (6.5, 5, '[2,1]'), (9.5, 5, '[2,3]'),
        (12, 5, '[3,1]'), (15, 5, '[3,2]')
    ]
    connections2 = [(2.5, 1), (2.5, 4), (8, 6.5), (8, 9.5), (13.5, 12), (13.5, 15)]
    for (px, i), (x, y, label) in zip(connections2, level2):
        ax.text(x, y, label, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor='black'))
        ax.plot([px, x], [7.1, y + 0.4], 'k-', lw=1.5)

    # Level 3 (leaves - complete permutations)
    level3 = [
        (0.5, 2.5, '[1,2,3]'), (1.5, 2.5, '[1,3,2]'),
        (3.5, 2.5, '[2,1,3]'), (4.5, 2.5, '[2,3,1]'),
        (6, 2.5, '[3,1,2]'), (7, 2.5, '[3,2,1]'),
    ]
    # Simplified: just show final permutations
    for i, (x, y, label) in enumerate(level3):
        ax.text(x + i * 1.5 + 0.5, 2.5, label, ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='#27AE60', edgecolor='black'),
               color='white', fontweight='bold')

    ax.text(8, 1, 'Total: 3! = 6 permutations', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow'))

    ax.set_xlim(-1, 17)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/permutations_tree.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/permutations_tree.png")


def generate_subsets_tree():
    """Generate subsets backtracking tree."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Subsets of [1, 2, 3]: Include/Exclude Decision Tree', fontsize=14, fontweight='bold')

    # Show the include/exclude pattern
    ax.text(7, 7, '{}', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    # Level 1: include/exclude 1
    ax.text(3.5, 5, '{1}', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor='black'))
    ax.text(10.5, 5, '{}', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='#FADBD8', edgecolor='black'))
    ax.annotate('include 1', xy=(4.5, 5.5), xytext=(5.5, 6.5), fontsize=9, color='green')
    ax.annotate('exclude 1', xy=(9.5, 5.5), xytext=(8, 6.5), fontsize=9, color='red')
    ax.plot([7, 3.5], [6.7, 5.4], 'g-', lw=1.5)
    ax.plot([7, 10.5], [6.7, 5.4], 'r-', lw=1.5)

    # Level 2: include/exclude 2
    leaves = [
        (1.5, 3, '{1,2}', '#D5F5E3'),
        (5.5, 3, '{1}', '#FADBD8'),
        (8.5, 3, '{2}', '#D5F5E3'),
        (12.5, 3, '{}', '#FADBD8'),
    ]
    for x, y, label, color in leaves:
        ax.text(x, y, label, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor=color, edgecolor='black'))

    # Final subsets at bottom
    ax.text(7, 1, 'All 2³ = 8 subsets: {}, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}',
           ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow'))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/subsets_tree.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/subsets_tree.png")


def generate_parentheses():
    """Generate generate parentheses visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title('Generate Parentheses (n=3): Backtracking with Pruning', fontsize=14, fontweight='bold')

    # Show the key constraint
    ax.text(7, 9.5, 'Constraints: open < n, close < open', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Sample tree paths
    ax.text(7, 8, '""', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    ax.text(4, 6.5, '"("', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor='black'))
    ax.plot([7, 4], [7.7, 6.9], 'g-', lw=1.5)
    ax.text(5, 7.3, 'add (', fontsize=9, color='green')

    ax.text(2, 5, '"(("', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor='black'))
    ax.text(6, 5, '"()"', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#AED6F1', edgecolor='black'))

    # Show valid results
    results = ['((()))', '(()())', '(())()', '()(())', '()()()']
    ax.text(7, 2, 'Valid results for n=3:', ha='center', fontsize=11, fontweight='bold')
    for i, r in enumerate(results):
        ax.text(2.5 + i * 2.5, 1, r, ha='center', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#27AE60', edgecolor='black'),
               color='white')

    ax.text(7, 0, f'Total: Catalan number C₃ = {len(results)}', ha='center', fontsize=10)

    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/generate_parentheses.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/generate_parentheses.png")


def generate_sudoku():
    """Generate sudoku backtracking visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Sudoku Solver: Backtracking Process', fontsize=14, fontweight='bold')

    # Partial board
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    for ax_idx, (ax, title) in enumerate([(axes[0], 'Input: Empty cells marked'),
                                          (axes[1], 'Process: Try digits 1-9')]):
        ax.set_title(title, fontsize=11)

        for i in range(9):
            for j in range(9):
                # Draw cell
                color = '#FADBD8' if board[i][j] == 0 else 'white'
                rect = patches.Rectangle((j, 8 - i), 1, 1, facecolor=color,
                                         edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)

                if board[i][j] != 0:
                    ax.text(j + 0.5, 8.5 - i, str(board[i][j]),
                           ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw 3x3 box borders
        for i in range(0, 10, 3):
            ax.axhline(y=i, color='black', linewidth=2)
            ax.axvline(x=i, color='black', linewidth=2)

        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.set_aspect('equal')
        ax.axis('off')

        if ax_idx == 1:
            # Highlight a cell being tried
            rect = patches.Rectangle((2, 6), 1, 1, facecolor='#F39C12',
                                     edgecolor='red', linewidth=3)
            ax.add_patch(rect)
            ax.text(2.5, 6.5, '?', ha='center', va='center', fontsize=12, color='red')
            ax.text(4.5, -0.5, 'Try 1,2,3,4... Check row/col/box constraints',
                   ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sudoku_backtracking.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/sudoku_backtracking.png")


def generate_backtracking_template():
    """Generate generic backtracking template visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Backtracking Template: Explore → Check → Backtrack', fontsize=14, fontweight='bold')

    # Flowchart
    boxes = [
        (6, 7, 'Start with empty solution', '#AED6F1'),
        (6, 5.5, 'For each choice:', '#D5F5E3'),
        (3, 4, 'Make choice\n(add to path)', '#27AE60'),
        (6, 4, 'Valid?', '#F39C12'),
        (9, 4, 'Goal\nreached?', '#E74C3C'),
        (3, 2.5, 'Recurse\n(explore further)', '#3498DB'),
        (6, 2.5, 'Backtrack\n(undo choice)', '#9B59B6'),
        (9, 2.5, 'Save\nsolution!', '#27AE60'),
    ]

    for x, y, text, color in boxes:
        rect = patches.FancyBboxPatch((x - 1.2, y - 0.5), 2.4, 1,
                                      boxstyle="round,pad=0.1",
                                      facecolor=color, edgecolor='black', lw=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows
    arrows = [
        ((6, 6.5), (6, 6)),
        ((6, 5), (3, 4.5)),
        ((4.2, 4), (4.8, 4)),
        ((7.2, 4), (7.8, 4)),
        ((3, 3.5), (3, 3)),
        ((6, 3.5), (6, 3)),
        ((9, 3.5), (9, 3)),
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Labels
    ax.text(5.4, 4.2, 'Yes', fontsize=9, color='green')
    ax.text(5.4, 3.8, 'No', fontsize=9, color='red')
    ax.text(8.4, 4.2, 'Yes', fontsize=9, color='green')
    ax.text(8.4, 3.8, 'No', fontsize=9, color='red')

    ax.set_xlim(0, 12)
    ax.set_ylim(1.5, 8)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/backtracking_template.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/backtracking_template.png")


if __name__ == '__main__':
    generate_fibonacci_tree()
    generate_permutations_tree()
    generate_subsets_tree()
    generate_parentheses()
    generate_sudoku()
    generate_backtracking_template()
    print("\n✅ All recursion visualizations generated!")
