#!/usr/bin/env python3
"""Generate binary search visualizations for the tutorial site."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/binary-search'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'active': '#3498DB',
    'eliminated': '#BDC3C7',
    'mid': '#F39C12',
    'found': '#27AE60',
    'target': '#E74C3C'
}


def generate_binary_search_steps():
    """Generate binary search step-by-step visualization."""
    arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    target = 13

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle(f'Binary Search for Target = {target}', fontsize=16, fontweight='bold')

    # Steps: (left, right, mid, comparison result)
    steps = [
        (0, 9, 4, "arr[4]=9 < 13, search right"),
        (5, 9, 7, "arr[7]=15 > 13, search left"),
        (5, 6, 5, "arr[5]=11 < 13, search right"),
        (6, 6, 6, "arr[6]=13 = 13, FOUND!"),
    ]

    for idx, (left, right, mid, desc) in enumerate(steps):
        ax = axes[idx]

        colors = []
        for i in range(len(arr)):
            if i < left or i > right:
                colors.append(COLORS['eliminated'])
            elif i == mid:
                colors.append(COLORS['found'] if arr[mid] == target else COLORS['mid'])
            else:
                colors.append(COLORS['active'])

        bars = ax.bar(range(len(arr)), arr, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for i, v in enumerate(arr):
            ax.text(i, v + 0.5, str(v), ha='center', fontsize=10, fontweight='bold')

        # Add pointer labels
        ax.annotate('L', (left, -1.5), ha='center', fontsize=12, fontweight='bold', color='blue')
        ax.annotate('R', (right, -1.5), ha='center', fontsize=12, fontweight='bold', color='blue')
        ax.annotate('M', (mid, arr[mid] + 2), ha='center', fontsize=12, fontweight='bold', color='orange')

        ax.set_title(f'Step {idx + 1}: {desc}', fontsize=11)
        ax.set_ylim(-3, 25)
        ax.set_xticks(range(len(arr)))
        ax.set_xlabel('Index')

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=COLORS['active'], label='Search Space'),
        patches.Patch(facecolor=COLORS['eliminated'], label='Eliminated'),
        patches.Patch(facecolor=COLORS['mid'], label='Mid Pointer'),
        patches.Patch(facecolor=COLORS['found'], label='Found'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(f'{OUTPUT_DIR}/binary_search_steps.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/binary_search_steps.png")


def generate_search_space_halving():
    """Generate O(log n) search space reduction visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Binary Search: Search Space Halves Each Step', fontsize=14, fontweight='bold')

    # Show search space reduction from 1024 to 1
    sizes = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    steps = list(range(len(sizes)))

    bars = ax.bar(steps, sizes, color='#3498DB', edgecolor='black', linewidth=1.5)

    # Color the bars in gradient
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.Blues(0.3 + 0.7 * (1 - i/len(sizes))))

    # Add labels
    for i, (step, size) in enumerate(zip(steps, sizes)):
        ax.text(step, size + 30, str(size), ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Step Number', fontsize=12)
    ax.set_ylabel('Remaining Search Space', fontsize=12)
    ax.set_xticks(steps)
    ax.set_xticklabels([f'Step {i}' for i in steps], rotation=45)

    # Add annotation
    ax.annotate('Only 10 steps to search\n1024 elements!', xy=(5, 32), xytext=(7, 400),
                fontsize=12, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/search_space_halving.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/search_space_halving.png")


def generate_left_right_bound():
    """Generate left bound vs right bound comparison."""
    arr = [1, 2, 2, 2, 2, 3, 4, 5]
    target = 2

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(f'Finding Target = {target} in Array with Duplicates', fontsize=14, fontweight='bold')

    scenarios = [
        ("Standard Binary Search", 3, "Returns any occurrence (index 3)"),
        ("Left Bound (bisect_left)", 1, "Returns first occurrence (index 1)"),
        ("Right Bound (bisect_right - 1)", 4, "Returns last occurrence (index 4)"),
    ]

    for idx, (title, result_idx, desc) in enumerate(scenarios):
        ax = axes[idx]

        colors = []
        for i in range(len(arr)):
            if arr[i] == target:
                if i == result_idx:
                    colors.append('#27AE60')  # Found
                else:
                    colors.append('#F39C12')  # Other matches
            else:
                colors.append('#BDC3C7')

        bars = ax.bar(range(len(arr)), [1]*len(arr), color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels inside bars
        for i, v in enumerate(arr):
            ax.text(i, 0.5, str(v), ha='center', va='center', fontsize=14, fontweight='bold')

        # Add index labels below
        for i in range(len(arr)):
            ax.text(i, -0.15, f'[{i}]', ha='center', fontsize=9, color='gray')

        # Mark result with arrow
        ax.annotate('↓', (result_idx, 1.1), ha='center', fontsize=20, color='#27AE60', fontweight='bold')

        ax.set_title(f'{title}: {desc}', fontsize=11)
        ax.set_ylim(-0.3, 1.4)
        ax.set_xlim(-0.6, len(arr) - 0.4)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/left_right_bound.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/left_right_bound.png")


def generate_linear_vs_binary():
    """Generate linear vs binary search comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Linear Search vs Binary Search', fontsize=16, fontweight='bold')

    arr = list(range(1, 17))
    target = 14

    # Linear search - checks elements one by one
    ax1 = axes[0]
    colors = ['#E74C3C' if i < 13 else '#27AE60' if i == 13 else '#BDC3C7' for i in range(16)]
    ax1.bar(range(16), [1]*16, color=colors, edgecolor='black')
    for i, v in enumerate(arr):
        ax1.text(i, 0.5, str(v), ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.set_title(f'Linear Search: 14 comparisons to find {target}', fontsize=12)
    ax1.set_ylim(0, 1.5)
    ax1.axis('off')
    ax1.text(8, 1.3, '→ → → → → → → → → → → → → ✓', ha='center', fontsize=10, color='#E74C3C')

    # Binary search - O(log n) comparisons
    ax2 = axes[1]
    # Steps: mid=7(8), mid=11(12), mid=13(14)
    checked = [7, 11, 13]
    colors2 = []
    for i in range(16):
        if i == 13:
            colors2.append('#27AE60')
        elif i in checked:
            colors2.append('#F39C12')
        else:
            colors2.append('#BDC3C7')
    ax2.bar(range(16), [1]*16, color=colors2, edgecolor='black')
    for i, v in enumerate(arr):
        ax2.text(i, 0.5, str(v), ha='center', va='center', fontsize=10, fontweight='bold')
    ax2.set_title(f'Binary Search: 4 comparisons to find {target}', fontsize=12)
    ax2.set_ylim(0, 1.5)
    ax2.axis('off')

    # Draw arrows showing binary search path
    ax2.annotate('', xy=(11, 1.2), xytext=(7, 1.2), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.annotate('', xy=(13, 1.1), xytext=(11, 1.2), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.text(8, 1.35, '3 jumps → found!', ha='center', fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/linear_vs_binary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/linear_vs_binary.png")


if __name__ == '__main__':
    generate_binary_search_steps()
    generate_search_space_halving()
    generate_left_right_bound()
    generate_linear_vs_binary()
    print("\n✅ All binary search visualizations generated!")
