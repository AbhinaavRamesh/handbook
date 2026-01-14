#!/usr/bin/env python3
"""Generate two pointer pattern visualizations for the tutorial site."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/two-pointer'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_converging_pointers():
    """Generate Two Sum II converging pointers visualization."""
    arr = [2, 7, 11, 15, 19, 21, 25]
    target = 26

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle(f'Two Pointers (Converging): Two Sum II, Target = {target}', fontsize=14, fontweight='bold')

    # Steps: (left, right, sum, action)
    steps = [
        (0, 6, 27, "2+25=27 > 26, move R left"),
        (0, 5, 23, "2+21=23 < 26, move L right"),
        (1, 5, 28, "7+21=28 > 26, move R left"),
        (1, 4, 26, "7+19=26 = target, FOUND!"),
    ]

    for idx, (left, right, s, desc) in enumerate(steps):
        ax = axes[idx]

        colors = ['#BDC3C7'] * len(arr)
        colors[left] = '#E74C3C'  # Left pointer
        colors[right] = '#3498DB'  # Right pointer

        if s == target:
            colors[left] = '#27AE60'
            colors[right] = '#27AE60'

        bars = ax.bar(range(len(arr)), arr, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for i, v in enumerate(arr):
            ax.text(i, v + 0.8, str(v), ha='center', fontsize=11, fontweight='bold')

        # Add pointer labels with arrows
        ax.annotate('L', (left, arr[left] + 3), ha='center', fontsize=14, fontweight='bold', color='#E74C3C')
        ax.annotate('R', (right, arr[right] + 3), ha='center', fontsize=14, fontweight='bold', color='#3498DB')

        # Draw sum calculation
        ax.text(len(arr)/2, -3, f'{arr[left]} + {arr[right]} = {s}', ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='yellow' if s == target else 'white', alpha=0.8))

        ax.set_title(f'Step {idx + 1}: {desc}', fontsize=11,
                    color='#27AE60' if s == target else 'black',
                    fontweight='bold' if s == target else 'normal')
        ax.set_ylim(-5, max(arr) + 6)
        ax.set_xticks(range(len(arr)))
        ax.set_xticklabels([f'[{i}]' for i in range(len(arr))])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/converging_pointers.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/converging_pointers.png")


def generate_fast_slow_pointers():
    """Generate fast/slow pointers for remove duplicates."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Fast & Slow Pointers: Remove Duplicates from Sorted Array', fontsize=14, fontweight='bold')

    original = [1, 1, 2, 2, 2, 3, 4, 4, 5]

    # Before
    ax1 = axes[0]
    ax1.set_xlim(-0.5, len(original) + 1)
    ax1.set_ylim(-0.5, 2)
    ax1.axis('off')
    ax1.set_title('Before: Array with duplicates', fontsize=12)

    for i, v in enumerate(original):
        color = '#E74C3C' if i > 0 and original[i] == original[i-1] else '#3498DB'
        rect = patches.FancyBboxPatch((i - 0.35, 0.3), 0.7, 0.8,
                                      boxstyle="round,pad=0.05", facecolor=color,
                                      edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(i, 0.7, str(v), ha='center', va='center', fontsize=14, fontweight='bold')

    ax1.text(len(original) + 0.5, 0.7, '← duplicates in red', fontsize=10, va='center')

    # After with pointers shown
    ax2 = axes[1]
    ax2.set_xlim(-0.5, len(original) + 1)
    ax2.set_ylim(-1, 3)
    ax2.axis('off')
    ax2.set_title('After: Slow pointer marks write position, Fast pointer scans', fontsize=12)

    result = [1, 2, 3, 4, 5]
    slow, fast = 4, 8  # Final positions

    for i, v in enumerate(original):
        if i <= slow:
            color = '#27AE60'  # Unique values
        else:
            color = '#BDC3C7'  # Will be overwritten
        rect = patches.FancyBboxPatch((i - 0.35, 0.3), 0.7, 0.8,
                                      boxstyle="round,pad=0.05", facecolor=color,
                                      edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)

        # Show result values for unique positions
        if i < len(result):
            ax2.text(i, 0.7, str(result[i]), ha='center', va='center', fontsize=14, fontweight='bold')
        else:
            ax2.text(i, 0.7, '?', ha='center', va='center', fontsize=14, color='gray')

    # Show pointers
    ax2.annotate('Slow', (slow, 1.4), ha='center', fontsize=12, fontweight='bold', color='#2980B9')
    ax2.annotate('Fast', (fast, 1.4), ha='center', fontsize=12, fontweight='bold', color='#E67E22')
    ax2.annotate('', xy=(slow, 1.15), xytext=(slow, 1.6), arrowprops=dict(arrowstyle='->', color='#2980B9', lw=2))
    ax2.annotate('', xy=(fast, 1.15), xytext=(fast, 1.6), arrowprops=dict(arrowstyle='->', color='#E67E22', lw=2))

    ax2.text(4, -0.5, 'Result: 5 unique elements [1, 2, 3, 4, 5]', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='#27AE60', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fast_slow_pointers.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/fast_slow_pointers.png")


def generate_container_water():
    """Generate container with most water visualization."""
    heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Container With Most Water\nArea = min(height[L], height[R]) × (R - L)',
                 fontsize=14, fontweight='bold')

    # Draw bars
    bars = ax.bar(range(len(heights)), heights, color='#3498DB', edgecolor='black', linewidth=1.5, width=0.4)

    # Highlight optimal container (indices 1 and 8)
    bars[1].set_color('#E74C3C')
    bars[8].set_color('#E74C3C')

    # Draw water area
    water = patches.Rectangle((1.2, 0), 6.6, 7, facecolor='#85C1E9', alpha=0.5, edgecolor='#2980B9', linewidth=2)
    ax.add_patch(water)

    # Draw height labels
    for i, h in enumerate(heights):
        ax.text(i, h + 0.3, str(h), ha='center', fontsize=11, fontweight='bold')

    # Draw pointers
    ax.annotate('L', (1, heights[1] + 1.5), ha='center', fontsize=14, fontweight='bold', color='#E74C3C')
    ax.annotate('R', (8, heights[8] + 1.5), ha='center', fontsize=14, fontweight='bold', color='#E74C3C')

    # Area calculation
    ax.text(4.5, 3.5, f'Area = min(8, 7) × 7 = 49', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

    # Explanation
    ax.text(4.5, -1.5, 'Move the shorter pointer inward to potentially find larger container',
           ha='center', fontsize=11, style='italic')

    ax.set_ylim(-2.5, 12)
    ax.set_xticks(range(len(heights)))
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Height', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/container_water.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/container_water.png")


def generate_three_sum():
    """Generate 3Sum visualization."""
    arr = [-4, -1, -1, 0, 1, 2]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('3Sum: Find triplets that sum to 0\nFix one pointer (i), use two pointers (j, k) for remaining',
                 fontsize=14, fontweight='bold')

    ax.set_xlim(-1, len(arr) + 2)
    ax.set_ylim(-1, 4)
    ax.axis('off')

    # Draw array
    for idx, v in enumerate(arr):
        rect = patches.FancyBboxPatch((idx - 0.35, 1.5), 0.7, 0.8,
                                      boxstyle="round,pad=0.05", facecolor='#ECF0F1',
                                      edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(idx, 1.9, str(v), ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(idx, 1.2, f'[{idx}]', ha='center', fontsize=9, color='gray')

    # Show one valid triplet: i=1 (-1), j=2 (-1), k=5 (2)
    # Actually: i=0 (-4), doesn't work well. Let's show i=3 (0), j=1 (-1), k=4 (1)
    # Better: i=1 (-1), j=4 (1), k=5 (2) but that's not 0
    # Let's show: i=2 (-1), j=3 (0), k=4 (1) → -1+0+1=0

    i, j, k = 2, 3, 4

    # Highlight the triplet
    for idx in [i, j, k]:
        rect = patches.FancyBboxPatch((idx - 0.35, 1.5), 0.7, 0.8,
                                      boxstyle="round,pad=0.05", facecolor='#27AE60',
                                      edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(idx, 1.9, str(arr[idx]), ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Pointer labels
    ax.annotate('i (fixed)', (i, 2.6), ha='center', fontsize=12, fontweight='bold', color='#E74C3C')
    ax.annotate('j', (j, 2.6), ha='center', fontsize=12, fontweight='bold', color='#3498DB')
    ax.annotate('k', (k, 2.6), ha='center', fontsize=12, fontweight='bold', color='#9B59B6')

    # Arrows
    ax.annotate('', xy=(i, 2.35), xytext=(i, 2.8), arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
    ax.annotate('', xy=(j, 2.35), xytext=(j, 2.8), arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2))
    ax.annotate('', xy=(k, 2.35), xytext=(k, 2.8), arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=2))

    # Sum calculation
    ax.text(3, 0.5, f'arr[{i}] + arr[{j}] + arr[{k}] = {arr[i]} + {arr[j]} + {arr[k]} = 0 ✓',
           ha='center', fontsize=13, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

    # Algorithm explanation
    ax.text(3, -0.3, 'For each i: use two pointers j and k to find pairs that sum to -arr[i]',
           ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/three_sum.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/three_sum.png")


if __name__ == '__main__':
    generate_converging_pointers()
    generate_fast_slow_pointers()
    generate_container_water()
    generate_three_sum()
    print("\n✅ All two pointer visualizations generated!")
