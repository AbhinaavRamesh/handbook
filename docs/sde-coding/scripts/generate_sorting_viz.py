#!/usr/bin/env python3
"""Generate sorting algorithm visualizations for the tutorial site."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/sorting'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme
COLORS = {
    'default': '#4A90A4',
    'pivot': '#E74C3C',
    'comparing': '#F39C12',
    'sorted': '#27AE60',
    'swapping': '#9B59B6',
    'highlight': '#3498DB'
}

def generate_quicksort_steps():
    """Generate QuickSort partitioning visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QuickSort Partitioning Steps', fontsize=16, fontweight='bold')

    # Initial array and steps
    steps = [
        ([8, 3, 7, 1, 5, 9, 2, 6], "Initial array, pivot=6", {'pivot': 7}),
        ([3, 1, 5, 2, 6, 9, 8, 7], "After first partition", {'sorted': [4], 'pivot': None}),
        ([2, 1, 3, 5, 6, 9, 8, 7], "Left partition, pivot=3", {'sorted': [2, 4], 'pivot': None}),
        ([1, 2, 3, 5, 6, 9, 8, 7], "Left-left partition", {'sorted': [0, 1, 2, 4]}),
        ([1, 2, 3, 5, 6, 7, 8, 9], "Right partition, pivot=7", {'sorted': [0, 1, 2, 4, 5, 6, 7]}),
        ([1, 2, 3, 5, 6, 7, 8, 9], "Fully sorted!", {'sorted': list(range(8))}),
    ]

    for idx, (arr, title, highlights) in enumerate(steps):
        ax = axes[idx // 3, idx % 3]
        colors = [COLORS['default']] * len(arr)

        if 'pivot' in highlights and highlights['pivot'] is not None:
            colors[highlights['pivot']] = COLORS['pivot']
        if 'sorted' in highlights:
            for i in highlights['sorted']:
                colors[i] = COLORS['sorted']

        bars = ax.bar(range(len(arr)), arr, color=colors, edgecolor='black', linewidth=1.5)

        for i, v in enumerate(arr):
            ax.text(i, v + 0.3, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 12)
        ax.set_xticks(range(len(arr)))
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=COLORS['pivot'], label='Pivot'),
        patches.Patch(facecolor=COLORS['sorted'], label='Sorted'),
        patches.Patch(facecolor=COLORS['default'], label='Unsorted'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{OUTPUT_DIR}/quicksort_steps.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/quicksort_steps.png")


def generate_mergesort_tree():
    """Generate MergeSort divide and conquer tree."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    ax.set_title('MergeSort: Divide and Conquer', fontsize=16, fontweight='bold', pad=20)

    def draw_array(ax, x, y, arr, color='lightblue'):
        width = len(arr) * 0.6
        box = patches.FancyBboxPatch((x - width/2, y - 0.3), width, 0.6,
                                      boxstyle="round,pad=0.02", facecolor=color,
                                      edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        for i, v in enumerate(arr):
            ax.text(x - width/2 + 0.3 + i*0.6, y, str(v), ha='center', va='center',
                   fontsize=10, fontweight='bold')

    # Divide phase (top-down)
    draw_array(ax, 8, 8, [38, 27, 43, 3, 9, 82, 10], '#AED6F1')
    ax.text(8, 8.6, 'Original', ha='center', fontsize=9, style='italic')

    # Level 1
    draw_array(ax, 4, 6, [38, 27, 43, 3], '#85C1E9')
    draw_array(ax, 12, 6, [9, 82, 10], '#85C1E9')
    ax.annotate('', xy=(4, 6.4), xytext=(8, 7.6), arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('', xy=(12, 6.4), xytext=(8, 7.6), arrowprops=dict(arrowstyle='->', color='gray'))

    # Level 2
    draw_array(ax, 2, 4, [38, 27], '#5DADE2')
    draw_array(ax, 6, 4, [43, 3], '#5DADE2')
    draw_array(ax, 10, 4, [9, 82], '#5DADE2')
    draw_array(ax, 14, 4, [10], '#5DADE2')

    # Level 3 (single elements)
    for i, (x, v) in enumerate([(1, 38), (3, 27), (5, 43), (7, 3), (9, 9), (11, 82), (13, 10)]):
        draw_array(ax, x, 2, [v], '#3498DB')

    # Merge phase (bottom-up) - show with green
    ax.text(8, 0.5, '↑ MERGE PHASE ↑', ha='center', fontsize=12, fontweight='bold', color='green')

    # Draw merge arrows
    ax.annotate('', xy=(2, 2.4), xytext=(2, 3.6), arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=(6, 2.4), xytext=(6, 3.6), arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Labels
    ax.text(-0.5, 8, 'DIVIDE', fontsize=11, fontweight='bold', rotation=90, va='center', color='#2980B9')
    ax.text(-0.5, 2, 'CONQUER', fontsize=11, fontweight='bold', rotation=90, va='center', color='#27AE60')

    plt.savefig(f'{OUTPUT_DIR}/mergesort_tree.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/mergesort_tree.png")


def generate_sorting_comparison():
    """Generate sorting algorithm comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Sorting Algorithm Comparison', fontsize=16, fontweight='bold')

    # Left: Time complexity comparison
    sizes = [10, 50, 100, 500, 1000]

    # Theoretical operations
    bubble = [n**2 for n in sizes]
    merge = [n * np.log2(n) for n in sizes]

    ax1.plot(sizes, bubble, 'o-', label='Bubble Sort O(n²)', color='#E74C3C', linewidth=2, markersize=8)
    ax1.plot(sizes, merge, 's-', label='Merge Sort O(n log n)', color='#27AE60', linewidth=2, markersize=8)
    ax1.set_xlabel('Input Size (n)', fontsize=12)
    ax1.set_ylabel('Operations', fontsize=12)
    ax1.set_title('Time Complexity Growth', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Right: Bar chart comparison
    algorithms = ['Bubble\nO(n²)', 'Selection\nO(n²)', 'Insertion\nO(n²)',
                  'Merge\nO(n log n)', 'Quick\nO(n log n)', 'Heap\nO(n log n)']
    # Relative speed (lower is faster)
    speeds = [100, 90, 70, 15, 12, 18]
    colors = ['#E74C3C', '#E74C3C', '#F39C12', '#27AE60', '#27AE60', '#27AE60']

    bars = ax2.bar(algorithms, speeds, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Relative Time (lower = faster)', fontsize=12)
    ax2.set_title('Algorithm Speed Comparison (n=10,000)', fontsize=12)
    ax2.set_ylim(0, 120)

    # Add speed labels
    for bar, speed in zip(bars, speeds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{speed}x', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sorting_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/sorting_comparison.png")


def generate_stability():
    """Generate sorting stability visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Sorting Stability: Preserving Relative Order of Equal Keys', fontsize=14, fontweight='bold')

    # Cards with same value but different suits (colors)
    # Format: (value, color_name, color_code)
    original = [(3, 'red', '#E74C3C'), (1, 'blue', '#3498DB'), (3, 'green', '#27AE60'),
                (2, 'purple', '#9B59B6'), (1, 'orange', '#F39C12')]

    stable_sorted = [(1, 'blue', '#3498DB'), (1, 'orange', '#F39C12'), (2, 'purple', '#9B59B6'),
                     (3, 'red', '#E74C3C'), (3, 'green', '#27AE60')]

    unstable_sorted = [(1, 'orange', '#F39C12'), (1, 'blue', '#3498DB'), (2, 'purple', '#9B59B6'),
                       (3, 'green', '#27AE60'), (3, 'red', '#E74C3C')]

    def draw_cards(ax, cards, title):
        ax.set_xlim(-0.5, len(cards) - 0.5)
        ax.set_ylim(-0.5, 2)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')

        for i, (val, name, color) in enumerate(cards):
            rect = patches.FancyBboxPatch((i - 0.35, 0.2), 0.7, 1.2,
                                          boxstyle="round,pad=0.05", facecolor=color,
                                          edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i, 0.8, str(val), ha='center', va='center', fontsize=20,
                   fontweight='bold', color='white')
            ax.text(i, 0.3, name, ha='center', va='center', fontsize=8, color='white')

    draw_cards(axes[0], original, 'Original Array')
    draw_cards(axes[1], stable_sorted, 'Stable Sort (MergeSort)\n✓ Blue 1 before Orange 1')
    draw_cards(axes[2], unstable_sorted, 'Unstable Sort (QuickSort)\n✗ Order of equal keys changed')

    # Add checkmark and X
    axes[1].text(2, 1.7, '✓', fontsize=30, color='green', ha='center')
    axes[2].text(2, 1.7, '✗', fontsize=30, color='red', ha='center')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sorting_stability.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/sorting_stability.png")


if __name__ == '__main__':
    generate_quicksort_steps()
    generate_mergesort_tree()
    generate_sorting_comparison()
    generate_stability()
    print("\n✅ All sorting visualizations generated!")
