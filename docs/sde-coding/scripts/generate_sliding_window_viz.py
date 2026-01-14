#!/usr/bin/env python3
"""Generate sliding window pattern visualizations for the tutorial site."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/sliding-window'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_fixed_window():
    """Generate fixed-size sliding window visualization."""
    arr = [2, 1, 5, 1, 3, 2]
    k = 3

    fig, axes = plt.subplots(len(arr) - k + 1, 1, figsize=(10, 10))
    fig.suptitle(f'Fixed Sliding Window (k={k}): Maximum Sum Subarray', fontsize=14, fontweight='bold')

    sums = [sum(arr[i:i+k]) for i in range(len(arr) - k + 1)]
    max_sum = max(sums)
    max_idx = sums.index(max_sum)

    for i in range(len(arr) - k + 1):
        ax = axes[i]
        colors = ['#BDC3C7'] * len(arr)

        # Highlight window
        for j in range(i, i + k):
            colors[j] = '#27AE60' if i == max_idx else '#3498DB'

        bars = ax.bar(range(len(arr)), arr, color=colors, edgecolor='black', linewidth=2)

        # Add value labels
        for j, v in enumerate(arr):
            ax.text(j, v + 0.15, str(v), ha='center', fontsize=12, fontweight='bold')

        # Draw window border
        rect = patches.Rectangle((i - 0.4, 0), k - 0.2, max(arr) + 0.5,
                                  linewidth=3, edgecolor='#2C3E50', facecolor='none', linestyle='--')
        ax.add_patch(rect)

        current_sum = sums[i]
        status = "← MAX!" if i == max_idx else ""
        ax.set_title(f'Window [{i}:{i+k}] → Sum = {current_sum} {status}', fontsize=11,
                    color='#27AE60' if i == max_idx else 'black', fontweight='bold' if i == max_idx else 'normal')
        ax.set_ylim(0, max(arr) + 1.5)
        ax.set_xticks(range(len(arr)))
        ax.set_xticklabels([f'[{j}]' for j in range(len(arr))])
        ax.set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fixed_window.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/fixed_window.png")


def generate_variable_window():
    """Generate variable-size sliding window visualization."""
    s = "ADOBECODEBANC"
    t = "ABC"

    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle(f'Variable Sliding Window: Minimum Window Substring\nFind minimum window in "{s}" containing all of "{t}"',
                 fontsize=13, fontweight='bold')

    # Key steps in the algorithm
    steps = [
        (0, 5, "ADOBEC", "Found all chars, try shrinking"),
        (3, 5, "BEC", "Missing 'A', expand right"),
        (3, 10, "BECODEBA", "Found all chars, try shrinking"),
        (5, 10, "ODEBA", "Missing 'C', expand right"),
        (9, 12, "BANC", "Found all chars - this is minimum!"),
    ]

    for idx, (left, right, window, desc) in enumerate(steps):
        ax = axes[idx]
        ax.set_xlim(-0.5, len(s) + 0.5)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')

        # Draw each character
        for i, c in enumerate(s):
            if left <= i <= right:
                if c in t:
                    color = '#27AE60'  # Target char in window
                else:
                    color = '#3498DB'  # Non-target in window
            else:
                color = '#ECF0F1'  # Outside window

            rect = patches.FancyBboxPatch((i - 0.35, 0.2), 0.7, 0.7,
                                          boxstyle="round,pad=0.05", facecolor=color,
                                          edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(i, 0.55, c, ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(i, 0.05, f'{i}', ha='center', fontsize=8, color='gray')

        # Draw pointers
        ax.annotate('L', (left, 1.05), ha='center', fontsize=14, fontweight='bold', color='#E74C3C')
        ax.annotate('R', (right, 1.05), ha='center', fontsize=14, fontweight='bold', color='#27AE60')

        # Draw arrows
        ax.annotate('', xy=(left, 0.95), xytext=(left, 1.2),
                   arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
        ax.annotate('', xy=(right, 0.95), xytext=(right, 1.2),
                   arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))

        is_final = idx == len(steps) - 1
        ax.set_title(f'Window: "{window}" (len={len(window)}) - {desc}', fontsize=10,
                    color='#27AE60' if is_final else 'black', fontweight='bold' if is_final else 'normal')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/variable_window.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/variable_window.png")


def generate_window_comparison():
    """Generate sliding window vs two pointer comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Sliding Window vs Two Pointers', fontsize=16, fontweight='bold')

    # Sliding Window - both pointers move same direction
    ax1 = axes[0]
    ax1.set_xlim(-1, 10)
    ax1.set_ylim(-1, 3)
    ax1.axis('off')
    ax1.set_title('Sliding Window\nBoth pointers move →', fontsize=12)

    for i in range(8):
        rect = patches.FancyBboxPatch((i, 1), 0.8, 0.8, boxstyle="round,pad=0.02",
                                      facecolor='#3498DB' if 2 <= i <= 4 else '#ECF0F1',
                                      edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(i + 0.4, 1.4, str(i), ha='center', fontsize=12)

    ax1.annotate('L', (2.4, 2.2), fontsize=14, fontweight='bold', color='#E74C3C', ha='center')
    ax1.annotate('R', (4.4, 2.2), fontsize=14, fontweight='bold', color='#27AE60', ha='center')
    ax1.annotate('', xy=(6, 0.5), xytext=(1, 0.5), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(3.5, 0.2, 'Both move right →', fontsize=10, ha='center')

    # Two Pointers - converging from ends
    ax2 = axes[1]
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(-1, 3)
    ax2.axis('off')
    ax2.set_title('Two Pointers (Converging)\nPointers move toward each other', fontsize=12)

    for i in range(8):
        rect = patches.FancyBboxPatch((i, 1), 0.8, 0.8, boxstyle="round,pad=0.02",
                                      facecolor='#E74C3C' if i == 1 else '#27AE60' if i == 6 else '#ECF0F1',
                                      edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(i + 0.4, 1.4, str(i), ha='center', fontsize=12)

    ax2.annotate('L', (1.4, 2.2), fontsize=14, fontweight='bold', color='#E74C3C', ha='center')
    ax2.annotate('R', (6.4, 2.2), fontsize=14, fontweight='bold', color='#27AE60', ha='center')
    ax2.annotate('', xy=(3.5, 0.5), xytext=(1, 0.5), arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
    ax2.annotate('', xy=(4.5, 0.5), xytext=(7, 0.5), arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
    ax2.text(4, 0.2, '← converge →', fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/window_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/window_comparison.png")


def generate_longest_substring():
    """Generate longest substring without repeating chars visualization."""
    s = "abcabcbb"

    fig, axes = plt.subplots(6, 1, figsize=(12, 14))
    fig.suptitle('Longest Substring Without Repeating Characters\nInput: "abcabcbb"',
                 fontsize=14, fontweight='bold')

    steps = [
        (0, 0, {'a'}, "Start: add 'a'"),
        (0, 1, {'a', 'b'}, "Expand: add 'b'"),
        (0, 2, {'a', 'b', 'c'}, "Expand: add 'c' ← Best so far!"),
        (0, 3, {'a', 'b', 'c'}, "Found 'a' duplicate, shrink left"),
        (1, 3, {'b', 'c', 'a'}, "After shrink, window valid"),
        (3, 5, {'a', 'b', 'c'}, "Continue... final answer = 3"),
    ]

    for idx, (left, right, seen, desc) in enumerate(steps):
        ax = axes[idx]
        ax.set_xlim(-0.5, len(s) + 2)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')

        for i, c in enumerate(s):
            if left <= i <= right:
                color = '#27AE60'
            else:
                color = '#ECF0F1'

            rect = patches.FancyBboxPatch((i - 0.35, 0.3), 0.7, 0.7,
                                          boxstyle="round,pad=0.05", facecolor=color,
                                          edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(i, 0.65, c, ha='center', va='center', fontsize=14, fontweight='bold')

        # Show seen set
        ax.text(len(s) + 0.5, 0.65, f'seen: {seen}', fontsize=10, va='center')

        is_best = "Best" in desc
        ax.set_title(f'Step {idx + 1}: {desc}', fontsize=10,
                    color='#27AE60' if is_best else 'black', fontweight='bold' if is_best else 'normal')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/longest_substring.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/longest_substring.png")


if __name__ == '__main__':
    generate_fixed_window()
    generate_variable_window()
    generate_window_comparison()
    generate_longest_substring()
    print("\n✅ All sliding window visualizations generated!")
