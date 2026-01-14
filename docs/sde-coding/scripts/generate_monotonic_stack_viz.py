#!/usr/bin/env python3
"""Generate monotonic stack visualizations for the tutorial site."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/monotonic-stack'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_next_greater_element():
    """Generate Next Greater Element visualization."""
    arr = [4, 5, 2, 10, 8]
    nge = [5, 10, 10, -1, -1]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-1, 12)
    ax.set_title('Next Greater Element: Using Monotonic Decreasing Stack', fontsize=14, fontweight='bold')

    # Draw bars
    bar_width = 0.6
    bars = ax.bar(range(len(arr)), arr, width=bar_width, color='#3498DB', edgecolor='black', linewidth=2)

    # Add value labels
    for i, v in enumerate(arr):
        ax.text(i, v + 0.4, str(v), ha='center', fontsize=14, fontweight='bold')
        ax.text(i, -0.8, f'[{i}]', ha='center', fontsize=10, color='gray')

    # Draw NGE arrows
    for i, (val, next_val) in enumerate(zip(arr, nge)):
        if next_val != -1:
            # Find index of next greater
            for j in range(i + 1, len(arr)):
                if arr[j] == next_val:
                    # Draw curved arrow
                    ax.annotate('', xy=(j, arr[j] + 1.5), xytext=(i, val + 1.5),
                               arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2,
                                              connectionstyle='arc3,rad=-0.3'))
                    break

    # Results table
    ax.text(6, 10, 'Result:', fontsize=12, fontweight='bold')
    for i, (v, n) in enumerate(zip(arr, nge)):
        ax.text(6, 9 - i * 0.8, f'NGE[{v}] = {n}', fontsize=11,
               family='monospace')

    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xticks(range(len(arr)))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/next_greater_element.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/next_greater_element.png")


def generate_daily_temperatures():
    """Generate Daily Temperatures visualization."""
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    result = [1, 1, 4, 2, 1, 1, 0, 0]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Daily Temperatures: Days Until Warmer Temperature', fontsize=14, fontweight='bold')

    # Draw temperature bars
    colors = ['#3498DB'] * len(temps)
    colors[2] = '#27AE60'  # 75 -> 76 takes 4 days
    colors[6] = '#E74C3C'  # 76 has no warmer day

    bars = ax.bar(range(len(temps)), temps, color=colors, edgecolor='black', linewidth=2)

    # Add temperature labels
    for i, t in enumerate(temps):
        ax.text(i, t + 1, f'{t}°', ha='center', fontsize=11, fontweight='bold')
        ax.text(i, 65, f'Day {i}', ha='center', fontsize=9, rotation=45)

    # Draw arrows to next warmer day
    arrows = [(0, 1), (1, 2), (2, 6), (3, 5), (4, 5), (5, 6)]
    for start, end in arrows:
        ax.annotate('', xy=(end, temps[end] + 3), xytext=(start, temps[start] + 3),
                   arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1.5,
                                  connectionstyle='arc3,rad=-0.2'))

    # Show result array
    ax.text(0, 60, f'Result: {result}', fontsize=12, family='monospace',
           bbox=dict(boxstyle='round', facecolor='#FCF3CF'))

    ax.set_ylim(55, 85)
    ax.set_ylabel('Temperature', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/daily_temperatures.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/daily_temperatures.png")


def generate_stack_evolution():
    """Generate stack state evolution visualization."""
    arr = [2, 1, 5, 6, 2, 3]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title('Monotonic Decreasing Stack: State Evolution', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Simulate stack evolution
    steps = []
    stack = []
    nge = [-1] * len(arr)

    for i in range(len(arr) - 1, -1, -1):
        step_stack = stack.copy()
        popped = []

        while stack and arr[stack[-1]] <= arr[i]:
            popped.append(stack.pop())

        if stack:
            nge[i] = arr[stack[-1]]

        stack.append(i)
        steps.append((i, arr[i], step_stack.copy(), popped, stack.copy(), nge[i]))

    # Draw grid
    cell_width = 1.8
    cell_height = 0.8
    start_x = 0.5
    start_y = 8

    # Headers
    headers = ['Index', 'Value', 'Stack Before', 'Popped', 'Stack After', 'NGE']
    for j, h in enumerate(headers):
        ax.text(start_x + j * cell_width + cell_width/2, start_y + 0.5, h,
               ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw steps (reversed to show left-to-right processing)
    for row, (i, val, before, popped, after, nge_val) in enumerate(reversed(steps)):
        y = start_y - (row + 1) * cell_height

        cells = [
            str(i),
            str(val),
            str([arr[x] for x in before]) if before else '[]',
            str([arr[x] for x in popped]) if popped else '[]',
            str([arr[x] for x in after]),
            str(nge_val)
        ]

        for j, cell in enumerate(cells):
            x = start_x + j * cell_width

            # Color coding
            if j == 3 and popped:  # Popped - red
                color = '#FADBD8'
            elif j == 4:  # After - green
                color = '#D5F5E3'
            else:
                color = '#EBF5FB'

            rect = patches.FancyBboxPatch((x, y - cell_height/2), cell_width - 0.1, cell_height - 0.1,
                                          boxstyle="round,pad=0.02", facecolor=color,
                                          edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + cell_width/2, y, cell, ha='center', va='center', fontsize=9)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)

    # Legend
    ax.text(0.5, 1, 'Process right to left, maintaining decreasing stack', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/stack_evolution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/stack_evolution.png")


def generate_largest_rectangle():
    """Generate Largest Rectangle in Histogram visualization."""
    heights = [2, 1, 5, 6, 2, 3]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Largest Rectangle in Histogram', fontsize=14, fontweight='bold')

    # Draw histogram bars
    bars = ax.bar(range(len(heights)), heights, color='#3498DB', edgecolor='black', linewidth=2, width=0.8)

    # Highlight the largest rectangle (indices 2-3, height 5, area = 10)
    rect = patches.Rectangle((1.6, 0), 2.8, 5, linewidth=3,
                             edgecolor='#E74C3C', facecolor='#F5B7B1', alpha=0.5)
    ax.add_patch(rect)

    # Add height labels
    for i, h in enumerate(heights):
        ax.text(i, h + 0.2, str(h), ha='center', fontsize=12, fontweight='bold')

    # Annotation
    ax.annotate('Largest Rectangle\nArea = 5 × 2 = 10', xy=(2.5, 2.5), xytext=(5, 4),
               fontsize=12, ha='center',
               arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2),
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Height', fontsize=12)
    ax.set_xticks(range(len(heights)))
    ax.set_ylim(0, 8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/largest_rectangle.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/largest_rectangle.png")


def generate_increasing_vs_decreasing():
    """Generate comparison of increasing vs decreasing monotonic stacks."""
    arr = [3, 1, 4, 1, 5, 9, 2, 6]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Monotonic Stack Types: What Each Finds', fontsize=14, fontweight='bold')

    for ax_idx, (ax, title, desc, color) in enumerate([
        (axes[0], 'Monotonic Increasing Stack', 'Finds Next Smaller Element', '#3498DB'),
        (axes[1], 'Monotonic Decreasing Stack', 'Finds Next Greater Element', '#E74C3C')
    ]):
        ax.set_xlim(-0.5, 9)
        ax.set_ylim(0, 12)
        ax.set_title(f'{title}\n{desc}', fontsize=11)

        # Draw bars
        ax.bar(range(len(arr)), arr, color=color, edgecolor='black', linewidth=1.5, alpha=0.7)

        # Add labels
        for i, v in enumerate(arr):
            ax.text(i, v + 0.3, str(v), ha='center', fontsize=11, fontweight='bold')

        # Show stack property
        if ax_idx == 0:
            ax.text(4, 11, 'Stack: [3] → [1] → [1,4] → [1,1] → ...', fontsize=9,
                   ha='center', bbox=dict(boxstyle='round', facecolor='#D6EAF8'))
            ax.text(4, 10, 'Pop when: current < stack.top()', fontsize=9, ha='center', style='italic')
        else:
            ax.text(4, 11, 'Stack: [3] → [3,1] → [4] → [4,1] → ...', fontsize=9,
                   ha='center', bbox=dict(boxstyle='round', facecolor='#FADBD8'))
            ax.text(4, 10, 'Pop when: current > stack.top()', fontsize=9, ha='center', style='italic')

        ax.set_xlabel('Index', fontsize=10)
        ax.set_xticks(range(len(arr)))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/increasing_vs_decreasing.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/increasing_vs_decreasing.png")


if __name__ == '__main__':
    generate_next_greater_element()
    generate_daily_temperatures()
    generate_stack_evolution()
    generate_largest_rectangle()
    generate_increasing_vs_decreasing()
    print("\n✅ All monotonic stack visualizations generated!")
