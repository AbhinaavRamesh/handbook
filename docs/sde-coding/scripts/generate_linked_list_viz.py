#!/usr/bin/env python3
"""Generate linked list visualizations for the tutorial site."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/linked-lists'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_node(ax, x, y, value, color='#3498DB', width=0.8, height=0.5):
    """Draw a linked list node."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05", facecolor=color,
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, str(value), ha='center', va='center', fontsize=14, fontweight='bold',
           color='white' if color != '#ECF0F1' else 'black')


def draw_arrow(ax, start, end, color='black', style='-'):
    """Draw arrow between nodes."""
    ax.annotate('', xy=end, xytext=start,
               arrowprops=dict(arrowstyle='->', color=color, lw=2, linestyle=style,
                              connectionstyle='arc3,rad=0'))


def generate_list_reversal():
    """Generate linked list reversal step-by-step visualization."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('Linked List Reversal: Step by Step', fontsize=16, fontweight='bold')

    # States: (list_values, arrows_reversed_up_to, prev_pos, curr_pos)
    steps = [
        ("Initial", [1, 2, 3, 4], [], None, 0),
        ("Step 1: Reverse 1→2", [1, 2, 3, 4], [0], 0, 1),
        ("Step 2: Reverse 2→3", [1, 2, 3, 4], [0, 1], 1, 2),
        ("Step 3: Reverse 3→4", [1, 2, 3, 4], [0, 1, 2], 2, 3),
    ]

    for idx, (title, values, reversed_arrows, prev, curr) in enumerate(steps):
        ax = axes[idx]
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 2)
        ax.axis('off')
        ax.set_title(title, fontsize=12)

        # Draw nodes
        positions = [(i * 1.5 + 0.5, 0.5) for i in range(len(values))]

        for i, (pos, val) in enumerate(zip(positions, values)):
            if prev is not None and i == prev:
                color = '#E74C3C'  # prev
            elif curr is not None and i == curr:
                color = '#27AE60'  # curr
            else:
                color = '#3498DB'
            draw_node(ax, pos[0], pos[1], val, color)

        # Draw arrows
        for i in range(len(values) - 1):
            start = (positions[i][0] + 0.4, positions[i][1])
            end = (positions[i+1][0] - 0.4, positions[i+1][1])

            if i in reversed_arrows:
                # Draw reversed arrow
                draw_arrow(ax, end, start, color='#27AE60')
            else:
                # Normal forward arrow
                draw_arrow(ax, start, end)

        # Draw NULL
        ax.text(positions[-1][0] + 1.2, positions[-1][1], 'NULL', fontsize=10, va='center')
        if reversed_arrows:
            ax.text(positions[0][0] - 1.2, positions[0][1], 'NULL', fontsize=10, va='center', ha='right')

        # Legend
        if idx == 0:
            ax.text(6, 1.5, 'prev', color='#E74C3C', fontsize=10, fontweight='bold')
            ax.text(6.8, 1.5, 'curr', color='#27AE60', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/list_reversal.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/list_reversal.png")


def generate_cycle_detection():
    """Generate Floyd's cycle detection visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Floyd's Cycle Detection (Tortoise & Hare)", fontsize=16, fontweight='bold')

    # List with cycle: 1 -> 2 -> 3 -> 4 -> 5 -> 3 (back to 3)
    # Ax1: Structure
    ax1 = axes[0]
    ax1.set_xlim(-1, 10)
    ax1.set_ylim(-1, 3)
    ax1.axis('off')
    ax1.set_title('Linked List with Cycle', fontsize=12)

    positions = [(0, 1), (1.5, 1), (3, 1), (4.5, 1.8), (6, 1), (4.5, 0.2)]
    values = [1, 2, 3, 4, 5, '→3']

    for i, ((x, y), val) in enumerate(zip(positions, values)):
        if i < 5:
            draw_node(ax1, x, y, val)

    # Draw arrows
    for i in range(4):
        start = (positions[i][0] + 0.4, positions[i][1])
        end = (positions[i+1][0] - 0.4, positions[i+1][1])
        draw_arrow(ax1, start, end)

    # Cycle arrow (5 -> 3)
    ax1.annotate('', xy=(3.4, 1), xytext=(6.4, 1),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2,
                               connectionstyle='arc3,rad=-0.5'))
    ax1.text(5, -0.3, 'Cycle back to node 3', fontsize=10, color='#E74C3C', ha='center')

    # Ax2: Detection steps
    ax2 = axes[1]
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(-0.5, 4)
    ax2.axis('off')
    ax2.set_title('Detection: Slow (🐢) moves 1 step, Fast (🐇) moves 2 steps', fontsize=12)

    # Show meeting point
    steps_text = """
Step 1: slow=1, fast=2
Step 2: slow=2, fast=4
Step 3: slow=3, fast=3 ← MEET! Cycle detected!

After meeting: Reset slow to head,
move both by 1 step to find cycle start.
"""
    ax2.text(0, 3.5, steps_text, fontsize=11, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#E8F6F3'))

    # Draw simplified diagram
    for i, ((x, y), val) in enumerate(zip([(0, 1), (1.5, 1), (3, 1), (4.5, 1), (6, 1)], [1, 2, 3, 4, 5])):
        color = '#27AE60' if val == 3 else '#3498DB'  # Highlight meeting point
        draw_node(ax2, x, y, val, color)

    ax2.text(3, 0.2, '↑ Meeting Point', fontsize=10, ha='center', color='#27AE60', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cycle_detection.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/cycle_detection.png")


def generate_merge_lists():
    """Generate merge two sorted lists visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 12)
    ax.set_ylim(-0.5, 5)
    ax.axis('off')
    ax.set_title('Merge Two Sorted Linked Lists', fontsize=16, fontweight='bold')

    # List 1: 1 -> 3 -> 5
    y1 = 4
    for i, (x, val) in enumerate([(0, 1), (1.5, 3), (3, 5)]):
        draw_node(ax, x, y1, val, '#3498DB')
        if i < 2:
            draw_arrow(ax, (x + 0.4, y1), (x + 1.1, y1))
    ax.text(-0.8, y1, 'L1:', fontsize=12, fontweight='bold', va='center')

    # List 2: 2 -> 4 -> 6
    y2 = 2.5
    for i, (x, val) in enumerate([(0, 2), (1.5, 4), (3, 6)]):
        draw_node(ax, x, y2, val, '#E74C3C')
        if i < 2:
            draw_arrow(ax, (x + 0.4, y2), (x + 1.1, y2))
    ax.text(-0.8, y2, 'L2:', fontsize=12, fontweight='bold', va='center')

    # Merged result: 1 -> 2 -> 3 -> 4 -> 5 -> 6
    y3 = 0.5
    merged = [1, 2, 3, 4, 5, 6]
    colors = ['#3498DB', '#E74C3C', '#3498DB', '#E74C3C', '#3498DB', '#E74C3C']

    for i, (val, color) in enumerate(zip(merged, colors)):
        x = i * 1.5 + 3
        draw_node(ax, x, y3, val, color)
        if i < len(merged) - 1:
            draw_arrow(ax, (x + 0.4, y3), (x + 1.1, y3), color='#27AE60')

    ax.text(2.2, y3, 'Result:', fontsize=12, fontweight='bold', va='center')

    # Algorithm note
    ax.text(6, 4.5, 'Compare heads, take smaller, advance that pointer',
           fontsize=11, style='italic', ha='center')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/merge_lists.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/merge_lists.png")


def generate_lru_cache():
    """Generate LRU Cache structure visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    ax.set_title('LRU Cache: Doubly Linked List + HashMap', fontsize=16, fontweight='bold')

    # Doubly linked list (horizontal)
    # Most recent -----> Least recent
    items = [('A', 1), ('B', 2), ('C', 3)]
    y = 2.5

    ax.text(-0.5, y, 'HEAD\n(MRU)', fontsize=10, ha='center', va='center', color='#27AE60')
    ax.text(10, y, 'TAIL\n(LRU)', fontsize=10, ha='center', va='center', color='#E74C3C')

    for i, (key, val) in enumerate(items):
        x = 2 + i * 3
        # Draw node (key:value)
        rect = FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor='#AED6F1' if i == 0 else '#F5B7B1' if i == 2 else '#FCF3CF',
                              edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, f'{key}:{val}', ha='center', va='center', fontsize=12, fontweight='bold')

        # Draw bidirectional arrows
        if i < len(items) - 1:
            next_x = x + 3
            ax.annotate('', xy=(next_x - 0.9, y + 0.15), xytext=(x + 0.9, y + 0.15),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
            ax.annotate('', xy=(x + 0.9, y - 0.15), xytext=(next_x - 0.9, y - 0.15),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # HashMap below
    ax.text(5, 0.5, 'HashMap: { A → node_A, B → node_B, C → node_C }', fontsize=11, ha='center',
           bbox=dict(boxstyle='round', facecolor='#D5D8DC'))

    # Operations
    ops_text = """
get(key):
  - If in map: move to head (MRU), return value
  - Else: return -1

put(key, value):
  - If in map: update, move to head
  - Else: add to head, evict tail if capacity exceeded
"""
    ax.text(11, 3.5, ops_text, fontsize=10, va='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='#E8F8F5'))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lru_cache.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/lru_cache.png")


if __name__ == '__main__':
    generate_list_reversal()
    generate_cycle_detection()
    generate_merge_lists()
    generate_lru_cache()
    print("\n✅ All linked list visualizations generated!")
