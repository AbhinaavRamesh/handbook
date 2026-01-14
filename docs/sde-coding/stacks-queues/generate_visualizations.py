#!/usr/bin/env python3
"""
Generate visualization images for Stack & Queue problems.
Run this script to create all PNG files in the assets directory.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set up output directory
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def save_fig(name):
    """Save figure to assets directory."""
    filepath = os.path.join(ASSETS_DIR, f'{name}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Created: {filepath}")


def draw_stack(ax, elements, title="Stack", x_offset=0, colors=None):
    """Draw a stack visualization."""
    box_height = 0.6
    box_width = 1.2
    spacing = 0.1

    if colors is None:
        colors = ['#4CAF50'] * len(elements)

    for i, elem in enumerate(elements):
        y = i * (box_height + spacing)
        color = colors[i] if i < len(colors) else '#4CAF50'
        rect = patches.FancyBboxPatch(
            (x_offset, y), box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x_offset + box_width/2, y + box_height/2, str(elem),
                ha='center', va='center', fontsize=12, fontweight='bold')

    if elements:
        ax.annotate('TOP', xy=(x_offset + box_width + 0.1, (len(elements)-1) * (box_height + spacing) + box_height/2),
                    fontsize=10, color='red', fontweight='bold')

    ax.text(x_offset + box_width/2, -0.5, title, ha='center', fontsize=11, fontweight='bold')


def draw_queue(ax, elements, title="Queue", y_offset=0, colors=None):
    """Draw a queue visualization."""
    box_width = 0.8
    box_height = 0.6
    spacing = 0.1

    if colors is None:
        colors = ['#2196F3'] * len(elements)

    for i, elem in enumerate(elements):
        x = i * (box_width + spacing)
        color = colors[i] if i < len(colors) else '#2196F3'
        rect = patches.FancyBboxPatch(
            (x, y_offset), box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + box_width/2, y_offset + box_height/2, str(elem),
                ha='center', va='center', fontsize=12, fontweight='bold')

    if elements:
        ax.text(-0.3, y_offset + box_height/2, 'FRONT', fontsize=9, color='green', fontweight='bold', va='center')
        ax.text(len(elements) * (box_width + spacing) + 0.1, y_offset + box_height/2, 'REAR',
                fontsize=9, color='blue', fontweight='bold', va='center')

    ax.text(len(elements) * (box_width + spacing) / 2, y_offset - 0.4, title, ha='center', fontsize=11, fontweight='bold')


# 1. Valid Parentheses Visualization
def create_valid_parentheses():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle('Valid Parentheses: Stack States for "{[()]}"', fontsize=14, fontweight='bold')

    states = [
        (['{'], 'After "{"'),
        (['{', '['], 'After "{["'),
        (['{', '[', '('], 'After "{[("'),
        ([], 'Final (empty)')
    ]

    for ax, (stack, title) in zip(axes, states):
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        draw_stack(ax, stack, title)

    plt.tight_layout()
    save_fig('valid-parentheses')


# 2. Queue Using Stacks
def create_queue_using_stacks():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Queue Using Two Stacks', fontsize=14, fontweight='bold')

    # Row 1: Push operation
    operations = [
        ('Push 1, 2, 3', [1, 2, 3], []),
        ('Dequeue: Move all', [], [3, 2, 1]),
        ('Pop from stack2', [], [3, 2])
    ]

    for i, (title, s1, s2) in enumerate(operations):
        ax = axes[0, i]
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11)
        draw_stack(ax, s1, 'Stack1 (in)', x_offset=0, colors=['#4CAF50']*len(s1))
        draw_stack(ax, s2, 'Stack2 (out)', x_offset=2, colors=['#FF9800']*len(s2))

    # Row 2: Continue operations
    operations2 = [
        ('Result: 1', [], [3, 2]),
        ('Dequeue again', [], [3]),
        ('Result: 2', [], [3])
    ]

    for i, (title, s1, s2) in enumerate(operations2):
        ax = axes[1, i]
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11)
        draw_stack(ax, s1, 'Stack1 (in)', x_offset=0)
        draw_stack(ax, s2, 'Stack2 (out)', x_offset=2, colors=['#FF9800']*len(s2))

    plt.tight_layout()
    save_fig('queue-using-stacks')


# 3. Stack Using Queues
def create_stack_using_queues():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    fig.suptitle('Stack Using One Queue (Push Costly)', fontsize=14, fontweight='bold')

    states = [
        ([1], 'Push 1'),
        ([2, 1], 'Push 2: rotate'),
        ([3, 2, 1], 'Push 3: rotate'),
        ([2, 1], 'Pop: returns 3')
    ]

    for ax, (queue, title) in zip(axes, states):
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        draw_queue(ax, queue, title)

    plt.tight_layout()
    save_fig('stack-using-queues')


# 4. Min Stack
def create_min_stack():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle('Min Stack: Two Stack Approach', fontsize=14, fontweight='bold')

    states = [
        ([5], [5], 'Push 5'),
        ([5, 3], [5, 3], 'Push 3'),
        ([5, 3, 7], [5, 3], 'Push 7'),
        ([5, 3], [5, 3], 'Pop 7')
    ]

    for ax, (main, mins, title) in zip(axes, states):
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11)
        draw_stack(ax, main, 'Main Stack', x_offset=0)
        draw_stack(ax, mins, 'Min Stack', x_offset=2, colors=['#E91E63']*len(mins))

    plt.tight_layout()
    save_fig('min-stack')


# 5. Next Greater Element
def create_next_greater_element():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Next Greater Element: Monotonic Decreasing Stack', fontsize=14, fontweight='bold')

    # Input: [2, 1, 2, 4, 3]
    states = [
        ([0], 'i=0: push 0 (val=2)', [2, 1, 2, 4, 3], [-1, -1, -1, -1, -1]),
        ([0, 1], 'i=1: push 1 (val=1)', [2, 1, 2, 4, 3], [-1, -1, -1, -1, -1]),
        ([0, 2], 'i=2: pop 1, result[1]=2', [2, 1, 2, 4, 3], [-1, 2, -1, -1, -1]),
        ([3], 'i=3: pop all, push 3', [2, 1, 2, 4, 3], [4, 2, 4, -1, -1]),
        ([3, 4], 'i=4: push 4 (val=3)', [2, 1, 2, 4, 3], [4, 2, 4, -1, -1]),
        ([], 'Final Result', [2, 1, 2, 4, 3], [4, 2, 4, -1, -1])
    ]

    for idx, (stack, title, arr, result) in enumerate(states):
        ax = axes[idx // 3, idx % 3]
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10)

        # Draw array
        for i, (v, r) in enumerate(zip(arr, result)):
            color = '#4CAF50' if i in stack else '#E0E0E0'
            rect = patches.Rectangle((i * 0.8, 2.5), 0.7, 0.5, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(i * 0.8 + 0.35, 2.75, str(v), ha='center', va='center', fontsize=10)
            ax.text(i * 0.8 + 0.35, 2.2, str(r), ha='center', va='center', fontsize=9, color='blue')

        draw_stack(ax, [arr[i] for i in stack], 'Stack', x_offset=4.5)

    plt.tight_layout()
    save_fig('next-greater-element')


# 6. Daily Temperatures
def create_daily_temperatures():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title('Daily Temperatures: Days Until Warmer', fontsize=14, fontweight='bold')

    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    result = [1, 1, 4, 2, 1, 1, 0, 0]

    x = np.arange(len(temps))
    bars = ax.bar(x, temps, color=['#FF5722' if r > 0 else '#9E9E9E' for r in result], edgecolor='black')

    ax.set_xlabel('Day Index')
    ax.set_ylabel('Temperature')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Day {i}\n{t}' for i, t in enumerate(temps)])

    for i, (t, r) in enumerate(zip(temps, result)):
        ax.text(i, t + 1, f'Wait: {r}', ha='center', fontsize=9, fontweight='bold')
        if r > 0:
            ax.annotate('', xy=(i + r, temps[i + r] - 1), xytext=(i, t + 0.5),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax.set_ylim(65, 82)
    plt.tight_layout()
    save_fig('daily-temperatures')


# 7. Reverse Polish Notation
def create_reverse_polish_notation():
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle('Evaluate RPN: ["2", "1", "+", "3", "*"]', fontsize=14, fontweight='bold')

    states = [
        (['2'], 'Read "2"', 'Push 2'),
        (['2', '1'], 'Read "1"', 'Push 1'),
        (['3'], 'Read "+"', 'Pop 1,2; Push 2+1=3'),
        (['3', '3'], 'Read "3"', 'Push 3'),
        (['9'], 'Read "*"', 'Pop 3,3; Push 3*3=9'),
        (['9'], 'Done', 'Result = 9'),
    ]

    for idx, (stack, token, action) in enumerate(states):
        row = idx // 4
        col = idx % 4
        if idx >= 6:
            break
        ax = axes[row, col]
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(-1.5, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{token}', fontsize=11, fontweight='bold')
        draw_stack(ax, stack, action[:20])

    # Hide unused subplots
    for idx in range(6, 8):
        axes[idx // 4, idx % 4].axis('off')

    plt.tight_layout()
    save_fig('reverse-polish-notation')


# 8. Basic Calculator
def create_basic_calculator():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Basic Calculator: "1 + (2 - 3)"', fontsize=14, fontweight='bold')

    states = [
        ([1], [], 'Read 1'),
        ([1], ['+'], 'Read +'),
        ([1], ['+'], 'Read (: save state'),
        ([1, 2], ['+'], 'Read 2'),
        ([1, 2], ['+', '-'], 'Read -'),
        ([1, -1], ['+'], 'Read 3, ): eval 2-3=-1'),
    ]

    for idx, (nums, ops, title) in enumerate(states):
        ax = axes[idx // 3, idx % 3]
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10)
        draw_stack(ax, nums, 'Numbers', x_offset=0)
        draw_stack(ax, ops, 'Operators', x_offset=2.5, colors=['#FF9800']*len(ops))

    plt.tight_layout()
    save_fig('basic-calculator')


# 9. Basic Calculator II
def create_basic_calculator_ii():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle('Basic Calculator II: "3+2*2"', fontsize=14, fontweight='bold')

    states = [
        ([3], '+', 'Read 3, op=+'),
        ([3, 2], '*', 'Read +2, op=*'),
        ([3, 4], None, 'Read *2, push 2*2'),
        ([7], None, 'Sum stack'),
        ([7], None, 'Result: 7')
    ]

    for ax, (stack, op, title) in zip(axes, states):
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        draw_stack(ax, stack, title)
        if op:
            ax.text(1.8, 1, f'op: {op}', fontsize=10, color='red')

    plt.tight_layout()
    save_fig('basic-calculator-ii')


# 10. Decode String
def create_decode_string():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Decode String: "3[a2[c]]"', fontsize=14, fontweight='bold')

    states = [
        ([''], [3], 'Read 3['),
        (['', 'a'], [3], 'Read a'),
        (['', 'a'], [3, 2], 'Read 2['),
        (['', 'a', 'c'], [3, 2], 'Read c'),
        (['', 'acc'], [3], 'Read ]: acc'),
        (['accaccacc'], [], 'Read ]: x3')
    ]

    for idx, (strs, nums, title) in enumerate(states):
        ax = axes[idx // 3, idx % 3]
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10)
        draw_stack(ax, strs, 'Strings', x_offset=0)
        draw_stack(ax, nums, 'Numbers', x_offset=2.5, colors=['#9C27B0']*len(nums))

    plt.tight_layout()
    save_fig('decode-string')


# 11. Largest Rectangle in Histogram
def create_largest_rectangle_histogram():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram bars
    ax1 = axes[0]
    heights = [2, 1, 5, 6, 2, 3]
    x = np.arange(len(heights))
    colors = ['#4CAF50'] * len(heights)
    colors[2] = colors[3] = '#FF5722'  # Highlight max rectangle

    ax1.bar(x, heights, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=2, xmin=0.35, xmax=0.65)
    ax1.set_title('Largest Rectangle = 10 (height=5, width=2)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Height')

    for i, h in enumerate(heights):
        ax1.text(i, h + 0.2, str(h), ha='center', fontsize=11, fontweight='bold')

    # Right: Stack states
    ax2 = axes[1]
    ax2.set_xlim(-1, 8)
    ax2.set_ylim(-1, 5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Monotonic Increasing Stack', fontsize=12, fontweight='bold')

    # Show stack progression
    stacks = [
        (0.5, [0], 'i=0'),
        (2, [1], 'i=1: pop 0'),
        (3.5, [1, 2], 'i=2'),
        (5, [1, 2, 3], 'i=3'),
        (6.5, [1, 4], 'i=4: pop, calc')
    ]

    for x_pos, stack, label in stacks:
        for j, idx in enumerate(stack):
            rect = patches.Rectangle((x_pos - 0.3, j * 0.5), 0.6, 0.4,
                                     facecolor='#2196F3', edgecolor='black')
            ax2.add_patch(rect)
            ax2.text(x_pos, j * 0.5 + 0.2, str(idx), ha='center', va='center', fontsize=9)
        ax2.text(x_pos, -0.5, label, ha='center', fontsize=8)

    plt.tight_layout()
    save_fig('largest-rectangle-histogram')


# 12. Trapping Rain Water
def create_trapping_rain_water():
    fig, ax = plt.subplots(figsize=(12, 5))

    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    x = np.arange(len(height))

    # Draw bars
    ax.bar(x, height, color='#795548', edgecolor='black', label='Elevation')

    # Calculate and draw water
    water = [0, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 0]
    for i, (h, w) in enumerate(zip(height, water)):
        if w > 0:
            ax.bar(i, w, bottom=h, color='#2196F3', edgecolor='#1565C0', alpha=0.7)

    ax.set_title(f'Trapping Rain Water: Total = {sum(water)} units', fontsize=14, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Height')
    ax.legend()

    plt.tight_layout()
    save_fig('trapping-rain-water')


# 13. Sliding Window Maximum
def create_sliding_window_maximum():
    fig, ax = plt.subplots(figsize=(14, 5))

    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    maxes = [3, 3, 5, 5, 6, 7]

    ax.set_xlim(-0.5, len(nums) + 0.5)
    ax.set_ylim(-1, 10)

    # Draw array
    for i, n in enumerate(nums):
        color = '#E0E0E0'
        rect = patches.Rectangle((i, 5), 0.9, 1, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(i + 0.45, 5.5, str(n), ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(i + 0.45, 4.3, f'i={i}', ha='center', fontsize=9)

    # Draw windows
    colors = ['#FFCDD2', '#C8E6C9', '#BBDEFB', '#FFE0B2', '#E1BEE7', '#B2EBF2']
    for w in range(len(maxes)):
        rect = patches.Rectangle((w - 0.05, 5 - 0.05), k + 0.1, 1.1,
                                 facecolor='none', edgecolor=plt.cm.tab10(w % 10),
                                 linewidth=3, linestyle='--')
        ax.add_patch(rect)
        ax.text(w + k/2, 7, f'max={maxes[w]}', ha='center', fontsize=10,
               color=plt.cm.tab10(w % 10), fontweight='bold')

    ax.set_title(f'Sliding Window Maximum (k={k}): Output = {maxes}', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    save_fig('sliding-window-maximum')


# 14. Remove K Digits
def create_remove_k_digits():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle('Remove K Digits: num="1432219", k=3', fontsize=14, fontweight='bold')

    states = [
        (['1'], 'Read 1'),
        (['1', '4'], 'Read 4'),
        (['1', '3'], 'Read 3: pop 4'),
        (['1', '2'], 'Read 2: pop 3'),
        (['1', '2', '1', '9'], 'Final: 1219')
    ]

    for ax, (stack, title) in zip(axes, states):
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        draw_stack(ax, stack, title)

    plt.tight_layout()
    save_fig('remove-k-digits')


# 15. Simplify Path
def create_simplify_path():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle('Simplify Path: "/a/./b/../../c/"', fontsize=14, fontweight='bold')

    states = [
        (['a'], 'Read "a"'),
        (['a'], 'Read "." (skip)'),
        (['a', 'b'], 'Read "b"'),
        (['a'], 'Read ".." (pop)'),
        ([], 'Read ".." (pop)')
    ]

    for ax, (stack, title) in zip(axes, states):
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        draw_stack(ax, stack if stack else ['(empty)'], title)

    plt.tight_layout()
    save_fig('simplify-path')


# 16. Asteroid Collision
def create_asteroid_collision():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Asteroid Collision: [5, 10, -5]', fontsize=14, fontweight='bold')

    states = [
        ([5], 'Push 5 (right)'),
        ([5, 10], 'Push 10 (right)'),
        ([5, 10], '-5 collides with 10'),
        ([5, 10], '10 > |-5|: -5 destroyed'),
        ([5, 10], 'Result: [5, 10]'),
        ([], '')
    ]

    for idx, (stack, title) in enumerate(states[:5]):
        ax = axes[idx // 3, idx % 3]
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-1, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        colors = ['#4CAF50' if x > 0 else '#F44336' for x in stack]
        draw_stack(ax, stack, title)

    axes[1, 2].axis('off')
    plt.tight_layout()
    save_fig('asteroid-collision')


# 17. Task Scheduler
def create_task_scheduler():
    fig, ax = plt.subplots(figsize=(14, 4))

    # Tasks: A, A, A, B, B, B with n=2
    schedule = ['A', 'B', 'idle', 'A', 'B', 'idle', 'A', 'B']
    colors = {'A': '#4CAF50', 'B': '#2196F3', 'idle': '#E0E0E0'}

    for i, task in enumerate(schedule):
        rect = patches.Rectangle((i, 0), 0.9, 1, facecolor=colors[task], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(i + 0.45, 0.5, task, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(i + 0.45, -0.3, f't={i}', ha='center', fontsize=9)

    ax.set_xlim(-0.5, len(schedule) + 0.5)
    ax.set_ylim(-1, 2)
    ax.set_title('Task Scheduler: [A,A,A,B,B,B], n=2 -> Total time = 8', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    save_fig('task-scheduler')


# 18. First Negative in Window
def create_first_negative_window():
    fig, ax = plt.subplots(figsize=(14, 5))

    arr = [12, -1, -7, 8, -15, 30, 16, 28]
    k = 3
    results = [-1, -1, -7, -15, -15, 0]

    ax.set_xlim(-0.5, len(arr) + 0.5)
    ax.set_ylim(-2, 8)

    # Draw array
    for i, n in enumerate(arr):
        color = '#F44336' if n < 0 else '#4CAF50'
        rect = patches.Rectangle((i, 4), 0.9, 1, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(i + 0.45, 4.5, str(n), ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw windows
    for w in range(len(results)):
        rect = patches.Rectangle((w - 0.05, 3.95), k + 0.1, 1.1,
                                 facecolor='none', edgecolor='blue', linewidth=2, linestyle='--')
        ax.add_patch(rect)
        ax.text(w + k/2, 3.3, f'First neg: {results[w]}', ha='center', fontsize=9, color='blue')

    ax.set_title(f'First Negative in Every Window of Size {k}', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    save_fig('first-negative-window')


# 19. Rotting Oranges
def create_rotting_oranges():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle('Rotting Oranges: BFS Spread', fontsize=14, fontweight='bold')

    # 0=empty, 1=fresh, 2=rotten
    grids = [
        [[2, 1, 1], [1, 1, 0], [0, 1, 1]],
        [[2, 2, 1], [2, 1, 0], [0, 1, 1]],
        [[2, 2, 2], [2, 2, 0], [0, 1, 1]],
        [[2, 2, 2], [2, 2, 0], [0, 2, 2]]
    ]
    titles = ['t=0', 't=1', 't=2', 't=3 (done)']

    colors = {0: '#FFFFFF', 1: '#4CAF50', 2: '#F44336'}

    for ax, grid, title in zip(axes, grids, titles):
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11)

        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                rect = patches.Rectangle((j, 2-i), 0.9, 0.9, facecolor=colors[val],
                                         edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                label = {0: '', 1: 'F', 2: 'R'}[val]
                ax.text(j + 0.45, 2-i + 0.45, label, ha='center', va='center',
                       fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_fig('rotting-oranges')


# 20. Stock Span
def create_stock_span():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Stock prices
    ax1 = axes[0]
    prices = [100, 80, 60, 70, 60, 75, 85]
    spans = [1, 1, 1, 2, 1, 4, 6]
    x = np.arange(len(prices))

    bars = ax1.bar(x, prices, color='#2196F3', edgecolor='black')
    ax1.set_title('Stock Prices and Spans', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price')

    for i, (p, s) in enumerate(zip(prices, spans)):
        ax1.text(i, p + 2, f'span={s}', ha='center', fontsize=9, fontweight='bold')
        # Draw span range
        ax1.hlines(y=p-5, xmin=i-s+1, xmax=i, color='red', linewidth=2)

    # Right: Stack visualization
    ax2 = axes[1]
    ax2.set_xlim(-1, 8)
    ax2.set_ylim(-1, 5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Monotonic Decreasing Stack (indices)', fontsize=12, fontweight='bold')

    stacks = [
        (0.5, [0], 'd=0'),
        (2, [0, 1], 'd=1'),
        (3.5, [0, 1, 2], 'd=2'),
        (5, [0, 1, 3], 'd=3'),
        (6.5, [0, 6], 'd=6')
    ]

    for x_pos, stack, label in stacks:
        for j, idx in enumerate(stack):
            rect = patches.Rectangle((x_pos - 0.3, j * 0.5), 0.6, 0.4,
                                     facecolor='#FF9800', edgecolor='black')
            ax2.add_patch(rect)
            ax2.text(x_pos, j * 0.5 + 0.2, str(idx), ha='center', va='center', fontsize=9)
        ax2.text(x_pos, -0.5, label, ha='center', fontsize=8)

    plt.tight_layout()
    save_fig('stock-span')


def main():
    """Generate all visualization images."""
    print("Generating Stack & Queue visualizations...")
    print("-" * 50)

    create_valid_parentheses()
    create_queue_using_stacks()
    create_stack_using_queues()
    create_min_stack()
    create_next_greater_element()
    create_daily_temperatures()
    create_reverse_polish_notation()
    create_basic_calculator()
    create_basic_calculator_ii()
    create_decode_string()
    create_largest_rectangle_histogram()
    create_trapping_rain_water()
    create_sliding_window_maximum()
    create_remove_k_digits()
    create_simplify_path()
    create_asteroid_collision()
    create_task_scheduler()
    create_first_negative_window()
    create_rotting_oranges()
    create_stock_span()

    print("-" * 50)
    print("All visualizations generated successfully!")


if __name__ == '__main__':
    main()
