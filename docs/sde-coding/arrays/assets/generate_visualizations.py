#!/usr/bin/env python3
"""
Generate PNG visualizations for 17 Array problem files.
Each visualization shows the algorithm step-by-step with colored cells,
pointer positions, and before/after states.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Set up the output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Common styling
COLORS = {
    'default': '#E3E3E3',
    'highlight': '#4CAF50',
    'secondary': '#2196F3',
    'warning': '#FF9800',
    'danger': '#F44336',
    'success': '#4CAF50',
    'pointer1': '#E91E63',
    'pointer2': '#9C27B0',
    'water': '#64B5F6',
    'bar': '#795548',
    'zero': '#BDBDBD',
    'nonzero': '#4CAF50',
    'sorted': '#81C784',
    'unsorted': '#FFB74D',
}

def save_fig(fig, filename):
    """Save figure to the output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {filepath}")


def draw_array(ax, arr, x_start=0, y=0, cell_width=0.8, cell_height=0.6,
               colors=None, labels=None, pointers=None, title=None):
    """Draw an array with colored cells and optional pointers."""
    if colors is None:
        colors = [COLORS['default']] * len(arr)

    for i, val in enumerate(arr):
        x = x_start + i * (cell_width + 0.1)
        color = colors[i] if i < len(colors) else COLORS['default']

        # Draw cell
        rect = FancyBboxPatch((x, y), cell_width, cell_height,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)

        # Draw value
        ax.text(x + cell_width/2, y + cell_height/2, str(val),
                ha='center', va='center', fontsize=12, fontweight='bold')

        # Draw index below
        ax.text(x + cell_width/2, y - 0.2, str(i),
                ha='center', va='top', fontsize=9, color='gray')

    # Draw pointers
    if pointers:
        for ptr_name, ptr_idx, ptr_color in pointers:
            if 0 <= ptr_idx < len(arr):
                x = x_start + ptr_idx * (cell_width + 0.1) + cell_width/2
                ax.annotate(ptr_name, xy=(x, y + cell_height + 0.1),
                           xytext=(x, y + cell_height + 0.4),
                           ha='center', fontsize=10, fontweight='bold',
                           color=ptr_color,
                           arrowprops=dict(arrowstyle='->', color=ptr_color, lw=2))

    if title:
        ax.text(x_start - 0.3, y + cell_height/2, title,
                ha='right', va='center', fontsize=11, fontweight='bold')


# ============================================================================
# 1. Move Zeros Visualization
# ============================================================================
def create_move_zeros_viz():
    """Two pointer technique for moving zeros to end."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    fig.suptitle('Move Zeros - Two Pointer Technique', fontsize=16, fontweight='bold', y=0.98)

    states = [
        ([0, 1, 0, 3, 12], 0, 0, "Initial"),
        ([0, 1, 0, 3, 12], 0, 1, "read=1, nums[1]=1, swap"),
        ([1, 0, 0, 3, 12], 1, 3, "read=3, nums[3]=3, swap"),
        ([1, 3, 0, 0, 12], 2, 4, "read=4, nums[4]=12, swap"),
        ([1, 3, 12, 0, 0], 3, 5, "Final Result"),
    ]

    for ax, (arr, write, read, label) in zip(axes, states):
        ax.set_xlim(-1, 10)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')

        colors = []
        for i, val in enumerate(arr):
            if val == 0:
                colors.append(COLORS['zero'])
            else:
                colors.append(COLORS['nonzero'])

        pointers = []
        if write < len(arr):
            pointers.append(('write', write, COLORS['pointer1']))
        if read < len(arr):
            pointers.append(('read', read, COLORS['pointer2']))

        draw_array(ax, arr, colors=colors, pointers=pointers, title=label)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=COLORS['nonzero'], edgecolor='black', label='Non-zero'),
        patches.Patch(facecolor=COLORS['zero'], edgecolor='black', label='Zero'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_fig(fig, 'move-zeros.png')


# ============================================================================
# 2. Remove Duplicates Visualization
# ============================================================================
def create_remove_duplicates_viz():
    """Read-write pointer pattern for removing duplicates."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 10))
    fig.suptitle('Remove Duplicates - Read-Write Pointer Pattern', fontsize=16, fontweight='bold', y=0.98)

    states = [
        ([0, 0, 1, 1, 1, 2, 2, 3, 3, 4], 1, 1, "Initial: write=1"),
        ([0, 1, 1, 1, 1, 2, 2, 3, 3, 4], 2, 2, "Found 1 (new), write to idx 1"),
        ([0, 1, 2, 1, 1, 2, 2, 3, 3, 4], 3, 5, "Found 2 (new), write to idx 2"),
        ([0, 1, 2, 3, 1, 2, 2, 3, 3, 4], 4, 7, "Found 3 (new), write to idx 3"),
        ([0, 1, 2, 3, 4, 2, 2, 3, 3, 4], 5, 9, "Found 4 (new), write to idx 4"),
    ]

    for ax, (arr, write, read, label) in zip(axes, states):
        ax.set_xlim(-1, 15)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')

        colors = []
        for i in range(len(arr)):
            if i < write:
                colors.append(COLORS['sorted'])
            else:
                colors.append(COLORS['default'])

        pointers = [('write', write, COLORS['pointer1'])]
        if read < len(arr):
            pointers.append(('read', read, COLORS['pointer2']))

        draw_array(ax, arr, colors=colors, pointers=pointers, title=label)

    legend_elements = [
        patches.Patch(facecolor=COLORS['sorted'], edgecolor='black', label='Unique elements'),
        patches.Patch(facecolor=COLORS['default'], edgecolor='black', label='To be processed'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_fig(fig, 'remove-duplicates.png')


# ============================================================================
# 3. Buy Sell Stock Visualization
# ============================================================================
def create_buy_sell_stock_viz():
    """Track minimum price for maximum profit."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Best Time to Buy and Sell Stock', fontsize=16, fontweight='bold')

    prices = [7, 1, 5, 3, 6, 4]
    days = range(len(prices))

    # Price chart
    ax1.plot(days, prices, 'b-o', markersize=10, linewidth=2)
    ax1.fill_between(days, prices, alpha=0.3)

    # Mark buy and sell points
    buy_day, sell_day = 1, 4
    ax1.scatter([buy_day], [prices[buy_day]], color='green', s=200, zorder=5, marker='^', label='Buy (Day 1, $1)')
    ax1.scatter([sell_day], [prices[sell_day]], color='red', s=200, zorder=5, marker='v', label='Sell (Day 4, $6)')

    # Draw profit arrow
    ax1.annotate('', xy=(sell_day, prices[sell_day]), xytext=(buy_day, prices[buy_day]),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax1.text(2.5, 3.5, 'Profit = $5', fontsize=12, fontweight='bold', color='green')

    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Price Chart with Buy/Sell Points', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(days)

    # Tracking visualization
    ax2.axis('off')
    tracking_data = [
        ("Day 0", 7, 7, 0),
        ("Day 1", 1, 1, 0),
        ("Day 2", 5, 1, 4),
        ("Day 3", 3, 1, 2),
        ("Day 4", 6, 1, 5),
        ("Day 5", 4, 1, 3),
    ]

    y_pos = 0.9
    ax2.text(0.1, y_pos, "Day", fontweight='bold', fontsize=11)
    ax2.text(0.3, y_pos, "Price", fontweight='bold', fontsize=11)
    ax2.text(0.5, y_pos, "Min Price", fontweight='bold', fontsize=11)
    ax2.text(0.7, y_pos, "Profit", fontweight='bold', fontsize=11)

    for i, (day, price, min_p, profit) in enumerate(tracking_data):
        y_pos = 0.8 - i * 0.12
        color = 'green' if profit == 5 else 'black'
        weight = 'bold' if profit == 5 else 'normal'
        ax2.text(0.1, y_pos, day, fontsize=10)
        ax2.text(0.3, y_pos, f"${price}", fontsize=10)
        ax2.text(0.5, y_pos, f"${min_p}", fontsize=10)
        ax2.text(0.7, y_pos, f"${profit}", fontsize=10, color=color, fontweight=weight)

    ax2.set_title('Tracking Minimum Price & Maximum Profit', fontsize=14, pad=10)

    plt.tight_layout()
    save_fig(fig, 'buy-sell-stock.png')


# ============================================================================
# 4. Contains Duplicate Visualization
# ============================================================================
def create_contains_duplicate_viz():
    """Hash set lookup for duplicate detection."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Contains Duplicate - Hash Set Lookup', fontsize=16, fontweight='bold')

    arr = [1, 2, 3, 1]

    # Draw array
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 4)
    ax.axis('off')

    # Title
    ax.text(0, 3.5, 'Array: [1, 2, 3, 1]', fontsize=14, fontweight='bold')

    # Steps
    steps = [
        (0, 1, "{}", "1 not in set, add 1", "{1}"),
        (1, 2, "{1}", "2 not in set, add 2", "{1, 2}"),
        (2, 3, "{1, 2}", "3 not in set, add 3", "{1, 2, 3}"),
        (3, 1, "{1, 2, 3}", "1 IN SET! Return True", "FOUND!"),
    ]

    y_start = 2.8
    for i, (idx, val, before, action, after) in enumerate(steps):
        y = y_start - i * 0.7

        # Step indicator
        color = COLORS['danger'] if "FOUND" in after else COLORS['highlight']
        ax.add_patch(FancyBboxPatch((0, y-0.25), 0.4, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black'))
        ax.text(0.2, y, str(i+1), ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # Check value
        ax.text(0.6, y, f"Check arr[{idx}] = {val}", fontsize=11, va='center')

        # Set before
        ax.text(3.5, y, f"Set: {before}", fontsize=10, va='center', family='monospace')

        # Arrow and action
        ax.text(6, y, action, fontsize=10, va='center',
               color=COLORS['danger'] if "IN SET" in action else 'black',
               fontweight='bold' if "IN SET" in action else 'normal')

        # Set after
        ax.text(10, y, f"-> {after}", fontsize=10, va='center', family='monospace',
               color=COLORS['danger'] if "FOUND" in after else COLORS['highlight'])

    plt.tight_layout()
    save_fig(fig, 'contains-duplicate.png')


# ============================================================================
# 5. Two Sum Visualization
# ============================================================================
def create_two_sum_viz():
    """Hash map complement lookup."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Two Sum - Hash Map Complement', fontsize=16, fontweight='bold')

    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 5)
    ax.axis('off')

    # Problem statement
    ax.text(0, 4.5, 'nums = [2, 7, 11, 15], target = 9', fontsize=14, fontweight='bold')
    ax.text(0, 4.0, 'Find two indices whose values sum to target', fontsize=11, style='italic')

    # Draw the array
    arr = [2, 7, 11, 15]
    for i, val in enumerate(arr):
        x = i * 1.2
        rect = FancyBboxPatch((x, 2.8), 1, 0.7,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['highlight'] if i <= 1 else COLORS['default'],
                              edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.5, 3.15, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(x + 0.5, 2.5, f"[{i}]", ha='center', va='center', fontsize=10, color='gray')

    # Draw connection
    ax.annotate('', xy=(0.5, 3.6), xytext=(1.7, 3.6),
               arrowprops=dict(arrowstyle='<->', color=COLORS['pointer1'], lw=3))
    ax.text(1.1, 3.8, '2 + 7 = 9', ha='center', fontsize=12, fontweight='bold', color=COLORS['pointer1'])

    # Steps
    steps = [
        ("Step 1:", "num=2, complement=9-2=7", "7 not in map", "Add {2: 0}"),
        ("Step 2:", "num=7, complement=9-7=2", "2 FOUND at index 0!", "Return [0, 1]"),
    ]

    y = 1.8
    for step, check, result, action in steps:
        ax.text(0, y, step, fontsize=11, fontweight='bold')
        ax.text(1.5, y, check, fontsize=10)
        ax.text(6, y, result, fontsize=10,
               color=COLORS['success'] if "FOUND" in result else 'black',
               fontweight='bold' if "FOUND" in result else 'normal')
        ax.text(10, y, action, fontsize=10, family='monospace',
               color=COLORS['success'] if "Return" in action else 'black')
        y -= 0.5

    # Hash map visualization
    ax.text(0, 0.5, 'Hash Map Evolution:', fontsize=12, fontweight='bold')
    ax.add_patch(FancyBboxPatch((3, 0.1), 3, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor='lightyellow', edgecolor='black'))
    ax.text(4.5, 0.5, '{value: index}', ha='center', va='center', fontsize=11, family='monospace')

    plt.tight_layout()
    save_fig(fig, 'two-sum.png')


# ============================================================================
# 6. Maximum Subarray (Kadane's Algorithm) Visualization
# ============================================================================
def create_maximum_subarray_viz():
    """Kadane's Algorithm visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Maximum Subarray - Kadane's Algorithm", fontsize=16, fontweight='bold')

    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    current_sums = [-2, 1, -2, 4, 3, 5, 6, 1, 5]
    max_sums = [-2, 1, 1, 4, 4, 5, 6, 6, 6]

    # Upper plot: Array with values
    ax1.set_xlim(-1, 12)
    ax1.set_ylim(-1, 3)
    ax1.axis('off')

    ax1.text(0, 2.5, 'Array values:', fontsize=12, fontweight='bold')

    # Draw array
    for i, val in enumerate(nums):
        x = i * 1.1
        # Highlight the maximum subarray [4, -1, 2, 1]
        if 3 <= i <= 6:
            color = COLORS['highlight']
        elif val < 0:
            color = COLORS['warning']
        else:
            color = COLORS['default']

        rect = FancyBboxPatch((x, 1.5), 1, 0.7,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(x + 0.5, 1.85, str(val), ha='center', va='center', fontsize=12, fontweight='bold')
        ax1.text(x + 0.5, 1.2, f"[{i}]", ha='center', va='center', fontsize=9, color='gray')

    # Bracket for max subarray
    ax1.annotate('', xy=(3, 2.35), xytext=(7.1, 2.35),
                arrowprops=dict(arrowstyle='-', color='green', lw=2))
    ax1.text(5.05, 2.55, 'Max Subarray: [4,-1,2,1] = 6', ha='center', fontsize=11,
            fontweight='bold', color='green')

    # Lower plot: Tracking table
    ax2.axis('off')
    ax2.set_xlim(-0.5, 12)
    ax2.set_ylim(-1, 4)

    # Headers
    headers = ['Index', 'Value', 'Decision', 'Current Sum', 'Max Sum']
    x_positions = [0, 1.5, 3, 6, 8.5]

    for header, x_pos in zip(headers, x_positions):
        ax2.text(x_pos, 3.5, header, fontsize=11, fontweight='bold')

    # Data rows
    decisions = ['Start', 'Start new (1>-1)', 'Extend (-2>-3)', 'Start new (4>2)',
                'Extend (3>-1)', 'Extend (5>2)', 'Extend (6>1)', 'Extend (1>-5)', 'Extend (5>4)']

    for i in range(len(nums)):
        y = 3 - i * 0.35
        is_max = (i == 6)  # Index of maximum
        color = 'green' if is_max else 'black'
        weight = 'bold' if is_max else 'normal'

        ax2.text(x_positions[0], y, str(i), fontsize=10, color=color, fontweight=weight)
        ax2.text(x_positions[1], y, str(nums[i]), fontsize=10, color=color, fontweight=weight)
        ax2.text(x_positions[2], y, decisions[i], fontsize=9, color=color)
        ax2.text(x_positions[3], y, str(current_sums[i]), fontsize=10, color=color, fontweight=weight)
        ax2.text(x_positions[4], y, str(max_sums[i]), fontsize=10, color=color, fontweight=weight)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=COLORS['highlight'], edgecolor='black', label='Maximum Subarray'),
        patches.Patch(facecolor=COLORS['warning'], edgecolor='black', label='Negative Value'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    save_fig(fig, 'maximum-subarray.png')


# ============================================================================
# 7. Container With Most Water Visualization
# ============================================================================
def create_container_water_viz():
    """Two pointers for maximum area."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Container With Most Water - Two Pointers', fontsize=16, fontweight='bold')

    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]

    # Upper plot: Bar chart visualization
    ax1.bar(range(len(height)), height, color=COLORS['bar'], edgecolor='black', linewidth=1.5)

    # Highlight the optimal container (indices 1 and 8)
    ax1.axhline(y=7, xmin=1/9, xmax=8/9, color=COLORS['water'], linewidth=3, linestyle='--')

    # Fill water area
    water_x = [1, 1, 8, 8]
    water_y = [0, 7, 7, 0]
    ax1.fill(water_x, water_y, color=COLORS['water'], alpha=0.3)

    # Mark pointers
    ax1.annotate('L', xy=(1, height[1]+0.5), fontsize=14, fontweight='bold', color=COLORS['pointer1'], ha='center')
    ax1.annotate('R', xy=(8, height[8]+0.5), fontsize=14, fontweight='bold', color=COLORS['pointer2'], ha='center')

    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Height', fontsize=12)
    ax1.set_title('height = [1, 8, 6, 2, 5, 4, 8, 3, 7]\nMax Area = width(7) x height(7) = 49', fontsize=12)
    ax1.set_xticks(range(len(height)))
    ax1.grid(True, alpha=0.3, axis='y')

    # Lower plot: Algorithm steps
    ax2.axis('off')
    ax2.set_xlim(-0.5, 12)
    ax2.set_ylim(-0.5, 4)

    steps = [
        (0, 8, 1, 7, 8, "h[0]=1 < h[8]=7, move L"),
        (1, 8, 8, 7, 49, "h[1]=8 > h[8]=7, move R **MAX**"),
        (1, 7, 8, 3, 18, "h[1]=8 > h[7]=3, move R"),
        (1, 6, 8, 8, 40, "h[1]=8 = h[6]=8, move R"),
    ]

    headers = ['Left', 'Right', 'h[L]', 'h[R]', 'Area', 'Action']
    x_pos = [0, 1.5, 3, 4.5, 6, 7.5]

    for header, x in zip(headers, x_pos):
        ax2.text(x, 3.5, header, fontsize=11, fontweight='bold')

    for i, (l, r, hl, hr, area, action) in enumerate(steps):
        y = 3 - i * 0.6
        is_max = area == 49
        color = 'green' if is_max else 'black'
        weight = 'bold' if is_max else 'normal'

        ax2.text(x_pos[0], y, str(l), fontsize=10, color=color, fontweight=weight)
        ax2.text(x_pos[1], y, str(r), fontsize=10, color=color, fontweight=weight)
        ax2.text(x_pos[2], y, str(hl), fontsize=10, color=color, fontweight=weight)
        ax2.text(x_pos[3], y, str(hr), fontsize=10, color=color, fontweight=weight)
        ax2.text(x_pos[4], y, str(area), fontsize=10, color=color, fontweight=weight)
        ax2.text(x_pos[5], y, action, fontsize=9, color=color, fontweight=weight)

    ax2.text(0, 0.5, 'Key: Area = min(h[L], h[R]) x (R - L)', fontsize=11, style='italic')

    plt.tight_layout()
    save_fig(fig, 'container-water.png')


# ============================================================================
# 8. Product Except Self Visualization
# ============================================================================
def create_product_except_self_viz():
    """Prefix/suffix product visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Product of Array Except Self - Prefix/Suffix Products', fontsize=16, fontweight='bold')

    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 7)
    ax.axis('off')

    nums = [1, 2, 3, 4]
    prefix = [1, 1, 2, 6]  # Product of elements before index
    suffix = [24, 12, 4, 1]  # Product of elements after index
    result = [24, 12, 8, 6]

    # Row labels and data
    rows = [
        ('nums', nums, COLORS['default']),
        ('prefix (left)', prefix, '#BBDEFB'),
        ('suffix (right)', suffix, '#C8E6C9'),
        ('result', result, COLORS['highlight']),
    ]

    y_pos = 5.5
    for label, data, color in rows:
        ax.text(0, y_pos, label + ':', fontsize=12, fontweight='bold', va='center')

        for i, val in enumerate(data):
            x = 3 + i * 1.5
            rect = FancyBboxPatch((x, y_pos - 0.35), 1.2, 0.7,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + 0.6, y_pos, str(val), ha='center', va='center', fontsize=12, fontweight='bold')

        y_pos -= 1.2

    # Index row
    ax.text(0, y_pos + 0.3, 'index:', fontsize=11, va='center', color='gray')
    for i in range(4):
        x = 3 + i * 1.5
        ax.text(x + 0.6, y_pos + 0.3, str(i), ha='center', va='center', fontsize=10, color='gray')

    # Explanation
    y_pos = 1.5
    ax.text(0, y_pos, 'How it works:', fontsize=12, fontweight='bold')
    explanations = [
        'result[i] = prefix[i] x suffix[i]',
        'result[0] = 1 x 24 = 24  (product of [2,3,4])',
        'result[1] = 1 x 12 = 12  (product of [1,3,4])',
        'result[2] = 2 x 4 = 8    (product of [1,2,4])',
        'result[3] = 6 x 1 = 6    (product of [1,2,3])',
    ]
    for i, exp in enumerate(explanations):
        ax.text(0.5, y_pos - 0.4 * (i + 1), exp, fontsize=10, family='monospace')

    plt.tight_layout()
    save_fig(fig, 'product-except-self.png')


# ============================================================================
# 9. Merge Intervals Visualization
# ============================================================================
def create_merge_intervals_viz():
    """Sort and merge intervals visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Merge Intervals - Sort and Merge', fontsize=16, fontweight='bold')

    # Timeline visualization
    intervals = [[1,3], [2,6], [8,10], [15,18]]
    merged = [[1,6], [8,10], [15,18]]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    # Before merging
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 5)
    ax1.set_title('Before Merge: [[1,3], [2,6], [8,10], [15,18]]', fontsize=12)

    for i, (start, end) in enumerate(intervals):
        y = 4 - i * 0.8
        ax1.barh(y, end - start, left=start, height=0.5, color=colors[i], edgecolor='black', linewidth=1.5)
        ax1.text((start + end) / 2, y, f'[{start},{end}]', ha='center', va='center', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Timeline', fontsize=11)
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3, axis='x')

    # After merging
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 4)
    ax2.set_title('After Merge: [[1,6], [8,10], [15,18]]', fontsize=12)

    merged_colors = [COLORS['highlight'], '#45B7D1', '#96CEB4']
    for i, (start, end) in enumerate(merged):
        y = 3 - i * 0.8
        ax2.barh(y, end - start, left=start, height=0.5, color=merged_colors[i], edgecolor='black', linewidth=2)
        ax2.text((start + end) / 2, y, f'[{start},{end}]', ha='center', va='center', fontsize=10, fontweight='bold')

    # Show merge explanation
    ax2.annotate('Merged!', xy=(3.5, 2.5), xytext=(5, 3.5),
                fontsize=10, fontweight='bold', color=COLORS['highlight'],
                arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=2))

    ax2.set_xlabel('Timeline', fontsize=11)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_fig(fig, 'merge-intervals.png')


# ============================================================================
# 10. Rotate Array Visualization
# ============================================================================
def create_rotate_array_viz():
    """Three reversals technique."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle('Rotate Array - Three Reversals Method', fontsize=16, fontweight='bold', y=0.98)

    states = [
        ([1, 2, 3, 4, 5, 6, 7], "Original Array", None),
        ([7, 6, 5, 4, 3, 2, 1], "Step 1: Reverse entire array", list(range(7))),
        ([5, 6, 7, 4, 3, 2, 1], "Step 2: Reverse first k=3 elements", [0, 1, 2]),
        ([5, 6, 7, 1, 2, 3, 4], "Step 3: Reverse remaining elements (Final)", [3, 4, 5, 6]),
    ]

    for ax, (arr, title, highlight) in zip(axes, states):
        ax.set_xlim(-1, 12)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')

        colors = []
        for i in range(len(arr)):
            if highlight and i in highlight:
                colors.append(COLORS['highlight'])
            elif i < 3:
                colors.append('#BBDEFB')
            else:
                colors.append('#C8E6C9')

        draw_array(ax, arr, colors=colors, title=title)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#BBDEFB', edgecolor='black', label='Last k elements (will be first)'),
        patches.Patch(facecolor='#C8E6C9', edgecolor='black', label='First n-k elements (will be last)'),
        patches.Patch(facecolor=COLORS['highlight'], edgecolor='black', label='Being reversed'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    save_fig(fig, 'rotate-array.png')


# ============================================================================
# 11. Find Peak Element Visualization
# ============================================================================
def create_find_peak_viz():
    """Binary search for peak element."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Find Peak Element - Binary Search', fontsize=16, fontweight='bold')

    nums = [1, 2, 1, 3, 5, 6, 4]

    # Upper: Array as mountain plot
    ax1.plot(range(len(nums)), nums, 'b-o', markersize=12, linewidth=2)
    ax1.fill_between(range(len(nums)), nums, alpha=0.3)

    # Mark peaks
    peaks = [1, 5]  # Indices 1 and 5 are peaks
    for peak in peaks:
        ax1.scatter([peak], [nums[peak]], color='red', s=200, zorder=5, marker='*')
        ax1.annotate(f'Peak\n({peak}, {nums[peak]})', xy=(peak, nums[peak]),
                    xytext=(peak, nums[peak] + 1),
                    ha='center', fontsize=10, fontweight='bold', color='red')

    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('nums = [1, 2, 1, 3, 5, 6, 4] - Peaks at indices 1 and 5', fontsize=12)
    ax1.set_xticks(range(len(nums)))
    ax1.grid(True, alpha=0.3)

    # Lower: Binary search steps
    ax2.axis('off')
    ax2.set_xlim(-0.5, 12)
    ax2.set_ylim(-0.5, 4)

    steps = [
        (0, 6, 3, "nums[3]=3 < nums[4]=5, go right"),
        (4, 6, 5, "nums[5]=6 > nums[6]=4, FOUND PEAK!"),
    ]

    ax2.text(0, 3.5, 'Binary Search Steps:', fontsize=12, fontweight='bold')

    headers = ['Left', 'Right', 'Mid', 'Decision']
    x_pos = [0, 1.5, 3, 5]

    for header, x in zip(headers, x_pos):
        ax2.text(x, 2.8, header, fontsize=11, fontweight='bold')

    for i, (l, r, m, decision) in enumerate(steps):
        y = 2.3 - i * 0.6
        is_found = "FOUND" in decision
        color = 'green' if is_found else 'black'

        ax2.text(x_pos[0], y, str(l), fontsize=10, color=color)
        ax2.text(x_pos[1], y, str(r), fontsize=10, color=color)
        ax2.text(x_pos[2], y, str(m), fontsize=10, color=color)
        ax2.text(x_pos[3], y, decision, fontsize=10, color=color, fontweight='bold' if is_found else 'normal')

    ax2.text(0, 0.5, 'Key Insight: If nums[mid] < nums[mid+1], peak is on the right', fontsize=11, style='italic')

    plt.tight_layout()
    save_fig(fig, 'find-peak.png')


# ============================================================================
# 12. Search in Rotated Sorted Array Visualization
# ============================================================================
def create_search_rotated_viz():
    """Binary search in rotated array."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle('Search in Rotated Sorted Array - Binary Search', fontsize=16, fontweight='bold')

    nums = [4, 5, 6, 7, 0, 1, 2]
    target = 0

    # Upper: Visual representation
    ax1.set_xlim(-1, 10)
    ax1.set_ylim(-1, 9)

    # Plot the values as bars
    colors = []
    for i, val in enumerate(nums):
        if val == target:
            colors.append(COLORS['danger'])
        elif i <= 3:  # Left sorted portion
            colors.append('#BBDEFB')
        else:  # Right sorted portion
            colors.append('#C8E6C9')

    ax1.bar(range(len(nums)), nums, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, val in enumerate(nums):
        ax1.text(i, val + 0.3, str(val), ha='center', fontsize=12, fontweight='bold')

    # Mark the pivot
    ax1.axvline(x=3.5, color='red', linestyle='--', linewidth=2, label='Pivot')
    ax1.annotate('Pivot', xy=(3.5, 7.5), fontsize=11, color='red', fontweight='bold')

    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('nums = [4, 5, 6, 7, 0, 1, 2], target = 0', fontsize=12)
    ax1.set_xticks(range(len(nums)))
    ax1.legend()

    # Lower: Binary search steps
    ax2.axis('off')
    ax2.set_xlim(-0.5, 14)
    ax2.set_ylim(-0.5, 4)

    steps = [
        (0, 6, 3, 7, "Left [4,5,6,7] sorted, 0 not in [4,7], go right"),
        (4, 6, 5, 1, "Right [0,1,2] sorted, 0 in [0,2], go left"),
        (4, 4, 4, 0, "nums[4] = 0 = target, FOUND!"),
    ]

    headers = ['L', 'R', 'Mid', 'nums[mid]', 'Decision']
    x_pos = [0, 1, 2.5, 4.5, 6.5]

    for header, x in zip(headers, x_pos):
        ax2.text(x, 3.5, header, fontsize=11, fontweight='bold')

    for i, (l, r, m, val, decision) in enumerate(steps):
        y = 2.8 - i * 0.7
        is_found = "FOUND" in decision
        color = 'green' if is_found else 'black'

        ax2.text(x_pos[0], y, str(l), fontsize=10)
        ax2.text(x_pos[1], y, str(r), fontsize=10)
        ax2.text(x_pos[2], y, str(m), fontsize=10)
        ax2.text(x_pos[3], y, str(val), fontsize=10)
        ax2.text(x_pos[4], y, decision, fontsize=9, color=color, fontweight='bold' if is_found else 'normal')

    ax2.text(0, 0.5, 'Key: One half is always sorted - check if target is in that half', fontsize=11, style='italic')

    plt.tight_layout()
    save_fig(fig, 'search-rotated.png')


# ============================================================================
# 13. Subarray Sum Equals K Visualization
# ============================================================================
def create_subarray_sum_k_viz():
    """Prefix sum with hash map."""
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.suptitle('Subarray Sum Equals K - Prefix Sum + Hash Map', fontsize=16, fontweight='bold')

    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    nums = [1, 2, 3]
    k = 3

    # Show array
    ax.text(0, 7, f'nums = {nums}, k = {k}', fontsize=14, fontweight='bold')

    # Draw array
    for i, val in enumerate(nums):
        x = i * 1.5
        rect = FancyBboxPatch((x, 5.8), 1.2, 0.8,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['default'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.6, 6.2, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(x + 0.6, 5.5, f'[{i}]', ha='center', va='center', fontsize=10, color='gray')

    # Prefix sum explanation
    ax.text(0, 4.8, 'Prefix Sums:', fontsize=12, fontweight='bold')
    prefix_sums = [0, 1, 3, 6]
    for i, ps in enumerate(prefix_sums):
        ax.text(i * 1.5 + 0.6, 4.3, str(ps), ha='center', fontsize=12,
               fontweight='bold', color=COLORS['pointer1'])

    # Step by step
    ax.text(0, 3.5, 'Process:', fontsize=12, fontweight='bold')

    steps = [
        ("Init", 0, 0, "{0: 1}", "Start with 0 in map"),
        ("i=0", 1, 1, "{0:1, 1:1}", "Need 1-3=-2, not found"),
        ("i=1", 3, 3, "{0:1, 1:1, 3:1}", "Need 3-3=0, FOUND! count=1"),
        ("i=2", 6, 6, "{0:1, 1:1, 3:1, 6:1}", "Need 6-3=3, FOUND! count=2"),
    ]

    headers = ['Step', 'prefix', 'need', 'map', 'action']
    x_pos = [0, 1.5, 3, 4.5, 8]

    y = 3
    for header, x in zip(headers, x_pos):
        ax.text(x, y, header, fontsize=10, fontweight='bold')

    for step, psum, need, map_str, action in steps:
        y -= 0.5
        is_found = "FOUND" in action
        color = COLORS['success'] if is_found else 'black'

        ax.text(x_pos[0], y, step, fontsize=10)
        ax.text(x_pos[1], y, str(psum), fontsize=10)
        ax.text(x_pos[2], y, str(psum - k), fontsize=10)
        ax.text(x_pos[3], y, map_str, fontsize=9, family='monospace')
        ax.text(x_pos[4], y, action, fontsize=9, color=color, fontweight='bold' if is_found else 'normal')

    # Key formula
    ax.text(0, 0.3, 'Key Formula: If prefix[j] - k exists in map, subarray sum equals k',
           fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    save_fig(fig, 'subarray-sum-k.png')


# ============================================================================
# 14. Sliding Window Maximum Visualization
# ============================================================================
def create_sliding_window_max_viz():
    """Monotonic deque visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Sliding Window Maximum - Monotonic Deque', fontsize=16, fontweight='bold')

    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3

    # Upper: Show array with sliding window
    ax1.set_xlim(-0.5, 10)
    ax1.set_ylim(-2, 10)

    # Draw bars
    for i, val in enumerate(nums):
        color = COLORS['highlight'] if 2 <= i <= 4 else COLORS['default']
        ax1.bar(i, val + 4, bottom=-2, color=color, edgecolor='black', linewidth=1.5, width=0.7)
        ax1.text(i, val + 2.5, str(val), ha='center', fontsize=12, fontweight='bold')

    # Mark window
    ax1.add_patch(patches.Rectangle((1.65, -2), 2.7, 10, fill=False,
                                    edgecolor=COLORS['pointer1'], linewidth=3, linestyle='--'))
    ax1.text(3, 8.5, f'Window (k={k})', ha='center', fontsize=11, color=COLORS['pointer1'])

    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3', fontsize=12)
    ax1.axhline(y=-2, color='black', linewidth=1)

    # Lower: Steps table
    ax2.axis('off')
    ax2.set_xlim(-0.5, 14)
    ax2.set_ylim(-1, 5)

    ax2.text(0, 4.5, 'Monotonic Deque (decreasing) stores indices:', fontsize=12, fontweight='bold')

    steps = [
        ("i=2", "[1,3,-1]", "[1,2]", "[3,-1]", "3"),
        ("i=3", "[3,-1,-3]", "[1,2,3]", "[3,-1,-3]", "3"),
        ("i=4", "[-1,-3,5]", "[4]", "[5]", "5"),
        ("i=5", "[-3,5,3]", "[4,5]", "[5,3]", "5"),
        ("i=6", "[5,3,6]", "[6]", "[6]", "6"),
        ("i=7", "[3,6,7]", "[7]", "[7]", "7"),
    ]

    headers = ['Step', 'Window', 'Deque (idx)', 'Deque (val)', 'Max']
    x_pos = [0, 1.5, 4, 7, 10]

    y = 4
    for header, x in zip(headers, x_pos):
        ax2.text(x, y, header, fontsize=10, fontweight='bold')

    for step, window, deque_idx, deque_val, max_val in steps:
        y -= 0.55
        ax2.text(x_pos[0], y, step, fontsize=10)
        ax2.text(x_pos[1], y, window, fontsize=9, family='monospace')
        ax2.text(x_pos[2], y, deque_idx, fontsize=9, family='monospace')
        ax2.text(x_pos[3], y, deque_val, fontsize=9, family='monospace')
        ax2.text(x_pos[4], y, max_val, fontsize=10, fontweight='bold', color=COLORS['success'])

    ax2.text(0, 0.3, 'Result: [3, 3, 5, 5, 6, 7]', fontsize=12, fontweight='bold', color=COLORS['success'])

    plt.tight_layout()
    save_fig(fig, 'sliding-window-max.png')


# ============================================================================
# 15. Trapping Rain Water Visualization
# ============================================================================
def create_trapping_rain_water_viz():
    """Two pointers for trapping rain water."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle('Trapping Rain Water - Two Pointers', fontsize=16, fontweight='bold')

    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    water = [0, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 0]  # Water at each position

    # Upper: Bar chart with water
    x = np.arange(len(height))
    ax1.bar(x, height, color=COLORS['bar'], edgecolor='black', linewidth=1.5, label='Bars')
    ax1.bar(x, water, bottom=height, color=COLORS['water'], edgecolor='black', linewidth=0.5, alpha=0.7, label='Water')

    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Height', fontsize=12)
    ax1.set_title(f'Total Water Trapped = {sum(water)} units', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3, axis='y')

    # Lower: Two pointer explanation
    ax2.axis('off')
    ax2.set_xlim(-0.5, 14)
    ax2.set_ylim(-0.5, 5)

    ax2.text(0, 4.5, 'Two Pointer Algorithm:', fontsize=12, fontweight='bold')
    ax2.text(0, 4, 'water[i] = min(maxLeft[i], maxRight[i]) - height[i]', fontsize=11, family='monospace')

    # Sample steps
    steps = [
        (0, 11, 0, 1, "h[0]=0 < h[11]=1, water=0-0=0"),
        (1, 11, 1, 1, "h[1]=1 = h[11]=1, water=1-1=0"),
        (2, 11, 1, 1, "h[2]=0 < h[11]=1, water=1-0=1"),
    ]

    headers = ['L', 'R', 'maxL', 'maxR', 'Calculation']
    x_pos = [0, 1, 2.5, 4, 5.5]

    y = 3.2
    for header, x in zip(headers, x_pos):
        ax2.text(x, y, header, fontsize=10, fontweight='bold')

    for l, r, maxl, maxr, calc in steps:
        y -= 0.5
        ax2.text(x_pos[0], y, str(l), fontsize=10)
        ax2.text(x_pos[1], y, str(r), fontsize=10)
        ax2.text(x_pos[2], y, str(maxl), fontsize=10)
        ax2.text(x_pos[3], y, str(maxr), fontsize=10)
        ax2.text(x_pos[4], y, calc, fontsize=9)

    ax2.text(0, 0.5, 'Key: Process from both ends, always move the smaller height pointer',
            fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    save_fig(fig, 'trapping-rain-water.png')


# ============================================================================
# 16. First Missing Positive Visualization
# ============================================================================
def create_first_missing_positive_viz():
    """Cycle sort / index as hash visualization."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('First Missing Positive - Cycle Sort (Index as Hash)', fontsize=16, fontweight='bold', y=0.98)

    states = [
        ([3, 4, -1, 1], "Initial: nums = [3, 4, -1, 1]", []),
        ([-1, 4, 3, 1], "Swap nums[0]=3 to index 2", [2]),
        ([-1, 1, 3, 4], "Swap nums[1]=4 to index 3", [3]),
        ([1, -1, 3, 4], "Swap nums[1]=1 to index 0", [0]),
        ([1, -1, 3, 4], "Final: Check for first mismatch at index 1", [1]),
    ]

    for ax, (arr, title, highlight) in zip(axes, states):
        ax.set_xlim(-1, 12)
        ax.set_ylim(-0.5, 2)
        ax.axis('off')

        colors = []
        for i in range(len(arr)):
            if i in highlight:
                colors.append(COLORS['highlight'])
            elif arr[i] == i + 1:
                colors.append(COLORS['success'])
            elif arr[i] <= 0 or arr[i] > len(arr):
                colors.append(COLORS['warning'])
            else:
                colors.append(COLORS['default'])

        # Draw array
        for i, val in enumerate(arr):
            x = i * 1.5
            rect = FancyBboxPatch((x, 0.8), 1.2, 0.8,
                                  boxstyle="round,pad=0.02",
                                  facecolor=colors[i], edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + 0.6, 1.2, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(x + 0.6, 0.5, f'idx {i}', ha='center', va='center', fontsize=9, color='gray')
            ax.text(x + 0.6, 0.1, f'expect {i+1}', ha='center', va='center', fontsize=8, color='blue')

        ax.text(6.5, 1.2, title, va='center', fontsize=11, fontweight='bold')

    # Legend
    legend_elements = [
        patches.Patch(facecolor=COLORS['success'], edgecolor='black', label='Correct position'),
        patches.Patch(facecolor=COLORS['warning'], edgecolor='black', label='Invalid value'),
        patches.Patch(facecolor=COLORS['highlight'], edgecolor='black', label='Just moved'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_fig(fig, 'first-missing-positive.png')


# ============================================================================
# 17. Median of Two Sorted Arrays Visualization
# ============================================================================
def create_median_two_arrays_viz():
    """Binary search partition visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle('Median of Two Sorted Arrays - Binary Search Partition', fontsize=16, fontweight='bold')

    nums1 = [1, 3, 8]
    nums2 = [7, 11, 18, 19]

    # Upper: Show partition concept
    ax1.set_xlim(-0.5, 14)
    ax1.set_ylim(-1, 5)
    ax1.axis('off')

    ax1.text(0, 4.5, 'Find partition where left <= right', fontsize=12, fontweight='bold')

    # Draw nums1
    ax1.text(0, 3.5, 'nums1:', fontsize=11, fontweight='bold')
    for i, val in enumerate(nums1):
        x = 1.5 + i * 1.2
        color = '#BBDEFB' if i < 2 else '#C8E6C9'
        rect = FancyBboxPatch((x, 3.1), 1, 0.7,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(x + 0.5, 3.45, str(val), ha='center', va='center', fontsize=12, fontweight='bold')

    # Partition line for nums1
    ax1.axvline(x=3.9, ymin=0.55, ymax=0.75, color='red', linewidth=3)
    ax1.text(3.9, 2.9, '|', fontsize=20, ha='center', color='red', fontweight='bold')

    # Draw nums2
    ax1.text(0, 2, 'nums2:', fontsize=11, fontweight='bold')
    for i, val in enumerate(nums2):
        x = 1.5 + i * 1.2
        color = '#BBDEFB' if i < 2 else '#C8E6C9'
        rect = FancyBboxPatch((x, 1.6), 1, 0.7,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(x + 0.5, 1.95, str(val), ha='center', va='center', fontsize=12, fontweight='bold')

    # Partition line for nums2
    ax1.axvline(x=3.9, ymin=0.25, ymax=0.45, color='red', linewidth=3)
    ax1.text(3.9, 1.4, '|', fontsize=20, ha='center', color='red', fontweight='bold')

    # Labels
    ax1.text(2.5, 0.8, 'Left Half', ha='center', fontsize=11, color='blue')
    ax1.text(5.5, 0.8, 'Right Half', ha='center', fontsize=11, color='green')

    # Merged visualization
    ax1.text(8, 3.5, 'Merged:', fontsize=11, fontweight='bold')
    merged = [1, 3, 7, 8, 11, 18, 19]
    for i, val in enumerate(merged):
        x = 8 + i * 0.7
        color = COLORS['highlight'] if i == 3 else ('#BBDEFB' if i < 3 else '#C8E6C9')
        rect = FancyBboxPatch((x, 2.8), 0.6, 0.6,
                              boxstyle="round,pad=0.01",
                              facecolor=color, edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(x + 0.3, 3.1, str(val), ha='center', va='center', fontsize=10)

    ax1.annotate('Median = 8', xy=(10.8, 3.1), xytext=(11.5, 3.8),
                fontsize=11, fontweight='bold', color=COLORS['success'],
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    # Lower: Binary search steps
    ax2.axis('off')
    ax2.set_xlim(-0.5, 14)
    ax2.set_ylim(-0.5, 4)

    ax2.text(0, 3.5, 'Binary Search on smaller array (nums1):', fontsize=12, fontweight='bold')

    steps = [
        ("i=1, j=2", "L1=1, R1=8, L2=11, R2=18", "1<=18, 11>8", "Need more from nums1"),
        ("i=2, j=1", "L1=3, R1=8, L2=7, R2=11", "3<=11, 7<=8", "Valid! Median=max(3,7)=7"),
    ]

    y = 2.8
    for partition, values, check, result in steps:
        ax2.text(0, y, partition, fontsize=10, family='monospace')
        ax2.text(3, y, values, fontsize=9)
        ax2.text(8, y, check, fontsize=9)
        color = COLORS['success'] if "Valid" in result else 'black'
        ax2.text(11, y, result, fontsize=9, color=color, fontweight='bold' if "Valid" in result else 'normal')
        y -= 0.6

    ax2.text(0, 0.5, 'Key: Binary search partition where max(left) <= min(right)',
            fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    save_fig(fig, 'median-two-arrays.png')


# ============================================================================
# Main execution
# ============================================================================
def main():
    """Generate all visualizations."""
    print("Generating visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    # Generate all visualizations
    create_move_zeros_viz()
    create_remove_duplicates_viz()
    create_buy_sell_stock_viz()
    create_contains_duplicate_viz()
    create_two_sum_viz()
    create_maximum_subarray_viz()
    create_container_water_viz()
    create_product_except_self_viz()
    create_merge_intervals_viz()
    create_rotate_array_viz()
    create_find_peak_viz()
    create_search_rotated_viz()
    create_subarray_sum_k_viz()
    create_sliding_window_max_viz()
    create_trapping_rain_water_viz()
    create_first_missing_positive_viz()
    create_median_two_arrays_viz()

    print("-" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
