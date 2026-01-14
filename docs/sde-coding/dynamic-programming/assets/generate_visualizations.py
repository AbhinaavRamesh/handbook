#!/usr/bin/env python3
"""
Generate visualization images for Dynamic Programming problems.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set the output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_fig(name):
    """Save figure to the assets directory."""
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Generated: {name}")


def climbing_stairs_visualization():
    """Generate climbing stairs decision tree visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create a tree structure for n=4
    positions = {
        (0, 4): (6, 7),      # Start
        (1, 3): (3, 5.5),    # After 1 step
        (2, 2): (9, 5.5),    # After 2 steps
        (1, 2): (1.5, 4),    # 1->1
        (2, 1): (4.5, 4),    # 1->2
        (2, 1, 'b'): (7.5, 4),  # 2->1
        (4, 0): (10.5, 4),   # 2->2
        (1, 1): (0.5, 2.5),
        (2, 0): (2.5, 2.5),
        (2, 0, 'b'): (3.5, 2.5),
        (3, 0): (5.5, 2.5),
        (3, 0, 'b'): (6.5, 2.5),
        (3, 0, 'c'): (8.5, 2.5),
        (1, 0): (0, 1),
        (2, 0, 'c'): (1, 1),
    }

    # Draw nodes
    node_labels = {
        (6, 7): 'Start\nn=4',
        (3, 5.5): 'n=3',
        (9, 5.5): 'n=2',
        (1.5, 4): 'n=2',
        (4.5, 4): 'n=1',
        (7.5, 4): 'n=1',
        (10.5, 4): 'n=0\n(goal!)',
        (0.5, 2.5): 'n=1',
        (2.5, 2.5): 'n=0\n(goal!)',
        (3.5, 2.5): 'n=0\n(goal!)',
        (5.5, 2.5): 'n=0\n(goal!)',
        (6.5, 2.5): 'n=0\n(goal!)',
        (8.5, 2.5): 'n=0\n(goal!)',
        (0, 1): 'n=0\n(goal!)',
        (1, 1): 'n=0\n(goal!)',
    }

    for pos, label in node_labels.items():
        color = '#90EE90' if 'goal' in label else '#87CEEB'
        circle = plt.Circle(pos, 0.4, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw edges with labels
    edges = [
        ((6, 7), (3, 5.5), '+1'),
        ((6, 7), (9, 5.5), '+2'),
        ((3, 5.5), (1.5, 4), '+1'),
        ((3, 5.5), (4.5, 4), '+2'),
        ((9, 5.5), (7.5, 4), '+1'),
        ((9, 5.5), (10.5, 4), '+2'),
        ((1.5, 4), (0.5, 2.5), '+1'),
        ((1.5, 4), (2.5, 2.5), '+2'),
        ((4.5, 4), (3.5, 2.5), '+1'),
        ((7.5, 4), (5.5, 2.5), '+1'),
        ((0.5, 2.5), (0, 1), '+1'),
    ]

    for start, end, label in edges:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
        mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
        ax.text(mid[0]-0.3, mid[1]+0.2, label, fontsize=9, color='red', fontweight='bold')

    ax.set_xlim(-1, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Climbing Stairs: Decision Tree for n=4\n(5 ways to reach the top)', fontsize=14, fontweight='bold')

    save_fig('climbing-stairs-tree.png')


def climbing_stairs_dp_table():
    """Generate DP table filling visualization."""
    fig, ax = plt.subplots(figsize=(14, 4))

    # DP values for n=5
    dp = [1, 1, 2, 3, 5, 8]
    n = len(dp)

    # Draw cells
    for i in range(n):
        rect = plt.Rectangle((i*2, 0), 2, 2, fill=True,
                             facecolor='#E6F3FF', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i*2 + 1, 1, str(dp[i]), ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(i*2 + 1, -0.5, f'dp[{i}]', ha='center', va='center', fontsize=12)

    # Add arrows showing recurrence
    for i in range(2, n):
        # Arrow from dp[i-1]
        ax.annotate('', xy=(i*2 + 0.3, 2.3), xytext=((i-1)*2 + 1.7, 2.3),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        # Arrow from dp[i-2]
        ax.annotate('', xy=(i*2 + 0.3, 2.7), xytext=((i-2)*2 + 1.7, 2.7),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.text(6, 3.5, 'dp[i] = dp[i-1] + dp[i-2]', ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Climbing Stairs: DP Table (Fibonacci Pattern)', fontsize=14, fontweight='bold')

    save_fig('climbing-stairs-dp.png')


def house_robber_visualization():
    """Generate house robber visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Houses with values
    houses = [2, 7, 9, 3, 1]
    n = len(houses)

    # Top plot: Houses
    ax = axes[0]
    for i in range(n):
        # Draw house
        house = plt.Rectangle((i*2.5, 0), 2, 1.5, fill=True,
                              facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(house)
        # Roof
        roof = plt.Polygon([(i*2.5, 1.5), (i*2.5 + 1, 2.5), (i*2.5 + 2, 1.5)],
                          facecolor='#CD853F', edgecolor='black', linewidth=2)
        ax.add_patch(roof)
        # Value
        ax.text(i*2.5 + 1, 0.75, f'${houses[i]}', ha='center', va='center',
               fontsize=14, fontweight='bold')
        ax.text(i*2.5 + 1, -0.5, f'House {i}', ha='center', va='center', fontsize=10)

    # Highlight optimal selection (houses 1 and 2)
    for i in [1, 2, 4]:
        highlight = plt.Rectangle((i*2.5 - 0.1, -0.1), 2.2, 2.7, fill=False,
                                  edgecolor='green', linewidth=3, linestyle='--')
        ax.add_patch(highlight)

    ax.text(6, 3.5, 'Optimal: Rob houses 1, 2, 4 = $7 + $9 + $1 = $17',
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-1, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('House Robber: Cannot Rob Adjacent Houses', fontsize=14, fontweight='bold')

    # Bottom plot: DP table
    ax = axes[1]
    dp = [0, 2, 7, 11, 11, 12]  # dp[i] = max money up to house i-1

    for i in range(n + 1):
        color = '#90EE90' if i == n else '#E6F3FF'
        rect = plt.Rectangle((i*2, 0), 2, 1.5, fill=True,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i*2 + 1, 0.75, str(dp[i]), ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(i*2 + 1, -0.5, f'dp[{i}]', ha='center', va='center', fontsize=10)

    ax.text(6, 2.5, 'dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])',
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-1, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    save_fig('house-robber.png')


def coin_change_visualization():
    """Generate coin change DP visualization."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Amount = 11, coins = [1, 2, 5]
    coins = [1, 2, 5]
    amount = 11
    dp = [0, 1, 1, 2, 2, 1, 2, 2, 3, 3, 2, 3]  # Minimum coins needed

    # Draw DP table
    cell_width = 1.2
    cell_height = 1.0

    for i in range(amount + 1):
        x = i * cell_width
        color = '#90EE90' if i == amount else '#E6F3FF'
        rect = plt.Rectangle((x, 4), cell_width, cell_height, fill=True,
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + cell_width/2, 4 + cell_height/2, str(dp[i]),
               ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x + cell_width/2, 3.5, str(i), ha='center', va='center', fontsize=10)

    ax.text(-0.5, 4.5, 'dp[]:', ha='right', va='center', fontsize=12, fontweight='bold')
    ax.text(-0.5, 3.5, 'amount:', ha='right', va='center', fontsize=10)

    # Show coin selection for amount=11
    ax.text(7, 2.5, 'Building dp[11]:', ha='center', va='center', fontsize=14, fontweight='bold')

    explanations = [
        ('Use coin 1: dp[11-1] + 1 = dp[10] + 1 = 2 + 1 = 3', '#FFB6C1'),
        ('Use coin 2: dp[11-2] + 1 = dp[9] + 1 = 3 + 1 = 4', '#FFB6C1'),
        ('Use coin 5: dp[11-5] + 1 = dp[6] + 1 = 2 + 1 = 3', '#98FB98'),
    ]

    for idx, (text, color) in enumerate(explanations):
        rect = plt.Rectangle((2, 1.8 - idx*0.6), 10, 0.5, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(7, 2.05 - idx*0.6, text, ha='center', va='center', fontsize=10)

    ax.text(7, 0, 'dp[11] = min(3, 4, 3) = 3 coins (5 + 5 + 1)',
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-1, 15)
    ax.set_ylim(-0.5, 6)
    ax.axis('off')
    ax.set_title('Coin Change: Minimum Coins for Amount=11 with coins=[1,2,5]',
                fontsize=14, fontweight='bold')

    save_fig('coin-change.png')


def lis_visualization():
    """Generate LIS visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Array
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    n = len(nums)
    dp = [1, 1, 1, 2, 2, 3, 4, 4]  # LIS ending at each position

    # Top: Array with dp values
    ax = axes[0]
    for i in range(n):
        # Number box
        rect = plt.Rectangle((i*1.8, 2), 1.5, 1, facecolor='#E6F3FF',
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i*1.8 + 0.75, 2.5, str(nums[i]), ha='center', va='center',
               fontsize=14, fontweight='bold')

        # DP value box
        color = '#90EE90' if dp[i] == 4 else '#FFE4B5'
        rect2 = plt.Rectangle((i*1.8, 0.5), 1.5, 1, facecolor=color,
                              edgecolor='black', linewidth=2)
        ax.add_patch(rect2)
        ax.text(i*1.8 + 0.75, 1, str(dp[i]), ha='center', va='center',
               fontsize=14, fontweight='bold')

        ax.text(i*1.8 + 0.75, 3.3, f'i={i}', ha='center', va='center', fontsize=10)

    ax.text(-0.5, 2.5, 'nums:', ha='right', va='center', fontsize=12, fontweight='bold')
    ax.text(-0.5, 1, 'dp:', ha='right', va='center', fontsize=12, fontweight='bold')

    ax.set_xlim(-1, 15)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Longest Increasing Subsequence: [10,9,2,5,3,7,101,18]',
                fontsize=14, fontweight='bold')

    # Bottom: Show the actual LIS
    ax = axes[1]

    # Highlight the LIS: 2, 3, 7, 101 (or 2, 5, 7, 101)
    lis_indices = [2, 4, 5, 6]  # 2, 3, 7, 101

    for i in range(n):
        x = i * 1.8
        color = '#90EE90' if i in lis_indices else '#DCDCDC'
        rect = plt.Rectangle((x, 1), 1.5, 1, facecolor=color,
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.75, 1.5, str(nums[i]), ha='center', va='center',
               fontsize=14, fontweight='bold')

    # Draw arrows for LIS
    for i in range(len(lis_indices) - 1):
        start_x = lis_indices[i] * 1.8 + 1.5
        end_x = lis_indices[i+1] * 1.8
        ax.annotate('', xy=(end_x, 1.5), xytext=(start_x, 1.5),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.text(7, 0.3, 'LIS = [2, 3, 7, 101] with length 4', ha='center', va='center',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-1, 15)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')

    plt.tight_layout()
    save_fig('lis.png')


def lcs_visualization():
    """Generate LCS DP table visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))

    text1 = "ABCDE"
    text2 = "ACE"

    # DP table
    m, n = len(text1), len(text2)
    dp = [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 2, 2],
        [0, 1, 2, 2],
        [0, 1, 2, 3],
    ]

    cell_size = 1.0

    # Draw header row (text2)
    for j, char in enumerate("_" + text2):
        x = (j + 1) * cell_size
        rect = plt.Rectangle((x, (m + 1) * cell_size), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + cell_size/2, (m + 1.5) * cell_size, char,
               ha='center', va='center', fontsize=14, fontweight='bold')

    # Draw header column (text1)
    for i, char in enumerate("_" + text1):
        y = (m - i) * cell_size
        rect = plt.Rectangle((0, y), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(cell_size/2, y + cell_size/2, char,
               ha='center', va='center', fontsize=14, fontweight='bold')

    # Draw DP table
    for i in range(m + 1):
        for j in range(n + 1):
            x = (j + 1) * cell_size
            y = (m - i) * cell_size

            # Highlight path for LCS
            is_match = (i > 0 and j > 0 and text1[i-1] == text2[j-1] and
                       dp[i][j] == dp[i-1][j-1] + 1)
            color = '#90EE90' if is_match else '#E6F3FF'

            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, str(dp[i][j]),
                   ha='center', va='center', fontsize=14, fontweight='bold')

    # Add legend
    ax.text(7, 4, 'LCS("ABCDE", "ACE") = "ACE" (length 3)',
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.text(7, 3, 'Green cells: Character match (diagonal move)',
           ha='center', va='center', fontsize=11)

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Longest Common Subsequence: DP Table', fontsize=14, fontweight='bold')

    save_fig('lcs.png')


def edit_distance_visualization():
    """Generate edit distance DP table visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))

    word1 = "horse"
    word2 = "ros"

    m, n = len(word1), len(word2)

    # DP table
    dp = [
        [0, 1, 2, 3],
        [1, 1, 2, 3],
        [2, 2, 1, 2],
        [3, 2, 2, 2],
        [4, 3, 3, 2],
        [5, 4, 4, 3],
    ]

    cell_size = 1.2

    # Draw header row (word2)
    for j, char in enumerate(" " + word2):
        x = (j + 1) * cell_size
        rect = plt.Rectangle((x, (m + 1) * cell_size), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        display_char = "''" if char == " " else char
        ax.text(x + cell_size/2, (m + 1.5) * cell_size, display_char,
               ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw header column (word1)
    for i, char in enumerate(" " + word1):
        y = (m - i) * cell_size
        rect = plt.Rectangle((0, y), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        display_char = "''" if char == " " else char
        ax.text(cell_size/2, y + cell_size/2, display_char,
               ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw DP table with path highlighting
    path = [(5, 3), (4, 2), (3, 2), (2, 1), (1, 1), (0, 0)]

    for i in range(m + 1):
        for j in range(n + 1):
            x = (j + 1) * cell_size
            y = (m - i) * cell_size

            color = '#90EE90' if (i, j) in path else '#E6F3FF'

            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, str(dp[i][j]),
                   ha='center', va='center', fontsize=14, fontweight='bold')

    # Add operations legend
    ax.text(8, 5, 'Operations:', ha='left', va='center', fontsize=12, fontweight='bold')
    ax.text(8, 4.3, '1. Replace h with r', ha='left', va='center', fontsize=11)
    ax.text(8, 3.7, '2. Keep o', ha='left', va='center', fontsize=11)
    ax.text(8, 3.1, '3. Delete r', ha='left', va='center', fontsize=11)
    ax.text(8, 2.5, '4. Keep s', ha='left', va='center', fontsize=11)
    ax.text(8, 1.9, '5. Delete e', ha='left', va='center', fontsize=11)

    ax.text(8, 0.8, 'Min operations = 3', ha='left', va='center',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Edit Distance: "horse" -> "ros"', fontsize=14, fontweight='bold')

    save_fig('edit-distance.png')


def knapsack_visualization():
    """Generate knapsack DP visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Items: (weight, value)
    items = [(2, 6), (3, 10), (4, 12)]
    capacity = 7

    # Top: Item visualization
    ax = axes[0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, (w, v) in enumerate(items):
        x = i * 4 + 1
        rect = plt.Rectangle((x, 0), 2, 2, facecolor=colors[i],
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1, 1, f'Item {i+1}\nW={w}, V={v}',
               ha='center', va='center', fontsize=11, fontweight='bold')

    ax.text(7, 3, f'Knapsack Capacity = {capacity}', ha='center', va='center',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')
    ax.set_title('0/1 Knapsack Problem', fontsize=14, fontweight='bold')

    # Bottom: DP table
    ax = axes[1]

    # DP table [items+1][capacity+1]
    dp = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 6, 6, 6, 6, 6, 6],
        [0, 0, 6, 10, 10, 16, 16, 16],
        [0, 0, 6, 10, 12, 16, 18, 22],
    ]

    cell_size = 1.0

    # Header row (capacity)
    for j in range(capacity + 1):
        x = (j + 1) * cell_size
        rect = plt.Rectangle((x, 4 * cell_size), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + cell_size/2, 4.5 * cell_size, str(j),
               ha='center', va='center', fontsize=11, fontweight='bold')

    # Header column (items)
    for i in range(4):
        y = (3 - i) * cell_size
        rect = plt.Rectangle((0, y), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        label = str(i) if i > 0 else '0'
        ax.text(cell_size/2, y + cell_size/2, label,
               ha='center', va='center', fontsize=11, fontweight='bold')

    # DP cells
    for i in range(4):
        for j in range(capacity + 1):
            x = (j + 1) * cell_size
            y = (3 - i) * cell_size

            color = '#90EE90' if (i == 3 and j == capacity) else '#E6F3FF'
            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                 facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, str(dp[i][j]),
                   ha='center', va='center', fontsize=11, fontweight='bold')

    ax.text(-0.5, 4.5, 'cap->', ha='right', va='center', fontsize=10)
    ax.text(-0.5, 2, 'item', ha='right', va='center', fontsize=10)

    ax.text(12, 2, 'Optimal: Items 1 & 3\nValue = 6 + 16 = 22',
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_xlim(-1, 14)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    plt.tight_layout()
    save_fig('knapsack.png')


def unique_paths_visualization():
    """Generate unique paths grid visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    m, n = 3, 7

    # DP values
    dp = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 3, 6, 10, 15, 21, 28],
    ]

    cell_size = 1.5

    for i in range(m):
        for j in range(n):
            x = j * cell_size
            y = (m - 1 - i) * cell_size

            # Color based on position
            if i == 0 or j == 0:
                color = '#FFE4B5'  # Base cases
            elif i == m-1 and j == n-1:
                color = '#90EE90'  # Goal
            else:
                color = '#E6F3FF'

            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, str(dp[i][j]),
                   ha='center', va='center', fontsize=14, fontweight='bold')

    # Add robot and goal markers
    ax.text(cell_size/2, (m-1) * cell_size + cell_size + 0.3, 'Robot',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text((n-1) * cell_size + cell_size/2, -0.5, 'Goal',
           ha='center', va='center', fontsize=11, fontweight='bold')

    # Add arrows showing directions
    ax.annotate('', xy=(2.5, 4.3), xytext=(1.5, 4.3),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=(1.5, 3.5), xytext=(1.5, 4.5),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(2.2, 5, 'Can only move right or down', ha='center', fontsize=10)

    ax.text(5.5, 5.5, 'dp[i][j] = dp[i-1][j] + dp[i][j-1]',
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-1, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Unique Paths: {m}x{n} Grid - Answer: 28 paths', fontsize=14, fontweight='bold')

    save_fig('unique-paths.png')


def minimum_path_sum_visualization():
    """Generate minimum path sum visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    grid = [
        [1, 3, 1],
        [1, 5, 1],
        [4, 2, 1],
    ]

    dp = [
        [1, 4, 5],
        [2, 7, 6],
        [6, 8, 7],
    ]

    m, n = 3, 3
    cell_size = 2.0

    for i in range(m):
        for j in range(n):
            x = j * cell_size
            y = (m - 1 - i) * cell_size

            # Path cells are highlighted
            is_path = (i, j) in [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
            color = '#90EE90' if is_path else '#E6F3FF'

            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            # Show grid value and dp value
            ax.text(x + cell_size/2, y + cell_size*0.65, f'{grid[i][j]}',
                   ha='center', va='center', fontsize=16, fontweight='bold')
            ax.text(x + cell_size/2, y + cell_size*0.25, f'(sum: {dp[i][j]})',
                   ha='center', va='center', fontsize=10, color='gray')

    # Draw path arrows
    path_points = [(0.5, 2.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5), (2.5, 0.5)]
    for i in range(len(path_points) - 1):
        ax.annotate('',
                   xy=(path_points[i+1][0]*2, path_points[i+1][1]*2),
                   xytext=(path_points[i][0]*2, path_points[i][1]*2),
                   arrowprops=dict(arrowstyle='->', color='red', lw=3))

    ax.text(3, 7, 'Minimum Path: 1->3->1->1->1 = 7',
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Minimum Path Sum', fontsize=14, fontweight='bold')

    save_fig('minimum-path-sum.png')


def maximum_subarray_visualization():
    """Generate maximum subarray (Kadane's) visualization."""
    fig, ax = plt.subplots(figsize=(14, 6))

    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    n = len(nums)

    # Track current and max sums
    current = [-2, 1, -2, 4, 3, 5, 6, 1, 5]
    max_ending = [4, 5, 6]  # indices of max subarray

    cell_width = 1.4

    # Draw array
    for i in range(n):
        x = i * cell_width

        # Highlight max subarray
        in_max = 3 <= i <= 6
        color = '#90EE90' if in_max else '#E6F3FF'

        rect = plt.Rectangle((x, 2), cell_width - 0.1, 1.2,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_width/2 - 0.05, 2.6, str(nums[i]),
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(x + cell_width/2 - 0.05, 1.5, f'i={i}', ha='center', fontsize=10)

    # Draw current sum line
    for i in range(n):
        x = i * cell_width + cell_width/2 - 0.05
        y = current[i] / 3 + 5  # Scale for visualization
        ax.plot(x, y, 'ro', markersize=8)
        if i > 0:
            prev_x = (i-1) * cell_width + cell_width/2 - 0.05
            prev_y = current[i-1] / 3 + 5
            ax.plot([prev_x, x], [prev_y, y], 'r-', linewidth=2)

    ax.text(6, 7.5, 'Current Sum at each position', ha='center', fontsize=11)

    ax.text(6, 0.5, 'Max Subarray: [4, -1, 2, 1] = 6',
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 8)
    ax.axis('off')
    ax.set_title("Kadane's Algorithm: Maximum Subarray Sum", fontsize=14, fontweight='bold')

    save_fig('maximum-subarray.png')


def word_break_visualization():
    """Generate word break visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))

    s = "leetcode"
    wordDict = ["leet", "code"]

    # Draw the string
    cell_width = 1.2
    for i, char in enumerate(s):
        x = i * cell_width + 1
        rect = plt.Rectangle((x, 4), cell_width - 0.1, 1,
                             facecolor='#E6F3FF', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_width/2 - 0.05, 4.5, char,
               ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(x + cell_width/2 - 0.05, 3.5, str(i), ha='center', fontsize=10)

    # Highlight word segments
    # "leet" segment
    leet_rect = plt.Rectangle((1, 4), 4 * cell_width - 0.1, 1,
                              fill=False, edgecolor='green', linewidth=3)
    ax.add_patch(leet_rect)
    ax.text(3, 5.3, '"leet"', ha='center', fontsize=11, color='green', fontweight='bold')

    # "code" segment
    code_rect = plt.Rectangle((1 + 4 * cell_width, 4), 4 * cell_width - 0.1, 1,
                              fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(code_rect)
    ax.text(7.8, 5.3, '"code"', ha='center', fontsize=11, color='blue', fontweight='bold')

    # DP array
    dp = [True, False, False, False, True, False, False, False, True]

    for i in range(len(s) + 1):
        x = i * cell_width + 1
        color = '#90EE90' if dp[i] else '#FFB6C1'
        rect = plt.Rectangle((x, 1.5), cell_width - 0.1, 0.8,
                             facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + cell_width/2 - 0.05, 1.9, 'T' if dp[i] else 'F',
               ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x + cell_width/2 - 0.05, 1, f'dp[{i}]', ha='center', fontsize=9)

    ax.text(6, 6.5, 'wordDict = ["leet", "code"]', ha='center', fontsize=12, fontweight='bold')
    ax.text(6, 0, 'dp[i] = True if s[0:i] can be segmented',
           ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 7.5)
    ax.axis('off')
    ax.set_title('Word Break: "leetcode"', fontsize=14, fontweight='bold')

    save_fig('word-break.png')


def decode_ways_visualization():
    """Generate decode ways visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Mapping
    ax.text(7, 7, "'A' = 1, 'B' = 2, ..., 'Z' = 26", ha='center',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#E6F3FF', alpha=0.8))

    s = "226"

    # Draw the string
    cell_width = 2
    for i, char in enumerate(s):
        x = i * cell_width + 3
        rect = plt.Rectangle((x, 4.5), cell_width - 0.2, 1.2,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_width/2 - 0.1, 5.1, char,
               ha='center', va='center', fontsize=20, fontweight='bold')

    # Show possible decodings
    decodings = [
        ('"2" "2" "6"', 'B B F'),
        ('"22" "6"', 'V F'),
        ('"2" "26"', 'B Z'),
    ]

    for i, (code, decode) in enumerate(decodings):
        y = 3 - i * 0.8
        ax.text(4, y, code, ha='center', fontsize=12, fontweight='bold')
        ax.text(4, y, ' -> ', ha='left', fontsize=12)
        ax.text(7, y, decode, ha='center', fontsize=12, fontweight='bold', color='green')

    # DP array
    dp = [1, 1, 2, 3]

    for i in range(len(s) + 1):
        x = i * 2 + 2
        color = '#90EE90' if i == len(s) else '#E6F3FF'
        rect = plt.Rectangle((x, 0), 1.8, 0.8,
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.9, 0.4, str(dp[i]),
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(x + 0.9, -0.4, f'dp[{i}]', ha='center', fontsize=10)

    ax.text(10, 0.4, '= 3 ways', ha='left', fontsize=14, fontweight='bold')

    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 8)
    ax.axis('off')
    ax.set_title('Decode Ways: "226"', fontsize=14, fontweight='bold')

    save_fig('decode-ways.png')


def partition_subset_visualization():
    """Generate partition equal subset sum visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    nums = [1, 5, 11, 5]
    total = sum(nums)  # 22
    target = total // 2  # 11

    # Draw original array
    cell_width = 2
    for i, num in enumerate(nums):
        x = i * cell_width + 2
        rect = plt.Rectangle((x, 5), cell_width - 0.2, 1,
                             facecolor='#E6F3FF', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_width/2 - 0.1, 5.5, str(num),
               ha='center', va='center', fontsize=16, fontweight='bold')

    ax.text(6, 6.5, f'nums = {nums}, total = {total}', ha='center', fontsize=12, fontweight='bold')
    ax.text(6, 4, f'Target for each subset = {total}/2 = {target}', ha='center', fontsize=12)

    # Show the two subsets
    subset1 = [1, 5, 5]
    subset2 = [11]

    # Subset 1
    ax.text(3, 2.8, 'Subset 1:', ha='center', fontsize=11, fontweight='bold')
    for i, num in enumerate(subset1):
        x = i * 1.5 + 1.5
        rect = plt.Rectangle((x, 1.8), 1.3, 0.8,
                             facecolor='#90EE90', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.65, 2.2, str(num), ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(3.8, 1.4, f'Sum = {sum(subset1)}', ha='center', fontsize=11)

    # Subset 2
    ax.text(8, 2.8, 'Subset 2:', ha='center', fontsize=11, fontweight='bold')
    rect = plt.Rectangle((7.3, 1.8), 1.3, 0.8,
                         facecolor='#87CEEB', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7.95, 2.2, '11', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(8, 1.4, f'Sum = {sum(subset2)}', ha='center', fontsize=11)

    ax.text(6, 0.5, 'Both subsets sum to 11. Answer: True',
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    ax.set_title('Partition Equal Subset Sum', fontsize=14, fontweight='bold')

    save_fig('partition-subset.png')


def target_sum_visualization():
    """Generate target sum visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))

    nums = [1, 1, 1, 1, 1]
    target = 3

    # Draw array
    cell_width = 1.8
    for i, num in enumerate(nums):
        x = i * cell_width + 2
        rect = plt.Rectangle((x, 7), cell_width - 0.2, 1,
                             facecolor='#E6F3FF', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_width/2 - 0.1, 7.5, str(num),
               ha='center', va='center', fontsize=16, fontweight='bold')

    ax.text(7, 8.5, f'nums = {nums}, target = {target}', ha='center', fontsize=12, fontweight='bold')

    # Show some valid expressions
    expressions = [
        ('+1 +1 +1 +1 -1', '= 3'),
        ('+1 +1 +1 -1 +1', '= 3'),
        ('+1 +1 -1 +1 +1', '= 3'),
        ('+1 -1 +1 +1 +1', '= 3'),
        ('-1 +1 +1 +1 +1', '= 3'),
    ]

    for i, (expr, result) in enumerate(expressions):
        y = 5.5 - i * 0.7
        ax.text(5, y, expr, ha='center', fontsize=11, fontfamily='monospace')
        ax.text(9, y, result, ha='center', fontsize=11, fontweight='bold', color='green')

    ax.text(7, 1.5, 'Total ways = 5', ha='center', va='center', fontsize=16, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.text(7, 0.5, 'Each number can be assigned + or -', ha='center', fontsize=11)

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis('off')
    ax.set_title('Target Sum: Count Ways to Reach Target', fontsize=14, fontweight='bold')

    save_fig('target-sum.png')


def matrix_chain_visualization():
    """Generate matrix chain multiplication visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Matrices: A1(10x30), A2(30x5), A3(5x60)
    matrices = [
        ('A1', '10x30'),
        ('A2', '30x5'),
        ('A3', '5x60'),
    ]

    # Draw matrices
    for i, (name, dims) in enumerate(matrices):
        x = i * 4 + 2
        rect = plt.Rectangle((x, 5), 3, 2,
                             facecolor='#E6F3FF', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.5, 6.3, name, ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(x + 1.5, 5.5, dims, ha='center', va='center', fontsize=12)

    # Show parenthesizations
    ax.text(7, 4, 'Two ways to parenthesize:', ha='center', fontsize=12, fontweight='bold')

    ax.text(4, 3, '(A1 * A2) * A3', ha='center', fontsize=12, fontfamily='monospace')
    ax.text(4, 2.3, '= (10x30 * 30x5) * 5x60', ha='center', fontsize=10)
    ax.text(4, 1.6, '= 10*30*5 + 10*5*60', ha='center', fontsize=10)
    ax.text(4, 0.9, '= 1500 + 3000 = 4500', ha='center', fontsize=11, fontweight='bold', color='green')

    ax.text(10, 3, 'A1 * (A2 * A3)', ha='center', fontsize=12, fontfamily='monospace')
    ax.text(10, 2.3, '= 10x30 * (30x5 * 5x60)', ha='center', fontsize=10)
    ax.text(10, 1.6, '= 30*5*60 + 10*30*60', ha='center', fontsize=10)
    ax.text(10, 0.9, '= 9000 + 18000 = 27000', ha='center', fontsize=11, fontweight='bold', color='red')

    ax.text(7, 0, 'Optimal: (A1 * A2) * A3 with 4500 multiplications',
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 8)
    ax.axis('off')
    ax.set_title('Matrix Chain Multiplication: Order Matters!', fontsize=14, fontweight='bold')

    save_fig('matrix-chain.png')


def burst_balloons_visualization():
    """Generate burst balloons visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))

    nums = [3, 1, 5, 8]

    # Draw balloons
    balloon_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, (num, color) in enumerate(zip(nums, balloon_colors)):
        x = i * 3 + 2
        # Balloon body
        circle = plt.Circle((x + 1, 5.5), 0.8, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x + 1, 5.5, str(num), ha='center', va='center', fontsize=16, fontweight='bold')
        # Balloon string
        ax.plot([x + 1, x + 1], [4.7, 4], 'k-', linewidth=2)

    ax.text(7, 7.5, 'nums = [3, 1, 5, 8]', ha='center', fontsize=12, fontweight='bold')

    # Show optimal bursting order
    ax.text(7, 3, 'Optimal Order:', ha='center', fontsize=12, fontweight='bold')

    steps = [
        'Burst 1: coins = 3*1*5 = 15',
        'Burst 5: coins = 3*5*8 = 120',
        'Burst 3: coins = 1*3*8 = 24',
        'Burst 8: coins = 1*8*1 = 8',
    ]

    for i, step in enumerate(steps):
        ax.text(7, 2.2 - i * 0.6, step, ha='center', fontsize=11)

    ax.text(7, -0.3, 'Total = 15 + 120 + 24 + 8 = 167',
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 8.5)
    ax.axis('off')
    ax.set_title('Burst Balloons: Maximum Coins', fontsize=14, fontweight='bold')

    save_fig('burst-balloons.png')


def regex_matching_visualization():
    """Generate regex matching DP visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))

    s = "aab"
    p = "c*a*b"

    # DP table for this example
    dp = [
        [True, False, True, False, True, False],
        [False, False, False, True, True, False],
        [False, False, False, False, True, False],
        [False, False, False, False, False, True],
    ]

    cell_size = 1.3

    # Header row (pattern)
    for j, char in enumerate(" " + p):
        x = (j + 1) * cell_size
        rect = plt.Rectangle((x, (len(s) + 1) * cell_size), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        display = "''" if char == " " else char
        ax.text(x + cell_size/2, (len(s) + 1.5) * cell_size, display,
               ha='center', va='center', fontsize=12, fontweight='bold')

    # Header column (string)
    for i, char in enumerate(" " + s):
        y = (len(s) - i) * cell_size
        rect = plt.Rectangle((0, y), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        display = "''" if char == " " else char
        ax.text(cell_size/2, y + cell_size/2, display,
               ha='center', va='center', fontsize=12, fontweight='bold')

    # DP cells
    for i in range(len(s) + 1):
        for j in range(len(p) + 1):
            x = (j + 1) * cell_size
            y = (len(s) - i) * cell_size

            color = '#90EE90' if dp[i][j] else '#FFB6C1'
            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, 'T' if dp[i][j] else 'F',
                   ha='center', va='center', fontsize=12, fontweight='bold')

    ax.text(10, 4, 'Rules:', ha='left', fontsize=11, fontweight='bold')
    ax.text(10, 3.3, "'.' matches any char", ha='left', fontsize=10)
    ax.text(10, 2.6, "'*' matches 0+ of prev", ha='left', fontsize=10)

    ax.text(10, 1.5, 'Result: True', ha='left', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Regular Expression Matching: s="aab", p="c*a*b"', fontsize=14, fontweight='bold')

    save_fig('regex-matching.png')


def wildcard_matching_visualization():
    """Generate wildcard matching DP visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))

    s = "adceb"
    p = "*a*b"

    # DP table
    dp = [
        [True, True, False, False, False],
        [False, True, True, True, True],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, True, True, True],
        [False, False, False, False, True],
    ]

    cell_size = 1.3

    # Header row (pattern)
    for j, char in enumerate(" " + p):
        x = (j + 1) * cell_size
        rect = plt.Rectangle((x, (len(s) + 1) * cell_size), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        display = "''" if char == " " else char
        ax.text(x + cell_size/2, (len(s) + 1.5) * cell_size, display,
               ha='center', va='center', fontsize=12, fontweight='bold')

    # Header column (string)
    for i, char in enumerate(" " + s):
        y = (len(s) - i) * cell_size
        rect = plt.Rectangle((0, y), cell_size, cell_size,
                             facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        display = "''" if char == " " else char
        ax.text(cell_size/2, y + cell_size/2, display,
               ha='center', va='center', fontsize=12, fontweight='bold')

    # DP cells
    for i in range(len(s) + 1):
        for j in range(len(p) + 1):
            x = (j + 1) * cell_size
            y = (len(s) - i) * cell_size

            color = '#90EE90' if dp[i][j] else '#FFB6C1'
            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, 'T' if dp[i][j] else 'F',
                   ha='center', va='center', fontsize=12, fontweight='bold')

    ax.text(9, 5, 'Rules:', ha='left', fontsize=11, fontweight='bold')
    ax.text(9, 4.3, "'?' matches any single char", ha='left', fontsize=10)
    ax.text(9, 3.6, "'*' matches any sequence", ha='left', fontsize=10)

    ax.text(9, 2.2, 'Result: True', ha='left', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Wildcard Matching: s="adceb", p="*a*b"', fontsize=14, fontweight='bold')

    save_fig('wildcard-matching.png')


if __name__ == '__main__':
    print("Generating DP visualization images...")

    climbing_stairs_visualization()
    climbing_stairs_dp_table()
    house_robber_visualization()
    coin_change_visualization()
    lis_visualization()
    lcs_visualization()
    edit_distance_visualization()
    knapsack_visualization()
    unique_paths_visualization()
    minimum_path_sum_visualization()
    maximum_subarray_visualization()
    word_break_visualization()
    decode_ways_visualization()
    partition_subset_visualization()
    target_sum_visualization()
    matrix_chain_visualization()
    burst_balloons_visualization()
    regex_matching_visualization()
    wildcard_matching_visualization()

    print("\nAll visualizations generated successfully!")
