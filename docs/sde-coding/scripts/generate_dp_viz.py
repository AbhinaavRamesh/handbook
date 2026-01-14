#!/usr/bin/env python3
"""Generate Dynamic Programming visualizations."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/dp'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_climbing_stairs():
    """Generate climbing stairs DP visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Climbing Stairs: DP State Transition', fontsize=14, fontweight='bold')

    n = 6
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    # Draw stairs
    for i in range(n + 1):
        rect = patches.FancyBboxPatch((i * 1.5, 0), 1.2, dp[i] * 0.3 + 0.5,
                                      boxstyle="round,pad=0.02",
                                      facecolor=plt.cm.Blues(0.3 + 0.1 * i),
                                      edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(i * 1.5 + 0.6, dp[i] * 0.3 + 0.7, f'dp[{i}]={dp[i]}',
               ha='center', fontsize=10, fontweight='bold')
        ax.text(i * 1.5 + 0.6, 0.2, f'Step {i}', ha='center', fontsize=9)

    # Draw arrows showing transitions
    for i in range(2, n + 1):
        # Arrow from i-1
        ax.annotate('', xy=(i * 1.5, dp[i] * 0.3 + 0.3),
                   xytext=((i-1) * 1.5 + 1.2, dp[i-1] * 0.3 + 0.3),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        # Arrow from i-2
        ax.annotate('', xy=(i * 1.5, dp[i] * 0.3 + 0.1),
                   xytext=((i-2) * 1.5 + 1.2, dp[i-2] * 0.3 + 0.3),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=1.5,
                                  connectionstyle='arc3,rad=0.2'))

    ax.text(5, 4.5, 'dp[i] = dp[i-1] + dp[i-2]', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.text(5, 4, '(Fibonacci pattern!)', fontsize=10, style='italic')

    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/climbing_stairs.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/climbing_stairs.png")


def generate_knapsack():
    """Generate 0/1 Knapsack DP table visualization."""
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    W = 7
    n = len(weights)

    # Build DP table
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('0/1 Knapsack: DP Table (items × capacity)', fontsize=14, fontweight='bold')

    # Draw heatmap
    im = ax.imshow(dp, cmap='YlGnBu', aspect='auto')

    # Add text annotations
    for i in range(n + 1):
        for w in range(W + 1):
            text = ax.text(w, i, dp[i][w], ha='center', va='center',
                          fontsize=10, fontweight='bold',
                          color='white' if dp[i][w] > 5 else 'black')

    ax.set_xticks(range(W + 1))
    ax.set_xticklabels([f'W={w}' for w in range(W + 1)])
    ax.set_yticks(range(n + 1))
    ax.set_yticklabels(['No items'] + [f'Item {i}: w={weights[i-1]}, v={values[i-1]}' for i in range(1, n + 1)])
    ax.set_xlabel('Capacity', fontsize=12)
    ax.set_ylabel('Items considered', fontsize=12)

    # Highlight optimal
    rect = patches.Rectangle((W - 0.5, n - 0.5), 1, 1, linewidth=3,
                             edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(W + 0.5, n, f'← Optimal: {dp[n][W]}', fontsize=11, fontweight='bold',
           va='center', color='red')

    plt.colorbar(im, ax=ax, label='Maximum Value')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/knapsack_table.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/knapsack_table.png")


def generate_coin_change():
    """Generate coin change DP visualization."""
    coins = [1, 2, 5]
    amount = 11

    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    parent = [-1] * (amount + 1)

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = coin

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(f'Coin Change: Minimum coins for amount {amount} using coins {coins}',
                fontsize=14, fontweight='bold')

    # Draw DP array
    for i in range(amount + 1):
        val = dp[i] if dp[i] != float('inf') else '∞'
        color = plt.cm.Greens(0.3 + 0.05 * (dp[i] if dp[i] != float('inf') else 0))
        rect = patches.FancyBboxPatch((i * 1.1, 0), 0.9, 1,
                                      boxstyle="round,pad=0.02",
                                      facecolor=color if dp[i] != float('inf') else '#FADBD8',
                                      edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(i * 1.1 + 0.45, 0.5, str(val), ha='center', va='center',
               fontsize=11, fontweight='bold')
        ax.text(i * 1.1 + 0.45, -0.3, f'{i}', ha='center', fontsize=9)

    # Trace back solution
    path = []
    curr = amount
    while curr > 0 and parent[curr] != -1:
        path.append(parent[curr])
        curr -= parent[curr]

    ax.text(6, 1.8, f'Solution: {path} (total {len(path)} coins)', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.8, 2.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/coin_change.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/coin_change.png")


def generate_unique_paths():
    """Generate unique paths grid DP visualization."""
    m, n = 4, 5

    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Unique Paths: DP Grid (paths to each cell)', fontsize=14, fontweight='bold')

    # Draw grid
    for i in range(m):
        for j in range(n):
            color = plt.cm.Blues(0.2 + 0.03 * dp[i][j])
            rect = patches.Rectangle((j, m - 1 - i), 1, 1, facecolor=color,
                                     edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(j + 0.5, m - 0.5 - i, str(dp[i][j]),
                   ha='center', va='center', fontsize=14, fontweight='bold')

    # Mark start and end
    ax.text(0.5, m - 0.15, 'START', ha='center', fontsize=9, color='green', fontweight='bold')
    ax.text(n - 0.5, 0.15, 'END', ha='center', fontsize=9, color='red', fontweight='bold')

    # Show formula
    ax.text(n/2, -0.8, 'dp[i][j] = dp[i-1][j] + dp[i][j-1]', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-1.2, m + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/unique_paths.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/unique_paths.png")


def generate_target_sum():
    """Generate target sum visualization."""
    nums = [1, 1, 1, 1, 1]
    target = 3

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'Target Sum: Find ways to reach {target} with +/- signs', fontsize=14, fontweight='bold')

    # Show decision tree (partial)
    ax.text(7, 7, f'nums = {nums}, target = {target}', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='lightblue'))

    # Level 0
    ax.text(7, 5.5, '0', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

    # Level 1: +1 and -1
    ax.text(4, 4, '+1', ha='center', fontsize=12,
           bbox=dict(boxstyle='circle', facecolor='#D5F5E3', edgecolor='black'))
    ax.text(10, 4, '-1', ha='center', fontsize=12,
           bbox=dict(boxstyle='circle', facecolor='#FADBD8', edgecolor='black'))
    ax.annotate('', xy=(4.3, 4.3), xytext=(6.7, 5.3), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(9.7, 4.3), xytext=(7.3, 5.3), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Level 2
    for x, val in [(2.5, '+2'), (5.5, '0'), (8.5, '0'), (11.5, '-2')]:
        color = '#D5F5E3' if '+' in val else '#FADBD8' if '-' in val else '#FEF9E7'
        ax.text(x, 2.5, val, ha='center', fontsize=11,
               bbox=dict(boxstyle='circle', facecolor=color, edgecolor='black'))

    ax.text(7, 1, '... continues branching (+/-) for each number ...', ha='center',
           fontsize=10, style='italic')

    ax.text(7, 0, f'Answer: 5 ways to reach target {target}', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow'))

    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 8)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/target_sum.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/target_sum.png")


def generate_dp_patterns():
    """Generate DP patterns overview."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Common DP Patterns', fontsize=16, fontweight='bold')

    patterns = [
        ('1D Linear', 'dp[i] depends on dp[i-1], dp[i-2]...',
         [(0, 1), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8)]),
        ('2D Grid', 'dp[i][j] depends on dp[i-1][j], dp[i][j-1]',
         None),
        ('Interval DP', 'dp[i][j] = min over k of dp[i][k] + dp[k][j]',
         None),
        ('Knapsack', 'dp[i][w] = max(take, skip)',
         None)
    ]

    for ax, (name, formula, data) in zip(axes.flat, patterns):
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.text(0.5, 0.6, formula, ha='center', va='center', fontsize=10,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightyellow'))

        if data:
            x = [d[0] for d in data]
            y = [d[1] for d in data]
            ax.bar(x, y, color='steelblue', edgecolor='black')
            for i, (xi, yi) in enumerate(data):
                ax.text(xi, yi + 0.2, f'dp[{xi}]={yi}', ha='center', fontsize=8)

        ax.set_xlim(-0.5, 6)
        ax.set_ylim(0, 1 if not data else max(d[1] for d in data) + 2)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dp_patterns.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/dp_patterns.png")


if __name__ == '__main__':
    generate_climbing_stairs()
    generate_knapsack()
    generate_coin_change()
    generate_unique_paths()
    generate_target_sum()
    generate_dp_patterns()
    print("\n✅ All DP visualizations generated!")
