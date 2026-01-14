#!/usr/bin/env python3
"""Generate Heap visualizations."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/heaps'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_heap_tree(ax, heap, title="Heap Structure"):
    """Draw a heap as a binary tree."""
    if not heap:
        return

    n = len(heap)
    # Calculate positions for each node
    positions = {}
    max_depth = int(np.log2(n)) + 1 if n > 0 else 0

    for i in range(n):
        depth = int(np.log2(i + 1))
        pos_in_level = i - (2**depth - 1)
        nodes_at_level = 2**depth

        x = (pos_in_level + 0.5) / nodes_at_level
        y = 1 - (depth + 0.5) / (max_depth + 0.5)
        positions[i] = (x, y)

    # Draw edges first
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            ax.plot([positions[i][0], positions[left][0]],
                   [positions[i][1], positions[left][1]], 'k-', lw=1.5)
        if right < n:
            ax.plot([positions[i][0], positions[right][0]],
                   [positions[i][1], positions[right][1]], 'k-', lw=1.5)

    # Draw nodes
    for i, val in enumerate(heap):
        x, y = positions[i]
        color = '#E74C3C' if i == 0 else '#3498DB'
        circle = plt.Circle((x, y), 0.04, color=color, ec='black', lw=2)
        ax.add_patch(circle)
        ax.text(x, y, str(val), ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


def generate_heap_structure():
    """Generate heap structure visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Heap Property: Parent vs Children', fontsize=16, fontweight='bold')

    # Min-Heap
    min_heap = [1, 3, 2, 7, 6, 4, 5]
    draw_heap_tree(axes[0], min_heap, "Min-Heap: parent ≤ children")

    # Max-Heap
    max_heap = [9, 7, 8, 3, 4, 5, 6]
    draw_heap_tree(axes[1], max_heap, "Max-Heap: parent ≥ children")

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/heap_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/heap_structure.png")


def generate_top_k_elements():
    """Generate top-k elements visualization."""
    nums = [3, 2, 1, 5, 6, 4]
    k = 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Find Top {k} Largest Using Min-Heap of Size {k}', fontsize=14, fontweight='bold')

    # Simulate the process
    heap = []
    states = []

    import heapq
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)
        states.append((num, heap.copy()))

    for idx, (num, h) in enumerate(states):
        ax = axes[idx // 3, idx % 3]

        # Draw heap
        draw_heap_tree(ax, sorted(h), f"After processing {num}")

        # Show current number
        ax.text(0.5, -0.05, f'Heap: {sorted(h)}', ha='center', fontsize=10,
               transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_k_elements.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/top_k_elements.png")


def generate_median_stream():
    """Generate find median from stream visualization."""
    stream = [5, 2, 3, 4, 1, 6]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Find Median from Data Stream: Two Heaps', fontsize=14, fontweight='bold')

    # Show the two-heap structure
    ax.text(7, 7, 'Two Heaps Approach:', ha='center', fontsize=12, fontweight='bold')

    # Max-heap (left half)
    ax.add_patch(patches.FancyBboxPatch((1, 3), 5, 3, boxstyle="round,pad=0.1",
                                        facecolor='#AED6F1', edgecolor='black', lw=2))
    ax.text(3.5, 5.5, 'Max-Heap', ha='center', fontsize=11, fontweight='bold')
    ax.text(3.5, 4.5, '(smaller half)', ha='center', fontsize=10)
    ax.text(3.5, 3.5, '[2, 1] → top=2', ha='center', fontsize=10, family='monospace')

    # Min-heap (right half)
    ax.add_patch(patches.FancyBboxPatch((8, 3), 5, 3, boxstyle="round,pad=0.1",
                                        facecolor='#FADBD8', edgecolor='black', lw=2))
    ax.text(10.5, 5.5, 'Min-Heap', ha='center', fontsize=11, fontweight='bold')
    ax.text(10.5, 4.5, '(larger half)', ha='center', fontsize=10)
    ax.text(10.5, 3.5, '[3, 4, 5, 6] → top=3', ha='center', fontsize=10, family='monospace')

    # Arrow showing median
    ax.annotate('', xy=(6.5, 4.5), xytext=(7.5, 4.5),
               arrowprops=dict(arrowstyle='<->', color='green', lw=3))
    ax.text(7, 2, 'Median = (max_heap.top + min_heap.top) / 2\n= (2 + 3) / 2 = 2.5',
           ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow'))

    # Rules
    ax.text(7, 0.5, 'Rules: 1) Balance sizes (differ by at most 1)\n'
                    '       2) All in max-heap ≤ All in min-heap',
           ha='center', fontsize=10, style='italic')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/median_stream.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/median_stream.png")


def generate_heap_operations():
    """Generate heap operations (insert/extract) visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Heap Operations: O(log n)', fontsize=14, fontweight='bold')

    # Initial heap
    heap1 = [1, 3, 2, 7, 6, 4, 5]
    draw_heap_tree(axes[0], heap1, "Initial Min-Heap")

    # After insert(0) - bubble up
    heap2 = [0, 1, 2, 3, 6, 4, 5, 7]
    draw_heap_tree(axes[1], heap2, "After insert(0): Bubble Up")
    axes[1].text(0.5, -0.1, '0 bubbles up to root', ha='center', fontsize=10,
                transform=axes[1].transAxes, color='green')

    # After extract_min - bubble down
    heap3 = [1, 3, 2, 7, 6, 4, 5]
    draw_heap_tree(axes[2], heap3, "After extract_min(): Bubble Down")
    axes[2].text(0.5, -0.1, 'Last element moves to root, bubbles down', ha='center',
                fontsize=10, transform=axes[2].transAxes, color='red')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/heap_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/heap_operations.png")


def generate_stock_trading():
    """Generate stock trading visualization."""
    prices = [7, 1, 5, 3, 6, 4]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Best Time to Buy and Sell Stock', fontsize=14, fontweight='bold')

    # Plot prices
    ax.plot(range(len(prices)), prices, 'b-o', linewidth=2, markersize=10)

    # Annotate buy and sell points
    buy_day, sell_day = 1, 4
    ax.annotate('BUY', (buy_day, prices[buy_day]), textcoords="offset points",
               xytext=(0, -25), ha='center', fontsize=11, fontweight='bold', color='green')
    ax.annotate('SELL', (sell_day, prices[sell_day]), textcoords="offset points",
               xytext=(0, 15), ha='center', fontsize=11, fontweight='bold', color='red')

    # Draw profit arrow
    ax.annotate('', xy=(sell_day, prices[sell_day]), xytext=(buy_day, prices[buy_day]),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(2.5, 4.5, f'Profit = {prices[sell_day]} - {prices[buy_day]} = {prices[sell_day] - prices[buy_day]}',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow'))

    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xticks(range(len(prices)))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/stock_trading.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/stock_trading.png")


if __name__ == '__main__':
    generate_heap_structure()
    generate_top_k_elements()
    generate_median_stream()
    generate_heap_operations()
    generate_stock_trading()
    print("\n✅ All heap visualizations generated!")
