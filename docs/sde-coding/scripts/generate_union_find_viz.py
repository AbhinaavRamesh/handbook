#!/usr/bin/env python3
"""Generate Union-Find visualizations for the tutorial site."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/sde-coding/union-find'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_union_operations():
    """Generate step-by-step union operations visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Union-Find: Building Connected Components', fontsize=16, fontweight='bold')

    edges = [(0, 1), (2, 3), (1, 2), (3, 4)]
    component_colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6']

    # Track parent array
    parent = list(range(5))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    def get_component(x):
        return find(x)

    for step, (u, v) in enumerate(edges):
        ax = axes[step // 2, step % 2]
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 3)
        ax.axis('off')

        union(u, v)

        # Group nodes by component
        components = {}
        for i in range(5):
            root = find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)

        # Draw nodes with component colors
        x_positions = [0, 1, 2, 3, 4]
        for i in range(5):
            root = find(i)
            color = component_colors[root]
            circle = plt.Circle((x_positions[i], 1.5), 0.35, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x_positions[i], 1.5, str(i), ha='center', va='center', fontsize=14, fontweight='bold')

        # Draw edge being added
        ax.plot([x_positions[u], x_positions[v]], [1.5, 1.5], 'g-', linewidth=3, alpha=0.5)

        # Show parent array
        ax.text(2.5, 0.5, f'parent = {parent.copy()}', ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        comp_str = ', '.join([f'{{{",".join(str(n) for n in nodes)}}}' for nodes in components.values()])
        ax.set_title(f'Step {step + 1}: union({u}, {v})\nComponents: {comp_str}', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/union_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/union_operations.png")


def generate_path_compression():
    """Generate path compression before/after visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Path Compression: Flattening the Tree', fontsize=16, fontweight='bold')

    # Before: long chain 4 -> 3 -> 2 -> 1 -> 0
    ax1 = axes[0]
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-0.5, 5.5)
    ax1.axis('off')
    ax1.set_title('Before: Tall Chain\nfind(4) traverses 4 nodes', fontsize=11)

    positions = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    for i, (x, y) in enumerate(positions):
        color = '#3498DB' if i == 0 else '#BDC3C7'
        circle = plt.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y, str(i), ha='center', va='center', fontsize=12, fontweight='bold')
        if i < 4:
            ax1.annotate('', xy=(x, y + 0.35), xytext=(x, y + 0.65),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax1.text(1, -0.3, 'Root', ha='center', fontsize=10, color='#3498DB')

    # During: find(4) traversing
    ax2 = axes[1]
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-0.5, 5.5)
    ax2.axis('off')
    ax2.set_title('During: find(4)\nTraversing path to root', fontsize=11)

    for i, (x, y) in enumerate(positions):
        color = '#F39C12' if i > 0 else '#3498DB'
        circle = plt.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x, y, str(i), ha='center', va='center', fontsize=12, fontweight='bold')
        if i < 4:
            ax2.annotate('', xy=(x, y + 0.35), xytext=(x, y + 0.65),
                        arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=3))

    ax2.text(2, 2, '← Following\n    parent\n    pointers', fontsize=10, color='#E74C3C')

    # After: flattened
    ax3 = axes[2]
    ax3.set_xlim(-1, 5)
    ax3.set_ylim(-0.5, 2.5)
    ax3.axis('off')
    ax3.set_title('After: Flat Tree\nAll nodes point to root', fontsize=11)

    root_pos = (2, 0.5)
    child_positions = [(0, 1.5), (1, 1.5), (2, 1.5), (3, 1.5), (4, 1.5)]

    # Draw root
    circle = plt.Circle(root_pos, 0.3, color='#3498DB', ec='black', linewidth=2)
    ax3.add_patch(circle)
    ax3.text(root_pos[0], root_pos[1], '0', ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw children pointing to root
    for i in range(1, 5):
        x, y = child_positions[i]
        circle = plt.Circle((x, y), 0.3, color='#27AE60', ec='black', linewidth=2)
        ax3.add_patch(circle)
        ax3.text(x, y, str(i), ha='center', va='center', fontsize=12, fontweight='bold')
        ax3.annotate('', xy=(root_pos[0], root_pos[1] + 0.35), xytext=(x, y - 0.35),
                    arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2,
                                   connectionstyle='arc3,rad=0.2'))

    ax3.text(2, -0.3, 'Now find(any) = O(1)!', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#27AE60', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/path_compression.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/path_compression.png")


def generate_cycle_detection():
    """Generate cycle detection with Union-Find visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Redundant Connection: Detecting Cycle with Union-Find', fontsize=14, fontweight='bold')

    edges = [(0, 1), (1, 2), (2, 0)]  # Last edge creates cycle

    parent = list(range(3))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    positions = {0: (0, 0), 1: (2, 0), 2: (1, 1.5)}

    for step in range(3):
        ax = axes[step // 2, step % 2] if step < 3 else None
        if step == 2:
            ax = axes[1, 0]

        ax.set_xlim(-1, 3)
        ax.set_ylim(-0.5, 2.5)
        ax.axis('off')

        u, v = edges[step]
        pu, pv = find(u), find(v)

        if pu == pv:
            # Cycle detected!
            ax.set_title(f'Edge ({u}, {v}): find({u})={pu}, find({v})={pv}\n⚠️ CYCLE DETECTED! Redundant edge!',
                        fontsize=11, color='#E74C3C')
            edge_color = '#E74C3C'
        else:
            parent[pu] = pv
            ax.set_title(f'Edge ({u}, {v}): find({u})={pu}, find({v})={pv}\n✓ Union performed',
                        fontsize=11, color='#27AE60')
            edge_color = '#27AE60'

        # Draw existing edges
        for i in range(step):
            eu, ev = edges[i]
            x1, y1 = positions[eu]
            x2, y2 = positions[ev]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

        # Draw current edge
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=3, linestyle='--' if pu == pv else '-', zorder=1)

        # Draw nodes
        for node, (x, y) in positions.items():
            circle = plt.Circle((x, y), 0.25, color='#3498DB', ec='black', linewidth=2, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, str(node), ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        ax.text(1, -0.3, f'parent = {parent.copy()}', ha='center', fontsize=10)

    # Final summary
    ax = axes[1, 1]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('Summary', fontsize=12, fontweight='bold')

    summary = """
Union-Find Cycle Detection:

1. Process edges one by one
2. For each edge (u, v):
   - If find(u) == find(v):
     → Cycle! This edge is redundant
   - Else: union(u, v)

3. Return the redundant edge
   (last one that creates cycle)
"""
    ax.text(0.1, 2.5, summary, fontsize=10, va='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='#ECF0F1'))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cycle_detection.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {OUTPUT_DIR}/cycle_detection.png")


if __name__ == '__main__':
    generate_union_operations()
    generate_path_compression()
    generate_cycle_detection()
    print("\n✅ All Union-Find visualizations generated!")
