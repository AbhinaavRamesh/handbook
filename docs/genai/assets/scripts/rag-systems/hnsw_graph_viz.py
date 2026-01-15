#!/usr/bin/env python3
"""
HNSW Graph Visualization
Visualizes the Hierarchical Navigable Small World graph structure for vector indexing.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'images', 'rag-systems')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_hnsw_structure():
    """Create visualization of HNSW hierarchical structure."""
    fig, ax = plt.subplots(figsize=(14, 10))

    np.random.seed(42)

    # Layer configurations
    layers = {
        2: {'nodes': 3, 'y_offset': 8, 'color': '#E8F5E9', 'label': 'Layer 2 (Coarsest)'},
        1: {'nodes': 8, 'y_offset': 5, 'color': '#FFF3E0', 'label': 'Layer 1'},
        0: {'nodes': 20, 'y_offset': 1.5, 'color': '#E3F2FD', 'label': 'Layer 0 (Finest)'}
    }

    all_positions = {}

    for layer, config in layers.items():
        n_nodes = config['nodes']
        y_base = config['y_offset']
        color = config['color']

        # Generate node positions
        x_positions = np.linspace(1, 13, n_nodes)
        y_positions = [y_base + np.random.uniform(-0.3, 0.3) for _ in range(n_nodes)]

        # Draw background for layer
        rect = mpatches.FancyBboxPatch(
            (0.3, y_base - 1), 13.4, 2,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='gray',
            linewidth=1.5,
            alpha=0.5
        )
        ax.add_patch(rect)
        ax.text(0.5, y_base + 0.8, config['label'], fontsize=10, fontweight='bold')

        # Draw nodes
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            node_id = f'L{layer}N{i}'
            all_positions[node_id] = (x, y)

            circle = plt.Circle((x, y), 0.25, color='white', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(i), ha='center', va='center', fontsize=8, fontweight='bold')

        # Draw edges within layer (small-world connections)
        for i in range(n_nodes):
            # Connect to neighbors
            neighbors_count = min(4, n_nodes - 1)
            for j in range(1, neighbors_count + 1):
                if i + j < n_nodes:
                    ax.plot([x_positions[i], x_positions[i + j]],
                           [y_positions[i], y_positions[i + j]],
                           'k-', linewidth=1, alpha=0.5)

            # Long-range connections (small-world property)
            if np.random.random() < 0.3 and i + 5 < n_nodes:
                ax.plot([x_positions[i], x_positions[i + 5]],
                       [y_positions[i], y_positions[i + 5]],
                       'b--', linewidth=1, alpha=0.3)

    # Draw inter-layer connections (skip connections)
    layer2_x = np.linspace(1, 13, layers[2]['nodes'])
    layer1_x = np.linspace(1, 13, layers[1]['nodes'])
    layer0_x = np.linspace(1, 13, layers[0]['nodes'])

    # Connect layer 2 to layer 1
    for i, x2 in enumerate(layer2_x):
        # Find closest nodes in layer 1
        closest_idx = np.argmin(np.abs(layer1_x - x2))
        ax.annotate('', xy=(layer1_x[closest_idx], layers[1]['y_offset'] + 0.5),
                   xytext=(x2, layers[2]['y_offset'] - 0.5),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.7))

    # Connect layer 1 to layer 0
    for i, x1 in enumerate(layer1_x):
        closest_idx = np.argmin(np.abs(layer0_x - x1))
        ax.annotate('', xy=(layer0_x[closest_idx], layers[0]['y_offset'] + 0.5),
                   xytext=(x1, layers[1]['y_offset'] - 0.5),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=1.5, alpha=0.7))

    # Highlight search path
    search_path = [
        (layer2_x[1], layers[2]['y_offset']),
        (layer1_x[3], layers[1]['y_offset']),
        (layer0_x[7], layers[0]['y_offset']),
        (layer0_x[8], layers[0]['y_offset']),
    ]

    for i in range(len(search_path) - 1):
        ax.annotate('', xy=search_path[i + 1], xytext=search_path[i],
                   arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8))

    # Mark query entry and result
    ax.plot(search_path[0][0], search_path[0][1], 'r*', markersize=20, label='Entry Point')
    ax.plot(search_path[-1][0], search_path[-1][1], 'g*', markersize=20, label='Nearest Neighbor')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.set_title('HNSW: Hierarchical Navigable Small World Graph', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=3, label='Search Path'),
        plt.Line2D([0], [0], color='green', linestyle='--', lw=1.5, label='Layer Connections'),
        plt.Line2D([0], [0], marker='*', color='red', markersize=15, linestyle='None', label='Entry Point'),
        plt.Line2D([0], [0], marker='*', color='green', markersize=15, linestyle='None', label='Result'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'hnsw_structure.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_index_comparison():
    """Compare different indexing algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Data points
    n_vectors = [10000, 100000, 1000000, 10000000, 100000000]
    n_labels = ['10K', '100K', '1M', '10M', '100M']

    # Query latency (ms)
    flat_latency = [10, 100, 1000, 10000, 100000]
    ivf_latency = [5, 15, 50, 200, 800]
    hnsw_latency = [1, 2, 5, 10, 20]

    # Recall@10
    flat_recall = [1.0, 1.0, 1.0, 1.0, 1.0]
    ivf_recall = [0.92, 0.90, 0.88, 0.85, 0.82]
    hnsw_recall = [0.98, 0.97, 0.96, 0.95, 0.94]

    # Memory usage (GB)
    flat_memory = [0.15, 1.5, 15, 150, 1500]
    ivf_memory = [0.16, 1.6, 16, 160, 1600]
    hnsw_memory = [0.25, 2.5, 25, 250, 2500]

    # Build time (seconds)
    flat_build = [1, 10, 100, 1000, 10000]
    ivf_build = [5, 50, 500, 5000, 50000]
    hnsw_build = [10, 100, 1000, 10000, 100000]

    # Plot 1: Query Latency
    ax1 = axes[0, 0]
    ax1.semilogy(n_labels, flat_latency, 'o-', label='Flat (Brute Force)',
                 linewidth=2, markersize=8, color='#F44336')
    ax1.semilogy(n_labels, ivf_latency, 's-', label='IVF',
                 linewidth=2, markersize=8, color='#FF9800')
    ax1.semilogy(n_labels, hnsw_latency, '^-', label='HNSW',
                 linewidth=2, markersize=8, color='#4CAF50')
    ax1.set_xlabel('Number of Vectors')
    ax1.set_ylabel('Query Latency (ms)')
    ax1.set_title('Query Latency Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    ax1.text(4, 150, '100ms SLA', fontsize=9, color='red')

    # Plot 2: Recall@10
    ax2 = axes[0, 1]
    ax2.plot(n_labels, flat_recall, 'o-', label='Flat (Exact)',
             linewidth=2, markersize=8, color='#F44336')
    ax2.plot(n_labels, ivf_recall, 's-', label='IVF',
             linewidth=2, markersize=8, color='#FF9800')
    ax2.plot(n_labels, hnsw_recall, '^-', label='HNSW',
             linewidth=2, markersize=8, color='#4CAF50')
    ax2.set_xlabel('Number of Vectors')
    ax2.set_ylabel('Recall@10')
    ax2.set_title('Recall Quality Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.75, 1.02)
    ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.5)
    ax2.text(4, 0.96, '95% Target', fontsize=9, color='green')

    # Plot 3: Memory Usage
    ax3 = axes[1, 0]
    ax3.semilogy(n_labels, flat_memory, 'o-', label='Flat',
                 linewidth=2, markersize=8, color='#F44336')
    ax3.semilogy(n_labels, ivf_memory, 's-', label='IVF',
                 linewidth=2, markersize=8, color='#FF9800')
    ax3.semilogy(n_labels, hnsw_memory, '^-', label='HNSW',
                 linewidth=2, markersize=8, color='#4CAF50')
    ax3.set_xlabel('Number of Vectors')
    ax3.set_ylabel('Memory Usage (GB)')
    ax3.set_title('Memory Requirements', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Trade-off summary
    ax4 = axes[1, 1]

    # Summary table as text
    ax4.axis('off')
    table_data = [
        ['Index Type', 'Latency', 'Recall', 'Memory', 'Best For'],
        ['Flat', 'O(n)', '100%', 'O(n)', '<10K vectors'],
        ['IVF', 'O(n/k)', '90-95%', 'O(n)', '10K-10M vectors'],
        ['HNSW', 'O(log n)', '95-99%', 'O(n*M)', '>100K vectors'],
        ['IVF-PQ', 'O(n/k)', '85-90%', 'O(n/4)', '>100M vectors'],
    ]

    table = ax4.table(cellText=table_data,
                      loc='center',
                      cellLoc='center',
                      colWidths=[0.2, 0.15, 0.15, 0.15, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header row
    for i in range(5):
        table[(0, i)].set_facecolor('#E3F2FD')
        table[(0, i)].set_text_props(fontweight='bold')

    # Color best cells
    table[(3, 1)].set_facecolor('#C8E6C9')  # HNSW latency
    table[(1, 2)].set_facecolor('#C8E6C9')  # Flat recall

    ax4.set_title('Index Selection Guide', fontsize=12, fontweight='bold', y=0.9)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'index_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_hnsw_parameters():
    """Visualize impact of HNSW parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # M parameter impact
    ax1 = axes[0]
    M_values = [4, 8, 16, 32, 64]
    recall = [0.85, 0.92, 0.96, 0.98, 0.99]
    memory_ratio = [0.6, 0.8, 1.0, 1.4, 2.0]
    build_time_ratio = [0.5, 0.7, 1.0, 1.5, 2.2]

    ax1_twin = ax1.twinx()

    line1 = ax1.plot(M_values, recall, 'o-', label='Recall@10',
                     linewidth=2, markersize=8, color='#4CAF50')
    line2 = ax1_twin.plot(M_values, memory_ratio, 's--', label='Memory (relative)',
                          linewidth=2, markersize=8, color='#F44336')
    line3 = ax1_twin.plot(M_values, build_time_ratio, '^--', label='Build Time (relative)',
                          linewidth=2, markersize=8, color='#FF9800')

    ax1.set_xlabel('M (Max Connections)')
    ax1.set_ylabel('Recall@10', color='#4CAF50')
    ax1_twin.set_ylabel('Relative Cost', color='#F44336')
    ax1.set_title('Impact of M Parameter', fontsize=12, fontweight='bold')

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.8, 1.0)
    ax1_twin.set_ylim(0, 2.5)

    # Highlight recommended
    ax1.axvline(x=16, color='gray', linestyle='--', alpha=0.5)
    ax1.text(16.5, 0.82, 'Recommended', fontsize=9, color='gray')

    # ef_search parameter impact
    ax2 = axes[1]
    ef_values = [16, 32, 64, 128, 256, 512]
    recall_ef = [0.88, 0.93, 0.96, 0.98, 0.99, 0.995]
    latency_ratio = [0.3, 0.5, 1.0, 1.8, 3.5, 7.0]

    ax2_twin = ax2.twinx()

    line1 = ax2.plot(ef_values, recall_ef, 'o-', label='Recall@10',
                     linewidth=2, markersize=8, color='#4CAF50')
    line2 = ax2_twin.plot(ef_values, latency_ratio, 's--', label='Latency (relative)',
                          linewidth=2, markersize=8, color='#F44336')

    ax2.set_xlabel('ef_search')
    ax2.set_ylabel('Recall@10', color='#4CAF50')
    ax2_twin.set_ylabel('Relative Latency', color='#F44336')
    ax2.set_title('Impact of ef_search Parameter', fontsize=12, fontweight='bold')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.85, 1.0)
    ax2_twin.set_ylim(0, 8)

    # Highlight trade-off zone
    ax2.axvspan(64, 128, alpha=0.2, color='green')
    ax2.text(96, 0.86, 'Optimal\nRange', ha='center', fontsize=9, color='green')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'hnsw_parameters.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    create_hnsw_structure()
    create_index_comparison()
    create_hnsw_parameters()
    print("HNSW graph visualizations complete!")
