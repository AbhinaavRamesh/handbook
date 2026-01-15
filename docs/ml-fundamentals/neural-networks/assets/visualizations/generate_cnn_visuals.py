#!/usr/bin/env python3
"""
Generate visualizations for CNNs module.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def create_cnn_convolution():
    """Animate convolution operation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input matrix
    np.random.seed(42)
    input_size = 5
    input_data = np.random.randint(0, 10, (input_size, input_size))

    # Filter
    filter_data = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])
    filter_size = 3

    # Output
    output_size = input_size - filter_size + 1
    output_data = np.zeros((output_size, output_size))

    frames = output_size * output_size

    def animate(frame):
        for ax in axes:
            ax.clear()

        i = frame // output_size
        j = frame % output_size

        # Compute convolution at this position
        region = input_data[i:i+filter_size, j:j+filter_size]
        output_data[i, j] = np.sum(region * filter_data)

        # Draw input
        ax = axes[0]
        ax.imshow(input_data, cmap='Blues', vmin=0, vmax=10)
        for x in range(input_size):
            for y in range(input_size):
                ax.text(y, x, str(input_data[x, y]), ha='center', va='center', fontsize=10)

        # Highlight current region
        rect = Rectangle((j - 0.5, i - 0.5), filter_size, filter_size,
                         fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
        ax.set_title('Input (5x5)', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw filter
        ax = axes[1]
        ax.imshow(filter_data, cmap='RdYlGn', vmin=-2, vmax=2)
        for x in range(filter_size):
            for y in range(filter_size):
                ax.text(y, x, str(filter_data[x, y]), ha='center', va='center',
                       fontsize=12, fontweight='bold')
        ax.set_title('Filter (3x3)', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw output
        ax = axes[2]
        ax.imshow(output_data, cmap='Oranges', vmin=-20, vmax=20)
        for x in range(output_size):
            for y in range(output_size):
                if output_data[x, y] != 0 or (x <= i and y <= j) or (x == i and y <= j):
                    ax.text(y, x, f'{int(output_data[x, y])}', ha='center', va='center', fontsize=10)
        rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
        ax.set_title('Output (3x3)', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        fig.suptitle(f'Convolution: Position ({i}, {j})\nSum = {int(output_data[i, j])}',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=500, blit=False)
    writer = PillowWriter(fps=2)
    anim.save(os.path.join(OUTPUT_DIR, 'cnn_convolution.gif'), writer=writer)
    plt.close()
    print("Generated: cnn_convolution.gif")


def create_cnn_pooling():
    """Show max vs average pooling."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Input
    input_data = np.array([[1, 3, 2, 4],
                          [5, 7, 6, 8],
                          [9, 11, 10, 12],
                          [13, 15, 14, 16]])

    # Max pooling
    max_pool = np.array([[7, 8],
                        [15, 16]])

    # Average pooling
    avg_pool = np.array([[4, 5],
                        [12, 13]])

    for ax, data, title, cmap in [(axes[0], input_data, 'Input (4x4)', 'Blues'),
                                  (axes[1], max_pool, 'Max Pooling (2x2)', 'Reds'),
                                  (axes[2], avg_pool, 'Average Pooling (2x2)', 'Greens')]:
        ax.imshow(data, cmap=cmap, vmin=0, vmax=16)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, str(data[i, j]), ha='center', va='center',
                       fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # Add annotations
    axes[0].text(2, -0.8, 'Pool size: 2x2, Stride: 2', ha='center', fontsize=10)
    axes[1].text(0.5, -0.5, 'Takes maximum\nin each region', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[2].text(0.5, -0.5, 'Takes average\nin each region', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    fig.suptitle('Pooling Operations: Downsampling Feature Maps', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_pooling.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: cnn_pooling.svg")


def create_cnn_resnet():
    """Show ResNet skip connection architecture."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Main path
    blocks = [
        (2, 5, 'Input\nx', 'lightblue'),
        (5, 5, 'Conv\nBN\nReLU', 'lightgreen'),
        (8, 5, 'Conv\nBN', 'lightgreen'),
        (11, 5, 'ReLU', 'lightyellow'),
    ]

    for x, y, label, color in blocks:
        box = FancyBboxPatch((x - 0.8, y - 1.2), 1.6, 2.4,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Forward arrows
    for i in range(len(blocks) - 1):
        ax.annotate('', xy=(blocks[i+1][0] - 0.8, 5), xytext=(blocks[i][0] + 0.8, 5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Skip connection
    ax.annotate('', xy=(10.2, 6), xytext=(2.8, 6),
               arrowprops=dict(arrowstyle='->', color='red', lw=3,
                              connectionstyle='arc3,rad=0.3'))
    ax.text(6.5, 8, 'Skip Connection (Identity)', ha='center', fontsize=11,
           color='red', fontweight='bold')

    # Addition symbol
    ax.text(10.2, 5, '+', fontsize=20, fontweight='bold', color='red')

    # Formula
    ax.text(7, 1.5, r'$y = F(x) + x$', fontsize=16, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.text(7, 0.5, 'Gradient: ∂y/∂x = F\'(x) + 1 (always ≥ 1)', fontsize=11, ha='center')

    ax.set_title('ResNet Residual Block: Skip Connection for Gradient Flow', fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_resnet.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: cnn_resnet.svg")


def create_cnn_filters():
    """Show multiple filters/feature maps visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Input image representation
    input_x, input_y = 1.5, 4
    input_size = 2.0
    input_rect = FancyBboxPatch((input_x - input_size/2, input_y - input_size/2),
                                 input_size, input_size,
                                 boxstyle="round,pad=0.02",
                                 facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(input_x, input_y, 'Input\nImage', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(input_x, input_y - 1.5, '(H x W x 3)', ha='center', fontsize=9)

    # Draw multiple filters
    filter_colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA']
    filter_labels = ['Edge\nDetector', 'Corner\nDetector', 'Texture\nDetector', 'Blob\nDetector']
    filter_x_start = 4.5
    filter_size = 1.0

    for i in range(4):
        y_offset = 5.5 - i * 1.5
        rect = FancyBboxPatch((filter_x_start, y_offset - filter_size/2),
                              filter_size, filter_size,
                              boxstyle="round,pad=0.02",
                              facecolor=filter_colors[i], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(filter_x_start + filter_size/2, y_offset, f'F{i+1}',
                ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(filter_x_start + filter_size + 0.3, y_offset, filter_labels[i],
                ha='left', va='center', fontsize=8)

        # Arrow from input to filter
        ax.annotate('', xy=(filter_x_start - 0.1, y_offset),
                   xytext=(input_x + input_size/2 + 0.1, input_y),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1,
                                  connectionstyle=f'arc3,rad={0.2 - i*0.15}'))

    # Convolution symbol
    ax.text(7.5, 4, '*', fontsize=30, ha='center', va='center', fontweight='bold')
    ax.text(7.5, 3, 'Convolution', ha='center', va='center', fontsize=9)

    # Draw output feature maps (stacked)
    output_x = 10.5
    output_base_y = 4
    map_size = 1.8
    map_offset = 0.25

    for i in range(4):
        depth_offset = (3 - i) * map_offset
        rect = FancyBboxPatch((output_x - map_size/2 + depth_offset,
                               output_base_y - map_size/2 + depth_offset),
                              map_size, map_size,
                              boxstyle="round,pad=0.02",
                              facecolor=filter_colors[3-i], edgecolor='black', linewidth=1.5,
                              alpha=0.9)
        ax.add_patch(rect)

    ax.text(output_x + map_offset * 1.5, output_base_y + map_offset * 1.5,
            'Feature\nMaps', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(output_x + map_offset * 1.5, output_base_y - 1.5,
            "(H' x W' x 4)", ha='center', fontsize=9)

    # Arrows from filters to output
    ax.annotate('', xy=(output_x - map_size/2 - 0.1, output_base_y),
               xytext=(7.9, 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Labels
    ax.text(4.8, 7.2, '4 Filters\n(3x3 each)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    ax.text(output_x + map_offset * 1.5, 7.2, '4 Feature Maps\n(one per filter)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    # Key insight at bottom
    ax.text(7, 0.8, 'Each filter detects a specific pattern. N filters produce N feature maps (channels).',
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))

    ax.set_title('Multiple Filters = Multiple Feature Maps', fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_filters.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: cnn_filters.svg")


def create_cnn_receptive_field():
    """Animate receptive field growth."""
    fig, ax = plt.subplots(figsize=(10, 10))

    layers = 4
    frames = layers * 10

    def animate(frame):
        ax.clear()
        current_layer = min(frame // 10, layers - 1)

        # Draw input grid
        grid_size = 13
        for i in range(grid_size):
            for j in range(grid_size):
                rect = Rectangle((j, grid_size - 1 - i), 1, 1,
                                facecolor='white', edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)

        # Highlight receptive field
        center = grid_size // 2
        rf_sizes = [1, 3, 5, 7]  # Receptive field grows with 3x3 convolutions
        rf_size = rf_sizes[current_layer]
        rf_start = center - rf_size // 2

        for i in range(rf_size):
            for j in range(rf_size):
                x, y = rf_start + j, grid_size - 1 - (rf_start + i)
                color_intensity = 1 - (abs(i - rf_size//2) + abs(j - rf_size//2)) / rf_size
                rect = Rectangle((x, y), 1, 1,
                                facecolor=plt.cm.Reds(0.3 + color_intensity * 0.7),
                                edgecolor='red', linewidth=1)
                ax.add_patch(rect)

        # Mark center (output neuron)
        rect = Rectangle((center, grid_size - 1 - center), 1, 1,
                        facecolor='blue', edgecolor='darkblue', linewidth=2)
        ax.add_patch(rect)

        ax.set_xlim(-0.5, grid_size + 0.5)
        ax.set_ylim(-0.5, grid_size + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f'Receptive Field After {current_layer + 1} Conv Layers (3x3 each)\n'
                    f'RF Size: {rf_size}x{rf_size}', fontsize=13, fontweight='bold')

        ax.text(grid_size / 2, -1.5,
               'Blue = output neuron, Red = input region that affects it',
               ha='center', fontsize=10)

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=300, blit=False)
    writer = PillowWriter(fps=3)
    anim.save(os.path.join(OUTPUT_DIR, 'cnn_receptive_field.gif'), writer=writer)
    plt.close()
    print("Generated: cnn_receptive_field.gif")


if __name__ == '__main__':
    print("Generating CNN visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_cnn_convolution()
    create_cnn_pooling()
    create_cnn_resnet()
    create_cnn_filters()
    create_cnn_receptive_field()

    print("\nAll CNN visualizations generated!")
