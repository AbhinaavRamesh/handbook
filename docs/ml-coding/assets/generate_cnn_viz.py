"""
CNN/Convolution Visualization Generator
Generates visualizations for 2D Convolution concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from PIL import Image
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_convolution_sliding_window_gif():
    """
    Create an animated GIF showing the convolution sliding window operation.
    Shows a 3x3 kernel sliding over a 5x5 input matrix.
    """
    print("Creating convolution sliding window animation...")

    # Input matrix (5x5)
    input_matrix = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 0],
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 0],
        [1, 2, 3, 4, 5]
    ], dtype=float)

    # Kernel (3x3) - Sobel-like edge detector
    kernel = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=float)

    # Output matrix (3x3)
    output = np.zeros((3, 3))

    # Pre-compute all convolution results
    for i in range(3):
        for j in range(3):
            region = input_matrix[i:i+3, j:j+3]
            output[i, j] = np.sum(region * kernel)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    def animate(frame):
        for ax in axes:
            ax.clear()

        i, j = divmod(frame, 3)

        # Left: Input matrix with sliding window
        ax1 = axes[0]
        ax1.set_title('Input Matrix (5x5)', fontsize=12, fontweight='bold')

        # Draw input matrix
        for row in range(5):
            for col in range(5):
                color = '#E3F2FD' if not (i <= row < i+3 and j <= col < j+3) else '#FFEB3B'
                rect = patches.FancyBboxPatch(
                    (col, 4-row), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor='#333333', linewidth=1
                )
                ax1.add_patch(rect)
                ax1.text(col+0.45, 4-row+0.45, f'{int(input_matrix[row, col])}',
                        ha='center', va='center', fontsize=11, fontweight='bold')

        # Draw sliding window outline
        rect = patches.Rectangle(
            (j, 4-i-2), 3, 3,
            fill=False, edgecolor='#E91E63', linewidth=3, linestyle='-'
        )
        ax1.add_patch(rect)

        ax1.set_xlim(-0.2, 5.2)
        ax1.set_ylim(-0.2, 5.2)
        ax1.set_aspect('equal')
        ax1.axis('off')

        # Middle: Kernel
        ax2 = axes[1]
        ax2.set_title('Kernel (3x3)', fontsize=12, fontweight='bold')

        for row in range(3):
            for col in range(3):
                color = '#C8E6C9' if kernel[row, col] > 0 else '#FFCDD2' if kernel[row, col] < 0 else '#FFFFFF'
                rect = patches.FancyBboxPatch(
                    (col, 2-row), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor='#333333', linewidth=1
                )
                ax2.add_patch(rect)
                ax2.text(col+0.45, 2-row+0.45, f'{int(kernel[row, col])}',
                        ha='center', va='center', fontsize=11, fontweight='bold')

        ax2.set_xlim(-0.2, 3.2)
        ax2.set_ylim(-0.2, 3.2)
        ax2.set_aspect('equal')
        ax2.axis('off')

        # Right: Output matrix (showing computed values)
        ax3 = axes[2]
        ax3.set_title('Output Matrix (3x3)', fontsize=12, fontweight='bold')

        for row in range(3):
            for col in range(3):
                # Show computed values up to current position
                idx = row * 3 + col
                current_idx = i * 3 + j

                if idx < current_idx:
                    color = '#E8F5E9'
                    text = f'{int(output[row, col])}'
                elif idx == current_idx:
                    color = '#FFEB3B'
                    text = f'{int(output[row, col])}'
                else:
                    color = '#F5F5F5'
                    text = '?'

                rect = patches.FancyBboxPatch(
                    (col, 2-row), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor='#333333', linewidth=1
                )
                ax3.add_patch(rect)
                ax3.text(col+0.45, 2-row+0.45, text,
                        ha='center', va='center', fontsize=11, fontweight='bold')

        ax3.set_xlim(-0.2, 3.2)
        ax3.set_ylim(-0.2, 3.2)
        ax3.set_aspect('equal')
        ax3.axis('off')

        # Add computation formula at bottom
        region = input_matrix[i:i+3, j:j+3]
        result = int(output[i, j])
        fig.suptitle(f'Position ({i},{j}): Sum of element-wise products = {result}',
                    fontsize=11, y=0.02)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    anim = FuncAnimation(fig, animate, frames=9, interval=1000, repeat=True)
    anim.save(os.path.join(OUTPUT_DIR, 'cnn_sliding_window.gif'),
              writer=PillowWriter(fps=1), dpi=150)
    plt.close()
    print("  Saved: cnn_sliding_window.gif")


def create_filter_effects_gallery():
    """
    Create a 2x4 grid showing different filter effects:
    Top row: Original, Edge (Sobel X), Edge (Sobel Y), Edge (Laplacian)
    Bottom row: Original, Blur, Sharpen, Emboss
    """
    print("Creating filter effects gallery...")

    # Create a sample image with interesting features
    np.random.seed(42)
    size = 64

    # Create a synthetic image with edges and gradients
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
    image = np.zeros((size, size))

    # Add a circle
    circle = (x**2 + y**2) < 1
    image[circle] = 200

    # Add a rectangle
    rect = (np.abs(x) < 0.5) & (np.abs(y + 1) < 0.3)
    image[rect] = 150

    # Add gradient
    image += np.linspace(0, 50, size).reshape(1, -1)

    # Add some texture/noise
    image += np.random.randn(size, size) * 10
    image = np.clip(image, 0, 255)

    # Define kernels
    kernels = {
        'Original': None,
        'Edge (Sobel X)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Edge (Sobel Y)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Edge (Laplacian)': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        'Blur (Gaussian)': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    }

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=150)

    # Top row: Edge detection filters
    top_row = ['Original', 'Edge (Sobel X)', 'Edge (Sobel Y)', 'Edge (Laplacian)']
    # Bottom row: Enhancement filters
    bottom_row = ['Identity', 'Blur (Gaussian)', 'Sharpen', 'Emboss']

    for idx, name in enumerate(top_row):
        ax = axes[0, idx]
        kernel = kernels[name]

        if kernel is None:
            result = image
            cmap = 'gray'
        else:
            result = ndimage.convolve(image, kernel, mode='constant')
            cmap = 'RdBu_r' if 'Edge' in name else 'gray'

        im = ax.imshow(result, cmap=cmap)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.axis('off')

        # Add kernel visualization in corner
        if kernel is not None:
            inset_ax = ax.inset_axes([0.02, 0.02, 0.35, 0.35])
            inset_ax.imshow(kernel, cmap='RdBu_r', vmin=-4, vmax=4)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            for spine in inset_ax.spines.values():
                spine.set_edgecolor('#333')
                spine.set_linewidth(1)

    for idx, name in enumerate(bottom_row):
        ax = axes[1, idx]
        kernel = kernels[name]

        if kernel is None:
            result = image
        else:
            result = ndimage.convolve(image, kernel, mode='constant')

        cmap = 'gray'
        im = ax.imshow(result, cmap=cmap)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.axis('off')

        # Add kernel visualization in corner
        if kernel is not None:
            inset_ax = ax.inset_axes([0.02, 0.02, 0.35, 0.35])
            inset_ax.imshow(kernel, cmap='RdBu_r', vmin=-4, vmax=4)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            for spine in inset_ax.spines.values():
                spine.set_edgecolor('#333')
                spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_filter_effects.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: cnn_filter_effects.png")


def create_feature_map_activations():
    """
    Create visualization showing feature maps at different CNN layers.
    Simulates how features become more abstract through layers.
    """
    print("Creating feature map activations visualization...")

    # Create a simple input image (like a number or letter)
    size = 28
    input_image = np.zeros((size, size))

    # Draw a simple "7" shape
    input_image[4:6, 6:22] = 1  # Top horizontal
    input_image[4:24, 20:22] = 1  # Vertical stem

    # Add some noise
    np.random.seed(42)
    input_image += np.random.randn(size, size) * 0.1
    input_image = np.clip(input_image, 0, 1)

    # Define layer filters (simulating learned filters)
    # Layer 1: Edge detectors
    layer1_filters = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # Vertical edges
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  # Horizontal edges
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),    # Laplacian
        np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])    # Diagonal
    ]

    # Apply Layer 1
    layer1_outputs = []
    for filt in layer1_filters:
        out = ndimage.convolve(input_image, filt, mode='constant')
        out = np.maximum(0, out)  # ReLU
        layer1_outputs.append(out)

    # Layer 2: More complex patterns (combine Layer 1 outputs)
    layer2_outputs = []
    for i in range(4):
        combined = np.zeros_like(input_image)
        for j, l1_out in enumerate(layer1_outputs):
            # Random weights to simulate learned combinations
            weight = np.random.randn(3, 3)
            combined += ndimage.convolve(l1_out, weight, mode='constant')
        combined = np.maximum(0, combined)  # ReLU
        # Downsample (simulating pooling)
        combined = combined[::2, ::2]
        layer2_outputs.append(combined)

    # Layer 3: High-level features
    layer3_outputs = []
    for i in range(4):
        combined = np.zeros_like(layer2_outputs[0])
        for l2_out in layer2_outputs:
            weight = np.random.randn(3, 3)
            combined += ndimage.convolve(l2_out, weight, mode='constant')
        combined = np.maximum(0, combined)
        combined = combined[::2, ::2]
        layer3_outputs.append(combined)

    # Create visualization
    fig = plt.figure(figsize=(14, 8), dpi=150)

    # Create grid for subplots
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.1)

    # Input image
    ax_input = fig.add_subplot(gs[:, 0])
    ax_input.imshow(input_image, cmap='gray')
    ax_input.set_title('Input\n(28x28)', fontsize=11, fontweight='bold')
    ax_input.axis('off')

    # Layer 1 feature maps
    for i, fm in enumerate(layer1_outputs):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(fm, cmap='viridis')
        if i == 0:
            ax.set_title(f'Conv1\nFeature {i+1}', fontsize=10)
        else:
            ax.set_title(f'Feature {i+1}', fontsize=10)
        ax.axis('off')

    # Layer 2 feature maps
    for i, fm in enumerate(layer2_outputs):
        ax = fig.add_subplot(gs[1, i+1])
        ax.imshow(fm, cmap='plasma')
        if i == 0:
            ax.set_title(f'Conv2+Pool\nFeature {i+1}', fontsize=10)
        else:
            ax.set_title(f'Feature {i+1}', fontsize=10)
        ax.axis('off')

    # Layer 3 feature maps
    for i, fm in enumerate(layer3_outputs):
        ax = fig.add_subplot(gs[2, i+1])
        ax.imshow(fm, cmap='magma')
        if i == 0:
            ax.set_title(f'Conv3+Pool\nFeature {i+1}', fontsize=10)
        else:
            ax.set_title(f'Feature {i+1}', fontsize=10)
        ax.axis('off')

    # Add arrows showing flow
    fig.text(0.21, 0.78, '--->', fontsize=20, ha='center', fontfamily='monospace')
    fig.text(0.21, 0.48, '--->', fontsize=20, ha='center', fontfamily='monospace')
    fig.text(0.21, 0.18, '--->', fontsize=20, ha='center', fontfamily='monospace')

    # Add layer labels
    fig.text(0.58, 0.95, 'Layer 1: Edge Detectors', fontsize=11, ha='center', fontweight='bold', color='#2E7D32')
    fig.text(0.58, 0.63, 'Layer 2: Texture/Pattern Combinations', fontsize=11, ha='center', fontweight='bold', color='#7B1FA2')
    fig.text(0.58, 0.31, 'Layer 3: High-Level Features', fontsize=11, ha='center', fontweight='bold', color='#C62828')

    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_feature_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: cnn_feature_maps.png")


def create_pooling_visualization():
    """
    Create visualization comparing Max Pooling vs Average Pooling.
    """
    print("Creating pooling operation visualization...")

    # Create sample feature map
    np.random.seed(42)
    feature_map = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [3, 2, 8, 7],
        [4, 1, 3, 5]
    ], dtype=float)

    # Apply 2x2 pooling with stride 2
    pool_size = 2

    # Max pooling
    max_pooled = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            region = feature_map[i*2:(i+1)*2, j*2:(j+1)*2]
            max_pooled[i, j] = np.max(region)

    # Average pooling
    avg_pooled = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            region = feature_map[i*2:(i+1)*2, j*2:(j+1)*2]
            avg_pooled[i, j] = np.mean(region)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    colors = ['#FFCDD2', '#FFECB3', '#C8E6C9', '#BBDEFB']

    # Input feature map
    ax1 = axes[0]
    ax1.set_title('Input Feature Map (4x4)', fontsize=12, fontweight='bold')

    for i in range(4):
        for j in range(4):
            # Color based on 2x2 region
            region_idx = (i // 2) * 2 + (j // 2)
            rect = patches.FancyBboxPatch(
                (j, 3-i), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=colors[region_idx], edgecolor='#333333', linewidth=1
            )
            ax1.add_patch(rect)
            ax1.text(j+0.45, 3-i+0.45, f'{int(feature_map[i, j])}',
                    ha='center', va='center', fontsize=14, fontweight='bold')

    # Add 2x2 pool region outlines
    for i in range(2):
        for j in range(2):
            rect = patches.Rectangle(
                (j*2, (1-i)*2), 2, 2,
                fill=False, edgecolor='#333', linewidth=2
            )
            ax1.add_patch(rect)

    ax1.set_xlim(-0.2, 4.2)
    ax1.set_ylim(-0.2, 4.2)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Max Pooling result
    ax2 = axes[1]
    ax2.set_title('Max Pooling (2x2, stride=2)', fontsize=12, fontweight='bold')

    max_positions = [
        (1, 0),  # Region 0: max is 6 at position (1,1)
        (0, 3),  # Region 1: max is 4 at position (0,3)
        (0, 2),  # Region 2: max is 4 at position (3,0)
        (0, 2)   # Region 3: max is 8 at position (2,2)
    ]

    for i in range(2):
        for j in range(2):
            region_idx = i * 2 + j
            rect = patches.FancyBboxPatch(
                (j, 1-i), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=colors[region_idx], edgecolor='#333333', linewidth=2
            )
            ax2.add_patch(rect)
            ax2.text(j+0.45, 1-i+0.45, f'{int(max_pooled[i, j])}',
                    ha='center', va='center', fontsize=16, fontweight='bold')

    ax2.set_xlim(-0.2, 2.2)
    ax2.set_ylim(-0.2, 2.2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.text(1, -0.5, 'Takes MAX value\nfrom each region', ha='center', fontsize=10, style='italic')

    # Average Pooling result
    ax3 = axes[2]
    ax3.set_title('Average Pooling (2x2, stride=2)', fontsize=12, fontweight='bold')

    for i in range(2):
        for j in range(2):
            region_idx = i * 2 + j
            rect = patches.FancyBboxPatch(
                (j, 1-i), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=colors[region_idx], edgecolor='#333333', linewidth=2
            )
            ax3.add_patch(rect)
            ax3.text(j+0.45, 1-i+0.45, f'{avg_pooled[i, j]:.1f}',
                    ha='center', va='center', fontsize=14, fontweight='bold')

    ax3.set_xlim(-0.2, 2.2)
    ax3.set_ylim(-0.2, 2.2)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.text(1, -0.5, 'Takes AVERAGE value\nfrom each region', ha='center', fontsize=10, style='italic')

    # Add explanatory text
    fig.text(0.5, 0.02, 'Pooling reduces spatial dimensions by 2x (4x4 -> 2x2) while preserving important features',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_pooling_operation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: cnn_pooling_operation.png")


def create_receptive_field_diagram():
    """
    Create diagram showing how receptive field grows through CNN layers.
    """
    print("Creating receptive field growth diagram...")

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    # Layer configurations
    layers = [
        {'name': 'Input', 'size': 32, 'rf': 1, 'color': '#E3F2FD'},
        {'name': 'Conv1\n(3x3)', 'size': 30, 'rf': 3, 'color': '#C8E6C9'},
        {'name': 'Conv2\n(3x3)', 'size': 28, 'rf': 5, 'color': '#FFF9C4'},
        {'name': 'Pool\n(2x2)', 'size': 14, 'rf': 6, 'color': '#F3E5F5'},
        {'name': 'Conv3\n(3x3)', 'size': 12, 'rf': 10, 'color': '#FFCCBC'},
        {'name': 'Conv4\n(3x3)', 'size': 10, 'rf': 14, 'color': '#B2DFDB'},
    ]

    x_positions = np.linspace(1, 12, len(layers))

    for i, (layer, x) in enumerate(zip(layers, x_positions)):
        # Draw layer box (scaled by size)
        height = layer['size'] / 32 * 4  # Scale to fit

        rect = patches.FancyBboxPatch(
            (x - 0.4, 2 - height/2), 0.8, height,
            boxstyle="round,pad=0.02",
            facecolor=layer['color'], edgecolor='#333333', linewidth=2
        )
        ax.add_patch(rect)

        # Layer name
        ax.text(x, 2 + height/2 + 0.3, layer['name'],
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Size and RF annotation
        ax.text(x, 2 - height/2 - 0.3, f"Size: {layer['size']}x{layer['size']}",
                ha='center', va='top', fontsize=9)
        ax.text(x, 2 - height/2 - 0.6, f"RF: {layer['rf']}x{layer['rf']}",
                ha='center', va='top', fontsize=9, color='#E91E63', fontweight='bold')

        # Draw arrows
        if i < len(layers) - 1:
            ax.annotate('', xy=(x_positions[i+1] - 0.5, 2),
                       xytext=(x + 0.5, 2),
                       arrowprops=dict(arrowstyle='->', color='#666', lw=2))

    # Add receptive field visualization on top
    ax.text(6.5, 5, 'Receptive Field Growth Through Layers', fontsize=14,
            fontweight='bold', ha='center')

    # Draw RF growth visualization
    rf_sizes = [l['rf'] for l in layers]
    rf_colors = ['#64B5F6', '#81C784', '#FFD54F', '#BA68C8', '#FF8A65', '#4DD0E1']

    for i, (rf, color) in enumerate(zip(rf_sizes, rf_colors)):
        size = rf / max(rf_sizes) * 1.5
        circle = patches.Circle(
            (x_positions[i], 4.2), size/2,
            facecolor=color, edgecolor='#333', linewidth=1, alpha=0.7
        )
        ax.add_patch(circle)
        if rf > 1:
            ax.text(x_positions[i], 4.2, f'{rf}x{rf}',
                   ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlim(0, 13)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add formula
    formula_text = 'RF Formula: RFₙ = RFₙ₋₁ + (kₙ - 1) × ∏ᵢ₌₁ⁿ⁻¹ sᵢ'
    ax.text(6.5, -0.2, formula_text, ha='center', fontsize=11,
            style='italic', bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCC'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_receptive_field.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: cnn_receptive_field.png")


def create_stride_padding_visualization():
    """
    Create visualization comparing different stride and padding configurations.
    """
    print("Creating stride and padding visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=150)

    configs = [
        {'title': 'No Padding, Stride=1', 'padding': 0, 'stride': 1, 'out_size': '3x3'},
        {'title': 'Same Padding, Stride=1', 'padding': 1, 'stride': 1, 'out_size': '5x5'},
        {'title': 'Full Padding, Stride=1', 'padding': 2, 'stride': 1, 'out_size': '7x7'},
        {'title': 'No Padding, Stride=2', 'padding': 0, 'stride': 2, 'out_size': '2x2'},
        {'title': 'Same Padding, Stride=2', 'padding': 1, 'stride': 2, 'out_size': '3x3'},
        {'title': 'No Padding, Stride=3', 'padding': 0, 'stride': 3, 'out_size': '1x1'},
    ]

    input_size = 5
    kernel_size = 3

    for idx, config in enumerate(configs):
        ax = axes[idx // 3, idx % 3]
        ax.set_title(f"{config['title']}\nOutput: {config['out_size']}",
                    fontsize=11, fontweight='bold')

        padding = config['padding']
        padded_size = input_size + 2 * padding

        # Draw grid
        for i in range(padded_size + 1):
            ax.axhline(i, color='#CCC', linewidth=0.5)
            ax.axvline(i, color='#CCC', linewidth=0.5)

        # Fill original input area
        for i in range(input_size):
            for j in range(input_size):
                rect = patches.Rectangle(
                    (j + padding, padded_size - 1 - i - padding), 1, 1,
                    facecolor='#E3F2FD', edgecolor='#333', linewidth=1
                )
                ax.add_patch(rect)

        # Fill padding area (if any)
        if padding > 0:
            for i in range(padded_size):
                for j in range(padded_size):
                    if i < padding or i >= padded_size - padding or \
                       j < padding or j >= padded_size - padding:
                        rect = patches.Rectangle(
                            (j, padded_size - 1 - i), 1, 1,
                            facecolor='#FFECB3', edgecolor='#333', linewidth=1
                        )
                        ax.add_patch(rect)

        # Calculate output positions and show kernel positions
        stride = config['stride']
        out_h = (padded_size - kernel_size) // stride + 1
        out_w = (padded_size - kernel_size) // stride + 1

        # Show first kernel position
        rect = patches.Rectangle(
            (0, padded_size - kernel_size), kernel_size, kernel_size,
            fill=False, edgecolor='#E91E63', linewidth=3
        )
        ax.add_patch(rect)

        # Show stride arrows if stride > 1
        if stride > 1 and out_w > 1:
            ax.annotate('', xy=(stride + kernel_size/2, padded_size - kernel_size/2),
                       xytext=(kernel_size/2, padded_size - kernel_size/2),
                       arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))
            ax.text((stride + kernel_size)/2, padded_size - kernel_size/2 + 0.3,
                   f'stride={stride}', fontsize=9, ha='center', color='#4CAF50')

        ax.set_xlim(-0.2, padded_size + 0.2)
        ax.set_ylim(-0.2, padded_size + 0.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Legend
        if idx == 0:
            ax.text(padded_size/2, -0.7, 'Blue: Input | Yellow: Padding | Pink: Kernel',
                   ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_stride_padding.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: cnn_stride_padding.png")


if __name__ == '__main__':
    print("Generating CNN visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    create_convolution_sliding_window_gif()
    create_filter_effects_gallery()
    create_feature_map_activations()
    create_pooling_visualization()
    create_receptive_field_diagram()
    create_stride_padding_visualization()

    print("-" * 50)
    print("All CNN visualizations generated successfully!")
