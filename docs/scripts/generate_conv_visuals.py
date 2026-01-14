#!/usr/bin/env python3
"""
Generate visualizations for the 2D Convolution documentation page.
Creates SVG files for:
1. Step-by-step kernel sliding animation (multiple frames)
2. Input matrix with kernel overlay showing element-wise multiplication
3. Before/after padding comparison
4. Different stride effects visualization
5. Common kernels (edge detection, blur, sharpen) with their effects
6. Output feature map calculation breakdown
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import os

# Set style for clean, professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme
COLORS = {
    'input': '#E3F2FD',       # Light blue
    'kernel': '#FFECB3',      # Light amber
    'highlight': '#4CAF50',   # Green
    'output': '#F3E5F5',      # Light purple
    'text': '#212121',        # Dark gray
    'accent1': '#2196F3',     # Blue
    'accent2': '#FF9800',     # Orange
    'accent3': '#E91E63',     # Pink
    'padding': '#FFCDD2',     # Light red
    'multiply': '#C8E6C9',    # Light green
}


def create_input_matrix():
    """Create sample 5x5 input matrix."""
    return np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])


def create_kernel():
    """Create sample 3x3 kernel (Sobel X)."""
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])


def draw_matrix(ax, matrix, x_offset=0, y_offset=0, cell_size=1,
                bg_color='white', highlight_cells=None, highlight_color='#C8E6C9',
                show_values=True, title=None, font_size=10, value_format='{}'):
    """Draw a matrix with cell values."""
    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(cols):
            x = x_offset + j * cell_size
            y = y_offset + (rows - 1 - i) * cell_size

            # Determine cell color
            cell_color = bg_color
            if highlight_cells is not None and (i, j) in highlight_cells:
                cell_color = highlight_color

            # Draw cell
            rect = Rectangle((x, y), cell_size, cell_size,
                            linewidth=1, edgecolor='black',
                            facecolor=cell_color)
            ax.add_patch(rect)

            # Add value
            if show_values:
                val = matrix[i, j]
                if isinstance(val, (int, np.integer)):
                    text = str(int(val))
                else:
                    text = value_format.format(val)
                ax.text(x + cell_size/2, y + cell_size/2, text,
                       ha='center', va='center', fontsize=font_size,
                       color=COLORS['text'], fontweight='bold')

    if title:
        ax.text(x_offset + cols * cell_size / 2, y_offset + rows * cell_size + 0.3,
               title, ha='center', va='bottom', fontsize=12, fontweight='bold')


def plot_kernel_sliding_animation():
    """Create step-by-step kernel sliding visualization showing all 9 positions."""
    input_matrix = create_input_matrix()
    kernel = create_kernel()

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle('Kernel Sliding Window: Step-by-Step Convolution', fontsize=16, fontweight='bold', y=1.02)

    positions = [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]

    for idx, (row, col) in enumerate(positions):
        ax = axes[idx // 3, idx % 3]
        ax.set_xlim(-0.5, 8)
        ax.set_ylim(-0.5, 6)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw input matrix
        draw_matrix(ax, input_matrix, x_offset=0, y_offset=0, cell_size=1,
                   bg_color=COLORS['input'], show_values=True, font_size=9)

        # Highlight the region being convolved
        highlight_cells = set()
        for ki in range(3):
            for kj in range(3):
                highlight_cells.add((row + ki, col + kj))

        # Draw highlighted overlay
        for i, j in highlight_cells:
            x = j
            y = 5 - 1 - i
            rect = Rectangle((x, y), 1, 1, linewidth=2,
                            edgecolor=COLORS['highlight'], facecolor=COLORS['multiply'],
                            alpha=0.7, zorder=10)
            ax.add_patch(rect)

        # Draw kernel outline
        kernel_x = col
        kernel_y = 5 - 1 - row - 2
        kernel_rect = Rectangle((kernel_x, kernel_y), 3, 3,
                                linewidth=3, edgecolor=COLORS['accent3'],
                                facecolor='none', linestyle='--', zorder=15)
        ax.add_patch(kernel_rect)

        # Calculate output value
        region = input_matrix[row:row+3, col:col+3]
        output_val = np.sum(region * kernel)

        # Draw kernel on the right
        draw_matrix(ax, kernel, x_offset=5.5, y_offset=1.5, cell_size=0.8,
                   bg_color=COLORS['kernel'], show_values=True, font_size=8)

        # Draw output position indicator
        ax.text(5.5 + 1.2, 4.2, f'Position ({row},{col})', fontsize=9,
               fontweight='bold', color=COLORS['accent1'])
        ax.text(5.5 + 1.2, 0.8, f'Output = {output_val}', fontsize=10,
               fontweight='bold', color=COLORS['accent3'],
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['accent3']))

        # Step number
        ax.text(2.5, 5.8, f'Step {idx + 1}', fontsize=11, fontweight='bold',
               ha='center', color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'conv-kernel-sliding.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'conv-kernel-sliding.svg')}")


def plot_element_wise_multiplication():
    """Visualize element-wise multiplication and summation in detail."""
    input_matrix = create_input_matrix()
    kernel = create_kernel()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 18)
    ax.set_ylim(-2, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Element-wise Multiplication and Summation (Position 0,0)',
                fontsize=14, fontweight='bold', pad=20)

    # Input region (3x3)
    region = input_matrix[0:3, 0:3]

    # Draw input region
    ax.text(1.5, 6.8, 'Input Region', fontsize=11, fontweight='bold', ha='center')
    draw_matrix(ax, region, x_offset=0, y_offset=3.5, cell_size=1,
               bg_color=COLORS['input'], show_values=True, font_size=11)

    # Multiplication symbol
    ax.text(3.8, 5, r'$\times$', fontsize=24, ha='center', va='center')

    # Draw kernel
    ax.text(5.5, 6.8, 'Kernel', fontsize=11, fontweight='bold', ha='center')
    draw_matrix(ax, kernel, x_offset=4, y_offset=3.5, cell_size=1,
               bg_color=COLORS['kernel'], show_values=True, font_size=11)

    # Equals symbol
    ax.text(7.8, 5, '=', fontsize=24, ha='center', va='center')

    # Draw products
    products = region * kernel
    ax.text(9.5, 6.8, 'Products', fontsize=11, fontweight='bold', ha='center')
    draw_matrix(ax, products, x_offset=8, y_offset=3.5, cell_size=1,
               bg_color=COLORS['multiply'], show_values=True, font_size=10)

    # Sum arrow
    ax.annotate('', xy=(14, 5), xytext=(11.5, 5),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['accent1']))
    ax.text(12.7, 5.5, 'Sum', fontsize=10, color=COLORS['accent1'])

    # Output value
    output_val = np.sum(products)
    output_box = FancyBboxPatch((13.5, 4), 3, 2, boxstyle="round,pad=0.1",
                                facecolor=COLORS['output'], edgecolor=COLORS['accent3'],
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(15, 5, f'{output_val}', fontsize=18, fontweight='bold',
           ha='center', va='center', color=COLORS['accent3'])
    ax.text(15, 6.8, 'Output[0,0]', fontsize=11, fontweight='bold', ha='center')

    # Detailed calculation below
    ax.text(0, 2.5, 'Calculation:', fontsize=11, fontweight='bold')

    calc_parts = []
    for i in range(3):
        for j in range(3):
            calc_parts.append(f'({region[i,j]} x {kernel[i,j]})')

    row1 = ' + '.join(calc_parts[0:3])
    row2 = ' + '.join(calc_parts[3:6])
    row3 = ' + '.join(calc_parts[6:9])

    ax.text(0, 1.8, f'= {row1}', fontsize=9, family='monospace')
    ax.text(0, 1.1, f'  + {row2}', fontsize=9, family='monospace')
    ax.text(0, 0.4, f'  + {row3}', fontsize=9, family='monospace')

    products_flat = products.flatten()
    ax.text(0, -0.3, f'= {" + ".join([f"({p})" for p in products_flat[:5]])}...',
           fontsize=9, family='monospace')
    ax.text(0, -1.0, f'= {output_val}', fontsize=11, fontweight='bold',
           color=COLORS['accent3'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'conv-element-wise.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'conv-element-wise.svg')}")


def plot_padding_comparison():
    """Visualize before/after padding comparison."""
    input_matrix = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    # Padded version (padding=1)
    padded = np.pad(input_matrix, pad_width=1, mode='constant', constant_values=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Zero Padding: Before and After', fontsize=14, fontweight='bold')

    # No padding
    ax = axes[0]
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-1.5, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('No Padding (Valid)', fontsize=12)

    draw_matrix(ax, input_matrix, x_offset=1, y_offset=1, cell_size=1,
               bg_color=COLORS['input'], show_values=True, font_size=11)

    # Show kernel overlay
    kernel_rect = Rectangle((1, 2), 3, 3, linewidth=2,
                            edgecolor=COLORS['accent3'], facecolor='none',
                            linestyle='--')
    ax.add_patch(kernel_rect)
    ax.text(2.5, 5.5, '3x3 kernel fits', fontsize=9, ha='center', style='italic')
    ax.text(3, 0, 'Input: 4x4', fontsize=10, ha='center', fontweight='bold')
    ax.text(3, -0.6, 'Output: 2x2', fontsize=10, ha='center',
           color=COLORS['accent1'], fontweight='bold')

    # With padding
    ax = axes[1]
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-1.5, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('With Padding=1 (Same)', fontsize=12)

    # Draw padded zeros
    for i in range(6):
        for j in range(6):
            x = j + 1
            y = 6 - 1 - i + 1
            if i == 0 or i == 5 or j == 0 or j == 5:
                # Padding cells
                rect = Rectangle((x, y), 1, 1, linewidth=1,
                                edgecolor='black', facecolor=COLORS['padding'], alpha=0.7)
                ax.add_patch(rect)
                ax.text(x + 0.5, y + 0.5, '0', ha='center', va='center',
                       fontsize=10, color='gray')
            else:
                # Original input
                rect = Rectangle((x, y), 1, 1, linewidth=1,
                                edgecolor='black', facecolor=COLORS['input'])
                ax.add_patch(rect)
                ax.text(x + 0.5, y + 0.5, str(input_matrix[i-1, j-1]),
                       ha='center', va='center', fontsize=10, fontweight='bold')

    # Show kernel overlay at corner
    kernel_rect = Rectangle((1, 4), 3, 3, linewidth=2,
                            edgecolor=COLORS['accent3'], facecolor='none',
                            linestyle='--')
    ax.add_patch(kernel_rect)
    ax.text(2.5, 7.5, '3x3 kernel covers padding', fontsize=9, ha='center', style='italic')

    # Legend
    legend_x = 1
    legend_y = 0.3
    rect1 = Rectangle((legend_x, legend_y), 0.5, 0.5, facecolor=COLORS['padding'],
                      edgecolor='black')
    ax.add_patch(rect1)
    ax.text(legend_x + 0.7, legend_y + 0.25, '= Zero padding', fontsize=9, va='center')

    rect2 = Rectangle((legend_x + 3.5, legend_y), 0.5, 0.5, facecolor=COLORS['input'],
                      edgecolor='black')
    ax.add_patch(rect2)
    ax.text(legend_x + 4.2, legend_y + 0.25, '= Original input', fontsize=9, va='center')

    ax.text(4, -0.3, 'Padded: 6x6', fontsize=10, ha='center', fontweight='bold')
    ax.text(4, -0.9, 'Output: 4x4 (same as input!)', fontsize=10, ha='center',
           color=COLORS['highlight'], fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'conv-padding-comparison.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'conv-padding-comparison.svg')}")


def plot_stride_effects():
    """Visualize different stride effects."""
    input_matrix = np.ones((6, 6))
    for i in range(6):
        for j in range(6):
            input_matrix[i, j] = i * 6 + j + 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Effect of Stride on Output Size', fontsize=14, fontweight='bold')

    strides = [1, 2, 3]

    for idx, stride in enumerate(strides):
        ax = axes[idx]
        ax.set_xlim(-0.5, 8)
        ax.set_ylim(-1.5, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Stride = {stride}', fontsize=12)

        # Draw input matrix
        draw_matrix(ax, input_matrix, x_offset=1, y_offset=1.5, cell_size=1,
                   bg_color=COLORS['input'], show_values=True, font_size=8)

        # Calculate output size
        output_size = (6 - 3) // stride + 1

        # Show kernel positions
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
        positions = []
        for row in range(output_size):
            for col in range(output_size):
                positions.append((row * stride, col * stride))

        for pos_idx, (row, col) in enumerate(positions[:4]):
            alpha = 0.4 if pos_idx > 0 else 0.7
            kernel_rect = Rectangle((1 + col, 7.5 - row - 3 + 1), 3, 3,
                                    linewidth=2, edgecolor=colors[pos_idx % 4],
                                    facecolor=colors[pos_idx % 4], alpha=alpha)
            ax.add_patch(kernel_rect)

        # Output info
        ax.text(4, 0.5, f'Input: 6x6', fontsize=10, ha='center')
        ax.text(4, 0, f'Kernel: 3x3', fontsize=10, ha='center')
        ax.text(4, -0.6, f'Output: {output_size}x{output_size}', fontsize=11,
               ha='center', fontweight='bold', color=COLORS['accent3'])

        # Formula
        ax.text(4, -1.2, f'({6}-{3})//{stride}+1 = {output_size}', fontsize=9,
               ha='center', family='monospace', color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'conv-stride-effects.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'conv-stride-effects.svg')}")


def plot_common_kernels():
    """Visualize common kernels and their effects."""
    # Create a sample image with edges
    np.random.seed(42)
    img = np.zeros((8, 8))
    img[2:6, 2:6] = 1  # Square in center
    img += np.random.normal(0, 0.05, img.shape)
    img = np.clip(img, 0, 1)

    # Kernels
    kernels = {
        'Edge Detection\n(Sobel X)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Edge Detection\n(Sobel Y)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Box Blur': np.ones((3, 3)) / 9,
        'Gaussian Blur': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    }

    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    fig.suptitle('Common Convolution Kernels and Their Effects', fontsize=14, fontweight='bold')

    for idx, (name, kernel) in enumerate(kernels.items()):
        # Draw kernel
        ax_kernel = axes[0, idx]
        im = ax_kernel.imshow(kernel, cmap='RdBu_r', aspect='equal')
        ax_kernel.set_title(name, fontsize=9)

        # Add values to cells
        for i in range(3):
            for j in range(3):
                val = kernel[i, j]
                if abs(val) < 0.01:
                    text = '0'
                elif abs(val - round(val)) < 0.01:
                    text = str(int(round(val)))
                else:
                    text = f'{val:.2f}'
                ax_kernel.text(j, i, text, ha='center', va='center',
                              fontsize=8, fontweight='bold',
                              color='white' if abs(val) > 2 else 'black')

        ax_kernel.set_xticks([])
        ax_kernel.set_yticks([])

        # Apply convolution and show result
        ax_result = axes[1, idx]

        # Simple convolution
        result = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                region = img[i:i+3, j:j+3]
                result[i, j] = np.sum(region * kernel)

        # Normalize for display
        if name in ['Sharpen']:
            result = np.clip(result, 0, 1)
        else:
            result = (result - result.min()) / (result.max() - result.min() + 1e-8)

        ax_result.imshow(result, cmap='gray', aspect='equal')
        ax_result.set_title('Result', fontsize=9)
        ax_result.set_xticks([])
        ax_result.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'conv-common-kernels.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'conv-common-kernels.svg')}")


def plot_output_calculation():
    """Full output feature map calculation visualization."""
    input_matrix = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # Calculate output
    output = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            region = input_matrix[i:i+3, j:j+3]
            output[i, j] = np.sum(region * kernel)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 18)
    ax.set_ylim(-2, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Complete Output Feature Map Calculation', fontsize=14, fontweight='bold', pad=20)

    # Draw input matrix
    ax.text(2, 7.8, 'Input (4x4)', fontsize=11, fontweight='bold', ha='center')
    draw_matrix(ax, input_matrix, x_offset=0, y_offset=3.5, cell_size=1,
               bg_color=COLORS['input'], show_values=True, font_size=11)

    # Convolution symbol
    ax.text(5, 5.5, '*', fontsize=30, ha='center', va='center', fontweight='bold')

    # Draw kernel
    ax.text(7.5, 7.8, 'Kernel (3x3)', fontsize=11, fontweight='bold', ha='center')
    draw_matrix(ax, kernel, x_offset=6, y_offset=4.5, cell_size=1,
               bg_color=COLORS['kernel'], show_values=True, font_size=11)

    # Equals
    ax.text(10, 5.5, '=', fontsize=30, ha='center', va='center', fontweight='bold')

    # Draw output
    ax.text(12.5, 7.8, 'Output (2x2)', fontsize=11, fontweight='bold', ha='center')
    draw_matrix(ax, output, x_offset=11.5, y_offset=4.5, cell_size=1,
               bg_color=COLORS['output'], show_values=True, font_size=11)

    # Detailed calculations below
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']

    ax.text(0, 2.5, 'Position-by-position calculations:', fontsize=11, fontweight='bold')

    for idx, (row, col) in enumerate(positions):
        region = input_matrix[row:row+3, col:col+3]
        result = np.sum(region * kernel)

        # Color code the output cell
        x = 11.5 + col
        y = 4.5 + (2 - 1 - row)
        rect = Rectangle((x, y), 1, 1, linewidth=3, edgecolor=colors[idx],
                         facecolor='none', zorder=20)
        ax.add_patch(rect)

        # Show calculation
        y_pos = 1.8 - idx * 0.6
        ax.text(0, y_pos, f'Output[{row},{col}]:', fontsize=9, fontweight='bold', color=colors[idx])

        # Products
        products = region * kernel
        products_str = ' + '.join([f'{int(p)}' for p in products.flatten() if p != 0])
        ax.text(2.5, y_pos, f'= {products_str} = {int(result)}', fontsize=9, family='monospace')

    # Output size formula
    ax.text(0, -0.8, 'Output size formula:', fontsize=10, fontweight='bold')
    ax.text(0, -1.4, r'output_size = (input_size - kernel_size) / stride + 1', fontsize=9, family='monospace')
    ax.text(0, -1.9, f'             = (4 - 3) / 1 + 1 = 2', fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'conv-output-calculation.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'conv-output-calculation.svg')}")


def plot_cnn_pipeline():
    """Visualize full CNN pipeline: Conv -> Activation -> Pool."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Full CNN Pipeline: Convolution -> ReLU Activation -> Max Pooling',
                fontsize=14, fontweight='bold', pad=20)

    # Input
    input_data = np.array([
        [1, -2, 3, -1],
        [-3, 4, -2, 5],
        [2, -1, 6, -3],
        [-4, 3, -2, 1]
    ])

    ax.text(2, 5.8, 'Input', fontsize=11, fontweight='bold', ha='center')
    draw_matrix(ax, input_data, x_offset=0, y_offset=1.5, cell_size=1,
               bg_color=COLORS['input'], show_values=True, font_size=9)

    # Arrow
    ax.annotate('', xy=(5, 3.5), xytext=(4.5, 3.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(4.7, 4.2, 'Conv', fontsize=9, ha='center')

    # After convolution (simulated)
    conv_output = np.array([
        [-8, 12],
        [6, -4]
    ])

    ax.text(6.5, 5.8, 'After Conv', fontsize=11, fontweight='bold', ha='center')
    draw_matrix(ax, conv_output, x_offset=5.5, y_offset=2.5, cell_size=1,
               bg_color='#E8F5E9', show_values=True, font_size=10)

    # Arrow
    ax.annotate('', xy=(9, 3.5), xytext=(8, 3.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(8.5, 4.2, 'ReLU', fontsize=9, ha='center')

    # After ReLU
    relu_output = np.maximum(0, conv_output)

    ax.text(10.5, 5.8, 'After ReLU', fontsize=11, fontweight='bold', ha='center')
    for i in range(2):
        for j in range(2):
            x = 9.5 + j
            y = 2.5 + (1 - i)
            val = relu_output[i, j]
            # Highlight zeros
            if conv_output[i, j] < 0:
                color = COLORS['padding']  # Red for zeroed
            else:
                color = '#C8E6C9'
            rect = Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(x + 0.5, y + 0.5, str(int(val)), ha='center', va='center',
                   fontsize=10, fontweight='bold')

    # Arrow
    ax.annotate('', xy=(13, 3.5), xytext=(12, 3.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(12.5, 4.2, 'MaxPool', fontsize=9, ha='center')

    # After max pooling (2x2 -> 1x1)
    max_val = np.max(relu_output)

    ax.text(14.5, 5.8, 'After Pool', fontsize=11, fontweight='bold', ha='center')
    rect = Rectangle((13.5, 2.5), 2, 2, linewidth=2, edgecolor=COLORS['accent3'],
                     facecolor=COLORS['output'])
    ax.add_patch(rect)
    ax.text(14.5, 3.5, str(int(max_val)), ha='center', va='center',
           fontsize=16, fontweight='bold', color=COLORS['accent3'])

    # Legend at bottom
    ax.text(0, 0.5, 'Key:', fontsize=10, fontweight='bold')

    rect1 = Rectangle((1.5, 0.3), 0.5, 0.5, facecolor=COLORS['padding'], edgecolor='black')
    ax.add_patch(rect1)
    ax.text(2.2, 0.55, '= Negative values zeroed by ReLU', fontsize=9, va='center')

    rect2 = Rectangle((8, 0.3), 0.5, 0.5, facecolor='#C8E6C9', edgecolor='black')
    ax.add_patch(rect2)
    ax.text(8.7, 0.55, '= Positive values preserved', fontsize=9, va='center')

    # ReLU equation
    ax.text(0, -0.5, 'ReLU: f(x) = max(0, x)    |    MaxPool: takes maximum value in each region',
           fontsize=9, family='monospace', color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'conv-cnn-pipeline.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'conv-cnn-pipeline.svg')}")


def plot_convolution_formula():
    """Visualize the convolution formula with color-coded matrices."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-1, 16)
    ax.set_ylim(-3, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('2D Discrete Convolution Formula', fontsize=14, fontweight='bold', pad=20)

    # Formula
    ax.text(7.5, 7, r'$(I * K)[i,j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I[i+m, j+n] \cdot K[m,n]$',
           fontsize=14, ha='center', va='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#2196F3', pad=0.5))

    # Visual representation
    input_region = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    ax.text(0, 5.5, 'Where:', fontsize=11, fontweight='bold')
    ax.text(0, 4.8, 'I = Input image/matrix', fontsize=10, color=COLORS['accent1'])
    ax.text(0, 4.2, 'K = Kernel/Filter', fontsize=10, color=COLORS['accent2'])
    ax.text(0, 3.6, '(i,j) = Output position', fontsize=10, color=COLORS['accent3'])
    ax.text(0, 3.0, '(m,n) = Kernel indices', fontsize=10, color='gray')

    # Example calculation
    ax.text(7.5, 2.5, 'Example Calculation:', fontsize=12, fontweight='bold', ha='center')

    # Draw input region
    ax.text(2.5, 1.8, 'I[i:i+3, j:j+3]', fontsize=10, ha='center', color=COLORS['accent1'])
    draw_matrix(ax, input_region, x_offset=1, y_offset=-1.5, cell_size=1,
               bg_color=COLORS['input'], show_values=True, font_size=10)

    # Multiplication symbol
    ax.text(4.7, 0, r'$\odot$', fontsize=20, ha='center', va='center')

    # Draw kernel
    ax.text(6.5, 1.8, 'K', fontsize=10, ha='center', color=COLORS['accent2'])
    draw_matrix(ax, kernel, x_offset=5, y_offset=-1.5, cell_size=1,
               bg_color=COLORS['kernel'], show_values=True, font_size=10)

    # Equals
    ax.text(8.7, 0, '=', fontsize=20, ha='center', va='center')

    # Products
    products = input_region * kernel
    ax.text(10.5, 1.8, 'Products', fontsize=10, ha='center', color=COLORS['highlight'])
    draw_matrix(ax, products, x_offset=9, y_offset=-1.5, cell_size=1,
               bg_color=COLORS['multiply'], show_values=True, font_size=10)

    # Sum arrow and result
    ax.annotate('', xy=(14, 0), xytext=(12.5, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['accent3']))
    ax.text(13.2, 0.7, r'$\sum$', fontsize=16, ha='center')

    result = np.sum(products)
    result_box = FancyBboxPatch((13.5, -1), 2, 2, boxstyle="round,pad=0.1",
                                facecolor=COLORS['output'], edgecolor=COLORS['accent3'],
                                linewidth=2)
    ax.add_patch(result_box)
    ax.text(14.5, 0, str(int(result)), fontsize=16, fontweight='bold',
           ha='center', va='center', color=COLORS['accent3'])

    # Calculation breakdown
    ax.text(0, -2.5, f'= (-1)(1) + (0)(2) + (1)(3) + (-2)(4) + (0)(5) + (2)(6) + (-1)(7) + (0)(8) + (1)(9)',
           fontsize=9, family='monospace')
    ax.text(0, -3, f'= -1 + 0 + 3 - 8 + 0 + 12 - 7 + 0 + 9 = {int(result)}',
           fontsize=9, family='monospace', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'conv-formula.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'conv-formula.svg')}")


if __name__ == '__main__':
    print("Generating 2D Convolution visualizations...")
    print("=" * 50)

    plot_kernel_sliding_animation()
    plot_element_wise_multiplication()
    plot_padding_comparison()
    plot_stride_effects()
    plot_common_kernels()
    plot_output_calculation()
    plot_cnn_pipeline()
    plot_convolution_formula()

    print("=" * 50)
    print("\nAll convolution visualizations generated successfully!")
