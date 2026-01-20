#!/usr/bin/env python3
"""
Batch Normalization and Dropout Visualization Generator

This script generates visualizations for understanding:
1. Internal covariate shift (before/after BN distributions)
2. Running mean/variance evolution during training
3. Dropout mask visualization (network with dropped neurons)
4. LayerNorm vs BatchNorm vs GroupNorm comparison
5. Training vs inference mode comparison

Usage:
    python generate_batchnorm_viz.py

Outputs:
    - internal_covariate_shift.png
    - running_stats_evolution.png
    - dropout_mask_visualization.png
    - normalization_comparison.png
    - training_vs_inference.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from pathlib import Path

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')

# Output directory
OUTPUT_DIR = Path(__file__).parent


def generate_internal_covariate_shift():
    """
    Generate visualization showing internal covariate shift problem
    and how Batch Normalization addresses it.

    Shows activation distributions before and after BatchNorm across layers.
    """
    np.random.seed(42)

    fig = plt.figure(figsize=(14, 10))

    # Create a 2x2 grid
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # === Top Left: Before BatchNorm (layers show shifting distributions) ===
    ax1 = fig.add_subplot(gs[0, 0])

    # Simulate activations without BatchNorm - distribution shifts and scales
    layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5']
    n_samples = 1000

    # Without BatchNorm: mean and variance drift across layers
    means_no_bn = [0, 2, 5, 10, 18]
    stds_no_bn = [1, 1.5, 2.5, 4, 7]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, 5))

    for i, (mean, std, color, layer) in enumerate(zip(means_no_bn, stds_no_bn, colors, layers)):
        data = np.random.normal(mean, std, n_samples)
        ax1.hist(data, bins=50, alpha=0.5, color=color, label=layer, density=True)

    ax1.set_xlabel('Activation Value', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Without Batch Normalization\n(Internal Covariate Shift)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(-10, 40)

    # Add annotation
    ax1.annotate('Distributions shift\nand scale!', xy=(20, 0.08), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))

    # === Top Right: After BatchNorm (distributions are normalized) ===
    ax2 = fig.add_subplot(gs[0, 1])

    # With BatchNorm: all layers have ~N(0,1) distribution
    colors_bn = plt.cm.Blues(np.linspace(0.3, 0.9, 5))

    for i, (color, layer) in enumerate(zip(colors_bn, layers)):
        # After BN, distributions are centered at 0 with unit variance
        data = np.random.normal(0, 1, n_samples)
        ax2.hist(data, bins=50, alpha=0.5, color=color, label=layer, density=True)

    ax2.set_xlabel('Normalized Activation Value', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('With Batch Normalization\n(Stable Distributions)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-5, 5)

    # Add annotation
    ax2.annotate('All layers:\nmean=0, var=1', xy=(2, 0.35), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#BBDEFB', alpha=0.8))

    # === Bottom Left: Single layer before/after comparison ===
    ax3 = fig.add_subplot(gs[1, 0])

    # Raw activations (before BN)
    raw_activations = np.random.normal(5, 3, n_samples)

    ax3.hist(raw_activations, bins=50, alpha=0.7, color='#EF5350', label='Before BN', density=True)

    # Add mean and std lines
    ax3.axvline(np.mean(raw_activations), color='darkred', linestyle='--', linewidth=2, label=f'Mean={np.mean(raw_activations):.1f}')

    ax3.set_xlabel('Activation Value', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Before Normalization\n(Single Layer)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)

    # === Bottom Right: After BatchNorm transformation ===
    ax4 = fig.add_subplot(gs[1, 1])

    # After BN: normalized activations
    normalized = (raw_activations - np.mean(raw_activations)) / np.std(raw_activations)

    ax4.hist(normalized, bins=50, alpha=0.7, color='#42A5F5', label='After BN', density=True)
    ax4.axvline(np.mean(normalized), color='darkblue', linestyle='--', linewidth=2, label=f'Mean={np.mean(normalized):.2f}')

    ax4.set_xlabel('Normalized Activation Value', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('After Normalization\n(x_norm = (x - mean) / std)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)

    # Main title
    fig.suptitle('Internal Covariate Shift: The Problem BatchNorm Solves\n'
                 'Without BN, activation distributions shift during training, slowing convergence',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = OUTPUT_DIR / 'internal_covariate_shift.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def generate_running_stats_evolution():
    """
    Generate visualization showing how running mean and variance
    evolve during training in BatchNorm.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Simulate training with changing batch statistics
    n_iterations = 100
    momentum = 0.1

    # Simulate batch statistics that change over time (learning process)
    batch_means = 5 * np.exp(-np.arange(n_iterations) / 30) + np.random.randn(n_iterations) * 0.5
    batch_vars = 3 * np.exp(-np.arange(n_iterations) / 40) + 1 + np.random.randn(n_iterations) * 0.2
    batch_vars = np.abs(batch_vars)  # Ensure positive variance

    # Compute running statistics
    running_mean = np.zeros(n_iterations)
    running_var = np.zeros(n_iterations)
    running_mean[0] = 0  # Initialize to 0
    running_var[0] = 1   # Initialize to 1

    for i in range(1, n_iterations):
        running_mean[i] = (1 - momentum) * running_mean[i-1] + momentum * batch_means[i]
        running_var[i] = (1 - momentum) * running_var[i-1] + momentum * batch_vars[i]

    iterations = np.arange(n_iterations)

    # === Top Left: Running Mean Evolution ===
    ax1 = axes[0, 0]
    ax1.plot(iterations, batch_means, 'o', alpha=0.4, markersize=4, color='#90CAF9', label='Batch Mean')
    ax1.plot(iterations, running_mean, '-', linewidth=2.5, color='#1565C0', label='Running Mean')
    ax1.fill_between(iterations, running_mean - 0.5, running_mean + 0.5, alpha=0.2, color='#1565C0')
    ax1.set_xlabel('Training Iteration', fontsize=11)
    ax1.set_ylabel('Mean Value', fontsize=11)
    ax1.set_title('Running Mean Evolution\nrunning_mean = (1-m)*running_mean + m*batch_mean', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # === Top Right: Running Variance Evolution ===
    ax2 = axes[0, 1]
    ax2.plot(iterations, batch_vars, 'o', alpha=0.4, markersize=4, color='#FFAB91', label='Batch Variance')
    ax2.plot(iterations, running_var, '-', linewidth=2.5, color='#E64A19', label='Running Variance')
    ax2.fill_between(iterations, running_var - 0.3, running_var + 0.3, alpha=0.2, color='#E64A19')
    ax2.set_xlabel('Training Iteration', fontsize=11)
    ax2.set_ylabel('Variance Value', fontsize=11)
    ax2.set_title('Running Variance Evolution\nrunning_var = (1-m)*running_var + m*batch_var', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # === Bottom Left: Different Momentum Values ===
    ax3 = axes[1, 0]

    momentums = [0.01, 0.1, 0.5, 0.9]
    colors_mom = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']

    for mom, color in zip(momentums, colors_mom):
        running = np.zeros(n_iterations)
        running[0] = 0
        for i in range(1, n_iterations):
            running[i] = (1 - mom) * running[i-1] + mom * batch_means[i]
        ax3.plot(iterations, running, '-', linewidth=2, color=color, label=f'm={mom}')

    ax3.plot(iterations, batch_means, 'o', alpha=0.2, markersize=3, color='gray', label='Batch Mean')
    ax3.set_xlabel('Training Iteration', fontsize=11)
    ax3.set_ylabel('Running Mean', fontsize=11)
    ax3.set_title('Effect of Momentum (m) on Running Mean\nLower m = smoother, slower adaptation', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # === Bottom Right: Convergence of Statistics ===
    ax4 = axes[1, 1]

    # Show how running stats stabilize over longer training
    n_long = 500
    batch_means_long = 2 * np.sin(np.arange(n_long) * 0.05) + np.random.randn(n_long) * 0.3

    running_long = np.zeros(n_long)
    for i in range(1, n_long):
        running_long[i] = 0.9 * running_long[i-1] + 0.1 * batch_means_long[i]

    ax4.plot(np.arange(n_long), batch_means_long, alpha=0.3, linewidth=0.5, color='gray', label='Batch Statistics')
    ax4.plot(np.arange(n_long), running_long, linewidth=2, color='#7B1FA2', label='Running Statistics')

    # Add vertical line showing when stats stabilize
    ax4.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Statistics stabilize')

    ax4.set_xlabel('Training Iteration', fontsize=11)
    ax4.set_ylabel('Statistic Value', fontsize=11)
    ax4.set_title('Running Statistics Stabilization\nUsed for inference after training', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Main title
    fig.suptitle('BatchNorm Running Statistics: How They Are Computed and Why They Matter\n'
                 'Running stats accumulate during training and are used for inference',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = OUTPUT_DIR / 'running_stats_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def generate_dropout_mask_visualization():
    """
    Generate visualization showing dropout effect on a neural network.
    Shows neurons being dropped and the inverted dropout scaling.
    """
    np.random.seed(42)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

    # === Top: Network with dropout visualization ===
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-1, 6)
    ax1.axis('off')
    ax1.set_title('Dropout in Neural Network: Training Mode (p=0.5)\n'
                  'Red neurons are "dropped" (output set to 0), others scaled by 1/(1-p)=2',
                  fontsize=12, fontweight='bold', pad=20)

    # Layer configuration
    layer_sizes = [4, 6, 6, 4, 2]
    layer_x = [1, 3, 5, 7, 9]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Hidden 3', 'Output']

    # Generate dropout masks (no dropout on input/output)
    dropout_masks = []
    for i, size in enumerate(layer_sizes):
        if i == 0 or i == len(layer_sizes) - 1:
            # No dropout on input/output
            mask = np.ones(size, dtype=bool)
        else:
            # 50% dropout on hidden layers
            mask = np.random.rand(size) > 0.5
        dropout_masks.append(mask)

    # Draw neurons
    neuron_positions = []
    for layer_idx, (x, size, mask, name) in enumerate(zip(layer_x, layer_sizes, dropout_masks, layer_names)):
        positions = []
        y_start = (5 - size) / 2 + 0.5

        for neuron_idx in range(size):
            y = y_start + neuron_idx * 0.8
            positions.append((x, y))

            if mask[neuron_idx]:
                # Active neuron
                circle = Circle((x, y), 0.25, facecolor='#4CAF50', edgecolor='black', linewidth=1.5, zorder=5)
                if layer_idx > 0 and layer_idx < len(layer_sizes) - 1:
                    # Show scaling for hidden layers
                    ax1.text(x, y, 'x2', fontsize=7, ha='center', va='center', fontweight='bold', color='white')
            else:
                # Dropped neuron
                circle = Circle((x, y), 0.25, facecolor='#EF5350', edgecolor='black', linewidth=1.5, zorder=5, alpha=0.5)
                ax1.text(x, y, 'X', fontsize=10, ha='center', va='center', fontweight='bold', color='white')

            ax1.add_patch(circle)

        neuron_positions.append(positions)

        # Layer name
        ax1.text(x, -0.5, name, fontsize=10, ha='center', fontweight='bold')

    # Draw connections
    for layer_idx in range(len(layer_sizes) - 1):
        for i, (x1, y1) in enumerate(neuron_positions[layer_idx]):
            for j, (x2, y2) in enumerate(neuron_positions[layer_idx + 1]):
                # Check if either neuron is dropped
                if dropout_masks[layer_idx][i] and dropout_masks[layer_idx + 1][j]:
                    alpha = 0.6
                    color = '#2E7D32'
                    linewidth = 0.8
                else:
                    alpha = 0.15
                    color = 'gray'
                    linewidth = 0.5

                ax1.plot([x1 + 0.25, x2 - 0.25], [y1, y2], '-',
                        color=color, alpha=alpha, linewidth=linewidth, zorder=1)

    # Legend
    active_patch = mpatches.Patch(color='#4CAF50', label='Active (scaled by 2)')
    dropped_patch = mpatches.Patch(color='#EF5350', alpha=0.5, label='Dropped (output=0)')
    ax1.legend(handles=[active_patch, dropped_patch], loc='upper right', fontsize=10)

    # === Bottom Left: Dropout probability effect ===
    ax2 = fig.add_subplot(gs[1, 0])

    # Show output distribution with different dropout rates
    input_val = np.ones(10000) * 2  # Input value of 2

    dropout_rates = [0.0, 0.2, 0.5, 0.8]
    colors_drop = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']

    for p, color in zip(dropout_rates, colors_drop):
        if p == 0:
            output = input_val.copy()
        else:
            mask = np.random.rand(len(input_val)) > p
            output = input_val * mask * (1 / (1 - p))

        # Slight offset for visibility
        ax2.hist(output + p * 0.1, bins=30, alpha=0.6, color=color,
                label=f'p={p} (keep={1-p:.1f})', density=True)

    ax2.set_xlabel('Output Value (input=2)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Inverted Dropout: Output Distribution\nExpected value stays the same across dropout rates',
                  fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-0.5, 5)

    # === Bottom Right: Standard vs Inverted Dropout ===
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    # Create comparison table
    table_data = [
        ['', 'Standard Dropout', 'Inverted Dropout'],
        ['Training', 'mask only\nno scaling', 'mask + scale by 1/(1-p)'],
        ['Inference', 'scale by (1-p)', 'no change'],
        ['Expected Value', 'E[out] = x * (1-p)', 'E[out] = x'],
        ['Advantage', 'Simple training', 'Simple inference\n(modern standard)']
    ]

    # Draw table
    cell_height = 0.15
    cell_widths = [0.18, 0.38, 0.38]
    start_y = 0.85
    start_x = 0.03

    colors_table = [['#E0E0E0', '#FFCDD2', '#C8E6C9'],
                    ['#E0E0E0', '#FFEBEE', '#E8F5E9'],
                    ['#E0E0E0', '#FFEBEE', '#E8F5E9'],
                    ['#E0E0E0', '#FFEBEE', '#E8F5E9'],
                    ['#E0E0E0', '#FFEBEE', '#E8F5E9']]

    for row_idx, row in enumerate(table_data):
        x = start_x
        for col_idx, (cell, width) in enumerate(zip(row, cell_widths)):
            y = start_y - row_idx * cell_height
            rect = FancyBboxPatch((x, y - cell_height + 0.02), width - 0.01, cell_height - 0.02,
                                  boxstyle="round,pad=0.01",
                                  facecolor=colors_table[row_idx][col_idx],
                                  edgecolor='gray', linewidth=0.5)
            ax3.add_patch(rect)

            fontweight = 'bold' if row_idx == 0 or col_idx == 0 else 'normal'
            fontsize = 9 if row_idx == 0 else 8
            ax3.text(x + width/2, y - cell_height/2, cell,
                    ha='center', va='center', fontsize=fontsize, fontweight=fontweight)
            x += width

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Standard vs Inverted Dropout Comparison', fontsize=11, fontweight='bold', pad=10)

    # Main title
    fig.suptitle('Dropout: Regularization by Randomly Dropping Neurons\n'
                 'Creates ensemble effect - trains exponentially many sub-networks',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = OUTPUT_DIR / 'dropout_mask_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def generate_normalization_comparison():
    """
    Generate visualization comparing BatchNorm, LayerNorm, and GroupNorm.
    Shows which dimensions each method normalizes over.
    """
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.3)

    # === Create tensor visualization for each normalization type ===

    def draw_tensor_normalization(ax, norm_type, title, description):
        """Draw a 3D tensor representation showing normalization dimensions."""
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # Tensor dimensions: N (batch) x C (channels) x H x W (spatial)
        # We'll show as N x C grid
        N, C = 4, 6
        cell_size = 0.12
        start_x, start_y = 0.1, 0.3

        # Color based on normalization type
        colors = {
            'batch': '#BBDEFB',      # Blue - normalize across batch
            'layer': '#C8E6C9',      # Green - normalize across features
            'instance': '#FFF9C4',   # Yellow - normalize per sample per channel
            'group': '#F8BBD9'       # Pink - normalize per group
        }

        # Draw the tensor grid
        for n in range(N):
            for c in range(C):
                x = start_x + c * cell_size * 1.1
                y = start_y + (N - 1 - n) * cell_size * 1.1

                # Determine color based on normalization type
                if norm_type == 'batch':
                    # BatchNorm: same column (channel) has same color
                    color = plt.cm.Blues(0.3 + 0.1 * c)
                elif norm_type == 'layer':
                    # LayerNorm: same row (sample) has same color
                    color = plt.cm.Greens(0.3 + 0.15 * n)
                elif norm_type == 'instance':
                    # InstanceNorm: each cell different
                    color = plt.cm.YlOrBr(0.2 + 0.1 * (n + c) % 4)
                elif norm_type == 'group':
                    # GroupNorm: groups of channels
                    group = c // 2  # 3 groups of 2 channels
                    color = plt.cm.RdPu(0.3 + 0.15 * group + 0.05 * n)

                rect = FancyBboxPatch((x, y), cell_size, cell_size,
                                      boxstyle="round,pad=0.01",
                                      facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)

        # Add labels
        ax.text(start_x + C * cell_size * 1.1 / 2, start_y + N * cell_size * 1.1 + 0.05,
               'Channels (C)', ha='center', fontsize=9)
        ax.text(start_x - 0.05, start_y + N * cell_size * 1.1 / 2 - 0.02,
               'Batch\n(N)', ha='center', va='center', fontsize=9, rotation=90)

        # Add brackets/arrows showing normalization direction
        if norm_type == 'batch':
            # Arrow along batch dimension (column)
            ax.annotate('', xy=(start_x + 0.06, start_y - 0.02),
                       xytext=(start_x + 0.06, start_y + N * cell_size * 1.1),
                       arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
            ax.text(start_x + 0.15, start_y + N * cell_size * 1.1 / 2,
                   'normalize', fontsize=8, rotation=90, va='center', color='blue')
        elif norm_type == 'layer':
            # Arrow along feature dimension (row)
            ax.annotate('', xy=(start_x - 0.02, start_y + (N-1) * cell_size * 1.1 + 0.06),
                       xytext=(start_x + C * cell_size * 1.1, start_y + (N-1) * cell_size * 1.1 + 0.06),
                       arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        elif norm_type == 'group':
            # Show group brackets
            for g in range(3):
                x1 = start_x + g * 2 * cell_size * 1.1
                ax.plot([x1, x1 + 2 * cell_size * 1.1 - 0.01],
                       [start_y - 0.03, start_y - 0.03], 'k-', linewidth=2)
                ax.text(x1 + cell_size * 1.1, start_y - 0.06, f'G{g+1}',
                       ha='center', fontsize=8, fontweight='bold')

        # Description
        ax.text(0.5, 0.08, description, ha='center', va='top', fontsize=9,
               wrap=True, transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # === BatchNorm ===
    ax1 = fig.add_subplot(gs[0, 0])
    draw_tensor_normalization(ax1, 'batch', 'Batch Normalization',
        'Normalizes across batch dimension\nfor each channel separately.\n'
        'Uses running stats for inference.')

    # === LayerNorm ===
    ax2 = fig.add_subplot(gs[0, 1])
    draw_tensor_normalization(ax2, 'layer', 'Layer Normalization',
        'Normalizes across all features\nfor each sample separately.\n'
        'Same computation in train/inference.')

    # === GroupNorm ===
    ax3 = fig.add_subplot(gs[1, 0])
    draw_tensor_normalization(ax3, 'group', 'Group Normalization',
        'Normalizes across groups of channels\nper sample. Compromise between\nBatchNorm and LayerNorm.')

    # === Instance Norm ===
    ax4 = fig.add_subplot(gs[1, 1])
    draw_tensor_normalization(ax4, 'instance', 'Instance Normalization',
        'Normalizes each channel of each sample\nindependently. Used in style transfer\nand image generation.')

    # === Bottom: Comparison table and use cases ===
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Create detailed comparison table
    headers = ['Property', 'BatchNorm', 'LayerNorm', 'GroupNorm', 'InstanceNorm']
    data = [
        ['Normalizes over', 'N, H, W', 'C, H, W', 'G channels, H, W', 'H, W'],
        ['Batch dependent', 'Yes', 'No', 'No', 'No'],
        ['Running stats', 'Yes', 'No', 'No', 'No'],
        ['Best for', 'CNNs (large batch)', 'Transformers, RNNs', 'CNNs (small batch)', 'Style transfer'],
        ['Train/Eval diff', 'Yes', 'No', 'No', 'No']
    ]

    # Draw table
    cell_height = 0.11
    cell_widths = [0.16, 0.2, 0.2, 0.22, 0.2]
    start_y = 0.92
    start_x = 0.01

    header_colors = ['#E0E0E0', '#BBDEFB', '#C8E6C9', '#F8BBD9', '#FFF9C4']

    # Draw headers
    x = start_x
    for header, width, color in zip(headers, cell_widths, header_colors):
        rect = FancyBboxPatch((x, start_y - cell_height + 0.02), width - 0.01, cell_height - 0.02,
                              boxstyle="round,pad=0.01",
                              facecolor=color, edgecolor='gray', linewidth=0.5)
        ax5.add_patch(rect)
        ax5.text(x + width/2, start_y - cell_height/2, header,
                ha='center', va='center', fontsize=9, fontweight='bold')
        x += width

    # Draw data rows
    for row_idx, row in enumerate(data):
        x = start_x
        for col_idx, (cell, width) in enumerate(zip(row, cell_widths)):
            y = start_y - (row_idx + 1) * cell_height - cell_height

            if col_idx == 0:
                bg_color = '#F5F5F5'
                fontweight = 'bold'
            else:
                bg_color = 'white'
                fontweight = 'normal'

            rect = FancyBboxPatch((x, y + 0.02), width - 0.01, cell_height - 0.02,
                                  boxstyle="round,pad=0.01",
                                  facecolor=bg_color, edgecolor='#E0E0E0', linewidth=0.5)
            ax5.add_patch(rect)
            ax5.text(x + width/2, y + cell_height/2, cell,
                    ha='center', va='center', fontsize=8, fontweight=fontweight)
            x += width

    ax5.set_xlim(0, 1)
    ax5.set_ylim(0.2, 1)
    ax5.set_title('Normalization Methods Comparison', fontsize=12, fontweight='bold', pad=5)

    # Main title
    fig.suptitle('Normalization Techniques: BatchNorm vs LayerNorm vs GroupNorm\n'
                 'Different methods normalize over different dimensions',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_path = OUTPUT_DIR / 'normalization_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def generate_training_vs_inference():
    """
    Generate visualization comparing training vs inference behavior
    for BatchNorm and Dropout.
    """
    np.random.seed(42)

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

    # === Top: BatchNorm Training vs Inference ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('BatchNorm: Training Mode\n(Uses batch statistics)', fontsize=11, fontweight='bold')

    # Simulate different batches with different statistics
    n_samples = 50
    batch1 = np.random.normal(3, 2, n_samples)
    batch2 = np.random.normal(5, 1.5, n_samples)
    batch3 = np.random.normal(2, 2.5, n_samples)

    # Show batch statistics vary
    ax1.hist(batch1, bins=20, alpha=0.5, color='#E53935', label=f'Batch 1 (mean={np.mean(batch1):.1f})', density=True)
    ax1.hist(batch2, bins=20, alpha=0.5, color='#43A047', label=f'Batch 2 (mean={np.mean(batch2):.1f})', density=True)
    ax1.hist(batch3, bins=20, alpha=0.5, color='#1E88E5', label=f'Batch 3 (mean={np.mean(batch3):.1f})', density=True)
    ax1.set_xlabel('Activation Value', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.annotate('Each batch uses\nits OWN statistics', xy=(6, 0.25), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('BatchNorm: Inference Mode\n(Uses running statistics)', fontsize=11, fontweight='bold')

    # Show running statistics used uniformly
    running_mean = 3.5
    running_var = 2.0

    # All batches normalized with same stats
    norm1 = (batch1 - running_mean) / np.sqrt(running_var + 1e-5)
    norm2 = (batch2 - running_mean) / np.sqrt(running_var + 1e-5)
    norm3 = (batch3 - running_mean) / np.sqrt(running_var + 1e-5)

    ax2.hist(norm1, bins=20, alpha=0.5, color='#E53935', label='Batch 1 (normalized)', density=True)
    ax2.hist(norm2, bins=20, alpha=0.5, color='#43A047', label='Batch 2 (normalized)', density=True)
    ax2.hist(norm3, bins=20, alpha=0.5, color='#1E88E5', label='Batch 3 (normalized)', density=True)
    ax2.set_xlabel('Normalized Value', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.annotate('All batches use\nSAME running stats\n(fixed after training)', xy=(1.5, 0.35), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))

    # === Middle: Dropout Training vs Inference ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('Dropout: Training Mode\n(Random masking + scaling)', fontsize=11, fontweight='bold')

    # Show multiple forward passes with different dropout masks
    input_val = 2.0
    n_passes = 5
    p = 0.5

    for i in range(n_passes):
        mask = np.random.rand(100) > p
        output = input_val * mask * (1 / (1 - p))
        # Add small jitter for visualization
        y_pos = 0.2 + i * 0.15
        ax3.scatter(output, np.ones_like(output) * y_pos + np.random.rand(len(output)) * 0.1,
                   alpha=0.5, s=10, label=f'Pass {i+1}' if i < 3 else None)

    ax3.axvline(x=input_val, color='red', linestyle='--', linewidth=2, label='Original input=2')
    ax3.axvline(x=input_val * 2, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Scaled output=4')
    ax3.set_xlabel('Output Value', fontsize=10)
    ax3.set_ylabel('Forward Pass', fontsize=10)
    ax3.set_yticks([0.25, 0.4, 0.55, 0.7, 0.85])
    ax3.set_yticklabels(['Pass 1', 'Pass 2', 'Pass 3', 'Pass 4', 'Pass 5'])
    ax3.legend(fontsize=8, loc='upper right')
    ax3.set_xlim(-0.5, 5)
    ax3.annotate('Stochastic!\nDifferent outputs\neach pass', xy=(3.5, 0.5), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Dropout: Inference Mode\n(Identity function - no change)', fontsize=11, fontweight='bold')

    # Inference: deterministic output
    input_vals = np.linspace(0, 4, 100)
    output_vals = input_vals  # Identity

    ax4.plot(input_vals, output_vals, 'b-', linewidth=3, label='Inference: y = x')
    ax4.fill_between(input_vals, output_vals - 0.1, output_vals + 0.1, alpha=0.2, color='blue')
    ax4.plot([0, 4], [0, 4], 'g--', linewidth=2, alpha=0.5, label='Expected value')
    ax4.set_xlabel('Input Value', fontsize=10)
    ax4.set_ylabel('Output Value', fontsize=10)
    ax4.legend(fontsize=9, loc='upper left')
    ax4.annotate('Deterministic!\nSame output\nevery time', xy=(2.5, 1), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))
    ax4.set_xlim(0, 4)
    ax4.set_ylim(0, 4)
    ax4.set_aspect('equal')

    # === Bottom: Summary diagram ===
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    ax5.set_xlim(0, 14)
    ax5.set_ylim(0, 4)

    # Training side
    train_box = FancyBboxPatch((0.5, 1), 6, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor='#E53935', linewidth=2)
    ax5.add_patch(train_box)
    ax5.text(3.5, 3.2, 'TRAINING MODE', fontsize=12, fontweight='bold', ha='center', color='#C62828')
    ax5.text(3.5, 2.5, 'model.train()', fontsize=10, ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax5.text(3.5, 1.8, 'BatchNorm: batch statistics\nDropout: random masking + 1/(1-p) scaling',
            fontsize=9, ha='center', va='center')

    # Inference side
    eval_box = FancyBboxPatch((7.5, 1), 6, 2.5, boxstyle="round,pad=0.1",
                               facecolor='#E8F5E9', edgecolor='#43A047', linewidth=2)
    ax5.add_patch(eval_box)
    ax5.text(10.5, 3.2, 'INFERENCE MODE', fontsize=12, fontweight='bold', ha='center', color='#2E7D32')
    ax5.text(10.5, 2.5, 'model.eval()', fontsize=10, ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax5.text(10.5, 1.8, 'BatchNorm: running statistics\nDropout: identity (no change)',
            fontsize=9, ha='center', va='center')

    # Arrow between
    ax5.annotate('', xy=(7.3, 2.25), xytext=(6.7, 2.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Warning
    ax5.text(7, 0.4, 'CRITICAL: Always call model.eval() before inference!\n'
             'Forgetting this is a common source of bugs.',
            fontsize=10, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2))

    ax5.set_title('Training vs Inference: Critical Differences', fontsize=12, fontweight='bold', pad=10)

    # Main title
    fig.suptitle('Training vs Inference Mode: BatchNorm and Dropout Behavior\n'
                 'Understanding mode differences is crucial for correct model deployment',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_path = OUTPUT_DIR / 'training_vs_inference.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all batch normalization and dropout visualizations."""
    print("Generating Batch Normalization and Dropout Visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    # 1. Internal covariate shift
    print("\n1. Generating internal covariate shift visualization...")
    generate_internal_covariate_shift()

    # 2. Running statistics evolution
    print("\n2. Generating running statistics evolution visualization...")
    generate_running_stats_evolution()

    # 3. Dropout mask visualization
    print("\n3. Generating dropout mask visualization...")
    generate_dropout_mask_visualization()

    # 4. Normalization comparison
    print("\n4. Generating normalization comparison visualization...")
    generate_normalization_comparison()

    # 5. Training vs inference mode
    print("\n5. Generating training vs inference mode visualization...")
    generate_training_vs_inference()

    print("\n" + "-" * 50)
    print("All visualizations generated successfully!")
    print("\nGenerated files:")
    print(f"  - {OUTPUT_DIR / 'internal_covariate_shift.png'}")
    print(f"  - {OUTPUT_DIR / 'running_stats_evolution.png'}")
    print(f"  - {OUTPUT_DIR / 'dropout_mask_visualization.png'}")
    print(f"  - {OUTPUT_DIR / 'normalization_comparison.png'}")
    print(f"  - {OUTPUT_DIR / 'training_vs_inference.png'}")


if __name__ == "__main__":
    main()
