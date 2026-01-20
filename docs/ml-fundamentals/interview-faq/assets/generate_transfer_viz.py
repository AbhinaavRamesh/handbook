"""
Transfer Learning Visualizations
Generates matplotlib diagrams for transfer learning concepts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, ConnectionPatch
import numpy as np
import os

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_feature_hierarchy_diagram():
    """
    CNN Feature Hierarchy: edges -> textures -> parts -> objects
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define layers with colors (early = more transferable = green, later = less = orange/red)
    layers = [
        {'name': 'Conv1-2\nEdges & Corners', 'x': 1, 'color': '#2ecc71', 'transferability': 'Very High'},
        {'name': 'Conv3-4\nTextures & Patterns', 'x': 3.25, 'color': '#27ae60', 'transferability': 'High'},
        {'name': 'Conv5-6\nParts & Shapes', 'x': 5.5, 'color': '#f39c12', 'transferability': 'Medium'},
        {'name': 'Conv7+\nObjects & Concepts', 'x': 7.75, 'color': '#e74c3c', 'transferability': 'Low'},
    ]

    # Draw layers as boxes
    box_width = 1.8
    box_height = 2.5
    y_center = 3

    for layer in layers:
        # Main box
        rect = FancyBboxPatch(
            (layer['x'] - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)

        # Layer name
        ax.text(layer['x'], y_center + 0.3, layer['name'],
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white')

        # Transferability label
        ax.text(layer['x'], y_center - 0.8, f"({layer['transferability']})",
                ha='center', va='center', fontsize=9, color='white', style='italic')

    # Draw arrows between layers
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    for i in range(len(layers) - 1):
        arrow = FancyArrowPatch(
            (layers[i]['x'] + box_width/2 + 0.1, y_center),
            (layers[i+1]['x'] - box_width/2 - 0.1, y_center),
            arrowstyle=arrow_style,
            color='#34495e',
            mutation_scale=1
        )
        ax.add_patch(arrow)

    # Title and subtitle
    ax.text(5, 5.5, 'CNN Feature Hierarchy', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#2c3e50')
    ax.text(5, 5.1, 'Early layers learn general features, later layers learn task-specific features',
            ha='center', va='center', fontsize=10, color='#7f8c8d')

    # Legend for transferability
    ax.text(5, 0.8, 'Transferability:', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2c3e50')

    # Color bar legend
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    cax = fig.add_axes([0.3, 0.08, 0.4, 0.03])
    cax.imshow(gradient, aspect='auto', cmap=plt.cm.RdYlGn_r)
    cax.set_xticks([0, 99])
    cax.set_xticklabels(['Task-Specific', 'General/Transferable'])
    cax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'transfer_feature_hierarchy.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: transfer_feature_hierarchy.png")


def create_freeze_unfreeze_diagram():
    """
    Network diagram showing freeze/unfreeze strategies with colored layers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 6), dpi=150)

    strategies = [
        {'title': 'Feature Extraction\n(Freeze All)', 'freeze': [0, 1, 2, 3, 4], 'train': [5]},
        {'title': 'Gradual Unfreezing\n(Freeze Early)', 'freeze': [0, 1, 2], 'train': [3, 4, 5]},
        {'title': 'Full Fine-Tuning\n(Train All)', 'freeze': [], 'train': [0, 1, 2, 3, 4, 5]},
    ]

    layer_names = ['Input', 'Conv1', 'Conv2', 'Conv3', 'FC1', 'Output']

    for ax, strategy in zip(axes, strategies):
        ax.set_xlim(0, 4)
        ax.set_ylim(-0.5, 7)
        ax.axis('off')
        ax.set_title(strategy['title'], fontsize=11, fontweight='bold', pad=10)

        # Draw layers
        box_width = 3
        box_height = 0.8
        x_center = 2

        for i, name in enumerate(layer_names):
            y = 6 - i

            # Determine color
            if i in strategy['freeze']:
                color = '#3498db'  # Blue = frozen
                text_label = 'Frozen'
            else:
                color = '#e74c3c'  # Red = trainable
                text_label = 'Trainable'

            # Draw box
            rect = FancyBboxPatch(
                (x_center - box_width/2, y - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color,
                edgecolor='#2c3e50',
                linewidth=1.5,
                alpha=0.85
            )
            ax.add_patch(rect)

            # Layer name
            ax.text(x_center, y, name, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

            # Draw arrow to next layer (except last)
            if i < len(layer_names) - 1:
                ax.annotate('', xy=(x_center, y - box_height/2 - 0.1),
                           xytext=(x_center, y - box_height/2 - 0.4),
                           arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    # Add legend
    frozen_patch = mpatches.Patch(color='#3498db', label='Frozen (no gradients)')
    trainable_patch = mpatches.Patch(color='#e74c3c', label='Trainable (updated)')
    fig.legend(handles=[frozen_patch, trainable_patch], loc='lower center',
               ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle('Freeze/Unfreeze Strategies', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(os.path.join(SCRIPT_DIR, 'transfer_freeze_strategies.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: transfer_freeze_strategies.png")


def create_learning_curve_comparison():
    """
    Learning curves: Transfer Learning vs From-Scratch training.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Generate synthetic data
    epochs = np.arange(1, 51)

    # From scratch - starts low, slow improvement
    from_scratch = 0.5 + 0.35 * (1 - np.exp(-0.08 * epochs)) + np.random.normal(0, 0.01, len(epochs))
    from_scratch = np.clip(from_scratch, 0, 1)

    # Transfer learning (feature extraction) - starts higher, quick plateau
    transfer_fe = 0.75 + 0.12 * (1 - np.exp(-0.3 * epochs)) + np.random.normal(0, 0.008, len(epochs))
    transfer_fe = np.clip(transfer_fe, 0, 1)

    # Transfer learning (fine-tuning) - starts high, reaches highest
    transfer_ft = 0.72 + 0.20 * (1 - np.exp(-0.15 * epochs)) + np.random.normal(0, 0.008, len(epochs))
    transfer_ft = np.clip(transfer_ft, 0, 1)

    # Plot curves
    ax.plot(epochs, from_scratch, 'o-', color='#e74c3c', label='From Scratch',
            linewidth=2, markersize=3, alpha=0.8)
    ax.plot(epochs, transfer_fe, 's-', color='#3498db', label='Transfer (Feature Extraction)',
            linewidth=2, markersize=3, alpha=0.8)
    ax.plot(epochs, transfer_ft, '^-', color='#2ecc71', label='Transfer (Fine-Tuning)',
            linewidth=2, markersize=3, alpha=0.8)

    # Annotations
    ax.annotate('Higher starting\naccuracy', xy=(3, 0.76), xytext=(8, 0.65),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5))

    ax.annotate('Faster\nconvergence', xy=(10, 0.86), xytext=(18, 0.78),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

    ax.annotate('Highest final\nperformance', xy=(45, 0.91), xytext=(38, 0.95),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

    # Styling
    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('Learning Curves: Transfer Learning vs From-Scratch', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 52)
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)

    # Add shaded regions for key insights
    ax.axhspan(0.85, 0.95, alpha=0.1, color='#2ecc71', label='_nolegend_')
    ax.text(48, 0.47, 'Transfer learning benefits:\n- Higher starting point\n- Faster convergence\n- Better final accuracy',
            fontsize=9, ha='right', va='bottom', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'transfer_learning_curves.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: transfer_learning_curves.png")


def create_domain_similarity_heatmap():
    """
    Domain similarity matrix showing transfer effectiveness between domains.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Define domains
    domains = ['ImageNet\n(Natural)', 'Medical\nX-rays', 'Satellite\nImagery', 'Faces',
               'Text\n(NLP)', 'Documents', 'Audio\nSpectrograms']

    # Similarity matrix (symmetric, 1 = same domain, 0 = very different)
    # Higher values = better transfer
    similarity = np.array([
        [1.0,  0.4,  0.5,  0.7,  0.1,  0.3,  0.2],  # ImageNet
        [0.4,  1.0,  0.3,  0.4,  0.1,  0.5,  0.2],  # Medical
        [0.5,  0.3,  1.0,  0.4,  0.1,  0.4,  0.2],  # Satellite
        [0.7,  0.4,  0.4,  1.0,  0.1,  0.3,  0.2],  # Faces
        [0.1,  0.1,  0.1,  0.1,  1.0,  0.7,  0.2],  # Text
        [0.3,  0.5,  0.4,  0.3,  0.7,  1.0,  0.2],  # Documents
        [0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  1.0],  # Audio
    ])

    # Create heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(similarity, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Transfer Effectiveness', fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Poor', 'Low', 'Medium', 'Good', 'Excellent'])

    # Set ticks and labels
    ax.set_xticks(np.arange(len(domains)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels(domains, fontsize=9)
    ax.set_yticklabels(domains, fontsize=9)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    for i in range(len(domains)):
        for j in range(len(domains)):
            value = similarity[i, j]
            text_color = 'white' if value < 0.5 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')

    ax.set_title('Domain Similarity Matrix for Transfer Learning', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Domain', fontsize=11)
    ax.set_ylabel('Source Domain', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'transfer_domain_similarity.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: transfer_domain_similarity.png")


def create_lora_decomposition_diagram():
    """
    LoRA low-rank decomposition diagram: W' = W + BA
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'LoRA: Low-Rank Adaptation', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#2c3e50')
    ax.text(6, 7.0, "W' = W + BA  (trainable parameters << original)", ha='center', va='center',
            fontsize=11, color='#7f8c8d', style='italic')

    # Original weight matrix W (large, frozen)
    w_x, w_y = 1, 3.5
    w_width, w_height = 2, 3
    w_rect = FancyBboxPatch((w_x, w_y), w_width, w_height,
                            boxstyle="round,pad=0.02",
                            facecolor='#3498db', edgecolor='#2c3e50',
                            linewidth=2, alpha=0.8)
    ax.add_patch(w_rect)
    ax.text(w_x + w_width/2, w_y + w_height/2, 'W\n(d x k)\nFROZEN',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(w_x + w_width/2, w_y - 0.4, 'd=4096, k=4096', ha='center', fontsize=8, color='#7f8c8d')

    # Plus sign
    ax.text(4, 5, '+', ha='center', va='center', fontsize=24, fontweight='bold', color='#2c3e50')

    # Matrix B (tall, thin)
    b_x, b_y = 5, 3.5
    b_width, b_height = 0.6, 3
    b_rect = FancyBboxPatch((b_x, b_y), b_width, b_height,
                            boxstyle="round,pad=0.02",
                            facecolor='#e74c3c', edgecolor='#2c3e50',
                            linewidth=2, alpha=0.8)
    ax.add_patch(b_rect)
    ax.text(b_x + b_width/2, b_y + b_height/2, 'B\n(d x r)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Multiplication sign
    ax.text(6.2, 5, 'x', ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')

    # Matrix A (short, wide)
    a_x, a_y = 6.8, 4.5
    a_width, a_height = 2, 0.6
    a_rect = FancyBboxPatch((a_x, a_y), a_width, a_height,
                            boxstyle="round,pad=0.02",
                            facecolor='#e74c3c', edgecolor='#2c3e50',
                            linewidth=2, alpha=0.8)
    ax.add_patch(a_rect)
    ax.text(a_x + a_width/2, a_y + a_height/2, 'A (r x k)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Trainable label
    ax.text(6.5, 3.2, 'TRAINABLE', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#e74c3c')
    ax.text(6.5, 2.8, 'r=8 (rank)', ha='center', va='center', fontsize=9, color='#7f8c8d')

    # Equals sign
    ax.text(9.5, 5, '=', ha='center', va='center', fontsize=24, fontweight='bold', color='#2c3e50')

    # Result W' (same size as W)
    wp_x, wp_y = 10, 3.5
    wp_width, wp_height = 2, 3
    wp_rect = FancyBboxPatch((wp_x, wp_y), wp_width, wp_height,
                             boxstyle="round,pad=0.02",
                             facecolor='#9b59b6', edgecolor='#2c3e50',
                             linewidth=2, alpha=0.8)
    ax.add_patch(wp_rect)
    ax.text(wp_x + wp_width/2, wp_y + wp_height/2, "W'\n(d x k)\nAdapted",
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Parameter comparison box
    box_y = 1.2
    param_box = FancyBboxPatch((2, box_y - 0.4), 8, 1.2,
                               boxstyle="round,pad=0.02",
                               facecolor='#ecf0f1', edgecolor='#bdc3c7',
                               linewidth=1.5, alpha=0.9)
    ax.add_patch(param_box)

    ax.text(6, box_y + 0.4, 'Parameter Comparison', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2c3e50')
    ax.text(6, box_y - 0.05, 'Full Fine-Tuning: d x k = 4096 x 4096 = 16.7M params',
            ha='center', va='center', fontsize=9, color='#e74c3c')
    ax.text(6, box_y - 0.4, 'LoRA (r=8): d x r + r x k = 4096 x 8 + 8 x 4096 = 65K params (0.4%)',
            ha='center', va='center', fontsize=9, color='#27ae60')

    # Legend
    frozen_patch = mpatches.Patch(color='#3498db', label='Frozen weights')
    trainable_patch = mpatches.Patch(color='#e74c3c', label='Trainable (LoRA)')
    result_patch = mpatches.Patch(color='#9b59b6', label='Adapted output')
    ax.legend(handles=[frozen_patch, trainable_patch, result_patch],
              loc='upper right', fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'transfer_lora_decomposition.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: transfer_lora_decomposition.png")


if __name__ == '__main__':
    print("Generating Transfer Learning Visualizations...")
    print(f"Output directory: {SCRIPT_DIR}")
    print("-" * 50)

    create_feature_hierarchy_diagram()
    create_freeze_unfreeze_diagram()
    create_learning_curve_comparison()
    create_domain_similarity_heatmap()
    create_lora_decomposition_diagram()

    print("-" * 50)
    print("All visualizations generated successfully!")
