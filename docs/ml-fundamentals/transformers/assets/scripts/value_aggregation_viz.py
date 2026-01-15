"""
Value Aggregation Visualization

This script demonstrates how attention weights combine value vectors
to produce the output representation for each position.

Shows:
1. Attention weights as bar heights
2. Value vectors as colored vectors/bars
3. Weighted sum computation
4. Final output as the aggregated result

Key formula: output = sum(attention_weight_i * value_i)

This is the final step of the attention mechanism after computing
the attention weights from QK^T.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_value_aggregation_viz():
    """Create a visualization showing how values are weighted and summed."""

    fig = plt.figure(figsize=(16, 12))

    # Example: Query position attending to 4 value vectors
    tokens = ["The", "cat", "sat", "mat"]
    n_tokens = 4
    d_v = 4  # Value dimension (simplified for visualization)

    # Attention weights (from softmax)
    attention_weights = np.array([0.15, 0.45, 0.30, 0.10])

    # Value vectors (each token has a d_v-dimensional value)
    np.random.seed(42)
    values = np.array([
        [0.8, 0.2, 0.1, 0.5],   # The
        [0.3, 0.9, 0.6, 0.2],   # cat
        [0.5, 0.4, 0.8, 0.7],   # sat
        [0.2, 0.3, 0.4, 0.9],   # mat
    ])

    # Weighted values
    weighted_values = attention_weights[:, np.newaxis] * values

    # Output (weighted sum)
    output = np.sum(weighted_values, axis=0)

    # Create subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1], hspace=0.4, wspace=0.3)

    # ============== Panel 1: Attention Weights ==============
    ax1 = fig.add_subplot(gs[0, 0])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(tokens, attention_weights, color=colors, alpha=0.8, edgecolor='black')

    ax1.set_title('Step 1: Attention Weights (from softmax)\nHow much each token contributes',
                 fontsize=11, fontweight='bold')
    ax1.set_ylabel('Attention Weight', fontsize=10)
    ax1.set_ylim(0, 0.6)

    # Add value labels on bars
    for bar, weight in zip(bars, attention_weights):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{weight:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add annotation
    ax1.annotate(f'Sum = {np.sum(attention_weights):.2f}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # ============== Panel 2: Value Vectors Heatmap ==============
    ax2 = fig.add_subplot(gs[0, 1])

    im = ax2.imshow(values, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Step 2: Value Vectors (V)\nInformation to aggregate', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Value Dimension', fontsize=10)
    ax2.set_ylabel('Token', fontsize=10)
    ax2.set_xticks(range(d_v))
    ax2.set_yticks(range(n_tokens))
    ax2.set_xticklabels([f'd{i+1}' for i in range(d_v)])
    ax2.set_yticklabels(tokens)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Add values to cells
    for i in range(n_tokens):
        for j in range(d_v):
            ax2.text(j, i, f'{values[i,j]:.1f}', ha='center', va='center',
                    fontsize=9, color='white' if values[i,j] > 0.5 else 'black')

    # ============== Panel 3: Weighted Values ==============
    ax3 = fig.add_subplot(gs[1, :])

    # Create a visual showing the multiplication
    bar_width = 0.18
    x = np.arange(d_v)

    for i, (token, weight, color) in enumerate(zip(tokens, attention_weights, colors)):
        offset = (i - 1.5) * bar_width
        bars = ax3.bar(x + offset, weighted_values[i], bar_width,
                      label=f'{token} ({weight:.2f})', color=color, alpha=0.7, edgecolor='black')

    # Add output as a thick line
    ax3.plot(x, output, 'ko-', linewidth=3, markersize=10, label='Output (sum)', zorder=5)

    ax3.set_title('Step 3: Weighted Values & Output\n'
                 'weighted_value_i = attention_weight_i * value_i\n'
                 'output = sum(weighted_values)',
                 fontsize=11, fontweight='bold')
    ax3.set_xlabel('Value Dimension', fontsize=10)
    ax3.set_ylabel('Value', fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'd{i+1}' for i in range(d_v)])
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(0, max(output) * 1.3)

    # ============== Panel 4: Mathematical Formula ==============
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    formula_text = (
        'Mathematical Computation:\n\n'
        'output = $\\sum_{i=1}^{n}$ attention_weight$_i$ $\\cdot$ value$_i$\n\n'
        'For this example:\n'
        f'output = {attention_weights[0]:.2f}*V$_{{The}}$ + {attention_weights[1]:.2f}*V$_{{cat}}$ + '
        f'{attention_weights[2]:.2f}*V$_{{sat}}$ + {attention_weights[3]:.2f}*V$_{{mat}}$\n\n'
        f'output = [{output[0]:.2f}, {output[1]:.2f}, {output[2]:.2f}, {output[3]:.2f}]'
    )

    ax4.text(0.1, 0.5, formula_text, fontsize=11, verticalalignment='center',
            family='monospace', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # ============== Panel 5: Interpretation ==============
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    interpretation_text = (
        'Interpretation:\n\n'
        'The output vector is a blend of all value vectors,\n'
        'weighted by how relevant each token is to the query.\n\n'
        'In this example:\n'
        f'- "cat" contributes most ({attention_weights[1]:.0%})\n'
        f'- "sat" is second ({attention_weights[2]:.0%})\n'
        f'- "The" and "mat" contribute less\n\n'
        'The output inherits properties from attended tokens:\n'
        '- High d2 from "cat" (0.9 * 0.45 = 0.41)\n'
        '- High d3 from "sat" (0.8 * 0.30 = 0.24)'
    )

    ax5.text(0.1, 0.5, interpretation_text, fontsize=10, verticalalignment='center',
            family='monospace', transform=ax5.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.suptitle('Value Aggregation: The Final Step of Attention',
                fontsize=14, fontweight='bold', y=0.98)

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/value_aggregation_viz.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")

    # Create a second simplified diagram
    create_simplified_aggregation_diagram(tokens, attention_weights, values, output)

    return output_path

def create_simplified_aggregation_diagram(tokens, weights, values, output):
    """Create a simplified flow diagram of value aggregation."""

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Draw value vectors on the left
    for i, (token, weight, val, color) in enumerate(zip(tokens, weights, values, colors)):
        y = 6 - i * 1.5

        # Token label
        ax.text(0, y, token, ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

        # Value vector (as a small heatmap-like bar)
        for j, v in enumerate(val):
            rect = patches.Rectangle((1.5 + j*0.8, y-0.3), 0.7, 0.6,
                                     facecolor=plt.cm.Blues(v), edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(1.85 + j*0.8, y, f'{v:.1f}', ha='center', va='center', fontsize=8)

        # Multiplication symbol and weight
        ax.text(5.5, y, f'x {weight:.2f}', ha='center', va='center', fontsize=11, fontweight='bold')

        # Equals sign
        ax.text(7, y, '=', ha='center', va='center', fontsize=14, fontweight='bold')

        # Weighted value
        weighted = weight * val
        for j, wv in enumerate(weighted):
            rect = patches.Rectangle((7.5 + j*0.8, y-0.3), 0.7, 0.6,
                                     facecolor=plt.cm.Blues(wv), edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(7.85 + j*0.8, y, f'{wv:.2f}', ha='center', va='center', fontsize=7)

    # Draw sum arrows converging
    ax.annotate('', xy=(12, 3), xytext=(11, 6), arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(12, 3), xytext=(11, 4.5), arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(12, 3), xytext=(11, 3), arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(12, 3), xytext=(11, 1.5), arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Sum symbol
    ax.text(11.5, 0.3, '$\\Sigma$', ha='center', va='center', fontsize=20, fontweight='bold')

    # Output vector
    ax.text(12.5, 3.8, 'Output:', ha='center', va='center', fontsize=11, fontweight='bold')
    for j, o in enumerate(output):
        rect = patches.Rectangle((12 + j*0.8, 2.7), 0.7, 0.6,
                                 facecolor=plt.cm.Blues(o/max(output)), edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(12.35 + j*0.8, 3, f'{o:.2f}', ha='center', va='center', fontsize=8, fontweight='bold')

    # Title and labels
    ax.text(0, 7.5, 'Token', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3, 7.5, 'Value Vector (V)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5.5, 7.5, 'Weight', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(9, 7.5, 'Weighted Value', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_title('Value Aggregation Flow: attention_weights * V = output',
                fontsize=14, fontweight='bold', y=1.02)

    # Save
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/value_aggregation_flow.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    create_value_aggregation_viz()
