"""
Softmax Saturation and Gradient Behavior Visualization

This script demonstrates:
1. How softmax behaves at different scales (small vs large inputs)
2. The gradient of softmax at different operating points
3. Why saturation leads to vanishing gradients
4. The importance of keeping inputs in a moderate range

Key formula: d(softmax)/dx = softmax(x) * (1 - softmax(x))
When softmax -> 0 or 1, gradient -> 0 (vanishing gradient problem)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def softmax(x):
    """Compute softmax values for each set of scores."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def softmax_gradient(x):
    """
    Compute the gradient of softmax output w.r.t. input.
    For diagonal elements: d(softmax_i)/d(x_i) = s_i * (1 - s_i)
    """
    s = softmax(x)
    return s * (1 - s)

def create_softmax_saturation_viz():
    """Create a multi-panel visualization of softmax saturation and gradients."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # ============== Panel 1: Softmax at Different Scales ==============
    ax1 = fig.add_subplot(gs[0, 0])

    # Different input scales
    base_logits = np.array([1, 2, 3])
    scales = [1, 5, 10, 20]

    bar_width = 0.2
    x = np.arange(len(base_logits))

    for i, scale in enumerate(scales):
        scaled_logits = base_logits * scale
        probs = softmax(scaled_logits)
        ax1.bar(x + i * bar_width, probs, bar_width, label=f'Scale={scale}x',
                color=colors[i], alpha=0.8)

    ax1.set_title('Softmax Output at Different Input Scales\n[1, 2, 3] vs [10, 20, 30] vs [100, 200, 300]', fontsize=11)
    ax1.set_xlabel('Input Position', fontsize=10)
    ax1.set_ylabel('Softmax Probability', fontsize=10)
    ax1.set_xticks(x + bar_width * 1.5)
    ax1.set_xticklabels(['x_1', 'x_2', 'x_3'])
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_ylim(0, 1.1)

    # ============== Panel 2: Softmax Values vs Input ==============
    ax2 = fig.add_subplot(gs[0, 1])

    # For a 2-class softmax, show how output changes with input difference
    differences = np.linspace(-10, 10, 200)
    softmax_vals = 1 / (1 + np.exp(-differences))  # sigmoid = 2-class softmax

    ax2.plot(differences, softmax_vals, 'b-', linewidth=2, label='Softmax output')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Mark saturation regions
    ax2.axhspan(0, 0.1, alpha=0.2, color='red', label='Saturation (near 0)')
    ax2.axhspan(0.9, 1.0, alpha=0.2, color='red')

    ax2.set_title('Softmax Saturation Regions\n(2-class case: sigmoid)', fontsize=11)
    ax2.set_xlabel('Input Difference (x_1 - x_2)', fontsize=10)
    ax2.set_ylabel('Softmax(x_1)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.05, 1.05)

    # ============== Panel 3: Gradient Magnitude ==============
    ax3 = fig.add_subplot(gs[0, 2])

    # Gradient of sigmoid: s(1-s)
    gradients = softmax_vals * (1 - softmax_vals)

    ax3.plot(differences, gradients, 'g-', linewidth=2, label='Gradient magnitude')
    ax3.axhline(y=0.25, color='orange', linestyle='--', alpha=0.7, label='Max gradient')

    # Mark vanishing gradient regions
    ax3.fill_between(differences, 0, gradients, where=(gradients < 0.05),
                     alpha=0.3, color='red', label='Vanishing gradient')

    ax3.set_title('Softmax Gradient = s(1-s)\nVanishes at Saturation', fontsize=11)
    ax3.set_xlabel('Input Difference (x_1 - x_2)', fontsize=10)
    ax3.set_ylabel('Gradient Magnitude', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.set_ylim(-0.02, 0.3)

    # ============== Panel 4: 3D View - Softmax and Gradient Together ==============
    ax4 = fig.add_subplot(gs[1, 0])

    # Show softmax probability distribution for different temperature/scale
    x_vals = np.array([0, 0.5, 1, 1.5, 2])
    temperatures = [0.5, 1.0, 2.0, 5.0]

    for i, temp in enumerate(temperatures):
        scaled = x_vals / temp
        probs = np.exp(scaled) / np.sum(np.exp(scaled))
        ax4.plot(range(len(x_vals)), probs, 'o-', linewidth=2, markersize=8,
                label=f'T={temp}', color=colors[i])

    ax4.set_title('Effect of Temperature/Scale on Softmax\nLower T = Sharper (more saturated)', fontsize=11)
    ax4.set_xlabel('Position', fontsize=10)
    ax4.set_ylabel('Probability', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.set_ylim(0, 1)

    # ============== Panel 5: Comparison Table (as visual) ==============
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    # Create a comparison table
    table_data = [
        ['Input Scale', 'Softmax Output', 'Max Gradient', 'Training Status'],
        ['[1, 2, 3]', '[0.09, 0.24, 0.67]', '0.22', 'HEALTHY'],
        ['[5, 10, 15]', '[0.00, 0.01, 0.99]', '0.01', 'WARNING'],
        ['[10, 20, 30]', '[0.00, 0.00, 1.00]', '~0', 'VANISHING'],
    ]

    cell_colors = [
        ['lightgray'] * 4,
        ['lightgreen'] * 4,
        ['lightyellow'] * 4,
        ['lightcoral'] * 4,
    ]

    table = ax5.table(cellText=table_data, cellColours=cell_colors,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    ax5.set_title('Softmax Saturation Summary\nLarge inputs -> Gradient collapse', fontsize=11, y=0.95)

    # ============== Panel 6: Training Implications ==============
    ax6 = fig.add_subplot(gs[1, 2])

    # Simulate training loss curves
    epochs = np.arange(100)

    # Good scaling (gradients flow)
    np.random.seed(42)
    loss_good = 2.0 * np.exp(-0.05 * epochs) + 0.1 + 0.02 * np.random.randn(100)
    loss_good = np.clip(loss_good, 0.1, 2.5)

    # Bad scaling (gradients vanish)
    loss_bad = 2.0 * np.exp(-0.005 * epochs) + 0.5 + 0.02 * np.random.randn(100)
    loss_bad = np.clip(loss_bad, 0.5, 2.5)

    ax6.plot(epochs, loss_good, 'g-', linewidth=2, label='With $\\sqrt{d_k}$ scaling', alpha=0.8)
    ax6.plot(epochs, loss_bad, 'r-', linewidth=2, label='Without scaling', alpha=0.8)
    ax6.set_title('Training Dynamics:\nScaling Enables Learning', fontsize=11)
    ax6.set_xlabel('Training Epoch', fontsize=10)
    ax6.set_ylabel('Loss', fontsize=10)
    ax6.legend(fontsize=9)
    ax6.set_ylim(0, 2.5)

    # Add annotation
    ax6.annotate('Gradients vanish\nLoss plateaus',
                xy=(80, loss_bad[80]), xytext=(60, 1.8),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='red'),
                color='red')

    plt.suptitle('Softmax Saturation: Why Scaling Matters for Training',
                 fontsize=14, fontweight='bold', y=1.01)

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/softmax_saturation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_softmax_saturation_viz()
