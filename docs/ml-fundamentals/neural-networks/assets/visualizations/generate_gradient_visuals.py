#!/usr/bin/env python3
"""
Generate visualizations for Gradient Problems module.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = {
    'input': '#4ECDC4',
    'hidden': '#45B7D1',
    'output': '#FF6B6B',
    'gradient': '#9B59B6',
    'forward': '#27AE60',
    'backward': '#E74C3C',
    'skip': '#F39C12',
}


def create_vanishing_gradient_animation():
    """
    Animate vanishing gradients through deep network layers.
    Shows how gradients shrink exponentially as they flow backward.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    num_layers = 8
    frames = num_layers * 6

    def animate(frame):
        ax.clear()
        current_layer = min(frame // 6, num_layers - 1)

        ax.set_xlim(-1, 15)
        ax.set_ylim(0, 9)
        ax.axis('off')

        # Draw layers from left (input) to right (output)
        layer_x = [1.5 + i * 1.6 for i in range(num_layers)]

        for i, x in enumerate(layer_x):
            # Calculate gradient magnitude (vanishes going backward from output)
            # Layer index from output: num_layers - 1 - i
            layer_from_output = num_layers - 1 - i

            # Gradient has reached this layer?
            if layer_from_output <= current_layer:
                # Gradient shrinks by 0.6 at each layer
                grad_magnitude = 0.6 ** (current_layer - layer_from_output)
            else:
                grad_magnitude = 0

            # Color intensity based on gradient magnitude
            if grad_magnitude > 0:
                color = plt.cm.Reds(0.3 + 0.7 * grad_magnitude)
            else:
                color = 'lightgray'

            # Draw layer box
            box = FancyBboxPatch((x - 0.5, 3), 1, 2,
                                boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(box)

            # Layer label
            if i == 0:
                label = 'Input'
            elif i == num_layers - 1:
                label = 'Output'
            else:
                label = f'L{i}'
            ax.text(x, 4, label, ha='center', va='center', fontsize=10, fontweight='bold')

            # Draw gradient magnitude bar
            if grad_magnitude > 0:
                bar_height = grad_magnitude * 2.5
                rect = Rectangle((x - 0.3, 6), 0.6, bar_height,
                                facecolor=COLORS['gradient'], alpha=0.8, edgecolor='black')
                ax.add_patch(rect)
                ax.text(x, 6 + bar_height + 0.2, f'{grad_magnitude:.2f}',
                       ha='center', fontsize=8)

            # Draw backward gradient arrows
            if i > 0 and layer_from_output < current_layer:
                ax.annotate('', xy=(x - 1.1, 4), xytext=(x - 0.5, 4),
                           arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=2.5))

        # Title and info
        ax.set_title('Vanishing Gradients: Gradients Shrink Exponentially Through Layers',
                    fontsize=13, fontweight='bold')

        # Info box showing current computation
        layers_traversed = current_layer
        current_grad = 0.6 ** current_layer if current_layer > 0 else 1.0
        ax.text(7.5, 1.5, f'Gradient flowing backward from Output to Layer {num_layers - 1 - current_layer}',
               ha='center', fontsize=11)
        ax.text(7.5, 0.7, f'Gradient magnitude: 0.6^{layers_traversed} = {current_grad:.4f}',
               ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='orange'))

        # Gradient magnitude legend
        ax.text(14, 7, 'Gradient\nMagnitude', ha='center', fontsize=9)

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=350, blit=False)
    writer = PillowWriter(fps=3)
    anim.save(os.path.join(OUTPUT_DIR, 'grad_vanishing.gif'), writer=writer)
    plt.close()
    print("Generated: grad_vanishing.gif")


def create_skip_connection_gradient():
    """
    Create visualization showing how skip connections help gradient flow.
    Shows y = F(x) + x and gradient = F'(x) + 1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: Without skip connection
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Without Skip Connection', fontsize=13, fontweight='bold')

    # Draw layers
    layers_no_skip = [(2, 5, 'x'), (5, 5, 'F(x)'), (8, 5, 'y')]
    for x, y, label in layers_no_skip:
        circle = Circle((x, y), 0.7, facecolor=COLORS['hidden'], edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Forward arrows
    ax.annotate('', xy=(4.3, 5), xytext=(2.7, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=3))
    ax.annotate('', xy=(7.3, 5), xytext=(5.7, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=3))

    # Backward gradient arrows (below)
    ax.annotate('', xy=(2.7, 3.5), xytext=(4.3, 3.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=2))
    ax.annotate('', xy=(5.7, 3.5), xytext=(7.3, 3.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=2))

    # Labels
    ax.text(3.5, 5.8, 'F', fontsize=11)
    ax.text(3.5, 3, "F'(x)", fontsize=10, ha='center')
    ax.text(6.5, 3, '1', fontsize=10, ha='center')

    # Gradient formula
    ax.text(5, 1.5, r"$\frac{\partial y}{\partial x} = F'(x)$", fontsize=14, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.text(5, 0.5, "If F'(x) small, gradient vanishes!", fontsize=10, ha='center', color='red')

    # Gradient magnitude bar (small)
    rect = Rectangle((8.5, 2), 0.4, 0.5, facecolor=COLORS['gradient'], alpha=0.8, edgecolor='black')
    ax.add_patch(rect)
    ax.text(8.7, 2.8, 'Small\ngradient', fontsize=8, ha='center')

    # Right: With skip connection
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('With Skip Connection (ResNet)', fontsize=13, fontweight='bold')

    # Draw layers
    layers_skip = [(2, 5, 'x'), (5, 5, 'F(x)'), (8, 5, 'y')]
    for x, y, label in layers_skip:
        circle = Circle((x, y), 0.7, facecolor=COLORS['hidden'], edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Forward arrows (main path)
    ax.annotate('', xy=(4.3, 5), xytext=(2.7, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=3))
    ax.annotate('', xy=(7.3, 5), xytext=(5.7, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=3))

    # Skip connection (curved arrow above)
    ax.annotate('', xy=(7.3, 5.8), xytext=(2.7, 5.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['skip'], lw=3,
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(5, 8, 'Skip Connection (+x)', fontsize=10, ha='center',
           color=COLORS['skip'], fontweight='bold')

    # Plus symbol at output
    ax.text(7.3, 6.5, '+', fontsize=16, ha='center', fontweight='bold', color=COLORS['skip'])

    # Backward gradient arrows
    # Main path gradient
    ax.annotate('', xy=(2.7, 3.5), xytext=(4.3, 3.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=2))
    ax.annotate('', xy=(5.7, 3.5), xytext=(7.3, 3.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=2))

    # Skip connection gradient (curved below)
    ax.annotate('', xy=(2.7, 2.5), xytext=(7.3, 2.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['skip'], lw=2.5,
                              connectionstyle='arc3,rad=0.3'))

    # Labels
    ax.text(3.5, 5.8, 'F', fontsize=11)
    ax.text(3.5, 3, "F'(x)", fontsize=10, ha='center')
    ax.text(6.5, 3, '1', fontsize=10, ha='center')
    ax.text(5, 2, '+1 (identity)', fontsize=10, ha='center', color=COLORS['skip'], fontweight='bold')

    # Gradient formula
    ax.text(5, 1, r"$\frac{\partial y}{\partial x} = F'(x) + 1$", fontsize=14, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(5, 0.2, "Even if F'(x)=0, gradient = 1!", fontsize=10, ha='center', color='green', fontweight='bold')

    # Gradient magnitude bar (large due to +1)
    rect = Rectangle((8.5, 2), 0.4, 2, facecolor=COLORS['gradient'], alpha=0.8, edgecolor='black')
    ax.add_patch(rect)
    ax.text(8.7, 4.3, 'Strong\ngradient', fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'grad_skip.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: grad_skip.svg")


if __name__ == '__main__':
    print("Generating Gradient Problems visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_vanishing_gradient_animation()
    create_skip_connection_gradient()

    print("\nAll gradient visualizations generated!")
