#!/usr/bin/env python3
"""
Generate visualizations for Backpropagation module.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = {
    'input': '#4ECDC4',
    'operation': '#FFE66D',
    'output': '#FF6B6B',
    'gradient': '#9B59B6',
    'forward': '#27AE60',
    'backward': '#E74C3C',
}


def create_computation_graph():
    """Create computation graph visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Nodes
    nodes = [
        (2, 5, 'x', COLORS['input']),
        (2, 8, 'W', COLORS['input']),
        (4, 6.5, '*', COLORS['operation']),
        (2, 2, 'b', COLORS['input']),
        (6, 5, '+', COLORS['operation']),
        (8, 5, 'σ', COLORS['operation']),
        (10, 5, 'h', COLORS['output']),
        (12, 5, 'L', COLORS['output']),
    ]

    for x, y, label, color in nodes:
        circle = Circle((x, y), 0.5, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold')

    # Edges (forward)
    edges = [
        ((2.5, 5), (3.5, 6.5)),     # x -> *
        ((2.5, 8), (3.5, 6.5)),     # W -> *
        ((4.5, 6.5), (5.5, 5)),     # * -> +
        ((2.5, 2), (5.5, 4.5)),     # b -> +
        ((6.5, 5), (7.5, 5)),       # + -> σ
        ((8.5, 5), (9.5, 5)),       # σ -> h
        ((10.5, 5), (11.5, 5)),     # h -> L
    ]

    for start, end in edges:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=2))

    # Labels
    ax.text(3, 7.3, 'z=Wx', fontsize=10, ha='center')
    ax.text(5, 4, 'z+b', fontsize=10, ha='center')
    ax.text(7.25, 5.5, 'σ(z)', fontsize=10, ha='center')

    ax.set_title('Computation Graph: Forward Pass', fontsize=14, fontweight='bold')

    # Legend
    ax.text(1, 1, 'Forward: Green arrows', color=COLORS['forward'], fontsize=10)

    plt.savefig(os.path.join(OUTPUT_DIR, 'computation_graph.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: computation_graph.svg")


def create_backprop_flow():
    """Create animation showing gradient flow backwards."""
    fig, ax = plt.subplots(figsize=(14, 6))

    layers = ['Loss', 'Output', 'σ', 'Sum', 'MatMul', 'Input']
    x_pos = [10, 8, 6, 4, 2, 0]

    frames = len(layers) * 8

    def animate(frame):
        ax.clear()
        ax.set_xlim(-1, 12)
        ax.set_ylim(0, 6)
        ax.axis('off')

        current_layer = min(frame // 8, len(layers) - 1)

        # Draw layers
        for i, (name, x) in enumerate(zip(layers, x_pos)):
            color = COLORS['gradient'] if i <= current_layer else 'lightgray'
            box = FancyBboxPatch((x - 0.5, 2), 1.5, 2,
                                boxstyle="round,pad=0.1",
                                facecolor=color if i <= current_layer else 'white',
                                edgecolor='black', linewidth=2)
            ax.add_patch(box)
            ax.text(x + 0.25, 3, name, ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   color='white' if i <= current_layer else 'black')

        # Draw backward arrows
        for i in range(len(x_pos) - 1):
            if i < current_layer:
                ax.annotate('', xy=(x_pos[i+1] + 1, 3), xytext=(x_pos[i] - 0.5, 3),
                           arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=3))

        # Gradient annotations
        gradients = ['∂L/∂L=1', '∂L/∂h', '∂L/∂z', '∂L/∂(Wx+b)', '∂L/∂W, ∂L/∂x']
        for i, grad in enumerate(gradients):
            if i < current_layer:
                ax.text(x_pos[i] - 1, 1, grad, ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        ax.set_title('Backpropagation: Gradients Flow Backward', fontsize=14, fontweight='bold')
        ax.text(5, 5.5, f'Computing gradient at: {layers[current_layer]}',
               ha='center', fontsize=12, fontweight='bold')

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)
    writer = PillowWriter(fps=5)
    anim.save(os.path.join(OUTPUT_DIR, 'backprop_flow.gif'), writer=writer)
    plt.close()
    print("Generated: backprop_flow.gif")


def create_chain_rule():
    """Create chain rule visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Function chain
    funcs = ['x', 'f(x)', 'g(f(x))', 'h(g(f(x)))=L']
    x_pos = [2, 5, 8, 11]

    for x, label in zip(x_pos, funcs):
        circle = Circle((x, 5), 0.8, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 5, label, ha='center', va='center', fontsize=11, fontweight='bold')

    # Forward arrows
    for i in range(len(x_pos) - 1):
        ax.annotate('', xy=(x_pos[i+1] - 0.8, 5), xytext=(x_pos[i] + 0.8, 5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=2))
        ax.text((x_pos[i] + x_pos[i+1]) / 2, 5.7, ['f', 'g', 'h'][i], fontsize=12, ha='center')

    # Backward arrows with derivatives
    y_back = 2.5
    for i in range(len(x_pos) - 1):
        ax.annotate('', xy=(x_pos[i] + 0.5, y_back), xytext=(x_pos[i+1] - 0.5, y_back),
                   arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=2))

    # Derivative labels
    ax.text(9.5, 2, "∂h/∂g", fontsize=10, ha='center')
    ax.text(6.5, 2, "∂g/∂f", fontsize=10, ha='center')
    ax.text(3.5, 2, "∂f/∂x", fontsize=10, ha='center')

    # Chain rule formula
    ax.text(7, 0.8, r'$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial g} \cdot \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial x}$',
           ha='center', fontsize=14,
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))

    ax.set_title('The Chain Rule: Composing Derivatives', fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, 'chain_rule.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: chain_rule.svg")


def create_forward_vs_reverse():
    """Create forward vs reverse mode autodiff comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, mode, direction, color in [(axes[0], 'Forward Mode', '→', COLORS['forward']),
                                       (axes[1], 'Reverse Mode', '←', COLORS['backward'])]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_title(mode, fontsize=13, fontweight='bold')

        # Draw network
        layers = [(2, [6, 4, 2]), (5, [5, 3]), (8, [4])]
        for x, ys in layers:
            for y in ys:
                circle = Circle((x, y), 0.4, facecolor='lightblue', edgecolor='black', linewidth=2)
                ax.add_patch(circle)

        # Draw connections
        for i in range(len(layers) - 1):
            x1, y1s = layers[i]
            x2, y2s = layers[i + 1]
            for y1 in y1s:
                for y2 in y2s:
                    ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.3, linewidth=1)

        # Draw direction arrows
        if direction == '→':
            ax.annotate('', xy=(9, 4), xytext=(1, 4),
                       arrowprops=dict(arrowstyle='->', color=color, lw=4, alpha=0.7))
            ax.text(5, 1, 'Compute ∂y/∂x for\none input, all outputs\nO(n) for n inputs',
                   ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            ax.annotate('', xy=(1, 4), xytext=(9, 4),
                       arrowprops=dict(arrowstyle='->', color=color, lw=4, alpha=0.7))
            ax.text(5, 1, 'Compute ∂L/∂x for\none output, all inputs\nO(1) passes!',
                   ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    fig.suptitle('Why Backpropagation Uses Reverse Mode', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'forward_vs_reverse.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: forward_vs_reverse.svg")


if __name__ == '__main__':
    print("Generating Backpropagation visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_computation_graph()
    create_backprop_flow()
    create_chain_rule()
    create_forward_vs_reverse()

    print("\nAll backprop visualizations generated!")
