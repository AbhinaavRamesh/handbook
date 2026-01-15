#!/usr/bin/env python3
"""
Generate visualizations for RNN, LSTM, GRU module.
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
    'gate': '#F39C12',
    'cell': '#9B59B6',
}


def create_rnn_unrolled():
    """Show RNN unrolled through time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    timesteps = 4

    for t in range(timesteps):
        x = 2 + t * 3.5

        # Input
        circle = Circle((x, 1), 0.4, facecolor=COLORS['input'], edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 1, f'$x_{t+1}$', ha='center', va='center', fontsize=11, fontweight='bold')

        # Hidden state
        box = FancyBboxPatch((x - 0.6, 3.5), 1.2, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=COLORS['hidden'], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, 4.25, f'$h_{t+1}$', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # Output
        circle = Circle((x, 7), 0.4, facecolor=COLORS['output'], edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 7, f'$y_{t+1}$', ha='center', va='center', fontsize=11, fontweight='bold')

        # Input to hidden
        ax.annotate('', xy=(x, 3.5), xytext=(x, 1.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Hidden to output
        ax.annotate('', xy=(x, 6.5), xytext=(x, 5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Hidden to next hidden
        if t < timesteps - 1:
            ax.annotate('', xy=(x + 2.9, 4.25), xytext=(x + 0.6, 4.25),
                       arrowprops=dict(arrowstyle='->', color=COLORS['hidden'], lw=2.5))

    # Labels
    ax.text(0.5, 1, 'Input\nSequence', ha='center', va='center', fontsize=10)
    ax.text(0.5, 4.25, 'Hidden\nState', ha='center', va='center', fontsize=10)
    ax.text(0.5, 7, 'Output\nSequence', ha='center', va='center', fontsize=10)

    # Equation
    ax.text(8, 0.3, '$h_t = \\tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.set_title('RNN Unrolled Through Time: Parameter Sharing Across Timesteps', fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, 'rnn_unrolled.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: rnn_unrolled.svg")


def create_lstm_cell():
    """Show LSTM cell architecture."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Cell state line
    ax.plot([2, 12], [8, 8], 'purple', linewidth=4, label='Cell State')

    # Gates
    gates = [
        (4, 5, 'Forget\nGate\n$f_t$', '#E74C3C'),
        (6.5, 5, 'Input\nGate\n$i_t$', '#27AE60'),
        (9, 5, 'Output\nGate\n$o_t$', '#3498DB'),
    ]

    for x, y, label, color in gates:
        box = FancyBboxPatch((x - 0.7, y - 1), 1.4, 2,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Candidate
    box = FancyBboxPatch((6.5 - 0.7, 2.5), 1.4, 1.5,
                        boxstyle="round,pad=0.1",
                        facecolor='#F39C12', edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(6.5, 3.25, '$\\tilde{C}_t$', ha='center', va='center', fontsize=11, fontweight='bold')

    # Operations
    # Forget gate multiply
    ax.text(4, 7.5, '×', fontsize=20, ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(4, 7.8), xytext=(4, 6),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Input gate multiply and add
    ax.text(6.5, 7.5, '+', fontsize=20, ha='center', va='center', fontweight='bold')
    ax.text(6.5, 6.5, '×', fontsize=16, ha='center', va='center')
    ax.annotate('', xy=(6.5, 7.2), xytext=(6.5, 6),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(6.5, 6.5), xytext=(6.5, 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Output gate and tanh
    ax.text(9, 6.5, 'tanh', fontsize=10, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='white'))
    ax.annotate('', xy=(9, 7.5), xytext=(9, 8),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(10.5, 7, '×', fontsize=16, ha='center', va='center')
    ax.annotate('', xy=(10.5, 7.3), xytext=(9, 6),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Inputs
    ax.text(2, 1, '$[h_{t-1}, x_t]$', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightblue'))

    # Outputs
    ax.text(12, 8, '$C_t$', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='plum'))
    ax.text(12, 5.5, '$h_t$', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightblue'))

    # Legend
    ax.text(1, 9.5, 'Cell State (gradient highway)', fontsize=10, color='purple', fontweight='bold')

    ax.set_title('LSTM Cell: Gates Control Information Flow', fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, 'lstm_cell.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: lstm_cell.svg")


def create_gru_cell():
    """Show GRU cell architecture."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Gates
    gates = [
        (3, 4, 'Reset\nGate\n$r_t$', '#E74C3C'),
        (6, 4, 'Update\nGate\n$z_t$', '#3498DB'),
    ]

    for x, y, label, color in gates:
        box = FancyBboxPatch((x - 0.7, y - 1), 1.4, 2,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Candidate
    box = FancyBboxPatch((8.5, 3), 1.8, 2,
                        boxstyle="round,pad=0.1",
                        facecolor='#27AE60', edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(9.4, 4, 'Candidate\n$\\tilde{h}_t$', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Input and output
    ax.text(1, 2, '$[h_{t-1}, x_t]$', fontsize=11, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(11, 7, '$h_t$', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightblue'))

    # Formula
    ax.text(6, 1, '$h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    # Comparison with LSTM
    ax.text(6, 7.5, 'GRU: 2 gates vs LSTM\'s 3 gates\nFewer parameters, often similar performance',
           fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.set_title('GRU Cell: Simplified Gating Mechanism', fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, 'gru_cell.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: gru_cell.svg")


def create_rnn_gradient_vanishing():
    """Animate gradient vanishing through time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    timesteps = 8
    frames = timesteps * 5

    def animate(frame):
        ax.clear()
        current_t = min(frame // 5, timesteps - 1)

        ax.set_xlim(0, 16)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Draw timesteps
        for t in range(timesteps):
            x = 1.5 + t * 1.7

            # Gradient magnitude (vanishes going backward)
            if t <= timesteps - 1 - current_t:
                grad_magnitude = 0.9 ** (timesteps - 1 - t - (timesteps - 1 - current_t))
            else:
                grad_magnitude = 0

            color = plt.cm.Reds(grad_magnitude) if grad_magnitude > 0 else 'lightgray'

            # Hidden state box
            box = FancyBboxPatch((x - 0.5, 3), 1, 1.5,
                                boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x, 3.75, f'$h_{t+1}$', ha='center', va='center', fontsize=9)

            # Gradient bar
            bar_height = grad_magnitude * 2
            rect = Rectangle((x - 0.3, 5.5), 0.6, bar_height, facecolor='purple', alpha=0.8)
            ax.add_patch(rect)

            # Arrow between timesteps (backward for gradient)
            if t > 0 and t >= timesteps - 1 - current_t:
                ax.annotate('', xy=(x - 1.2, 3.75), xytext=(x - 0.5, 3.75),
                           arrowprops=dict(arrowstyle='->', color='purple', lw=2))

        # Labels
        ax.text(8, 1.5, f'Gradient flowing backward from t={timesteps} to t={timesteps - current_t}',
               ha='center', fontsize=11)
        ax.text(8, 0.5, f'Gradient magnitude: 0.9^{current_t} = {0.9**current_t:.4f}',
               ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow'))

        ax.set_title('Vanishing Gradients in RNNs: Information Gets Lost Over Time', fontsize=13, fontweight='bold')

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=400, blit=False)
    writer = PillowWriter(fps=3)
    anim.save(os.path.join(OUTPUT_DIR, 'rnn_gradient_vanishing.gif'), writer=writer)
    plt.close()
    print("Generated: rnn_gradient_vanishing.gif")


if __name__ == '__main__':
    print("Generating RNN visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_rnn_unrolled()
    create_lstm_cell()
    create_gru_cell()
    create_rnn_gradient_vanishing()

    print("\nAll RNN visualizations generated!")
