#!/usr/bin/env python3
"""
Generate visualizations for Weight Initialization module.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def relu(x):
    return np.maximum(0, x)


def create_init_bad():
    """Show effects of bad initialization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    np.random.seed(42)
    n_samples = 1000
    n_layers = 5
    layer_size = 100

    init_configs = [
        ('All Zeros', lambda n_in, n_out: np.zeros((n_in, n_out))),
        ('Too Small (std=0.001)', lambda n_in, n_out: np.random.randn(n_in, n_out) * 0.001),
        ('Too Large (std=2.0)', lambda n_in, n_out: np.random.randn(n_in, n_out) * 2.0),
        ('He Init (correct)', lambda n_in, n_out: np.random.randn(n_in, n_out) * np.sqrt(2/n_in)),
    ]

    for ax, (name, init_fn) in zip(axes.flatten(), init_configs):
        X = np.random.randn(n_samples, layer_size)
        activations = [X]

        for l in range(n_layers):
            W = init_fn(layer_size, layer_size)
            X = relu(X @ W)
            activations.append(X)

        # Plot activation distributions
        for i, act in enumerate(activations[1:]):
            mean = np.mean(act)
            std = np.std(act)
            ax.bar(i, std, alpha=0.7, label=f'Layer {i+1}' if i < 3 else '')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Activation Std')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f'L{i+1}' for i in range(n_layers)])

        # Add annotation
        final_std = np.std(activations[-1])
        if final_std < 1e-6:
            ax.text(2, ax.get_ylim()[1] * 0.8, 'Vanished!', color='red',
                   fontsize=12, fontweight='bold', ha='center')
        elif final_std > 100:
            ax.text(2, ax.get_ylim()[1] * 0.8, 'Exploded!', color='red',
                   fontsize=12, fontweight='bold', ha='center')

    fig.suptitle('Effect of Weight Initialization on Activation Variance', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'init_bad.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: init_bad.svg")


def create_init_variance():
    """Animate variance through layers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    np.random.seed(42)
    n_samples = 1000
    layer_size = 100
    n_layers = 10

    def simulate_layers(variance_factor):
        X = np.random.randn(n_samples, layer_size)
        stds = [np.std(X)]
        for _ in range(n_layers):
            W = np.random.randn(layer_size, layer_size) * np.sqrt(variance_factor / layer_size)
            X = relu(X @ W)
            stds.append(np.std(X))
        return stds

    frames = 50

    def animate(frame):
        for ax in axes:
            ax.clear()

        # Vary the variance factor
        var_factors = np.linspace(0.5, 3.0, frames)
        var_factor = var_factors[frame]

        stds = simulate_layers(var_factor)

        # Left plot: Activation std per layer
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(stds)))
        bars = axes[0].bar(range(len(stds)), stds, color=colors)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Activation Std')
        axes[0].set_title(f'Variance Factor = {var_factor:.2f}', fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, max(5, max(stds) * 1.1))
        axes[0].axhline(y=1, color='green', linestyle='--', label='Ideal (std=1)')
        axes[0].legend()

        # Right plot: Variance factor vs final std
        test_factors = np.linspace(0.5, 3.0, 20)
        final_stds = []
        for vf in test_factors:
            s = simulate_layers(vf)
            final_stds.append(s[-1])

        axes[1].plot(test_factors, final_stds, 'b-', linewidth=2)
        axes[1].axvline(x=var_factor, color='red', linestyle='--', linewidth=2)
        axes[1].axvline(x=2.0, color='green', linestyle=':', label='He init (factor=2)')
        axes[1].scatter([var_factor], [stds[-1]], color='red', s=100, zorder=5)
        axes[1].set_xlabel('Variance Factor (k in Var=k/n)')
        axes[1].set_ylabel('Final Layer Std')
        axes[1].set_title('Finding the Right Initialization', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].set_ylim(0, 5)

        fig.suptitle('Weight Initialization: Preserving Variance Through Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)
    writer = PillowWriter(fps=10)
    anim.save(os.path.join(OUTPUT_DIR, 'init_variance.gif'), writer=writer)
    plt.close()
    print("Generated: init_variance.gif")


def create_init_he_xavier():
    """Compare He vs Xavier initialization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    np.random.seed(42)
    n_samples = 1000
    layer_size = 256
    n_layers = 10

    def simulate(init_name, activation):
        X = np.random.randn(n_samples, layer_size)
        stds = []

        for l in range(n_layers):
            if init_name == 'Xavier':
                W = np.random.randn(layer_size, layer_size) * np.sqrt(2 / (layer_size + layer_size))
            else:  # He
                W = np.random.randn(layer_size, layer_size) * np.sqrt(2 / layer_size)

            X = X @ W
            if activation == 'relu':
                X = np.maximum(0, X)
            else:  # tanh
                X = np.tanh(X)

            stds.append(np.std(X))

        return stds

    # ReLU comparison
    ax = axes[0]
    stds_xavier = simulate('Xavier', 'relu')
    stds_he = simulate('He', 'relu')
    ax.plot(range(1, n_layers + 1), stds_xavier, 'b-o', linewidth=2, label='Xavier', markersize=8)
    ax.plot(range(1, n_layers + 1), stds_he, 'g-s', linewidth=2, label='He', markersize=8)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation Std')
    ax.set_title('ReLU Activation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 2)
    ax.text(5, 0.3, 'Xavier vanishes\nfor ReLU!', color='blue', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Tanh comparison
    ax = axes[1]
    stds_xavier = simulate('Xavier', 'tanh')
    stds_he = simulate('He', 'tanh')
    ax.plot(range(1, n_layers + 1), stds_xavier, 'b-o', linewidth=2, label='Xavier', markersize=8)
    ax.plot(range(1, n_layers + 1), stds_he, 'g-s', linewidth=2, label='He', markersize=8)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation Std')
    ax.set_title('Tanh Activation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 2)
    ax.text(5, 1.5, 'Both work\nfor Tanh', color='green', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    fig.suptitle('He vs Xavier: Match Initialization to Activation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'init_he_xavier.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: init_he_xavier.svg")


def create_init_activations():
    """Show activation distributions through layers."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    np.random.seed(42)
    n_samples = 5000
    layer_size = 256

    for row, (init_name, init_fn) in enumerate([
        ('Bad Init (std=0.01)', lambda n: np.random.randn(n, n) * 0.01),
        ('He Init', lambda n: np.random.randn(n, n) * np.sqrt(2/n))
    ]):
        X = np.random.randn(n_samples, layer_size)

        for col in range(5):
            ax = axes[row, col]
            W = init_fn(layer_size)
            X = np.maximum(0, X @ W)  # ReLU

            # Histogram of activations
            ax.hist(X.flatten(), bins=50, density=True, alpha=0.7,
                   color='blue' if row == 0 else 'green')
            ax.set_xlim(-0.5, 3)
            ax.set_ylim(0, 5)
            ax.set_title(f'Layer {col + 1}', fontsize=10)

            if col == 0:
                ax.set_ylabel(init_name, fontsize=10, fontweight='bold')

            # Stats
            mean, std = np.mean(X), np.std(X)
            ax.text(0.95, 0.95, f'μ={mean:.2f}\nσ={std:.2f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Activation Distributions: Bad vs Good Initialization', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'init_activations.gif'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: init_activations.gif")


if __name__ == '__main__':
    print("Generating Weight Initialization visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_init_bad()
    create_init_variance()
    create_init_he_xavier()
    create_init_activations()

    print("\nAll initialization visualizations generated!")
