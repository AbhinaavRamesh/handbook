"""
Visualization script for Vanishing Gradients FAQ
Generates matplotlib figures to illustrate gradient flow concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Set consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Output directory (same as script location)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_figure(fig, name):
    """Save figure with consistent settings."""
    filepath = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_gradient_magnitude_decay():
    """
    Figure 1: Gradient magnitude through layers (bar chart showing decay)
    Shows exponential decay of gradients in deep networks.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Vanishing gradients scenario
    num_layers = 10
    gradient_factor = 0.25  # Sigmoid max derivative
    layers = list(range(1, num_layers + 1))

    # Calculate gradient magnitudes (from output layer going backward)
    vanishing_gradients = [gradient_factor ** i for i in range(num_layers)]
    vanishing_gradients = vanishing_gradients[::-1]  # Reverse for display (layer 1 to 10)

    colors_vanish = plt.cm.Reds(np.linspace(0.3, 0.9, num_layers))
    axes[0].bar(layers, vanishing_gradients, color=colors_vanish, edgecolor='darkred', linewidth=1.2)
    axes[0].set_xlabel('Layer (from input to output)')
    axes[0].set_ylabel('Relative Gradient Magnitude')
    axes[0].set_title('Vanishing Gradients\n(Sigmoid: max derivative = 0.25)')
    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-7, 2)
    axes[0].axhline(y=1e-6, color='red', linestyle='--', linewidth=2, label='Effectively zero')
    axes[0].legend(loc='upper right')

    # Add annotation
    axes[0].annotate('Early layers\nreceive ~0 gradient',
                     xy=(2, vanishing_gradients[1]),
                     xytext=(4, 1e-3),
                     arrowprops=dict(arrowstyle='->', color='black'),
                     fontsize=10)

    # Exploding gradients scenario
    gradient_factor_explode = 1.5
    exploding_gradients = [gradient_factor_explode ** i for i in range(num_layers)]
    exploding_gradients = exploding_gradients[::-1]

    colors_explode = plt.cm.Blues(np.linspace(0.3, 0.9, num_layers))
    axes[1].bar(layers, exploding_gradients, color=colors_explode, edgecolor='darkblue', linewidth=1.2)
    axes[1].set_xlabel('Layer (from input to output)')
    axes[1].set_ylabel('Relative Gradient Magnitude')
    axes[1].set_title('Exploding Gradients\n(Weight eigenvalues > 1)')
    axes[1].set_yscale('log')
    axes[1].axhline(y=1000, color='blue', linestyle='--', linewidth=2, label='Numerical instability')
    axes[1].legend(loc='upper right')

    # Add annotation
    axes[1].annotate('Gradients\nexplode',
                     xy=(2, exploding_gradients[1]),
                     xytext=(5, 10),
                     arrowprops=dict(arrowstyle='->', color='black'),
                     fontsize=10)

    fig.suptitle('Gradient Flow Through Deep Networks', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'gradient_magnitude_decay')


def plot_activation_functions_with_derivatives():
    """
    Figure 2: Activation functions with their derivatives (2-row subplot)
    Shows why some activations cause vanishing gradients.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    x = np.linspace(-5, 5, 500)

    # Define activation functions and derivatives
    activations = {
        'Sigmoid': {
            'f': lambda x: 1 / (1 + np.exp(-x)),
            'df': lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
            'color': '#e74c3c'
        },
        'Tanh': {
            'f': lambda x: np.tanh(x),
            'df': lambda x: 1 - np.tanh(x)**2,
            'color': '#3498db'
        },
        'ReLU': {
            'f': lambda x: np.maximum(0, x),
            'df': lambda x: np.where(x > 0, 1, 0),
            'color': '#2ecc71'
        },
        'Leaky ReLU': {
            'f': lambda x: np.where(x > 0, x, 0.1 * x),
            'df': lambda x: np.where(x > 0, 1, 0.1),
            'color': '#9b59b6'
        }
    }

    for idx, (name, funcs) in enumerate(activations.items()):
        # Top row: activation functions
        y = funcs['f'](x)
        axes[0, idx].plot(x, y, color=funcs['color'], linewidth=2.5)
        axes[0, idx].axhline(y=0, color='gray', linewidth=0.5)
        axes[0, idx].axvline(x=0, color='gray', linewidth=0.5)
        axes[0, idx].set_title(f'{name}', fontweight='bold')
        axes[0, idx].set_xlabel('x')
        axes[0, idx].set_ylabel('f(x)')
        axes[0, idx].set_ylim(-1.5, 1.5) if name in ['Sigmoid', 'Tanh'] else axes[0, idx].set_ylim(-1, 3)

        # Bottom row: derivatives
        dy = funcs['df'](x)
        axes[1, idx].plot(x, dy, color=funcs['color'], linewidth=2.5)
        axes[1, idx].axhline(y=0, color='gray', linewidth=0.5)
        axes[1, idx].axvline(x=0, color='gray', linewidth=0.5)
        axes[1, idx].set_xlabel('x')
        axes[1, idx].set_ylabel("f'(x)")

        # Highlight saturation regions for sigmoid and tanh
        if name == 'Sigmoid':
            axes[1, idx].axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Max = 0.25')
            axes[1, idx].fill_between(x, 0, dy, where=(np.abs(x) > 2), alpha=0.3, color='red', label='Saturation')
            axes[1, idx].legend(fontsize=8)
            axes[1, idx].set_ylim(-0.05, 0.35)
        elif name == 'Tanh':
            axes[1, idx].axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='Max = 1.0')
            axes[1, idx].fill_between(x, 0, dy, where=(np.abs(x) > 2), alpha=0.3, color='blue', label='Saturation')
            axes[1, idx].legend(fontsize=8)
            axes[1, idx].set_ylim(-0.1, 1.2)
        elif name == 'ReLU':
            axes[1, idx].fill_between(x, 0, dy, where=(x < 0), alpha=0.3, color='red', label='Dead region')
            axes[1, idx].legend(fontsize=8)
            axes[1, idx].set_ylim(-0.2, 1.3)
        else:
            axes[1, idx].axhline(y=0.1, color='purple', linestyle='--', alpha=0.7, label='Min = 0.1')
            axes[1, idx].legend(fontsize=8)
            axes[1, idx].set_ylim(-0.1, 1.3)

    axes[0, 0].set_ylabel('Activation f(x)', fontweight='bold')
    axes[1, 0].set_ylabel("Derivative f'(x)", fontweight='bold')

    fig.suptitle('Activation Functions and Their Derivatives', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'activation_functions_derivatives')


def plot_sigmoid_vs_relu_gradient_flow():
    """
    Figure 3: Sigmoid vs ReLU gradient flow comparison
    Shows how gradients decay differently through layers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    num_layers = 15
    layers = list(range(1, num_layers + 1))

    # Simulate gradient flow
    # Sigmoid: derivative is 0.25 at best, often less
    sigmoid_gradient = np.cumprod([0.25] * num_layers)[::-1]

    # ReLU: derivative is 1 for positive, 0 for negative
    # Assume 50% neurons are active on average
    relu_gradient = np.cumprod([0.5] * num_layers)[::-1]  # Average case
    relu_best = np.ones(num_layers)  # Best case (all active)

    # Plot sigmoid gradient flow
    axes[0].semilogy(layers, sigmoid_gradient, 'r-o', linewidth=2, markersize=8, label='Sigmoid')
    axes[0].fill_between(layers, 1e-15, sigmoid_gradient, alpha=0.3, color='red')
    axes[0].axhline(y=1e-6, color='gray', linestyle='--', label='Vanishing threshold')
    axes[0].set_xlabel('Layer (from output to input)')
    axes[0].set_ylabel('Gradient Magnitude (log scale)')
    axes[0].set_title('Sigmoid: Severe Vanishing', fontweight='bold', color='red')
    axes[0].legend()
    axes[0].set_ylim(1e-12, 10)
    axes[0].grid(True, alpha=0.3)

    # Add text annotation
    vanish_layer = np.argmax(sigmoid_gradient < 1e-6) + 1
    axes[0].annotate(f'Vanishes by layer {vanish_layer}',
                     xy=(vanish_layer, 1e-6),
                     xytext=(vanish_layer + 2, 1e-4),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=10, color='red')

    # Plot ReLU gradient flow
    axes[1].semilogy(layers, relu_best, 'g-o', linewidth=2, markersize=8, label='ReLU (best: all active)')
    axes[1].semilogy(layers, relu_gradient, 'b-s', linewidth=2, markersize=8, label='ReLU (avg: 50% active)')
    axes[1].fill_between(layers, relu_gradient, relu_best, alpha=0.2, color='green')
    axes[1].axhline(y=1e-6, color='gray', linestyle='--', label='Vanishing threshold')
    axes[1].set_xlabel('Layer (from output to input)')
    axes[1].set_ylabel('Gradient Magnitude (log scale)')
    axes[1].set_title('ReLU: Much Better Flow', fontweight='bold', color='green')
    axes[1].legend()
    axes[1].set_ylim(1e-12, 10)
    axes[1].grid(True, alpha=0.3)

    # Add text annotation
    axes[1].annotate('Gradient preserved\nin best case',
                     xy=(8, 1),
                     xytext=(10, 0.1),
                     arrowprops=dict(arrowstyle='->', color='green'),
                     fontsize=10, color='green')

    fig.suptitle('Gradient Flow: Sigmoid vs ReLU Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'sigmoid_vs_relu_gradient_flow')


def plot_weight_initialization_effect():
    """
    Figure 4: Weight initialization effect on activations per layer
    Shows how different initialization strategies affect activation distributions.
    """
    np.random.seed(42)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    num_layers = 6
    layer_size = 500
    input_size = 500

    # Simulate forward pass through layers
    def simulate_forward(init_type, activation='relu'):
        x = np.random.randn(1000, input_size)  # Batch of 1000 samples
        activations_per_layer = [x.std()]

        for layer in range(num_layers):
            n_in, n_out = layer_size, layer_size

            if init_type == 'random':
                W = np.random.randn(n_in, n_out) * 1.0
            elif init_type == 'xavier':
                limit = np.sqrt(6 / (n_in + n_out))
                W = np.random.uniform(-limit, limit, (n_in, n_out))
            elif init_type == 'he':
                W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

            x = x @ W

            if activation == 'relu':
                x = np.maximum(0, x)
            elif activation == 'tanh':
                x = np.tanh(x)

            activations_per_layer.append(x.std())

        return activations_per_layer

    init_types = ['random', 'xavier', 'he']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    titles = ['Random (std=1.0)', 'Xavier/Glorot', 'He/Kaiming']

    # Top row: with tanh activation
    for idx, (init_type, color, title) in enumerate(zip(init_types, colors, titles)):
        acts = simulate_forward(init_type, 'tanh')
        axes[0, idx].bar(range(len(acts)), acts, color=color, alpha=0.7, edgecolor='black')
        axes[0, idx].axhline(y=1.0, color='gray', linestyle='--', label='Ideal std=1')
        axes[0, idx].set_xlabel('Layer')
        axes[0, idx].set_ylabel('Activation Std Dev')
        axes[0, idx].set_title(f'{title}\n(tanh activation)', fontweight='bold')
        axes[0, idx].legend(fontsize=8)
        axes[0, idx].set_ylim(0, max(acts) * 1.2 if max(acts) > 1.5 else 2)

    # Bottom row: with ReLU activation
    for idx, (init_type, color, title) in enumerate(zip(init_types, colors, titles)):
        acts = simulate_forward(init_type, 'relu')
        axes[1, idx].bar(range(len(acts)), acts, color=color, alpha=0.7, edgecolor='black')
        axes[1, idx].axhline(y=1.0, color='gray', linestyle='--', label='Ideal std=1')
        axes[1, idx].set_xlabel('Layer')
        axes[1, idx].set_ylabel('Activation Std Dev')
        axes[1, idx].set_title(f'{title}\n(ReLU activation)', fontweight='bold')
        axes[1, idx].legend(fontsize=8)
        axes[1, idx].set_ylim(0, max(acts) * 1.2 if max(acts) > 1.5 else 2)

    # Add recommendations
    axes[0, 1].annotate('Best for tanh', xy=(3, 0.8), fontsize=12, color='blue', fontweight='bold')
    axes[1, 2].annotate('Best for ReLU', xy=(3, 0.8), fontsize=12, color='green', fontweight='bold')

    fig.suptitle('Weight Initialization Effect on Activation Variance', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'weight_initialization_effect')


def plot_residual_connection_gradient_path():
    """
    Figure 5: Residual connection gradient path diagram
    Illustrates how skip connections create gradient highways.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # Left: Without residual connection
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Without Skip Connection\n(Gradient must flow through F)', fontweight='bold', fontsize=13, color='red')

    # Draw layers
    layer_positions = [8, 6, 4, 2]
    layer_labels = ['Output', 'Layer n', 'Layer n-1', 'Input']

    for pos, label in zip(layer_positions, layer_labels):
        rect = FancyBboxPatch((2, pos - 0.4), 6, 0.8, boxstyle="round,pad=0.05",
                               facecolor='lightblue', edgecolor='navy', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(5, pos, label, ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw gradient flow arrows (showing decay)
    gradient_colors = ['green', 'yellow', 'orange', 'red']
    gradient_alphas = [1.0, 0.7, 0.4, 0.2]

    for i in range(len(layer_positions) - 1):
        arrow = FancyArrowPatch((5, layer_positions[i] - 0.5), (5, layer_positions[i+1] + 0.5),
                                 arrowstyle='->', mutation_scale=20,
                                 color=gradient_colors[i], linewidth=3, alpha=gradient_alphas[i])
        ax1.add_patch(arrow)

    # Add gradient magnitude labels
    ax1.text(7.5, 7, '∇L = 1.0', fontsize=10, color='green')
    ax1.text(7.5, 5, '∇L = 0.25', fontsize=10, color='orange')
    ax1.text(7.5, 3, '∇L ≈ 0.06', fontsize=10, color='red')

    # Add equation
    ax1.text(5, 0.5, '∂L/∂x = ∂L/∂F(x) × ∂F(x)/∂x', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Right: With residual connection
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('With Skip Connection\n(Gradient highway through identity)', fontweight='bold', fontsize=13, color='green')

    # Draw layers
    for pos, label in zip(layer_positions, layer_labels):
        rect = FancyBboxPatch((2, pos - 0.4), 6, 0.8, boxstyle="round,pad=0.05",
                               facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(5, pos, label, ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw main gradient flow (through F)
    for i in range(len(layer_positions) - 1):
        arrow = FancyArrowPatch((4, layer_positions[i] - 0.5), (4, layer_positions[i+1] + 0.5),
                                 arrowstyle='->', mutation_scale=15,
                                 color='orange', linewidth=2, alpha=0.6)
        ax2.add_patch(arrow)

    # Draw skip connection gradient (identity path)
    arrow_skip = FancyArrowPatch((8.5, 8 - 0.5), (8.5, 2 + 0.5),
                                  arrowstyle='->', mutation_scale=25,
                                  color='green', linewidth=4, linestyle='-')
    ax2.add_patch(arrow_skip)
    ax2.text(9.2, 5, 'Identity\npath\n(+1)', fontsize=10, color='green', fontweight='bold', va='center')

    # Add gradient magnitude labels
    ax2.text(1.5, 7, '∇L = 1.0', fontsize=10, color='green')
    ax2.text(1.5, 5, '∇L ≈ 1.0', fontsize=10, color='green')
    ax2.text(1.5, 3, '∇L ≈ 1.0', fontsize=10, color='green')

    # Add equation
    ax2.text(5, 0.5, '∂L/∂x = ∂L/∂y × (∂F(x)/∂x + 1)', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    fig.suptitle('Residual Connections: Gradient Highway', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, 'residual_connection_gradient_path')


def plot_lstm_gate_visualization():
    """
    Figure 6: LSTM gate visualization
    Shows how LSTM gates control information and gradient flow.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: LSTM cell diagram
    ax1 = axes[0]
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('LSTM Cell Structure', fontweight='bold', fontsize=14)

    # Cell state line (highway)
    ax1.annotate('', xy=(11, 8), xytext=(1, 8),
                 arrowprops=dict(arrowstyle='->', color='green', linewidth=4))
    ax1.text(6, 8.5, 'Cell State (Gradient Highway)', ha='center', fontsize=11,
             fontweight='bold', color='green')

    # Hidden state line
    ax1.annotate('', xy=(11, 2), xytext=(1, 2),
                 arrowprops=dict(arrowstyle='->', color='blue', linewidth=3))
    ax1.text(6, 1.3, 'Hidden State', ha='center', fontsize=11, fontweight='bold', color='blue')

    # Gates
    gate_info = [
        (3, 5, 'Forget\nGate\n(f)', 'red', 'σ'),
        (5.5, 5, 'Input\nGate\n(i)', 'orange', 'σ'),
        (8, 5, 'Output\nGate\n(o)', 'purple', 'σ'),
    ]

    for x, y, label, color, act in gate_info:
        circle = plt.Circle((x, y), 0.8, color=color, alpha=0.3, ec=color, linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y, act, ha='center', va='center', fontsize=14, fontweight='bold')
        ax1.text(x, y - 1.5, label, ha='center', va='center', fontsize=9)

    # Candidate cell state
    rect = FancyBboxPatch((6.5, 6.5), 1.5, 1, boxstyle="round,pad=0.05",
                           facecolor='lightyellow', edgecolor='goldenrod', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(7.25, 7, 'tanh', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(7.25, 6, 'Candidate\n(g)', ha='center', va='center', fontsize=9)

    # Connections (simplified)
    ax1.annotate('', xy=(3, 8), xytext=(3, 5.8),
                 arrowprops=dict(arrowstyle='->', color='red', linewidth=2))
    ax1.annotate('', xy=(6.5, 8), xytext=(5.5, 5.8),
                 arrowprops=dict(arrowstyle='->', color='orange', linewidth=2))
    ax1.annotate('', xy=(8, 5.8), xytext=(8, 4),
                 arrowprops=dict(arrowstyle='->', color='purple', linewidth=2))

    # Operations
    ax1.text(3, 8, '×', fontsize=20, ha='center', va='center', fontweight='bold')
    ax1.text(6.5, 8, '+', fontsize=20, ha='center', va='center', fontweight='bold', color='green')

    # Key equation
    ax1.text(6, 0.3, 'cₜ = f × cₜ₋₁ + i × g  (Additive update!)', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Right: Gradient flow comparison
    ax2 = axes[1]

    time_steps = np.arange(1, 21)

    # Vanilla RNN gradient (product of weight matrices and tanh derivatives)
    rnn_gradient = 0.7 ** time_steps  # Typical decay with tanh

    # LSTM gradient (through cell state, forget gate near 1)
    forget_gate = 0.95  # Typically high
    lstm_gradient = forget_gate ** time_steps

    ax2.semilogy(time_steps, rnn_gradient, 'r-o', linewidth=2, markersize=6, label='Vanilla RNN')
    ax2.semilogy(time_steps, lstm_gradient, 'g-s', linewidth=2, markersize=6, label='LSTM (f=0.95)')
    ax2.fill_between(time_steps, rnn_gradient, lstm_gradient, alpha=0.2, color='green')

    ax2.axhline(y=1e-6, color='gray', linestyle='--', linewidth=2, label='Vanishing threshold')
    ax2.set_xlabel('Time Steps Back', fontsize=12)
    ax2.set_ylabel('Gradient Magnitude (log scale)', fontsize=12)
    ax2.set_title('Gradient Flow Through Time', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-10, 10)

    # Annotations
    ax2.annotate('LSTM preserves\ngradient much longer',
                 xy=(10, lstm_gradient[9]),
                 xytext=(12, 0.1),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, color='green')

    ax2.annotate('Vanilla RNN\ngradient vanishes quickly',
                 xy=(8, rnn_gradient[7]),
                 xytext=(10, 1e-6),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    fig.suptitle('LSTM: Solving Vanishing Gradients in Sequences', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    save_figure(fig, 'lstm_gate_visualization')


def main():
    """Generate all visualizations."""
    print("Generating Vanishing Gradients Visualizations...")
    print("=" * 50)

    plot_gradient_magnitude_decay()
    plot_activation_functions_with_derivatives()
    plot_sigmoid_vs_relu_gradient_flow()
    plot_weight_initialization_effect()
    plot_residual_connection_gradient_path()
    plot_lstm_gate_visualization()

    print("=" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
