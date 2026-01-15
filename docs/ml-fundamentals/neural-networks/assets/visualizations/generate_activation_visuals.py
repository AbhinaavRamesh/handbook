#!/usr/bin/env python3
"""
Generate visualizations for Activation Functions module.

Outputs:
- activation_nonlinearity.svg - Linear vs non-linear comparison
- activation_classic.svg - Sigmoid and tanh
- activation_relu_family.svg - ReLU variants
- activation_modern.svg - GELU, Swish, Mish
- activation_gradients.svg - Gradient comparison
- dying_relu.gif - Dying ReLU animation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = {
    'sigmoid': '#E74C3C',
    'tanh': '#3498DB',
    'relu': '#27AE60',
    'leaky_relu': '#9B59B6',
    'elu': '#F39C12',
    'gelu': '#1ABC9C',
    'swish': '#E91E63',
    'mish': '#00BCD4',
}


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def swish(x):
    return x * sigmoid(x)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

# Derivatives
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def relu_deriv(x):
    return np.where(x > 0, 1, 0)

def gelu_deriv(x):
    # Numerical derivative
    eps = 1e-5
    return (gelu(x + eps) - gelu(x - eps)) / (2 * eps)


def create_activation_nonlinearity():
    """Create visualization comparing linear vs non-linear networks."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate data for XOR-like pattern (non-linearly separable)
    np.random.seed(42)
    n_points = 100

    # Create XOR-like clusters
    cluster1 = np.random.randn(n_points // 4, 2) * 0.3 + np.array([0.5, 0.5])
    cluster2 = np.random.randn(n_points // 4, 2) * 0.3 + np.array([-0.5, -0.5])
    cluster3 = np.random.randn(n_points // 4, 2) * 0.3 + np.array([0.5, -0.5])
    cluster4 = np.random.randn(n_points // 4, 2) * 0.3 + np.array([-0.5, 0.5])

    class_a = np.vstack([cluster1, cluster2])  # Diagonal class A
    class_b = np.vstack([cluster3, cluster4])  # Diagonal class B

    # Left panel: Linear network (can only draw straight lines)
    ax = axes[0]
    ax.scatter(class_a[:, 0], class_a[:, 1], c='#E74C3C', s=50, label='Class A', alpha=0.7, edgecolors='white')
    ax.scatter(class_b[:, 0], class_b[:, 1], c='#3498DB', s=50, label='Class B', alpha=0.7, edgecolors='white')

    # Draw a linear decision boundary (best linear attempt - fails)
    x_line = np.linspace(-1.2, 1.2, 100)
    ax.plot(x_line, x_line * 0, 'g--', linewidth=2, alpha=0.7, label='Linear boundary')
    ax.plot(x_line * 0, x_line, 'g--', linewidth=2, alpha=0.7)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title('Linear Network\n(Cannot solve XOR problem)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add failed annotation
    ax.annotate('Linear functions\ncannot separate\nthese classes!', xy=(0, 0), fontsize=10,
               ha='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    # Right panel: Non-linear network (can draw curved boundaries)
    ax = axes[1]
    ax.scatter(class_a[:, 0], class_a[:, 1], c='#E74C3C', s=50, label='Class A', alpha=0.7, edgecolors='white')
    ax.scatter(class_b[:, 0], class_b[:, 1], c='#3498DB', s=50, label='Class B', alpha=0.7, edgecolors='white')

    # Create meshgrid for contour visualization
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))
    # XOR-like decision function: x * y > 0
    Z = xx * yy

    ax.contour(xx, yy, Z, levels=[0], colors=['#27AE60'], linewidths=2.5, linestyles=['-'])
    ax.contourf(xx, yy, Z, levels=[-10, 0], colors=['#3498DB'], alpha=0.15)
    ax.contourf(xx, yy, Z, levels=[0, 10], colors=['#E74C3C'], alpha=0.15)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title('Non-linear Network\n(Solves XOR with curved boundaries)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add success annotation
    ax.annotate('Non-linear activation\nenables curved\ndecision boundaries!', xy=(0, 0), fontsize=10,
               ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'activation_nonlinearity.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: activation_nonlinearity.svg")


def create_activation_classic():
    """Create comparison of sigmoid and tanh."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.linspace(-6, 6, 500)

    # Activations
    ax = axes[0]
    ax.plot(x, sigmoid(x), color=COLORS['sigmoid'], linewidth=2.5, label='Sigmoid')
    ax.plot(x, tanh(x), color=COLORS['tanh'], linewidth=2.5, label='Tanh')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Classic Activations', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)

    # Derivatives
    ax = axes[1]
    ax.plot(x, sigmoid_deriv(x), color=COLORS['sigmoid'], linewidth=2.5, label="Sigmoid' (max=0.25)")
    ax.plot(x, tanh_deriv(x), color=COLORS['tanh'], linewidth=2.5, label="Tanh' (max=1.0)")
    ax.axhline(y=0.25, color=COLORS['sigmoid'], linestyle=':', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel("f'(x)", fontsize=12)
    ax.set_title('Derivatives (Gradient Flow)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)

    # Annotate saturation
    ax.annotate('Saturation\n(vanishing gradient)', xy=(-4, 0.02), fontsize=9,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'activation_classic.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: activation_classic.svg")


def create_activation_relu_family():
    """Create comparison of ReLU variants."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.linspace(-4, 4, 500)

    # Activations
    ax = axes[0]
    ax.plot(x, relu(x), color=COLORS['relu'], linewidth=2.5, label='ReLU')
    ax.plot(x, leaky_relu(x, 0.1), color=COLORS['leaky_relu'], linewidth=2.5, label='Leaky ReLU (α=0.1)')
    ax.plot(x, elu(x), color=COLORS['elu'], linewidth=2.5, label='ELU')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('ReLU Family', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 4)

    # Derivatives
    ax = axes[1]
    ax.plot(x, relu_deriv(x), color=COLORS['relu'], linewidth=2.5, label="ReLU'")
    ax.plot(x, np.where(x > 0, 1, 0.1), color=COLORS['leaky_relu'], linewidth=2.5, label="Leaky ReLU'")
    elu_d = np.where(x > 0, 1, np.exp(x))
    ax.plot(x, elu_d, color=COLORS['elu'], linewidth=2.5, label="ELU'")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel("f'(x)", fontsize=12)
    ax.set_title('Derivatives', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.5)

    # Annotate dying ReLU region
    axes[1].annotate('Dead region\n(zero gradient)', xy=(-2, 0.05), fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'activation_relu_family.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: activation_relu_family.svg")


def create_activation_modern():
    """Create comparison of modern activations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.linspace(-4, 4, 500)

    # Activations
    ax = axes[0]
    ax.plot(x, relu(x), color=COLORS['relu'], linewidth=2, linestyle='--', alpha=0.5, label='ReLU (reference)')
    ax.plot(x, gelu(x), color=COLORS['gelu'], linewidth=2.5, label='GELU')
    ax.plot(x, swish(x), color=COLORS['swish'], linewidth=2.5, label='Swish/SiLU')
    ax.plot(x, mish(x), color=COLORS['mish'], linewidth=2.5, label='Mish')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Modern Activations', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 4)

    # Zoom on negative region
    ax = axes[1]
    x_zoom = np.linspace(-3, 1, 500)
    ax.plot(x_zoom, relu(x_zoom), color=COLORS['relu'], linewidth=2, linestyle='--', alpha=0.5, label='ReLU')
    ax.plot(x_zoom, gelu(x_zoom), color=COLORS['gelu'], linewidth=2.5, label='GELU')
    ax.plot(x_zoom, swish(x_zoom), color=COLORS['swish'], linewidth=2.5, label='Swish')
    ax.plot(x_zoom, mish(x_zoom), color=COLORS['mish'], linewidth=2.5, label='Mish')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Zoom: Smooth Transition Near Zero', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1)

    ax.annotate('Smooth curve\n(vs ReLU kink)', xy=(-0.5, -0.15), fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'activation_modern.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: activation_modern.svg")


def create_dying_relu():
    """Create animation showing dying ReLU phenomenon."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    frames = 40

    def animate(frame):
        for ax in axes:
            ax.clear()

        # Simulate weight becoming more negative
        progress = frame / frames
        weight = 2 - 4 * progress  # Goes from +2 to -2

        # Input data
        x = np.linspace(-2, 2, 100)

        # Neuron behavior
        z = weight * x  # Pre-activation
        a = relu(z)  # Post-activation

        # Plot 1: Weight evolution
        axes[0].bar(['Weight'], [weight], color='green' if weight > 0 else 'red')
        axes[0].set_ylim(-3, 3)
        axes[0].axhline(y=0, color='gray', linestyle='--')
        axes[0].set_title(f'Weight Value: {weight:.2f}', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Weight')

        # Plot 2: Pre-activation (z = wx)
        axes[1].plot(x, z, color='blue', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--')
        axes[1].fill_between(x, z, 0, where=(z < 0), color='red', alpha=0.3, label='Negative region')
        axes[1].set_xlabel('Input x')
        axes[1].set_ylabel('z = wx')
        axes[1].set_title('Pre-activation', fontsize=11, fontweight='bold')
        axes[1].set_ylim(-5, 5)
        axes[1].legend()

        # Plot 3: Post-activation (ReLU output)
        axes[2].plot(x, a, color='green', linewidth=2)
        axes[2].axhline(y=0, color='gray', linestyle='--')
        axes[2].set_xlabel('Input x')
        axes[2].set_ylabel('ReLU(z)')
        axes[2].set_title('Post-activation', fontsize=11, fontweight='bold')
        axes[2].set_ylim(-0.5, 5)

        if weight < 0:
            axes[2].text(0, 2, 'Neuron\nDEAD!', ha='center', fontsize=14, fontweight='bold',
                        color='red', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        fig.suptitle('Dying ReLU: When Weights Go Negative', fontsize=13, fontweight='bold')
        plt.tight_layout()

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=150, blit=False)
    writer = PillowWriter(fps=8)
    anim.save(os.path.join(OUTPUT_DIR, 'dying_relu.gif'), writer=writer)
    plt.close()
    print("Generated: dying_relu.gif")


def create_activation_gradients():
    """Create gradient comparison across activations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-4, 4, 500)

    ax.plot(x, sigmoid_deriv(x), color=COLORS['sigmoid'], linewidth=2, label='Sigmoid (max=0.25)')
    ax.plot(x, tanh_deriv(x), color=COLORS['tanh'], linewidth=2, label='Tanh (max=1.0)')
    ax.plot(x, relu_deriv(x), color=COLORS['relu'], linewidth=2, label='ReLU (0 or 1)')
    ax.plot(x, gelu_deriv(x), color=COLORS['gelu'], linewidth=2, label='GELU (smooth)')

    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.25, color=COLORS['sigmoid'], linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel("f'(x) (Gradient)", fontsize=12)
    ax.set_title('Activation Gradients: Impact on Backpropagation', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.4)

    # Annotations
    ax.annotate('Sigmoid gradient\nalways ≤ 0.25', xy=(2, 0.05), fontsize=9,
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    ax.annotate('ReLU: perfect gradient\nfor positive inputs', xy=(2, 0.95), fontsize=9,
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    plt.savefig(os.path.join(OUTPUT_DIR, 'activation_gradients.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: activation_gradients.svg")


if __name__ == '__main__':
    print("Generating Activation Function visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_activation_nonlinearity()
    create_activation_classic()
    create_activation_relu_family()
    create_activation_modern()
    create_dying_relu()
    create_activation_gradients()

    print("\nAll activation visualizations generated!")
