"""
Generate visualizations for Softmax and Cross-Entropy concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')

def softmax(logits, temperature=1.0):
    """Numerically stable softmax with temperature."""
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

def cross_entropy(probs, targets, epsilon=1e-12):
    """Cross-entropy loss."""
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    return -np.sum(targets * np.log(probs_clipped), axis=-1)

def mse_loss(probs, targets):
    """Mean squared error loss."""
    return np.mean((probs - targets) ** 2, axis=-1)


def plot_temperature_effect():
    """Plot softmax temperature effect with different T values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Fixed logits
    logits = np.array([2.0, 1.0, 0.1])
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(temperatures)))

    # Left plot: Bar chart comparison
    ax1 = axes[0]
    x = np.arange(3)
    width = 0.12

    for i, (T, color) in enumerate(zip(temperatures, colors)):
        probs = softmax(logits, temperature=T)
        offset = (i - len(temperatures)/2 + 0.5) * width
        bars = ax1.bar(x + offset, probs, width, label=f'T={T}', color=color, edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('Class', fontsize=11)
    ax1.set_ylabel('Probability', fontsize=11)
    ax1.set_title('Softmax Output at Different Temperatures', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Class 0\n(logit=2.0)', 'Class 1\n(logit=1.0)', 'Class 2\n(logit=0.1)'])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 1.05)

    # Right plot: Line plot showing temperature effect
    ax2 = axes[1]
    temp_range = np.linspace(0.1, 10, 100)

    for i, logit in enumerate(logits):
        probs_over_temp = []
        for T in temp_range:
            p = softmax(logits, temperature=T)
            probs_over_temp.append(p[i])
        ax2.plot(temp_range, probs_over_temp, linewidth=2.5,
                label=f'Class {i} (logit={logit})')

    ax2.set_xlabel('Temperature (T)', fontsize=11)
    ax2.set_ylabel('Probability', fontsize=11)
    ax2.set_title('Probability vs Temperature', fontsize=12, fontweight='bold')
    ax2.legend(loc='center right', fontsize=10)
    ax2.set_xlim(0.1, 10)
    ax2.set_ylim(0, 1)

    # Add annotations
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate('Standard\nSoftmax', xy=(1.0, 0.85), fontsize=9, ha='center', color='gray')
    ax2.annotate('Low T:\nHard decisions', xy=(0.5, 0.1), fontsize=9, ha='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax2.annotate('High T:\nSoft/uniform', xy=(7, 0.1), fontsize=9, ha='left',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/softmax_temperature.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: softmax_temperature.png")


def plot_probability_simplex():
    """Plot 2D probability simplex (triangle) for 3 classes with loss contours."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Create simplex vertices (equilateral triangle)
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

    def barycentric_to_cartesian(p):
        """Convert barycentric (probability) coordinates to 2D cartesian."""
        return p[0] * vertices[0] + p[1] * vertices[1] + p[2] * vertices[2]

    def cartesian_to_barycentric(point):
        """Convert cartesian to barycentric coordinates."""
        v0 = vertices[2] - vertices[0]
        v1 = vertices[1] - vertices[0]
        v2 = point - vertices[0]

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return np.array([u, w, v])

    # Left plot: Cross-entropy loss contours (true class = 0)
    ax1 = axes[0]

    # Draw simplex
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(triangle)

    # Create grid and compute loss
    resolution = 100
    x_range = np.linspace(-0.1, 1.1, resolution)
    y_range = np.linspace(-0.1, np.sqrt(3)/2 + 0.1, resolution)
    X, Y = np.meshgrid(x_range, y_range)

    loss_grid = np.full_like(X, np.nan)

    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            bary = cartesian_to_barycentric(point)

            # Check if inside simplex (all coordinates positive)
            if np.all(bary >= -0.01) and np.all(bary <= 1.01):
                bary = np.clip(bary, 0.001, 0.999)
                bary = bary / bary.sum()  # Normalize
                # Cross-entropy loss when true class is 0
                loss_grid[i, j] = -np.log(bary[0] + 1e-10)

    # Plot contours
    levels = np.linspace(0, 4, 20)
    contour = ax1.contourf(X, Y, loss_grid, levels=levels, cmap='RdYlGn_r', alpha=0.8)
    ax1.contour(X, Y, loss_grid, levels=levels[::2], colors='gray', linewidths=0.5, alpha=0.5)

    # Mark vertices (class corners)
    for i, (v, label) in enumerate(zip(vertices, ['p(0)=1', 'p(1)=1', 'p(2)=1'])):
        ax1.plot(v[0], v[1], 'ko', markersize=10)
        offset = [(-0.12, -0.08), (0.05, -0.08), (0.05, 0.05)][i]
        ax1.annotate(label, xy=v, xytext=(v[0]+offset[0], v[1]+offset[1]), fontsize=10, fontweight='bold')

    # Mark optimal point (true class = 0)
    optimal = barycentric_to_cartesian([1, 0, 0])
    ax1.plot(optimal[0], optimal[1], 'g*', markersize=20, markeredgecolor='white', markeredgewidth=1.5)

    ax1.set_xlim(-0.15, 1.15)
    ax1.set_ylim(-0.15, 1.05)
    ax1.set_aspect('equal')
    ax1.set_title('Cross-Entropy Loss on Probability Simplex\n(True class = 0)', fontsize=11, fontweight='bold')
    ax1.axis('off')

    cbar = plt.colorbar(contour, ax=ax1, shrink=0.7)
    cbar.set_label('CE Loss', fontsize=10)

    # Right plot: Gradient vectors
    ax2 = axes[1]

    triangle2 = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(triangle2)

    # Plot gradient vectors at sample points
    sample_points = []
    for i in range(5):
        for j in range(5-i):
            k = 4 - i - j
            if i + j + k == 4:
                p = np.array([i/4, j/4, k/4])
                p = np.clip(p, 0.1, 0.9)
                p = p / p.sum()
                sample_points.append(p)

    for p in sample_points:
        cart = barycentric_to_cartesian(p)

        # Gradient of CE loss w.r.t. probabilities (true class = 0)
        grad_p = np.array([-1/p[0], 0, 0])  # Only p[0] matters

        # Convert gradient direction to cartesian
        grad_cart = grad_p[0] * (vertices[0] - cart) + grad_p[1] * (vertices[1] - cart) + grad_p[2] * (vertices[2] - cart)
        grad_norm = np.linalg.norm(grad_cart)
        if grad_norm > 0:
            grad_cart = grad_cart / grad_norm * 0.08  # Normalize for visualization

        ax2.arrow(cart[0], cart[1], grad_cart[0], grad_cart[1],
                 head_width=0.02, head_length=0.01, fc='blue', ec='blue', alpha=0.7)

    # Mark vertices
    for i, (v, label) in enumerate(zip(vertices, ['p(0)=1', 'p(1)=1', 'p(2)=1'])):
        ax2.plot(v[0], v[1], 'ko', markersize=10)
        offset = [(-0.12, -0.08), (0.05, -0.08), (0.05, 0.05)][i]
        ax2.annotate(label, xy=v, xytext=(v[0]+offset[0], v[1]+offset[1]), fontsize=10, fontweight='bold')

    optimal = barycentric_to_cartesian([1, 0, 0])
    ax2.plot(optimal[0], optimal[1], 'g*', markersize=20, markeredgecolor='white', markeredgewidth=1.5)

    ax2.set_xlim(-0.15, 1.15)
    ax2.set_ylim(-0.15, 1.05)
    ax2.set_aspect('equal')
    ax2.set_title('Gradient Directions\n(pointing toward minimum loss)', fontsize=11, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/probability_simplex.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: probability_simplex.png")


def plot_ce_vs_mse():
    """Compare cross-entropy vs MSE loss."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Left plot: Loss curves as function of predicted probability
    ax1 = axes[0]

    p = np.linspace(0.001, 0.999, 500)

    # True class probability = 1 (we're predicting for the correct class)
    ce_loss = -np.log(p)
    mse_loss_vals = (1 - p) ** 2

    ax1.plot(p, ce_loss, 'b-', linewidth=2.5, label='Cross-Entropy: -log(p)')
    ax1.plot(p, mse_loss_vals, 'r-', linewidth=2.5, label='MSE: (1-p)^2')

    ax1.set_xlabel('Predicted Probability for True Class', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss vs Predicted Probability\n(When True Label = 1)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 6)

    # Add annotations
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate('CE penalizes\nlow confidence\nmuch more', xy=(0.15, 3), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Right plot: Gradient magnitude comparison
    ax2 = axes[1]

    # Gradients
    ce_grad = -1 / p  # d/dp of -log(p)
    mse_grad = -2 * (1 - p)  # d/dp of (1-p)^2

    ax2.plot(p, np.abs(ce_grad), 'b-', linewidth=2.5, label='|CE gradient|: 1/p')
    ax2.plot(p, np.abs(mse_grad), 'r-', linewidth=2.5, label='|MSE gradient|: 2(1-p)')

    ax2.set_xlabel('Predicted Probability for True Class', fontsize=11)
    ax2.set_ylabel('|Gradient| (magnitude)', fontsize=11)
    ax2.set_title('Gradient Magnitude Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 15)

    # Add annotations
    ax2.annotate('CE: Strong gradient\nwhen wrong', xy=(0.1, 8), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax2.annotate('MSE: Gradient\nvanishes when wrong', xy=(0.5, 1.5), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/ce_vs_mse_loss.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: ce_vs_mse_loss.png")


def plot_gradient_at_confidence():
    """Plot gradient magnitude at different confidence levels."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Left: Softmax + CE gradient (dL/dz = p - y)
    ax1 = axes[0]

    # For a 3-class problem, varying the logit of the true class
    logits_range = np.linspace(-5, 5, 100)

    # Compute softmax probabilities and gradient
    probs_true = []
    grads_true = []
    grads_other = []

    for logit_true in logits_range:
        logits = np.array([logit_true, 0.0, 0.0])  # True class has varying logit
        probs = softmax(logits)

        # Gradient: p - y (for true class, y=1, so grad = p - 1)
        grad = probs - np.array([1, 0, 0])

        probs_true.append(probs[0])
        grads_true.append(grad[0])
        grads_other.append(grad[1])

    ax1.plot(logits_range, grads_true, 'b-', linewidth=2.5, label='True class gradient (p-1)')
    ax1.plot(logits_range, grads_other, 'r--', linewidth=2.5, label='Other class gradient (p-0)')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax1.set_xlabel('Logit of True Class', fontsize=11)
    ax1.set_ylabel('Gradient (dL/dz)', fontsize=11)
    ax1.set_title('Softmax+CE Gradient: p - y', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-1.1, 0.6)

    # Annotations
    ax1.annotate('Confident & correct:\nsmall gradient', xy=(4, -0.1), fontsize=9,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax1.annotate('Confident & wrong:\nlarge gradient', xy=(-4, -0.9), fontsize=9,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Right: Gradient magnitude heatmap for 2 classes
    ax2 = axes[1]

    z1_range = np.linspace(-3, 3, 50)
    z2_range = np.linspace(-3, 3, 50)
    Z1, Z2 = np.meshgrid(z1_range, z2_range)

    grad_mag = np.zeros_like(Z1)

    for i in range(len(z1_range)):
        for j in range(len(z2_range)):
            logits = np.array([Z1[i, j], Z2[i, j]])
            probs = softmax(logits)
            # True class is 0
            grad = probs - np.array([1, 0])
            grad_mag[i, j] = np.linalg.norm(grad)

    im = ax2.contourf(Z1, Z2, grad_mag, levels=20, cmap='YlOrRd')
    ax2.contour(Z1, Z2, grad_mag, levels=10, colors='gray', linewidths=0.5, alpha=0.5)

    # Mark diagonal (equal logits)
    ax2.plot([-3, 3], [-3, 3], 'k--', linewidth=1.5, alpha=0.5, label='z1 = z2')

    ax2.set_xlabel('Logit z1 (true class)', fontsize=11)
    ax2.set_ylabel('Logit z2 (other class)', fontsize=11)
    ax2.set_title('Gradient Magnitude\n(True class = 0)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('||gradient||', fontsize=10)

    # Annotations
    ax2.annotate('Low gradient\n(correct)', xy=(2.5, -2), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.annotate('High gradient\n(wrong)', xy=(-2, 2.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/gradient_confidence.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: gradient_confidence.png")


def plot_label_smoothing():
    """Visualize label smoothing effect."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    num_classes = 5
    true_class = 2

    smoothing_values = [0.0, 0.1, 0.2, 0.3]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Left: Target distribution comparison
    ax1 = axes[0]
    x = np.arange(num_classes)
    width = 0.18

    for i, (alpha, color) in enumerate(zip(smoothing_values, colors)):
        target = np.zeros(num_classes)
        target[true_class] = 1 - alpha
        target += alpha / num_classes

        offset = (i - len(smoothing_values)/2 + 0.5) * width
        ax1.bar(x + offset, target, width, label=f'alpha={alpha}', color=color, edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('Class', fontsize=11)
    ax1.set_ylabel('Target Probability', fontsize=11)
    ax1.set_title('Label Smoothing: Target Distributions', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax1.legend(fontsize=10)
    ax1.axhline(y=1/num_classes, color='gray', linestyle='--', alpha=0.5, label='Uniform')

    # Right: Effect on loss landscape
    ax2 = axes[1]

    # Plot loss vs confidence for different smoothing values
    confidence_range = np.linspace(0.01, 0.99, 100)

    for alpha, color in zip(smoothing_values, colors):
        losses = []
        for conf in confidence_range:
            # Predicted distribution (assume true class gets 'conf', rest split uniformly)
            probs = np.ones(num_classes) * (1 - conf) / (num_classes - 1)
            probs[true_class] = conf

            # Smoothed target
            target = np.zeros(num_classes)
            target[true_class] = 1 - alpha
            target += alpha / num_classes

            # Cross-entropy loss
            loss = -np.sum(target * np.log(probs + 1e-10))
            losses.append(loss)

        ax2.plot(confidence_range, losses, color=color, linewidth=2.5, label=f'alpha={alpha}')

    ax2.set_xlabel('Predicted Confidence for True Class', fontsize=11)
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax2.set_title('Loss vs Confidence with Label Smoothing', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 5)

    # Annotations
    ax2.annotate('Smoothing prevents\nloss from going to 0', xy=(0.95, 0.3), fontsize=9,
                ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/label_smoothing.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: label_smoothing.png")


def main():
    """Generate all visualizations."""
    print("Generating Softmax and Cross-Entropy visualizations...")
    print("=" * 50)

    plot_temperature_effect()
    plot_probability_simplex()
    plot_ce_vs_mse()
    plot_gradient_at_confidence()
    plot_label_smoothing()

    print("=" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
