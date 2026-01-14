#!/usr/bin/env python3
"""
Generate visualizations for Logistic Regression Coding implementation page.
Creates SVG files for sigmoid vs softmax, OvR boundaries, binning, and probability surfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.colors import ListedColormap

# Set style for clean, professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images/'


def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                   1 / (1 + np.exp(-z)),
                   np.exp(z) / (1 + np.exp(z)))


def softmax(z):
    """Softmax function."""
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def generate_sigmoid_softmax_comparison():
    """Generate sigmoid vs softmax comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Sigmoid for binary classification
    ax1 = axes[0]
    z = np.linspace(-6, 6, 500)
    sig = sigmoid(z)

    ax1.plot(z, sig, 'b-', linewidth=2.5, label=r'$\sigma(z) = \frac{1}{1+e^{-z}}$')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Mark key points
    ax1.plot(0, 0.5, 'ro', markersize=10, zorder=5)
    ax1.annotate(r'$\sigma(0) = 0.5$', xy=(0.3, 0.55), fontsize=11, color='red')

    # Fill regions
    ax1.fill_between(z, sig, 0.5, where=(z > 0), alpha=0.2, color='green', label='Predict Class 1')
    ax1.fill_between(z, sig, 0.5, where=(z < 0), alpha=0.2, color='red', label='Predict Class 0')

    ax1.set_xlabel('z = w^T x + b')
    ax1.set_ylabel('P(y=1|x)')
    ax1.set_title('Sigmoid: Binary Classification')
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(loc='lower right', fontsize=10)

    # Right: Softmax for multi-class
    ax2 = axes[1]
    z1 = np.linspace(-3, 3, 100)
    z2 = 0  # Fixed at 0
    z3 = 1  # Fixed at 1

    probs = []
    for z in z1:
        logits = np.array([z, z2, z3])
        probs.append(softmax(logits))
    probs = np.array(probs)

    ax2.plot(z1, probs[:, 0], 'b-', linewidth=2.5, label='Class 0')
    ax2.plot(z1, probs[:, 1], 'g-', linewidth=2.5, label='Class 1')
    ax2.plot(z1, probs[:, 2], 'r-', linewidth=2.5, label='Class 2')

    ax2.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.annotate('All classes equal', xy=(-2.5, 0.38), fontsize=10, color='gray')

    ax2.set_xlabel(r'$z_0$ (logit for Class 0)')
    ax2.set_ylabel('Probability')
    ax2.set_title('Softmax: Multi-class (K=3)')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=10)

    # Add annotation about sum to 1
    ax2.annotate('Probabilities\nsum to 1', xy=(2, 0.7), fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'logreg-sigmoid-softmax-comparison.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated: logreg-sigmoid-softmax-comparison.svg')


def generate_ovr_decision_boundaries():
    """Generate One-vs-Rest decision boundary visualization."""
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Generate 3-class data
    n_per_class = 50
    X0 = np.random.randn(n_per_class, 2) * 0.5 + np.array([-1.5, 1])
    X1 = np.random.randn(n_per_class, 2) * 0.5 + np.array([1.5, 1])
    X2 = np.random.randn(n_per_class, 2) * 0.5 + np.array([0, -1])

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    class_names = ['Class 0', 'Class 1', 'Class 2']

    # Train 3 binary classifiers (simplified)
    def train_binary(X, y, positive_class):
        y_binary = (y == positive_class).astype(float)
        w = np.zeros(2)
        b = 0.0
        lr = 0.1
        for _ in range(500):
            z = np.dot(X, w) + b
            y_pred = sigmoid(z)
            errors = y_pred - y_binary
            dw = (1/len(y)) * np.dot(X.T, errors)
            db = (1/len(y)) * np.sum(errors)
            w -= lr * dw
            b -= lr * db
        return w, b

    classifiers = []
    for cls in range(3):
        w, b = train_binary(X, y, cls)
        classifiers.append((w, b))

    # Plot each binary classifier
    for idx, (ax, (w, b)) in enumerate(zip(axes, classifiers)):
        # Create meshgrid
        xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
        Z = sigmoid(w[0]*xx + w[1]*yy + b)

        # Plot probability contour
        contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap='RdBu_r', alpha=0.4)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

        # Plot decision boundary
        x_line = np.linspace(-3, 3, 100)
        y_line = -(w[0] * x_line + b) / w[1]
        ax.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')

        # Plot data points
        for cls in range(3):
            mask = y == cls
            if cls == idx:
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], s=60,
                          edgecolors='black', linewidths=2, label=f'{class_names[cls]} (Positive)', zorder=5)
            else:
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], s=40,
                          alpha=0.5, label=f'{class_names[cls]} (Negative)')

        ax.set_xlabel('Feature $x_1$')
        ax.set_ylabel('Feature $x_2$')
        ax.set_title(f'OvR: {class_names[idx]} vs Rest')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('One-vs-Rest: Each Classifier Separates One Class from Others', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'logreg-ovr-decision-boundaries.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated: logreg-ovr-decision-boundaries.svg')


def generate_binning_regression():
    """Generate visualization for binning continuous targets."""
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Generate regression data
    n_samples = 100
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_true = 2 * np.sin(X.flatten()) + X.flatten() + np.random.randn(n_samples) * 0.5

    # Left: Original continuous target
    ax1 = axes[0]
    ax1.scatter(X, y_true, c='blue', alpha=0.6, s=40, label='Data points')
    ax1.plot(X, 2 * np.sin(X) + X, 'r-', linewidth=2, label='True function')
    ax1.set_xlabel('Feature X')
    ax1.set_ylabel('Continuous Target y')
    ax1.set_title('Original Regression Problem')
    ax1.legend()

    # Middle: Binned target
    ax2 = axes[1]
    n_bins = 5
    bin_edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    y_binned = np.digitize(y_true, bin_edges[1:-1])

    colors = plt.cm.viridis(np.linspace(0, 1, n_bins))

    for i in range(n_bins):
        mask = y_binned == i
        ax2.scatter(X[mask], y_true[mask], c=[colors[i]], s=50, alpha=0.8,
                   label=f'Bin {i}: [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})')

    # Draw horizontal bin boundaries
    for edge in bin_edges:
        ax2.axhline(y=edge, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Feature X')
    ax2.set_ylabel('Target y (binned)')
    ax2.set_title(f'Target Discretized into {n_bins} Bins')
    ax2.legend(loc='upper left', fontsize=9)

    # Right: Softmax probability distribution
    ax3 = axes[2]

    # Simulate softmax output for a few example points
    example_points = [1, 3, 5, 7, 9]
    bar_width = 0.15

    for i, x_val in enumerate(example_points):
        # Generate fake probabilities (peaked around expected bin)
        true_bin = np.digitize([x_val], bin_edges[1:-1])[0]
        probs = np.exp(-0.5 * (np.arange(n_bins) - true_bin) ** 2)
        probs /= probs.sum()

        positions = np.arange(n_bins) + i * bar_width
        ax3.bar(positions, probs, bar_width, label=f'x={x_val}', alpha=0.7)

    ax3.set_xlabel('Bin Index')
    ax3.set_ylabel('Softmax Probability')
    ax3.set_title('Softmax Output per Example')
    ax3.set_xticks(np.arange(n_bins) + 2 * bar_width)
    ax3.set_xticklabels([f'Bin {i}' for i in range(n_bins)])
    ax3.legend(loc='upper right', fontsize=9)

    # Add expected value annotation
    ax3.annotate('Prediction = E[bin center]\n= sum(prob * center)',
                xy=(3.5, 0.7), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'logreg-binning-regression.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated: logreg-binning-regression.svg')


def generate_multiclass_probabilities():
    """Generate multi-class probability surface visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create 2D feature space
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Simulate 3-class softmax classifier
    np.random.seed(42)
    n_classes = 3

    # Weights for each class (2 features)
    W = np.array([
        [1.0, 0.5],    # Class 0
        [-0.5, 1.0],   # Class 1
        [-0.5, -1.0]   # Class 2
    ])
    b = np.array([0, 0.5, -0.5])

    # Compute logits and probabilities
    logits = np.dot(grid, W.T) + b
    probs = softmax(logits)

    # Reshape for plotting
    probs_reshaped = probs.reshape(200, 200, 3)
    predictions = np.argmax(probs, axis=1).reshape(200, 200)

    # Left: Decision regions with probabilities
    ax1 = axes[0]

    # Create RGB image from probabilities
    rgb_image = np.zeros((200, 200, 3))
    rgb_image[:, :, 0] = probs_reshaped[:, :, 0]  # Red for class 0
    rgb_image[:, :, 1] = probs_reshaped[:, :, 1]  # Green for class 1
    rgb_image[:, :, 2] = probs_reshaped[:, :, 2]  # Blue for class 2

    ax1.imshow(rgb_image, extent=[-3, 3, -3, 3], origin='lower', alpha=0.7)

    # Add contour lines for decision boundaries
    ax1.contour(xx, yy, predictions, levels=[0.5, 1.5], colors='white', linewidths=2, linestyles='--')

    # Add class labels
    ax1.annotate('Class 0', xy=(1.5, 1.5), fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    ax1.annotate('Class 1', xy=(-2, 1.5), fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    ax1.annotate('Class 2', xy=(-1, -2), fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))

    ax1.set_xlabel('Feature $x_1$')
    ax1.set_ylabel('Feature $x_2$')
    ax1.set_title('Softmax Decision Regions\n(Color = Probability)')

    # Right: Probability surface for each class
    ax2 = axes[1]

    # Show max probability at each point
    max_probs = np.max(probs, axis=1).reshape(200, 200)
    contour = ax2.contourf(xx, yy, max_probs, levels=np.linspace(0.33, 1.0, 15), cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label('Max Class Probability', fontsize=11)

    # Add decision boundary contours
    ax2.contour(xx, yy, predictions, levels=[0.5, 1.5], colors='white', linewidths=2)

    # Mark uncertainty regions
    uncertainty_mask = max_probs < 0.5
    ax2.contour(xx, yy, uncertainty_mask.astype(float), levels=[0.5], colors='red',
               linewidths=2, linestyles='--')
    ax2.annotate('High uncertainty\n(max prob < 0.5)', xy=(0, 0), fontsize=10,
                ha='center', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Feature $x_1$')
    ax2.set_ylabel('Feature $x_2$')
    ax2.set_title('Prediction Confidence\n(Higher = More Certain)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'logreg-multiclass-probabilities.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated: logreg-multiclass-probabilities.svg')


if __name__ == '__main__':
    print('Generating Logistic Regression Coding visualizations...')
    print(f'Output directory: {OUTPUT_DIR}')
    print()

    generate_sigmoid_softmax_comparison()
    generate_ovr_decision_boundaries()
    generate_binning_regression()
    generate_multiclass_probabilities()

    print()
    print('All visualizations generated successfully!')
