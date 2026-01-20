"""
Hyperparameter Tuning Visualizations
Generates matplotlib visualizations for hyperparameter tuning concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm
import os

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_grid_search_heatmap():
    """Create a 2D parameter grid heatmap showing performance landscape."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Create parameter grids
    learning_rates = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    regularizations = np.array([0.0001, 0.001, 0.01, 0.1, 1.0])

    # Simulated accuracy values (higher is better)
    # Best around lr=0.01-0.05, reg=0.001-0.01
    np.random.seed(42)
    accuracy = np.array([
        [0.72, 0.78, 0.75, 0.70, 0.55],
        [0.80, 0.92, 0.89, 0.82, 0.65],
        [0.76, 0.88, 0.86, 0.78, 0.60],
        [0.68, 0.82, 0.80, 0.72, 0.55],
        [0.50, 0.65, 0.62, 0.58, 0.45]
    ])

    # Create heatmap
    im = ax.imshow(accuracy, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Validation Accuracy')

    # Set ticks
    ax.set_xticks(np.arange(len(regularizations)))
    ax.set_yticks(np.arange(len(learning_rates)))
    ax.set_xticklabels([f'{r}' for r in regularizations])
    ax.set_yticklabels([f'{lr}' for lr in learning_rates])

    # Add text annotations
    for i in range(len(learning_rates)):
        for j in range(len(regularizations)):
            text = ax.text(j, i, f'{accuracy[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10, fontweight='bold')

    # Mark best configuration
    best_idx = np.unravel_index(np.argmax(accuracy), accuracy.shape)
    rect = Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                     linewidth=3, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

    ax.set_xlabel('Regularization Strength', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Grid Search: 2D Parameter Space Exploration\n(Blue box = Best configuration)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'grid_search_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: grid_search_heatmap.png")


def create_grid_vs_random_comparison():
    """Compare grid search vs random search coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    np.random.seed(42)
    n_samples = 25

    # Grid Search (left)
    ax1 = axes[0]
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    xx, yy = np.meshgrid(grid_x, grid_y)
    ax1.scatter(xx.flatten(), yy.flatten(), s=100, c='steelblue', edgecolors='black', linewidth=1, zorder=3)

    # Draw grid lines
    for x in grid_x:
        ax1.axvline(x, color='lightgray', linestyle='--', alpha=0.7, zorder=1)
    for y in grid_y:
        ax1.axhline(y, color='lightgray', linestyle='--', alpha=0.7, zorder=1)

    # Project points onto axes to show coverage
    ax1.scatter(xx.flatten(), np.zeros(25) - 0.05, s=30, c='steelblue', alpha=0.5, marker='|')
    ax1.scatter(np.zeros(25) - 0.05, yy.flatten(), s=30, c='steelblue', alpha=0.5, marker='_')

    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel('Hyperparameter 1 (Important)', fontsize=11)
    ax1.set_ylabel('Hyperparameter 2 (Less Important)', fontsize=11)
    ax1.set_title('Grid Search\n(5 unique values per dimension)', fontsize=13)
    ax1.set_aspect('equal')

    # Random Search (right)
    ax2 = axes[1]
    random_x = np.random.uniform(0.05, 0.95, n_samples)
    random_y = np.random.uniform(0.05, 0.95, n_samples)
    ax2.scatter(random_x, random_y, s=100, c='coral', edgecolors='black', linewidth=1, zorder=3)

    # Project points onto axes
    ax2.scatter(random_x, np.zeros(n_samples) - 0.05, s=30, c='coral', alpha=0.5, marker='|')
    ax2.scatter(np.zeros(n_samples) - 0.05, random_y, s=30, c='coral', alpha=0.5, marker='_')

    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel('Hyperparameter 1 (Important)', fontsize=11)
    ax2.set_ylabel('Hyperparameter 2 (Less Important)', fontsize=11)
    ax2.set_title('Random Search\n(25 unique values for important dim)', fontsize=13)
    ax2.set_aspect('equal')

    # Add annotation
    fig.text(0.5, 0.02, 'Random search explores more unique values of the important hyperparameter with same budget',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, 'grid_vs_random_search.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: grid_vs_random_search.png")


def create_bayesian_optimization_gif():
    """Create animated GIF showing Bayesian optimization with acquisition function."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # True objective function (unknown to optimizer)
    x = np.linspace(0, 10, 200)
    true_func = lambda x: np.sin(x) + 0.5 * np.sin(3 * x) - 0.1 * (x - 5) ** 2 + 5
    y_true = true_func(x)

    # Initial observations
    np.random.seed(42)
    observed_x = [1.0, 4.0, 8.0]
    observed_y = [true_func(xi) + np.random.normal(0, 0.1) for xi in observed_x]

    frames = []

    def gaussian_process_predict(x_new, x_obs, y_obs):
        """Simple GP prediction (not actual GP, just for visualization)."""
        mean = np.zeros_like(x_new)
        std = np.ones_like(x_new) * 2

        for xi, yi in zip(x_obs, y_obs):
            dist = np.abs(x_new - xi)
            weight = np.exp(-dist ** 2 / 2)
            mean += weight * yi
            std *= (1 - weight * 0.8)

        mean /= (len(x_obs) + 0.1)
        mean += 3  # Baseline
        std = np.maximum(std, 0.2)
        return mean, std

    def expected_improvement(x_new, x_obs, y_obs, xi=0.01):
        """Calculate Expected Improvement acquisition function."""
        mean, std = gaussian_process_predict(x_new, x_obs, y_obs)
        best_y = max(y_obs)

        z = (mean - best_y - xi) / (std + 1e-9)
        ei = (mean - best_y - xi) * norm.cdf(z) + std * norm.pdf(z)
        ei = np.maximum(ei, 0)
        return ei, mean, std

    def update(frame):
        nonlocal observed_x, observed_y

        ax1.clear()
        ax2.clear()

        # Calculate GP prediction and EI
        ei, mean, std = expected_improvement(x, observed_x, observed_y)

        # Upper plot: Objective function and GP
        ax1.plot(x, y_true, 'k--', label='True Objective (unknown)', alpha=0.5, linewidth=2)
        ax1.plot(x, mean, 'b-', label='GP Mean (surrogate)', linewidth=2)
        ax1.fill_between(x, mean - 2*std, mean + 2*std, alpha=0.2, color='blue', label='Uncertainty (2 std)')
        ax1.scatter(observed_x, observed_y, s=150, c='red', edgecolors='black',
                   linewidth=2, zorder=5, label='Observations')

        if frame < 5:  # Add new point based on EI
            next_x = x[np.argmax(ei)]
            ax1.axvline(next_x, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Next sample')

        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 8)
        ax1.set_xlabel('Hyperparameter Value', fontsize=11)
        ax1.set_ylabel('Objective (e.g., Accuracy)', fontsize=11)
        ax1.set_title(f'Bayesian Optimization - Iteration {len(observed_x) - 2}', fontsize=13)
        ax1.legend(loc='lower left', fontsize=9)

        # Lower plot: Acquisition function
        ax2.fill_between(x, 0, ei, alpha=0.4, color='green')
        ax2.plot(x, ei, 'g-', linewidth=2)

        if frame < 5:
            next_x = x[np.argmax(ei)]
            ax2.axvline(next_x, color='green', linestyle='--', linewidth=2)
            ax2.scatter([next_x], [np.max(ei)], s=100, c='green', edgecolors='black', linewidth=2, zorder=5)

        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, max(ei) * 1.2 + 0.1)
        ax2.set_xlabel('Hyperparameter Value', fontsize=11)
        ax2.set_ylabel('Expected Improvement', fontsize=11)
        ax2.set_title('Acquisition Function (where to sample next)', fontsize=11)

        plt.tight_layout()

        # Add new observation for next frame
        if frame < 5:
            next_x = x[np.argmax(ei)]
            observed_x = observed_x + [next_x]
            observed_y = observed_y + [true_func(next_x) + np.random.normal(0, 0.1)]

    # Create animation
    ani = FuncAnimation(fig, update, frames=6, interval=1500, repeat=True)
    ani.save(os.path.join(OUTPUT_DIR, 'bayesian_optimization.gif'),
             writer=PillowWriter(fps=0.7), dpi=100)
    plt.close()
    print("Created: bayesian_optimization.gif")


def create_learning_rate_schedules():
    """Compare different learning rate scheduling strategies."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    epochs = np.arange(0, 100)
    initial_lr = 0.1

    # Step Decay (reduce by 0.1x at epochs 30 and 60)
    step_decay = np.where(epochs < 30, initial_lr,
                         np.where(epochs < 60, initial_lr * 0.1, initial_lr * 0.01))

    # Exponential Decay
    exp_decay = initial_lr * np.exp(-0.05 * epochs)

    # Cosine Annealing
    cosine = initial_lr * 0.5 * (1 + np.cos(np.pi * epochs / 100))

    # Warmup + Linear Decay
    warmup_epochs = 10
    warmup_decay = np.where(epochs < warmup_epochs,
                            initial_lr * epochs / warmup_epochs,
                            initial_lr * (1 - (epochs - warmup_epochs) / (100 - warmup_epochs)))
    warmup_decay = np.maximum(warmup_decay, 0.001)

    # Plot all schedules
    ax.plot(epochs, step_decay, 'b-', linewidth=2.5, label='Step Decay (at 30, 60)')
    ax.plot(epochs, exp_decay, 'r-', linewidth=2.5, label='Exponential Decay')
    ax.plot(epochs, cosine, 'g-', linewidth=2.5, label='Cosine Annealing')
    ax.plot(epochs, warmup_decay, 'm-', linewidth=2.5, label='Warmup + Linear Decay')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule Comparison', fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, initial_lr * 1.1)
    ax.legend(loc='upper right', fontsize=10)

    # Add annotations
    ax.annotate('Warmup\nphase', xy=(5, 0.05), fontsize=9, ha='center', color='purple')
    ax.annotate('Step drops', xy=(30, 0.015), fontsize=9, ha='center', color='blue')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'learning_rate_schedules.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: learning_rate_schedules.png")


def create_cross_validation_diagram():
    """Create a diagram showing K-fold cross-validation splits."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    k_folds = 5
    colors = ['#4ECDC4', '#FF6B6B']  # Train, Validation

    # Draw fold boxes
    box_height = 0.6
    gap = 0.15

    for fold in range(k_folds):
        y_pos = (k_folds - 1 - fold) * (box_height + gap)

        for block in range(k_folds):
            x_start = block * 1.2

            if block == fold:
                # Validation fold
                color = colors[1]
                label = 'Val' if fold == 0 and block == 0 else ''
            else:
                # Training fold
                color = colors[0]
                label = 'Train' if fold == 0 and block == 1 else ''

            rect = FancyBboxPatch((x_start, y_pos), 1.0, box_height,
                                  boxstyle="round,pad=0.02,rounding_size=0.1",
                                  facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

        # Add fold label
        ax.text(-0.4, y_pos + box_height/2, f'Fold {fold + 1}',
                fontsize=11, va='center', ha='right', fontweight='bold')

    # Add legend
    train_patch = plt.Rectangle((0, 0), 1, 1, facecolor=colors[0], edgecolor='black', linewidth=1.5)
    val_patch = plt.Rectangle((0, 0), 1, 1, facecolor=colors[1], edgecolor='black', linewidth=1.5)
    ax.legend([train_patch, val_patch], ['Training Data', 'Validation Data'],
              loc='upper right', fontsize=10)

    # Add block labels
    for i in range(k_folds):
        ax.text(i * 1.2 + 0.5, -0.4, f'Block {i + 1}', ha='center', fontsize=10)

    ax.set_xlim(-0.8, k_folds * 1.2 + 0.2)
    ax.set_ylim(-0.8, k_folds * (box_height + gap))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('K-Fold Cross-Validation (K=5)\nEach fold takes turns as validation set',
                 fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cross_validation_folds.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: cross_validation_folds.png")


def create_validation_curve():
    """Create validation curve showing underfitting/overfitting regions."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Hyperparameter values (e.g., model complexity / polynomial degree)
    hyperparam = np.linspace(1, 15, 100)

    # Simulated scores
    np.random.seed(42)

    # Training score (keeps improving with complexity)
    train_score = 0.5 + 0.4 * (1 - np.exp(-0.5 * hyperparam))
    train_score = np.minimum(train_score, 0.99)

    # Validation score (improves then degrades due to overfitting)
    val_score = 0.5 + 0.35 * (1 - np.exp(-0.4 * hyperparam)) - 0.02 * (hyperparam - 5) ** 2
    val_score = np.maximum(val_score, 0.3)
    val_score = np.clip(val_score, 0.3, 0.9)

    # Find optimal point
    optimal_idx = np.argmax(val_score)
    optimal_x = hyperparam[optimal_idx]
    optimal_y = val_score[optimal_idx]

    # Plot curves
    ax.plot(hyperparam, train_score, 'b-', linewidth=2.5, label='Training Score')
    ax.plot(hyperparam, val_score, 'r-', linewidth=2.5, label='Validation Score')

    # Add confidence bands
    ax.fill_between(hyperparam, train_score - 0.02, train_score + 0.02, alpha=0.2, color='blue')
    ax.fill_between(hyperparam, val_score - 0.05, val_score + 0.05, alpha=0.2, color='red')

    # Mark optimal point
    ax.scatter([optimal_x], [optimal_y], s=150, c='green', edgecolors='black',
               linewidth=2, zorder=5, label=f'Optimal (x={optimal_x:.1f})')
    ax.axvline(optimal_x, color='green', linestyle='--', alpha=0.5, linewidth=2)

    # Add region annotations
    ax.axvspan(1, 4, alpha=0.1, color='orange')
    ax.axvspan(8, 15, alpha=0.1, color='purple')

    ax.text(2.5, 0.4, 'Underfitting\n(High Bias)', fontsize=11, ha='center',
            color='darkorange', fontweight='bold')
    ax.text(11.5, 0.4, 'Overfitting\n(High Variance)', fontsize=11, ha='center',
            color='purple', fontweight='bold')

    ax.set_xlabel('Model Complexity (Hyperparameter Value)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Validation Curve: Finding Optimal Model Complexity', fontsize=14)
    ax.set_xlim(1, 15)
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'validation_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: validation_curve.png")


if __name__ == '__main__':
    print("Generating hyperparameter tuning visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    create_grid_search_heatmap()
    create_grid_vs_random_comparison()
    create_bayesian_optimization_gif()
    create_learning_rate_schedules()
    create_cross_validation_diagram()
    create_validation_curve()

    print("-" * 50)
    print("All visualizations generated successfully!")
