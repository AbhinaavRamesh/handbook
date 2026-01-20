"""
Feature Scaling Visualizations
Generates educational plots for feature scaling concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Output directory
OUTPUT_DIR = Path(__file__).parent


def generate_scaling_comparison():
    """
    Create 2x2 comparison of scaling methods.
    Shows original data vs StandardScaler, MinMaxScaler, RobustScaler.
    """
    np.random.seed(42)

    # Generate data with different scales and an outlier
    n_samples = 200
    feature1 = np.random.normal(100, 15, n_samples)  # Large values
    feature2 = np.random.normal(5, 2, n_samples)     # Small values

    # Add outliers
    feature1 = np.append(feature1, [200, 210])
    feature2 = np.append(feature2, [15, 18])

    X = np.column_stack([feature1, feature2])

    # Scaling methods
    def standard_scale(X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X - mean) / std

    def minmax_scale(X):
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        return (X - X_min) / (X_max - X_min)

    def robust_scale(X):
        median = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        return (X - median) / iqr

    X_standard = standard_scale(X)
    X_minmax = minmax_scale(X)
    X_robust = robust_scale(X)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=150)

    datasets = [
        (X, 'Original Data', 'Vastly different scales'),
        (X_standard, 'StandardScaler', 'Mean=0, Std=1'),
        (X_minmax, 'MinMaxScaler', 'Range [0, 1]'),
        (X_robust, 'RobustScaler', 'Median=0, IQR scaled')
    ]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))

    for ax, (data, title, subtitle), color in zip(axes.flat, datasets, colors):
        # Plot points (regular and outliers)
        ax.scatter(data[:-2, 0], data[:-2, 1], alpha=0.6, s=30,
                   c=[color], edgecolors='white', linewidth=0.5)
        ax.scatter(data[-2:, 0], data[-2:, 1], alpha=0.9, s=80,
                   c='red', marker='*', label='Outliers', edgecolors='darkred')

        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

        # Add statistics as text
        stats_text = f'F1: [{data[:,0].min():.1f}, {data[:,0].max():.1f}]\nF2: [{data[:,1].min():.1f}, {data[:,1].max():.1f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scaling_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: scaling_comparison.png")


def generate_gradient_descent_scaling_gif():
    """
    Create animated GIF showing gradient descent with vs without scaling.
    Shows contour plots with optimization trajectories.
    """
    np.random.seed(42)

    # Create a simple quadratic loss function with different scales
    # Unscaled: elongated ellipse
    # Scaled: circular contours

    def loss_unscaled(w1, w2):
        return 0.1 * (w1 - 3)**2 + 10 * (w2 - 2)**2

    def grad_unscaled(w1, w2):
        return np.array([0.2 * (w1 - 3), 20 * (w2 - 2)])

    def loss_scaled(w1, w2):
        return (w1 - 3)**2 + (w2 - 2)**2

    def grad_scaled(w1, w2):
        return np.array([2 * (w1 - 3), 2 * (w2 - 2)])

    # Gradient descent
    def gradient_descent(grad_fn, start, lr, n_steps):
        path = [start.copy()]
        w = start.copy()
        for _ in range(n_steps):
            w = w - lr * grad_fn(w[0], w[1])
            path.append(w.copy())
        return np.array(path)

    # Run gradient descent
    start = np.array([-2.0, 5.0])
    n_steps = 50

    path_unscaled = gradient_descent(grad_unscaled, start, lr=0.04, n_steps=n_steps)
    path_scaled = gradient_descent(grad_scaled, start, lr=0.3, n_steps=n_steps)

    # Create animation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Create contour grids
    w1_range = np.linspace(-4, 8, 100)
    w2_range = np.linspace(-1, 7, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)

    Z_unscaled = loss_unscaled(W1, W2)
    Z_scaled = loss_scaled(W1, W2)

    # Static contours
    ax1, ax2 = axes

    levels_unscaled = np.logspace(-1, 2.5, 20)
    levels_scaled = np.logspace(-1, 2, 20)

    ax1.contour(W1, W2, Z_unscaled, levels=levels_unscaled, cmap='Blues', alpha=0.7)
    ax1.contourf(W1, W2, Z_unscaled, levels=levels_unscaled, cmap='Blues', alpha=0.3)
    ax1.plot(3, 2, 'r*', markersize=15, label='Optimum')
    ax1.set_xlabel('$w_1$', fontsize=12)
    ax1.set_ylabel('$w_2$', fontsize=12)
    ax1.set_title('Without Feature Scaling\n(Elongated contours = slow convergence)', fontsize=11, fontweight='bold')

    ax2.contour(W1, W2, Z_scaled, levels=levels_scaled, cmap='Greens', alpha=0.7)
    ax2.contourf(W1, W2, Z_scaled, levels=levels_scaled, cmap='Greens', alpha=0.3)
    ax2.plot(3, 2, 'r*', markersize=15, label='Optimum')
    ax2.set_xlabel('$w_1$', fontsize=12)
    ax2.set_ylabel('$w_2$', fontsize=12)
    ax2.set_title('With Feature Scaling\n(Circular contours = fast convergence)', fontsize=11, fontweight='bold')

    # Initialize trajectory lines
    line1, = ax1.plot([], [], 'r.-', lw=2, markersize=8, alpha=0.8)
    line2, = ax2.plot([], [], 'r.-', lw=2, markersize=8, alpha=0.8)
    point1, = ax1.plot([], [], 'ko', markersize=10)
    point2, = ax2.plot([], [], 'ko', markersize=10)

    step_text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    step_text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=10,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        point1.set_data([], [])
        point2.set_data([], [])
        step_text1.set_text('')
        step_text2.set_text('')
        return line1, line2, point1, point2, step_text1, step_text2

    def animate(i):
        idx = min(i, len(path_unscaled) - 1)
        idx2 = min(i, len(path_scaled) - 1)

        line1.set_data(path_unscaled[:idx+1, 0], path_unscaled[:idx+1, 1])
        line2.set_data(path_scaled[:idx2+1, 0], path_scaled[:idx2+1, 1])

        point1.set_data([path_unscaled[idx, 0]], [path_unscaled[idx, 1]])
        point2.set_data([path_scaled[idx2, 0]], [path_scaled[idx2, 1]])

        loss1 = loss_unscaled(path_unscaled[idx, 0], path_unscaled[idx, 1])
        loss2 = loss_scaled(path_scaled[idx2, 0], path_scaled[idx2, 1])

        step_text1.set_text(f'Step: {idx}\nLoss: {loss1:.2f}')
        step_text2.set_text(f'Step: {idx2}\nLoss: {loss2:.4f}')

        return line1, line2, point1, point2, step_text1, step_text2

    plt.tight_layout()

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_steps+10,
                                   interval=100, blit=True)
    anim.save(OUTPUT_DIR / 'gradient_descent_scaling.gif', writer='pillow', fps=10)
    plt.close()
    print("Generated: gradient_descent_scaling.gif")


def generate_distribution_transforms():
    """
    Create histograms showing distributions before and after scaling.
    """
    np.random.seed(42)

    # Generate different types of data
    normal_data = np.random.normal(50, 10, 500)
    skewed_data = np.random.exponential(10, 500) + 20
    bimodal_data = np.concatenate([np.random.normal(30, 5, 250),
                                   np.random.normal(70, 5, 250)])

    datasets = [
        (normal_data, 'Normal Distribution'),
        (skewed_data, 'Skewed Distribution'),
        (bimodal_data, 'Bimodal Distribution')
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), dpi=150)

    def standard_scale(x):
        return (x - x.mean()) / x.std()

    def minmax_scale(x):
        return (x - x.min()) / (x.max() - x.min())

    scalers = [
        (lambda x: x, 'Original'),
        (standard_scale, 'StandardScaler'),
        (minmax_scale, 'MinMaxScaler')
    ]

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, (data, data_name) in enumerate(datasets):
        for j, (scaler_fn, scaler_name) in enumerate(scalers):
            ax = axes[i, j]
            scaled_data = scaler_fn(data)

            ax.hist(scaled_data, bins=30, color=colors[j], alpha=0.7, edgecolor='white')

            # Add vertical lines for mean and median
            ax.axvline(scaled_data.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {scaled_data.mean():.2f}')
            ax.axvline(np.median(scaled_data), color='orange', linestyle=':', lw=2, label=f'Median: {np.median(scaled_data):.2f}')

            if i == 0:
                ax.set_title(scaler_name, fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(data_name, fontsize=11)

            # Add stats
            stats = f'$\\mu$={scaled_data.mean():.2f}\n$\\sigma$={scaled_data.std():.2f}'
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.legend(loc='upper left', fontsize=8)

    plt.suptitle('Distribution Shape After Scaling', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribution_transforms.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: distribution_transforms.png")


def generate_outlier_robustness():
    """
    Create visualization showing RobustScaler advantage with outliers.
    """
    np.random.seed(42)

    # Generate data with outliers
    n_samples = 100
    clean_data = np.random.normal(50, 5, n_samples)

    # Add outliers
    outliers = np.array([150, 160, 170, -20, -30])
    data_with_outliers = np.concatenate([clean_data, outliers])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    # Standard scaling
    def standard_scale(x):
        return (x - x.mean()) / x.std()

    # Robust scaling
    def robust_scale(x):
        median = np.median(x)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        return (x - median) / iqr

    # MinMax scaling
    def minmax_scale(x):
        return (x - x.min()) / (x.max() - x.min())

    # Original data
    ax1 = axes[0]
    ax1.scatter(range(len(data_with_outliers[:-5])), data_with_outliers[:-5],
                c='#3498db', alpha=0.6, s=50, label='Normal points')
    ax1.scatter(range(len(data_with_outliers)-5, len(data_with_outliers)),
                data_with_outliers[-5:], c='red', s=100, marker='*', label='Outliers')
    ax1.axhline(np.mean(data_with_outliers), color='blue', linestyle='--', lw=2,
                label=f'Mean: {np.mean(data_with_outliers):.1f}')
    ax1.axhline(np.median(data_with_outliers), color='green', linestyle=':', lw=2,
                label=f'Median: {np.median(data_with_outliers):.1f}')
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Original Data with Outliers', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # StandardScaler vs RobustScaler comparison
    ax2 = axes[1]

    standard_scaled = standard_scale(data_with_outliers)
    robust_scaled = robust_scale(data_with_outliers)

    x_positions = np.arange(len(data_with_outliers))
    width = 0.4

    ax2.bar(x_positions[:len(clean_data)] - width/2, standard_scaled[:len(clean_data)],
            width, alpha=0.7, color='#e74c3c', label='StandardScaler')
    ax2.bar(x_positions[:len(clean_data)] + width/2, robust_scaled[:len(clean_data)],
            width, alpha=0.7, color='#2ecc71', label='RobustScaler')

    # Highlight outlier effect
    ax2.axhline(0, color='black', linestyle='-', lw=0.5)
    ax2.axhline(1, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax2.axhline(-1, color='gray', linestyle='--', lw=1, alpha=0.5)

    ax2.set_xlabel('Sample Index (normal points only)', fontsize=11)
    ax2.set_ylabel('Scaled Value', fontsize=11)
    ax2.set_title('Scaling Effect on Normal Points\n(Outliers compressed normal data differently)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-5, len(clean_data) + 5)

    # Box plot comparison
    ax3 = axes[2]

    scaling_methods = {
        'Original': data_with_outliers,
        'StandardScaler': standard_scaled,
        'MinMaxScaler': minmax_scale(data_with_outliers),
        'RobustScaler': robust_scaled
    }

    bp = ax3.boxplot([v for v in scaling_methods.values()],
                      tick_labels=list(scaling_methods.keys()),
                      patch_artist=True)

    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_ylabel('Value', fontsize=11)
    ax3.set_title('Distribution Comparison\n(RobustScaler maintains better spread)', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)

    # Add annotation
    ax3.annotate('Outliers affect\nmean/std heavily',
                 xy=(2, scaling_methods['StandardScaler'].max()),
                 xytext=(2.5, scaling_methods['StandardScaler'].max() * 0.8),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'outlier_robustness.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: outlier_robustness.png")


def generate_scaler_decision_flowchart():
    """
    Create a visual decision flowchart for choosing the right scaler.
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, text, color, text_color='white', fontsize=10):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.03,rounding_size=0.2",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color, wrap=True)

    def draw_diamond(ax, x, y, size, text, color):
        diamond = plt.Polygon([(x, y + size), (x + size, y), (x, y - size), (x - size, y)],
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', wrap=True)

    def draw_arrow(ax, start, end, text='', text_pos=0.5):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        if text:
            mid_x = start[0] + (end[0] - start[0]) * text_pos
            mid_y = start[1] + (end[1] - start[1]) * text_pos
            ax.text(mid_x + 0.3, mid_y, text, fontsize=9, fontweight='bold')

    # Start
    draw_box(ax, 6, 7.5, 3, 0.6, 'Choose Scaler', '#2c3e50', fontsize=12)

    # First decision: Sparse data?
    draw_diamond(ax, 6, 6, 0.6, 'Sparse\nData?', '#e74c3c')
    draw_arrow(ax, (6, 7.2), (6, 6.6))

    # MaxAbsScaler
    draw_box(ax, 9.5, 6, 2.2, 0.8, 'MaxAbsScaler\n(preserves zeros)', '#27ae60')
    draw_arrow(ax, (6.6, 6), (8.4, 6), 'Yes')

    # Second decision: Outliers?
    draw_diamond(ax, 6, 4.5, 0.6, 'Outliers?', '#e74c3c')
    draw_arrow(ax, (6, 5.4), (6, 5.1), 'No')

    # RobustScaler
    draw_box(ax, 9.5, 4.5, 2.2, 0.8, 'RobustScaler\n(median/IQR)', '#27ae60')
    draw_arrow(ax, (6.6, 4.5), (8.4, 4.5), 'Yes')

    # Third decision: Bounded range needed?
    draw_diamond(ax, 6, 3, 0.7, 'Bounded\nRange?', '#e74c3c')
    draw_arrow(ax, (6, 3.9), (6, 3.7), 'No')

    # MinMaxScaler
    draw_box(ax, 9.5, 3, 2.2, 0.8, 'MinMaxScaler\n([0, 1] range)', '#27ae60')
    draw_arrow(ax, (6.7, 3), (8.4, 3), 'Yes')

    # Default: StandardScaler
    draw_box(ax, 6, 1.5, 2.5, 0.8, 'StandardScaler\n(mean=0, std=1)', '#27ae60')
    draw_arrow(ax, (6, 2.3), (6, 2.3), 'No')

    # Side notes
    ax.text(2, 6, 'For text/frequency\ndata, consider\nNormalizer (L1/L2)',
            fontsize=9, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.text(2, 3, 'Neural Networks:\nMinMaxScaler or\nStandardScaler',
            fontsize=9, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.text(2, 1.5, 'Tree-based models:\nScaling often\nnot required',
            fontsize=9, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.title('Feature Scaler Selection Guide', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scaler_selection_guide.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: scaler_selection_guide.png")


def generate_fit_transform_diagram():
    """
    Create diagram showing correct fit/transform workflow.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(6, 5.7, 'Fit/Transform Pattern: Correct vs Incorrect',
            ha='center', fontsize=14, fontweight='bold')

    # Correct workflow (left side)
    ax.text(3, 5.2, 'CORRECT', ha='center', fontsize=12, fontweight='bold', color='green')

    # Boxes
    def draw_box(x, y, w, h, text, color, alpha=1.0):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='black', linewidth=1.5, alpha=alpha)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, wrap=True)

    # Training data
    draw_box(3, 4.5, 2, 0.6, 'Train Data', '#3498db')

    # fit_transform
    draw_box(3, 3.5, 2.2, 0.6, 'fit_transform()\nLearn stats + scale', '#2ecc71')
    ax.annotate('', xy=(3, 3.8), xytext=(3, 4.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Scaled train
    draw_box(3, 2.5, 2, 0.6, 'Scaled Train', '#9b59b6')
    ax.annotate('', xy=(3, 2.8), xytext=(3, 3.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Test data
    draw_box(5, 4.5, 1.6, 0.6, 'Test Data', '#e74c3c')

    # transform only
    draw_box(5, 3.5, 1.8, 0.6, 'transform()\nUse train stats', '#f39c12')
    ax.annotate('', xy=(5, 3.8), xytext=(5, 4.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Arrow from scaler to test transform
    ax.annotate('', xy=(4.1, 3.5), xytext=(3.9, 3.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='--'))

    # Scaled test
    draw_box(5, 2.5, 1.8, 0.6, 'Scaled Test', '#9b59b6')
    ax.annotate('', xy=(5, 2.8), xytext=(5, 3.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Stats box
    draw_box(1.2, 3.5, 1.2, 0.8, 'Stats:\nmean, std', 'lightyellow')
    ax.annotate('', xy=(1.8, 3.5), xytext=(1.9, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))

    # Incorrect workflow (right side)
    ax.text(9, 5.2, 'INCORRECT (Data Leakage!)', ha='center', fontsize=12,
            fontweight='bold', color='red')

    # All data combined
    draw_box(9, 4.5, 2.5, 0.6, 'All Data\n(Train + Test)', '#e74c3c')

    # fit_transform on all
    draw_box(9, 3.5, 2.5, 0.6, 'fit_transform()\nLearns from ALL data', '#e74c3c', alpha=0.7)
    ax.annotate('', xy=(9, 3.8), xytext=(9, 4.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Warning
    ax.text(9, 2.5, 'Test data influences\ntraining statistics!',
            ha='center', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='mistyrose', edgecolor='red'))

    # Separator
    ax.axvline(6.5, color='gray', linestyle='--', lw=2, alpha=0.5)

    # Key points at bottom
    ax.text(3, 1.2, 'Key: Statistics computed\nonly from training data',
            ha='center', fontsize=10, color='green',
            bbox=dict(boxstyle='round', facecolor='honeydew', edgecolor='green'))

    ax.text(9, 1.2, 'Problem: Test info leaks\ninto training process',
            ha='center', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='mistyrose', edgecolor='red'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fit_transform_workflow.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fit_transform_workflow.png")


if __name__ == '__main__':
    print("Generating Feature Scaling Visualizations...")
    print("=" * 50)

    generate_scaling_comparison()
    generate_gradient_descent_scaling_gif()
    generate_distribution_transforms()
    generate_outlier_robustness()
    generate_scaler_decision_flowchart()
    generate_fit_transform_diagram()

    print("=" * 50)
    print("All visualizations generated successfully!")
