"""
Feature Engineering Visualizations for Interview FAQ
Generates matplotlib visualizations for key feature engineering concepts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch
import os

# Set consistent styling
plt.style.use('seaborn-v0_8-whitegrid')

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_correlation_heatmap():
    """Create a triangular correlation heatmap visualization."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Create sample correlation matrix
    np.random.seed(42)
    features = ['age', 'income', 'debt', 'credit_score', 'years_employed', 'num_accounts']
    n = len(features)

    # Generate a realistic correlation matrix
    corr = np.array([
        [1.00, 0.45, 0.12, 0.65, 0.72, 0.23],
        [0.45, 1.00, 0.35, 0.78, 0.55, 0.41],
        [0.12, 0.35, 1.00, -0.62, -0.15, 0.68],
        [0.65, 0.78, -0.62, 1.00, 0.48, -0.22],
        [0.72, 0.55, -0.15, 0.48, 1.00, 0.11],
        [0.23, 0.41, 0.68, -0.22, 0.11, 1.00]
    ])

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot only lower triangle
    masked_corr = np.ma.array(corr, mask=mask)

    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)

    # Mask upper triangle with white
    for i in range(n):
        for j in range(i+1, n):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='white'))

    # Add annotations in lower triangle only
    for i in range(n):
        for j in range(i+1):
            color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold')

    # Customize
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_title('Feature Correlation Heatmap (Triangular Matrix)', fontsize=14, fontweight='bold', pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=11)

    # Add note about multicollinearity
    ax.annotate('High correlation (>0.7)\nindicates multicollinearity',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlation_heatmap.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: feature_correlation_heatmap.png")


def create_log_transformation():
    """Create visualization showing log transformation effect on skewed data."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 6), dpi=150)

    # Generate skewed data (income-like distribution)
    np.random.seed(42)
    original = np.random.exponential(scale=50000, size=2000)
    original = original[original < 300000]  # Cap at 300k

    # Log transform
    log_transformed = np.log1p(original)

    # Colors
    original_color = '#e74c3c'
    transformed_color = '#27ae60'

    # Plot 1: Original distribution
    axes[0].hist(original, bins=40, color=original_color, alpha=0.7, edgecolor='white')
    axes[0].set_title('Original (Skewed)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Income ($)', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].axvline(np.mean(original), color='darkred', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(original):,.0f}')
    axes[0].axvline(np.median(original), color='darkred', linestyle=':', linewidth=2, label=f'Median: ${np.median(original):,.0f}')
    axes[0].legend(fontsize=8)
    skewness_orig = pd.Series(original).skew()
    axes[0].annotate(f'Skewness: {skewness_orig:.2f}', xy=(0.65, 0.85), xycoords='axes fraction',
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Arrow showing transformation
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].arrow(2, 5, 5, 0, head_width=0.5, head_length=0.3, fc='#3498db', ec='#3498db', linewidth=3)
    axes[1].text(5, 6.5, 'log(1 + x)', fontsize=14, ha='center', fontweight='bold', color='#3498db')
    axes[1].text(5, 3.5, 'Log Transform', fontsize=12, ha='center', style='italic')
    axes[1].axis('off')
    axes[1].set_title('Transformation', fontsize=12, fontweight='bold')

    # Plot 3: Transformed distribution
    axes[2].hist(log_transformed, bins=40, color=transformed_color, alpha=0.7, edgecolor='white')
    axes[2].set_title('Log Transformed (Normal-like)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('log(1 + Income)', fontsize=10)
    axes[2].set_ylabel('Frequency', fontsize=10)
    axes[2].axvline(np.mean(log_transformed), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {np.mean(log_transformed):.2f}')
    axes[2].axvline(np.median(log_transformed), color='darkgreen', linestyle=':', linewidth=2, label=f'Median: {np.median(log_transformed):.2f}')
    axes[2].legend(fontsize=8)
    skewness_trans = pd.Series(log_transformed).skew()
    axes[2].annotate(f'Skewness: {skewness_trans:.2f}', xy=(0.65, 0.85), xycoords='axes fraction',
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Log Transformation: Normalizing Skewed Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'log_transformation.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: log_transformation.png")


def create_encoding_comparison():
    """Create visualization comparing different encoding methods."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150)

    # Sample data
    categories = ['Red', 'Blue', 'Green', 'Yellow']

    # Plot 1: One-Hot Encoding
    ax1 = axes[0, 0]
    one_hot_data = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    im1 = ax1.imshow(one_hot_data, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(['is_Red', 'is_Blue', 'is_Green', 'is_Yellow'], rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.set_title('One-Hot Encoding', fontsize=11, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, str(one_hot_data[i, j]), ha='center', va='center',
                    color='white' if one_hot_data[i, j] == 1 else 'black', fontsize=10)
    ax1.set_ylabel('Original Category', fontsize=10)

    # Plot 2: Ordinal Encoding
    ax2 = axes[0, 1]
    ordinal_values = [0, 1, 2, 3]
    colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12']
    bars = ax2.barh(categories, ordinal_values, color=colors, edgecolor='white', linewidth=2)
    ax2.set_xlabel('Encoded Value', fontsize=10)
    ax2.set_title('Ordinal/Label Encoding', fontsize=11, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, ordinal_values)):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 4.5)
    ax2.annotate('Implies ordering\n(may be incorrect)', xy=(0.6, 0.15), xycoords='axes fraction',
                fontsize=8, style='italic', color='gray')

    # Plot 3: Target Encoding
    ax3 = axes[1, 0]
    target_values = [0.72, 0.45, 0.58, 0.31]  # Example target means
    bars3 = ax3.barh(categories, target_values, color=colors, edgecolor='white', linewidth=2)
    ax3.set_xlabel('Mean Target Value', fontsize=10)
    ax3.set_title('Target Encoding', fontsize=11, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars3, target_values)):
        ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10, fontweight='bold')
    ax3.set_xlim(0, 1.0)
    ax3.annotate('Risk: Data leakage\nUse CV encoding', xy=(0.55, 0.15), xycoords='axes fraction',
                fontsize=8, style='italic', color='#c0392b')

    # Plot 4: Comparison table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = [
        ['Method', 'Pros', 'Cons'],
        ['One-Hot', 'No ordering assumed', 'High dimensionality'],
        ['Ordinal', 'Single column', 'Implies false order'],
        ['Target', 'Captures relationship', 'Leakage risk']
    ]

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.4, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(3):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style alternating rows
    for i in range(1, 4):
        for j in range(3):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

    ax4.set_title('Encoding Methods Comparison', fontsize=11, fontweight='bold', pad=20)

    plt.suptitle('Categorical Variable Encoding Methods', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'encoding_comparison.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: encoding_comparison.png")


def create_polynomial_features():
    """Create visualization of polynomial feature expansion."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 5, 100)
    y_true = 0.5 * x**2 - 2*x + 3 + np.random.normal(0, 0.5, 100)

    # Plot 1: Linear vs Polynomial fit
    ax1 = axes[0]
    ax1.scatter(x, y_true, alpha=0.5, s=30, color='#3498db', label='Data points')

    # Linear fit
    z_linear = np.polyfit(x, y_true, 1)
    p_linear = np.poly1d(z_linear)
    ax1.plot(x, p_linear(x), 'r--', linewidth=2, label=f'Linear: y = {z_linear[0]:.2f}x + {z_linear[1]:.2f}')

    # Quadratic fit
    z_quad = np.polyfit(x, y_true, 2)
    p_quad = np.poly1d(z_quad)
    ax1.plot(x, p_quad(x), 'g-', linewidth=2, label=f'Poly(2): y = {z_quad[0]:.2f}x\u00b2 + {z_quad[1]:.2f}x + {z_quad[2]:.2f}')

    ax1.set_xlabel('Feature x', fontsize=11)
    ax1.set_ylabel('Target y', fontsize=11)
    ax1.set_title('Polynomial Feature Expansion', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')

    # Calculate R-squared values manually
    ss_res_linear = np.sum((y_true - p_linear(x))**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2_linear = 1 - (ss_res_linear / ss_tot)

    ss_res_quad = np.sum((y_true - p_quad(x))**2)
    r2_quad = 1 - (ss_res_quad / ss_tot)

    ax1.annotate(f'R\u00b2 Linear: {r2_linear:.3f}\nR\u00b2 Poly(2): {r2_quad:.3f}',
                xy=(0.65, 0.05), xycoords='axes fraction',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Plot 2: Feature expansion diagram
    ax2 = axes[1]
    ax2.axis('off')

    # Draw feature expansion
    original_box = plt.Rectangle((0.1, 0.6), 0.25, 0.25, fill=True, facecolor='#3498db',
                                   edgecolor='#2c3e50', linewidth=2, alpha=0.8)
    ax2.add_patch(original_box)
    ax2.text(0.225, 0.725, 'Original\n[x\u2081, x\u2082]', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Arrow
    ax2.annotate('', xy=(0.55, 0.725), xytext=(0.38, 0.725),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax2.text(0.465, 0.8, 'Polynomial\ndegree=2', ha='center', va='center', fontsize=9)

    # Expanded box
    expanded_box = plt.Rectangle((0.55, 0.5), 0.4, 0.45, fill=True, facecolor='#27ae60',
                                   edgecolor='#2c3e50', linewidth=2, alpha=0.8)
    ax2.add_patch(expanded_box)
    ax2.text(0.75, 0.725, 'Expanded\n[1, x\u2081, x\u2082, x\u2081\u00b2,\nx\u2081x\u2082, x\u2082\u00b2]',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Formula
    ax2.text(0.5, 0.3, 'n features \u2192 (n+d)! / (n! \u00d7 d!) terms',
            ha='center', va='center', fontsize=11, style='italic')
    ax2.text(0.5, 0.15, 'Example: 2 features, degree 2 = 6 terms',
            ha='center', va='center', fontsize=10)

    # Warning
    ax2.text(0.5, 0.02, 'Warning: Features grow combinatorially!',
            ha='center', va='center', fontsize=10, color='#c0392b', fontweight='bold')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Feature Expansion Process', fontsize=12, fontweight='bold')

    plt.suptitle('Polynomial Features: Capturing Non-Linear Relationships', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'polynomial_features.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: polynomial_features.png")


def create_feature_selection_venn():
    """Create Venn-like diagram for feature selection methods using circles."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Define circle positions and radius for a 3-circle Venn
    r = 1.5
    # Centers arranged in a triangle formation
    centers = [
        (-0.8, 0.5),   # Filter (left)
        (0.8, 0.5),    # Wrapper (right)
        (0, -0.8)      # Embedded (bottom)
    ]
    colors = ['#e74c3c', '#3498db', '#27ae60']
    labels = ['Filter\nMethods', 'Wrapper\nMethods', 'Embedded\nMethods']

    # Draw circles
    for i, (center, color, label) in enumerate(zip(centers, colors, labels)):
        circle = Circle(center, r, fill=True, facecolor=color, alpha=0.35,
                       edgecolor=color, linewidth=2)
        ax.add_patch(circle)

    # Add labels outside circles
    ax.text(-2.2, 1.5, 'Filter\nMethods', fontsize=12, fontweight='bold',
           color='#c0392b', ha='center')
    ax.text(2.2, 1.5, 'Wrapper\nMethods', fontsize=12, fontweight='bold',
           color='#2980b9', ha='center')
    ax.text(0, -2.8, 'Embedded\nMethods', fontsize=12, fontweight='bold',
           color='#27ae60', ha='center')

    # Add content in each region
    # Filter only (left)
    ax.text(-1.8, 0.6, 'Fast\nModel-agnostic\nChi-sq, MI\nCorrelation',
           fontsize=9, ha='center', va='center')

    # Wrapper only (right)
    ax.text(1.8, 0.6, 'RFE\nForward Selection\nBackward Elim.\nAccurate',
           fontsize=9, ha='center', va='center')

    # Embedded only (bottom)
    ax.text(0, -1.8, 'L1/L2 Reg.\nTree importance\nElastic Net\nEfficient',
           fontsize=9, ha='center', va='center')

    # Intersections
    # Filter + Wrapper (top center)
    ax.text(0, 1.0, 'SelectKBest\n+ RFE', fontsize=8, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Filter + Embedded (bottom left)
    ax.text(-0.7, -0.5, 'Stat\nThreshold', fontsize=8, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Wrapper + Embedded (bottom right)
    ax.text(0.7, -0.5, 'Model-based\nSelection', fontsize=8, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Center (all three)
    ax.text(0, 0.15, 'RFECV', fontsize=9, ha='center', va='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Feature Selection Methods Overview', fontsize=14, fontweight='bold', pad=20)

    # Add legend box at bottom
    legend_text = (
        "Filter: Statistical measures (fast, independent of model)\n"
        "Wrapper: Uses model performance (accurate, computationally expensive)\n"
        "Embedded: Built into model training (balanced approach)"
    )
    ax.text(0, -3.2, legend_text, ha='center', va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_selection_methods.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: feature_selection_methods.png")


def create_binning_strategies():
    """Create visualization comparing different binning strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150)

    # Generate sample data with varying density
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(20, 5, 200),
        np.random.normal(60, 10, 150),
        np.random.normal(85, 5, 100)
    ])
    data = data[(data >= 0) & (data <= 100)]

    # Plot 1: Original distribution
    ax1 = axes[0, 0]
    ax1.hist(data, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    ax1.set_title('Original Distribution', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)

    # Plot 2: Equal-width binning
    ax2 = axes[0, 1]
    n_bins = 5
    bins_equal_width = np.linspace(data.min(), data.max(), n_bins + 1)
    ax2.hist(data, bins=bins_equal_width, color='#e74c3c', alpha=0.7, edgecolor='white', linewidth=2)
    ax2.set_title('Equal-Width Binning', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Value', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    # Add bin edges
    for edge in bins_equal_width[1:-1]:
        ax2.axvline(edge, color='darkred', linestyle='--', alpha=0.5)
    ax2.annotate('Fixed interval width\nUneven bin counts', xy=(0.6, 0.75), xycoords='axes fraction',
                fontsize=8, style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Equal-frequency (quantile) binning
    ax3 = axes[1, 0]
    bins_quantile = np.percentile(data, np.linspace(0, 100, n_bins + 1))
    ax3.hist(data, bins=bins_quantile, color='#27ae60', alpha=0.7, edgecolor='white', linewidth=2)
    ax3.set_title('Equal-Frequency (Quantile) Binning', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Value', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    # Add bin edges
    for edge in bins_quantile[1:-1]:
        ax3.axvline(edge, color='darkgreen', linestyle='--', alpha=0.5)
    ax3.annotate('Equal samples per bin\nVariable interval width', xy=(0.55, 0.75), xycoords='axes fraction',
                fontsize=8, style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 4: Comparison summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create comparison table
    table_data = [
        ['Strategy', 'Best For', 'Drawback'],
        ['Equal-Width', 'Uniform data', 'Empty bins with skewed'],
        ['Quantile', 'Skewed data', 'Loses value range info'],
        ['K-Means', 'Natural clusters', 'Computationally costly'],
        ['Domain', 'Business rules', 'Requires expertise']
    ]

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.35, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style rows
    for i in range(1, 5):
        for j in range(3):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

    ax4.set_title('Binning Strategy Selection Guide', fontsize=11, fontweight='bold', pad=20)

    plt.suptitle('Binning Strategies for Continuous Variables', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'binning_strategies.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: binning_strategies.png")


def main():
    """Generate all visualizations."""
    print("Generating Feature Engineering visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    create_correlation_heatmap()
    create_log_transformation()
    create_encoding_comparison()
    create_polynomial_features()
    create_feature_selection_venn()
    create_binning_strategies()

    print("-" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
