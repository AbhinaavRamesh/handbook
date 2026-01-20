#!/usr/bin/env python3
"""
Naive Bayes Visualization Generator

This script generates educational visualizations for Naive Bayes:
1. Class-conditional Gaussian distributions (overlapping curves)
2. 2D decision boundary with posterior probability heatmap
3. Prior to posterior update visualization
4. Multinomial NB word probability bar charts

Outputs:
- gaussian_class_conditional.png: Overlapping Gaussian PDFs for each class
- nb_decision_boundary.png: 2D decision boundary with posterior heatmap
- prior_posterior_update.png: Bayesian update visualization
- multinomial_word_probs.png: Word probability bar charts for text classification
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# Color palette (colorblind-friendly)
COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
BACKGROUND_COLOR = '#F8F9FA'

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/'


def create_gaussian_class_conditional(output_path=None):
    """
    Create visualization of class-conditional Gaussian distributions.

    Shows:
    - Overlapping Gaussian PDFs for two classes
    - Decision boundary where posteriors are equal
    - Shaded regions showing classification areas
    """
    print("Generating class-conditional Gaussian distributions...")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    # Parameters for two classes
    mu_0, sigma_0 = 2, 1.2  # Class 0: lower mean
    mu_1, sigma_1 = 5, 1.0  # Class 1: higher mean
    prior_0, prior_1 = 0.4, 0.6  # Class priors

    # Create x range
    x = np.linspace(-2, 10, 500)

    # Compute class-conditional densities P(x|y)
    pdf_0 = norm.pdf(x, mu_0, sigma_0)
    pdf_1 = norm.pdf(x, mu_1, sigma_1)

    # Compute posteriors using Bayes theorem (unnormalized)
    posterior_0_unnorm = pdf_0 * prior_0
    posterior_1_unnorm = pdf_1 * prior_1

    # Normalize posteriors
    normalizer = posterior_0_unnorm + posterior_1_unnorm
    posterior_0 = posterior_0_unnorm / normalizer
    posterior_1 = posterior_1_unnorm / normalizer

    # Find decision boundary (where posteriors are equal)
    decision_idx = np.argmin(np.abs(posterior_0 - posterior_1))
    decision_x = x[decision_idx]

    # Plot class-conditional densities
    ax.fill_between(x, pdf_0, alpha=0.3, color=COLORS[0], label=f'P(x|Class 0) ~ N({mu_0}, {sigma_0})')
    ax.fill_between(x, pdf_1, alpha=0.3, color=COLORS[1], label=f'P(x|Class 1) ~ N({mu_1}, {sigma_1})')

    ax.plot(x, pdf_0, color=COLORS[0], linewidth=2.5)
    ax.plot(x, pdf_1, color=COLORS[1], linewidth=2.5)

    # Mark decision boundary
    ax.axvline(x=decision_x, color='#1D3557', linestyle='--', linewidth=2,
               label=f'Decision Boundary (x={decision_x:.2f})')

    # Shade classification regions
    ax.fill_between(x[x < decision_x], 0, np.minimum(pdf_0[x < decision_x], 0.5),
                    alpha=0.1, color=COLORS[0], hatch='//')
    ax.fill_between(x[x >= decision_x], 0, np.minimum(pdf_1[x >= decision_x], 0.5),
                    alpha=0.1, color=COLORS[1], hatch='\\\\')

    # Add annotations for priors
    ax.annotate(f'Prior P(Class 0) = {prior_0}', xy=(0, 0.35), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate(f'Prior P(Class 1) = {prior_1}', xy=(6.5, 0.35), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Mark means
    ax.scatter([mu_0, mu_1], [norm.pdf(mu_0, mu_0, sigma_0), norm.pdf(mu_1, mu_1, sigma_1)],
               s=100, c=[COLORS[0], COLORS[1]], edgecolors='white', linewidth=2, zorder=5)

    ax.set_xlabel('Feature Value (x)', fontsize=12)
    ax.set_ylabel('Probability Density P(x|Class)', fontsize=12)
    ax.set_title('Class-Conditional Gaussian Distributions\nNaive Bayes assumes features follow these distributions',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_xlim(-2, 10)
    ax.set_ylim(0, 0.45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_decision_boundary_heatmap(output_path=None):
    """
    Create 2D decision boundary with posterior probability heatmap.

    Shows:
    - Data points from two classes
    - Posterior probability heatmap
    - Decision boundary contour
    """
    print("Generating 2D decision boundary heatmap...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BACKGROUND_COLOR)

    # Generate 2D data for two classes
    np.random.seed(42)
    n_samples = 100

    # Class 0: centered at (2, 2)
    X0 = np.random.randn(n_samples, 2) * 1.2 + np.array([2, 2])
    # Class 1: centered at (5, 5)
    X1 = np.random.randn(n_samples, 2) * 1.0 + np.array([5, 5])

    X = np.vstack([X0, X1])
    y = np.array([0] * n_samples + [1] * n_samples)

    # Compute class parameters (Gaussian NB)
    mu_0 = X0.mean(axis=0)
    mu_1 = X1.mean(axis=0)
    var_0 = X0.var(axis=0) + 1e-9
    var_1 = X1.var(axis=0) + 1e-9
    prior_0 = 0.5
    prior_1 = 0.5

    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Compute posterior probabilities for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    def gaussian_log_likelihood(x, mu, var):
        """Compute log likelihood for Gaussian distribution."""
        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mu) ** 2) / var, axis=1)

    log_likelihood_0 = gaussian_log_likelihood(grid_points, mu_0, var_0)
    log_likelihood_1 = gaussian_log_likelihood(grid_points, mu_1, var_1)

    log_posterior_0 = np.log(prior_0) + log_likelihood_0
    log_posterior_1 = np.log(prior_1) + log_likelihood_1

    # Convert to probabilities using log-sum-exp trick
    max_log = np.maximum(log_posterior_0, log_posterior_1)
    posterior_0 = np.exp(log_posterior_0 - max_log)
    posterior_1 = np.exp(log_posterior_1 - max_log)
    normalizer = posterior_0 + posterior_1
    posterior_1 = posterior_1 / normalizer

    Z = posterior_1.reshape(xx.shape)

    # Left plot: Posterior probability heatmap
    ax1 = axes[0]
    ax1.set_facecolor(BACKGROUND_COLOR)

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('custom', [COLORS[0], 'white', COLORS[1]])

    contourf = ax1.contourf(xx, yy, Z, levels=50, cmap=cmap, alpha=0.8)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='#1D3557', linewidths=2.5, linestyles='--')

    # Plot data points
    ax1.scatter(X0[:, 0], X0[:, 1], c=COLORS[0], s=40, alpha=0.7,
                edgecolors='white', linewidth=0.5, label='Class 0')
    ax1.scatter(X1[:, 0], X1[:, 1], c=COLORS[1], s=40, alpha=0.7,
                edgecolors='white', linewidth=0.5, label='Class 1')

    # Mark means
    ax1.scatter([mu_0[0]], [mu_0[1]], marker='X', s=200, c=COLORS[0],
                edgecolors='white', linewidth=2, zorder=10, label='Mean Class 0')
    ax1.scatter([mu_1[0]], [mu_1[1]], marker='X', s=200, c=COLORS[1],
                edgecolors='white', linewidth=2, zorder=10, label='Mean Class 1')

    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label('P(Class 1 | x)', fontsize=11)

    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('Gaussian Naive Bayes Decision Boundary\n(Posterior Probability Heatmap)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Right plot: Classification regions
    ax2 = axes[1]
    ax2.set_facecolor(BACKGROUND_COLOR)

    # Predicted classes
    predictions = (Z > 0.5).astype(int)

    ax2.contourf(xx, yy, predictions, levels=[-0.5, 0.5, 1.5], colors=[COLORS[0], COLORS[1]], alpha=0.3)
    ax2.contour(xx, yy, Z, levels=[0.5], colors='#1D3557', linewidths=2.5, linestyles='-')

    # Plot data points
    ax2.scatter(X0[:, 0], X0[:, 1], c=COLORS[0], s=40, alpha=0.8,
                edgecolors='white', linewidth=0.5, label='Class 0')
    ax2.scatter(X1[:, 0], X1[:, 1], c=COLORS[1], s=40, alpha=0.8,
                edgecolors='white', linewidth=0.5, label='Class 1')

    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.set_title('Classification Regions\n(Decision Boundary at P(Class 1|x) = 0.5)',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_prior_posterior_update(output_path=None):
    """
    Create visualization showing Bayesian update from prior to posterior.

    Shows:
    - Prior probability
    - Likelihood (evidence from data)
    - Posterior probability after update
    """
    print("Generating prior to posterior update visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BACKGROUND_COLOR)

    # Scenario: Medical diagnosis
    # Disease prevalence (prior): 1%
    # Test sensitivity (P(positive|disease)): 90%
    # Test specificity (P(negative|no disease)): 95%

    prior_disease = 0.01
    prior_no_disease = 0.99
    sensitivity = 0.90  # P(+|D)
    specificity = 0.95  # P(-|~D)
    false_positive = 1 - specificity  # P(+|~D)

    # Calculate P(+) = P(+|D)P(D) + P(+|~D)P(~D)
    p_positive = sensitivity * prior_disease + false_positive * prior_no_disease

    # Calculate posterior P(D|+) using Bayes theorem
    posterior_disease = (sensitivity * prior_disease) / p_positive
    posterior_no_disease = 1 - posterior_disease

    # Plot 1: Prior probabilities
    ax1 = axes[0]
    ax1.set_facecolor(BACKGROUND_COLOR)
    bars1 = ax1.bar(['Disease', 'No Disease'], [prior_disease, prior_no_disease],
                    color=[COLORS[0], COLORS[1]], alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Prior P(Disease)\n(Base Rate)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars1, [prior_disease, prior_no_disease]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add formula
    ax1.text(0.5, 0.75, r'$P(D) = 0.01$', transform=ax1.transAxes, fontsize=11,
             ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Likelihood (test results given condition)
    ax2 = axes[1]
    ax2.set_facecolor(BACKGROUND_COLOR)

    x = np.arange(2)
    width = 0.35

    bars2a = ax2.bar(x - width/2, [sensitivity, 1-sensitivity], width,
                     label='Has Disease', color=COLORS[0], alpha=0.8, edgecolor='white', linewidth=2)
    bars2b = ax2.bar(x + width/2, [false_positive, specificity], width,
                     label='No Disease', color=COLORS[1], alpha=0.8, edgecolor='white', linewidth=2)

    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Likelihood P(Test|Condition)\n(Test Characteristics)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Test +', 'Test -'])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.0%}', ha='center', va='bottom', fontsize=10)

    # Plot 3: Posterior after positive test
    ax3 = axes[2]
    ax3.set_facecolor(BACKGROUND_COLOR)
    bars3 = ax3.bar(['Disease', 'No Disease'], [posterior_disease, posterior_no_disease],
                    color=[COLORS[0], COLORS[1]], alpha=0.8, edgecolor='white', linewidth=2)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title('Posterior P(Disease|Test+)\n(After Positive Test)', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars3, [posterior_disease, posterior_no_disease]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add Bayes formula
    ax3.text(0.5, 0.85, r'$P(D|+) = \frac{P(+|D) \cdot P(D)}{P(+)}$',
             transform=ax3.transAxes, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add annotation showing the surprising result
    fig.text(0.5, 0.02,
             'Note: Despite 90% sensitivity, a positive test only means ~15% chance of disease\n'
             'due to low prior probability (base rate fallacy).',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_multinomial_word_probs(output_path=None):
    """
    Create visualization of word probabilities in Multinomial Naive Bayes.

    Shows:
    - Top words for each class (spam vs ham)
    - Log probability comparison
    - Feature importance for classification
    """
    print("Generating Multinomial NB word probability chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BACKGROUND_COLOR)

    # Simulated word probabilities for spam detection
    # These are log probabilities P(word|class)
    words = ['free', 'win', 'money', 'click', 'offer', 'meeting', 'report', 'project', 'thanks', 'hello']

    # Higher values = more associated with that class (these are raw probabilities)
    spam_probs = np.array([0.15, 0.12, 0.10, 0.09, 0.08, 0.02, 0.01, 0.01, 0.02, 0.01])
    ham_probs = np.array([0.02, 0.01, 0.02, 0.01, 0.01, 0.12, 0.10, 0.11, 0.08, 0.09])

    # Normalize to make them valid probabilities
    spam_probs = spam_probs / spam_probs.sum() * 0.4  # Scale for visualization
    ham_probs = ham_probs / ham_probs.sum() * 0.4

    # Left plot: Side-by-side comparison
    ax1 = axes[0]
    ax1.set_facecolor(BACKGROUND_COLOR)

    x = np.arange(len(words))
    width = 0.35

    bars1 = ax1.barh(x - width/2, spam_probs, width, label='Spam',
                     color=COLORS[0], alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax1.barh(x + width/2, ham_probs, width, label='Ham (Not Spam)',
                     color=COLORS[1], alpha=0.8, edgecolor='white', linewidth=1)

    ax1.set_yticks(x)
    ax1.set_yticklabels(words, fontsize=11)
    ax1.set_xlabel('P(word | class)', fontsize=12)
    ax1.set_title('Word Probabilities by Class\n(Multinomial Naive Bayes)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.invert_yaxis()

    # Right plot: Log probability ratio (feature importance)
    ax2 = axes[1]
    ax2.set_facecolor(BACKGROUND_COLOR)

    # Compute log ratio: log P(word|spam) - log P(word|ham)
    # Positive = more indicative of spam, Negative = more indicative of ham
    log_ratio = np.log(spam_probs + 1e-10) - np.log(ham_probs + 1e-10)

    # Sort by absolute importance
    sorted_idx = np.argsort(np.abs(log_ratio))[::-1]

    colors = [COLORS[0] if lr > 0 else COLORS[1] for lr in log_ratio[sorted_idx]]

    bars3 = ax2.barh(range(len(words)), log_ratio[sorted_idx], color=colors,
                     alpha=0.8, edgecolor='white', linewidth=1)

    ax2.set_yticks(range(len(words)))
    ax2.set_yticklabels([words[i] for i in sorted_idx], fontsize=11)
    ax2.axvline(x=0, color='#1D3557', linewidth=2, linestyle='-')
    ax2.set_xlabel('Log Probability Ratio: log P(word|spam) - log P(word|ham)', fontsize=11)
    ax2.set_title('Feature Importance for Classification\n(Positive = Spam Indicator, Negative = Ham Indicator)',
                  fontsize=13, fontweight='bold')
    ax2.invert_yaxis()

    # Add annotations
    ax2.annotate('Spam\nIndicators', xy=(1.5, 1), fontsize=10, fontweight='bold',
                 color=COLORS[0], ha='center')
    ax2.annotate('Ham\nIndicators', xy=(-1.5, 8), fontsize=10, fontweight='bold',
                 color=COLORS[1], ha='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {output_path}")

    plt.close(fig)


def main():
    """Generate all Naive Bayes visualizations."""
    print("=" * 60)
    print("Naive Bayes Visualization Generator")
    print("=" * 60)
    print()

    # 1. Class-conditional Gaussian distributions
    create_gaussian_class_conditional(output_path=f'{OUTPUT_DIR}gaussian_class_conditional.png')

    # 2. 2D decision boundary with posterior heatmap
    create_decision_boundary_heatmap(output_path=f'{OUTPUT_DIR}nb_decision_boundary.png')

    # 3. Prior to posterior update visualization
    create_prior_posterior_update(output_path=f'{OUTPUT_DIR}prior_posterior_update.png')

    # 4. Multinomial NB word probability bar charts
    create_multinomial_word_probs(output_path=f'{OUTPUT_DIR}multinomial_word_probs.png')

    print()
    print("=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  1. {OUTPUT_DIR}gaussian_class_conditional.png")
    print(f"  2. {OUTPUT_DIR}nb_decision_boundary.png")
    print(f"  3. {OUTPUT_DIR}prior_posterior_update.png")
    print(f"  4. {OUTPUT_DIR}multinomial_word_probs.png")


if __name__ == '__main__':
    main()
