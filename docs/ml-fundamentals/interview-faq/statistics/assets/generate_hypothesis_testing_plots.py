"""
Hypothesis Testing & A/B Testing Visualizations
Generates plots for ML interview preparation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

output_dir = os.path.dirname(os.path.abspath(__file__))


def plot_type_i_type_ii_errors():
    """Visualize Type I and Type II errors"""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.linspace(-4, 8, 1000)

    # Null hypothesis distribution (H0 is true)
    null_mean = 0
    null_dist = stats.norm.pdf(x, null_mean, 1)

    # Alternative hypothesis distribution (H1 is true)
    alt_mean = 3
    alt_dist = stats.norm.pdf(x, alt_mean, 1)

    # Critical value (alpha = 0.05, one-tailed)
    critical_value = stats.norm.ppf(0.95)  # ~1.645

    # Plot distributions
    ax.plot(x, null_dist, 'b-', linewidth=2.5, label='H₀ Distribution (Null is True)')
    ax.plot(x, alt_dist, 'r-', linewidth=2.5, label='H₁ Distribution (Alternative is True)')

    # Fill Type I error (alpha)
    x_type1 = x[x >= critical_value]
    ax.fill_between(x_type1, 0, stats.norm.pdf(x_type1, null_mean, 1),
                    alpha=0.4, color='blue', label=f'Type I Error (α)\nFalse Positive')

    # Fill Type II error (beta)
    x_type2 = x[x <= critical_value]
    ax.fill_between(x_type2, 0, stats.norm.pdf(x_type2, alt_mean, 1),
                    alpha=0.4, color='red', label=f'Type II Error (β)\nFalse Negative')

    # Fill Power (1 - beta)
    x_power = x[x >= critical_value]
    ax.fill_between(x_power, 0, stats.norm.pdf(x_power, alt_mean, 1),
                    alpha=0.3, color='green', label='Power (1-β)\nCorrect Detection')

    # Critical value line
    ax.axvline(x=critical_value, color='black', linestyle='--', linewidth=2,
               label=f'Critical Value = {critical_value:.2f}')

    ax.set_xlabel('Test Statistic', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('Type I and Type II Errors in Hypothesis Testing', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-4, 8)
    ax.set_ylim(0, 0.45)

    # Add annotations
    ax.annotate('Reject H₀', xy=(2.5, 0.02), fontsize=12, fontweight='bold', color='green')
    ax.annotate('Fail to Reject H₀', xy=(-2, 0.02), fontsize=12, fontweight='bold', color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_type_i_ii_errors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: hypothesis_type_i_ii_errors.png")


def plot_p_value_visualization():
    """Visualize p-value concept"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(-4, 4, 1000)
    normal_dist = stats.norm.pdf(x)

    # Left plot: Significant result (low p-value)
    ax1 = axes[0]
    ax1.plot(x, normal_dist, 'b-', linewidth=2.5)
    ax1.fill_between(x, 0, normal_dist, alpha=0.2, color='blue')

    test_stat = 2.5
    x_tail = x[x >= test_stat]
    ax1.fill_between(x_tail, 0, stats.norm.pdf(x_tail), alpha=0.6, color='red',
                     label=f'p-value = {1 - stats.norm.cdf(test_stat):.4f}')
    ax1.axvline(x=test_stat, color='red', linestyle='--', linewidth=2)

    ax1.set_xlabel('Test Statistic', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Significant Result (p < 0.05)\nReject H₀', fontsize=13, fontweight='bold', color='green')
    ax1.legend(loc='upper right')
    ax1.annotate(f'Test stat = {test_stat}', xy=(test_stat, 0.02), fontsize=11, ha='left')

    # Right plot: Non-significant result (high p-value)
    ax2 = axes[1]
    ax2.plot(x, normal_dist, 'b-', linewidth=2.5)
    ax2.fill_between(x, 0, normal_dist, alpha=0.2, color='blue')

    test_stat = 1.2
    x_tail = x[x >= test_stat]
    ax2.fill_between(x_tail, 0, stats.norm.pdf(x_tail), alpha=0.6, color='orange',
                     label=f'p-value = {1 - stats.norm.cdf(test_stat):.4f}')
    ax2.axvline(x=test_stat, color='orange', linestyle='--', linewidth=2)

    ax2.set_xlabel('Test Statistic', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Non-Significant Result (p > 0.05)\nFail to Reject H₀', fontsize=13, fontweight='bold', color='orange')
    ax2.legend(loc='upper right')
    ax2.annotate(f'Test stat = {test_stat}', xy=(test_stat, 0.02), fontsize=11, ha='left')

    plt.suptitle('P-Value: Probability of Observing Data Given H₀ is True',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_p_value.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: hypothesis_p_value.png")


def plot_one_vs_two_tailed():
    """Compare one-tailed vs two-tailed tests"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(-4, 4, 1000)
    normal_dist = stats.norm.pdf(x)

    # Two-tailed test
    ax1 = axes[0]
    ax1.plot(x, normal_dist, 'b-', linewidth=2.5)
    ax1.fill_between(x, 0, normal_dist, alpha=0.2, color='blue')

    # Critical values for alpha = 0.05 (two-tailed)
    crit_low = stats.norm.ppf(0.025)
    crit_high = stats.norm.ppf(0.975)

    ax1.fill_between(x[x <= crit_low], 0, stats.norm.pdf(x[x <= crit_low]),
                     alpha=0.5, color='red', label='α/2 = 2.5%')
    ax1.fill_between(x[x >= crit_high], 0, stats.norm.pdf(x[x >= crit_high]),
                     alpha=0.5, color='red')

    ax1.axvline(x=crit_low, color='red', linestyle='--', linewidth=1.5)
    ax1.axvline(x=crit_high, color='red', linestyle='--', linewidth=1.5)

    ax1.annotate(f'z = {crit_low:.2f}', xy=(crit_low, 0.02), fontsize=10, ha='right')
    ax1.annotate(f'z = {crit_high:.2f}', xy=(crit_high, 0.02), fontsize=10, ha='left')

    ax1.set_xlabel('Test Statistic (z)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Two-Tailed Test (α = 0.05)\nH₁: μ ≠ μ₀', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')

    # One-tailed test (right)
    ax2 = axes[1]
    ax2.plot(x, normal_dist, 'b-', linewidth=2.5)
    ax2.fill_between(x, 0, normal_dist, alpha=0.2, color='blue')

    crit = stats.norm.ppf(0.95)
    ax2.fill_between(x[x >= crit], 0, stats.norm.pdf(x[x >= crit]),
                     alpha=0.5, color='red', label='α = 5%')
    ax2.axvline(x=crit, color='red', linestyle='--', linewidth=1.5)
    ax2.annotate(f'z = {crit:.2f}', xy=(crit, 0.02), fontsize=10, ha='left')

    ax2.set_xlabel('Test Statistic (z)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('One-Tailed Test (α = 0.05)\nH₁: μ > μ₀', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')

    plt.suptitle('One-Tailed vs Two-Tailed Hypothesis Tests', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_one_two_tailed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: hypothesis_one_two_tailed.png")


def plot_power_analysis():
    """Visualize statistical power and sample size relationship"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Power vs Sample Size
    ax1 = axes[0]
    sample_sizes = np.arange(10, 500, 10)
    effect_size = 0.3
    alpha = 0.05

    # Calculate power for different sample sizes
    powers = []
    for n in sample_sizes:
        se = 1 / np.sqrt(n)
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = (effect_size - 0) / se - z_alpha
        power = stats.norm.cdf(z_beta)
        powers.append(power)

    ax1.plot(sample_sizes, powers, 'b-', linewidth=2.5)
    ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, label='Target Power = 80%')

    # Find required sample size for 80% power
    required_n = sample_sizes[np.argmin(np.abs(np.array(powers) - 0.8))]
    ax1.axvline(x=required_n, color='green', linestyle='--', linewidth=1.5)
    ax1.scatter([required_n], [0.8], s=150, c='green', zorder=5)
    ax1.annotate(f'n ≈ {required_n}', xy=(required_n + 10, 0.75), fontsize=11, color='green')

    ax1.set_xlabel('Sample Size (n)', fontsize=12)
    ax1.set_ylabel('Statistical Power', fontsize=12)
    ax1.set_title('Power vs Sample Size\n(Effect Size = 0.3)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1)

    # Right: Power vs Effect Size
    ax2 = axes[1]
    effect_sizes = np.linspace(0.1, 1.0, 100)
    n = 100

    powers = []
    for d in effect_sizes:
        se = 1 / np.sqrt(n)
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = (d - 0) / se - z_alpha
        power = stats.norm.cdf(z_beta)
        powers.append(power)

    ax2.plot(effect_sizes, powers, 'r-', linewidth=2.5)
    ax2.axhline(y=0.8, color='blue', linestyle='--', linewidth=1.5, label='Target Power = 80%')
    ax2.fill_between(effect_sizes, 0, powers, alpha=0.2, color='red')

    ax2.set_xlabel('Effect Size (d)', fontsize=12)
    ax2.set_ylabel('Statistical Power', fontsize=12)
    ax2.set_title('Power vs Effect Size\n(n = 100)', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1)

    plt.suptitle('Power Analysis: Key Relationships', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_power_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: hypothesis_power_analysis.png")


def plot_ab_testing_example():
    """Visualize A/B testing results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    np.random.seed(42)

    # Left: Conversion rates comparison
    ax1 = axes[0]
    groups = ['Control (A)', 'Treatment (B)']
    conversions = [0.12, 0.15]
    errors = [0.01, 0.012]
    colors = ['#3498db', '#2ecc71']

    bars = ax1.bar(groups, conversions, yerr=errors, capsize=10, color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    ax1.set_ylabel('Conversion Rate', fontsize=12)
    ax1.set_title('A/B Test Results: Conversion Rates', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 0.2)

    # Add percentage labels
    for bar, conv in zip(bars, conversions):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f'{conv*100:.1f}%', ha='center', fontsize=12, fontweight='bold')

    # Add lift annotation
    lift = (conversions[1] - conversions[0]) / conversions[0] * 100
    ax1.annotate(f'Lift: +{lift:.1f}%', xy=(1, conversions[1]),
                 xytext=(1.3, conversions[1] + 0.02),
                 fontsize=12, color='green', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green'))

    # Right: Distribution of possible outcomes
    ax2 = axes[1]

    # Simulated distribution of differences
    n_simulations = 10000
    control_samples = np.random.binomial(1000, 0.12, n_simulations) / 1000
    treatment_samples = np.random.binomial(1000, 0.15, n_simulations) / 1000
    differences = treatment_samples - control_samples

    ax2.hist(differences, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Effect (H₀)')
    ax2.axvline(x=np.mean(differences), color='green', linestyle='-', linewidth=2,
                label=f'Observed Diff = {np.mean(differences):.3f}')

    # Calculate p-value (proportion below 0)
    p_value = np.mean(differences <= 0)

    ax2.set_xlabel('Difference (Treatment - Control)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'Distribution of Differences\np-value ≈ {p_value:.4f}', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')

    plt.suptitle('A/B Testing: Visualizing Statistical Significance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_ab_testing.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: hypothesis_ab_testing.png")


def plot_error_tradeoff_matrix():
    """Create confusion matrix style visualization for hypothesis testing"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the confusion matrix
    matrix_data = np.array([[0.95, 0.05], [0.20, 0.80]])

    # Plot heatmap
    im = ax.imshow(matrix_data, cmap='RdYlGn', vmin=0, vmax=1)

    # Labels
    decision_labels = ['Fail to Reject H₀', 'Reject H₀']
    truth_labels = ['H₀ True', 'H₀ False']

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(decision_labels, fontsize=12)
    ax.set_yticklabels(truth_labels, fontsize=12)

    # Add text annotations
    text_colors = ['green', 'red', 'red', 'green']
    outcomes = [
        'Correct!\n(True Negative)\n95%',
        'Type I Error\n(False Positive)\nα = 5%',
        'Type II Error\n(False Negative)\nβ = 20%',
        'Correct!\n(True Positive)\nPower = 80%'
    ]

    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            ax.text(j, i, outcomes[idx], ha='center', va='center',
                    fontsize=11, fontweight='bold', color='black')

    ax.set_xlabel('Decision', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reality', fontsize=14, fontweight='bold')
    ax.set_title('Hypothesis Testing: Decision Matrix\n(α = 0.05, Power = 0.80)',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Probability', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_decision_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: hypothesis_decision_matrix.png")


if __name__ == '__main__':
    print("Generating Hypothesis Testing visualizations...")
    plot_type_i_type_ii_errors()
    plot_p_value_visualization()
    plot_one_vs_two_tailed()
    plot_power_analysis()
    plot_ab_testing_example()
    plot_error_tradeoff_matrix()
    print("\nAll visualizations generated successfully!")
