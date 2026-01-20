"""
Generate visualizations for Hypothesis Testing FAQ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Common styling
FIGSIZE = (10, 6)
DPI = 150
COLORS = {
    'primary': '#2563eb',
    'secondary': '#dc2626',
    'tertiary': '#16a34a',
    'quaternary': '#9333ea',
    'fill': '#93c5fd',
    'fill2': '#fca5a5',
}


def plot_type_i_ii_errors():
    """Generate Type I and Type II errors visualization."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    x = np.linspace(-4, 8, 500)

    # Null hypothesis distribution (H0: μ = 0)
    null_dist = stats.norm.pdf(x, loc=0, scale=1)

    # Alternative hypothesis distribution (H1: μ = 3)
    alt_dist = stats.norm.pdf(x, loc=3, scale=1)

    # Critical value (α = 0.05, one-tailed)
    critical_value = stats.norm.ppf(0.95)

    # Plot distributions
    ax.plot(x, null_dist, color=COLORS['primary'], linewidth=2.5, label='H₀ (Null: μ = 0)')
    ax.plot(x, alt_dist, color=COLORS['secondary'], linewidth=2.5, label='H₁ (Alternative: μ = 3)')

    # Shade Type I error (α)
    x_type1 = x[x >= critical_value]
    ax.fill_between(x_type1, stats.norm.pdf(x_type1, loc=0, scale=1),
                    alpha=0.4, color=COLORS['primary'], label=f'Type I Error (α)')

    # Shade Type II error (β)
    x_type2 = x[x <= critical_value]
    ax.fill_between(x_type2, stats.norm.pdf(x_type2, loc=3, scale=1),
                    alpha=0.4, color=COLORS['secondary'], label=f'Type II Error (β)')

    # Mark critical value
    ax.axvline(x=critical_value, color='black', linestyle='--', linewidth=2)
    ax.annotate(f'Critical Value\n({critical_value:.2f})', xy=(critical_value, 0.35),
                xytext=(critical_value + 1.5, 0.38), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Labels
    ax.annotate('Type I Error (α)\nFalse Positive\nReject H₀ when true',
                xy=(2.2, 0.02), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor=COLORS['fill'], alpha=0.8))
    ax.annotate('Type II Error (β)\nFalse Negative\nFail to reject H₀ when false',
                xy=(0.8, 0.15), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor=COLORS['fill2'], alpha=0.8))

    ax.set_xlabel('Test Statistic', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Type I and Type II Errors in Hypothesis Testing', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-4, 8)
    ax.set_ylim(0, 0.45)

    plt.tight_layout()
    plt.savefig('hypothesis_type_i_ii_errors.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: hypothesis_type_i_ii_errors.png")


def plot_p_value():
    """Generate p-value visualization."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    x = np.linspace(-4, 4, 500)
    y = stats.norm.pdf(x)

    # Observed test statistic
    observed = 2.1

    # Plot distribution
    ax.plot(x, y, color=COLORS['primary'], linewidth=2.5, label='Null Distribution')
    ax.fill_between(x, y, alpha=0.2, color=COLORS['primary'])

    # Shade p-value region (two-tailed)
    x_pval_right = x[x >= observed]
    x_pval_left = x[x <= -observed]
    ax.fill_between(x_pval_right, stats.norm.pdf(x_pval_right),
                    alpha=0.6, color=COLORS['secondary'], label=f'P-value region')
    ax.fill_between(x_pval_left, stats.norm.pdf(x_pval_left),
                    alpha=0.6, color=COLORS['secondary'])

    # Mark observed statistic
    ax.axvline(x=observed, color=COLORS['tertiary'], linestyle='--', linewidth=2)
    ax.axvline(x=-observed, color=COLORS['tertiary'], linestyle='--', linewidth=2)

    ax.scatter([observed], [0], color=COLORS['tertiary'], s=150, zorder=5, marker='^')
    ax.annotate(f'Observed\nStatistic = {observed}', xy=(observed, 0.05),
                xytext=(observed + 0.8, 0.15), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Calculate and show p-value
    p_value = 2 * (1 - stats.norm.cdf(observed))
    ax.text(0, 0.25, f'P-value = {p_value:.4f}\n\nProbability of observing\na result this extreme\nif H₀ is true',
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_xlabel('Test Statistic', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('P-Value: The Probability of Observing Data as Extreme as Ours (Under H₀)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.45)

    plt.tight_layout()
    plt.savefig('hypothesis_p_value.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: hypothesis_p_value.png")


def plot_one_two_tailed():
    """Generate one-tailed vs two-tailed tests visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

    x = np.linspace(-4, 4, 500)
    y = stats.norm.pdf(x)
    alpha = 0.05

    # Two-tailed test
    ax = axes[0]
    ax.plot(x, y, color=COLORS['primary'], linewidth=2.5)
    ax.fill_between(x, y, alpha=0.2, color=COLORS['primary'])

    critical_two = stats.norm.ppf(1 - alpha/2)
    x_right = x[x >= critical_two]
    x_left = x[x <= -critical_two]
    ax.fill_between(x_right, stats.norm.pdf(x_right), alpha=0.6, color=COLORS['secondary'])
    ax.fill_between(x_left, stats.norm.pdf(x_left), alpha=0.6, color=COLORS['secondary'])

    ax.axvline(x=critical_two, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=-critical_two, color='black', linestyle='--', linewidth=1.5)

    ax.text(0, 0.2, 'Accept H₀\n(95%)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.text(2.5, 0.05, 'Reject\nα/2=2.5%', ha='center', fontsize=9)
    ax.text(-2.5, 0.05, 'Reject\nα/2=2.5%', ha='center', fontsize=9)

    ax.set_title('Two-Tailed Test\nH₁: μ ≠ μ₀ (either direction)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Statistic', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)

    # One-tailed test (right)
    ax = axes[1]
    ax.plot(x, y, color=COLORS['primary'], linewidth=2.5)
    ax.fill_between(x, y, alpha=0.2, color=COLORS['primary'])

    critical_one = stats.norm.ppf(1 - alpha)
    x_right = x[x >= critical_one]
    ax.fill_between(x_right, stats.norm.pdf(x_right), alpha=0.6, color=COLORS['secondary'])

    ax.axvline(x=critical_one, color='black', linestyle='--', linewidth=1.5)

    ax.text(-0.5, 0.2, 'Accept H₀\n(95%)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.text(2.3, 0.08, 'Reject\nα=5%', ha='center', fontsize=9)

    ax.set_title('One-Tailed Test (Right)\nH₁: μ > μ₀ (specific direction)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Statistic', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)

    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.45)

    plt.suptitle('One-Tailed vs Two-Tailed Tests (α = 0.05)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('hypothesis_one_two_tailed.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: hypothesis_one_two_tailed.png")


def plot_power_analysis():
    """Generate power analysis visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)

    # Effect of sample size
    ax = axes[0]
    effect_size = 0.5
    alpha = 0.05
    sample_sizes = np.arange(10, 201, 5)

    powers = []
    for n in sample_sizes:
        se = 1 / np.sqrt(n)
        critical_z = stats.norm.ppf(1 - alpha)
        z_power = effect_size / se - critical_z
        power = stats.norm.cdf(z_power)
        powers.append(power)

    ax.plot(sample_sizes, powers, color=COLORS['primary'], linewidth=2.5)
    ax.axhline(y=0.8, color=COLORS['secondary'], linestyle='--', label='Target Power (0.8)')
    ax.fill_between(sample_sizes, powers, 0.8, where=np.array(powers) < 0.8,
                    alpha=0.3, color=COLORS['secondary'])
    ax.set_xlabel('Sample Size (n)', fontsize=11)
    ax.set_ylabel('Statistical Power', fontsize=11)
    ax.set_title('Power vs Sample Size\n(d=0.5, α=0.05)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    # Effect of effect size
    ax = axes[1]
    n = 50
    effect_sizes = np.linspace(0.1, 1.5, 50)

    powers = []
    for d in effect_sizes:
        se = 1 / np.sqrt(n)
        critical_z = stats.norm.ppf(1 - alpha)
        z_power = d / se - critical_z
        power = stats.norm.cdf(z_power)
        powers.append(power)

    ax.plot(effect_sizes, powers, color=COLORS['tertiary'], linewidth=2.5)
    ax.axhline(y=0.8, color=COLORS['secondary'], linestyle='--', label='Target Power (0.8)')
    ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.7)
    ax.text(0.2, 0.1, 'Small', fontsize=9, ha='center')
    ax.text(0.5, 0.1, 'Medium', fontsize=9, ha='center')
    ax.text(0.8, 0.1, 'Large', fontsize=9, ha='center')
    ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=11)
    ax.set_ylabel('Statistical Power', fontsize=11)
    ax.set_title('Power vs Effect Size\n(n=50, α=0.05)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    # Effect of alpha
    ax = axes[2]
    n = 50
    effect_size = 0.5
    alphas = np.linspace(0.001, 0.2, 50)

    powers = []
    for a in alphas:
        se = 1 / np.sqrt(n)
        critical_z = stats.norm.ppf(1 - a)
        z_power = effect_size / se - critical_z
        power = stats.norm.cdf(z_power)
        powers.append(power)

    ax.plot(alphas, powers, color=COLORS['quaternary'], linewidth=2.5)
    ax.axhline(y=0.8, color=COLORS['secondary'], linestyle='--', label='Target Power (0.8)')
    ax.axvline(x=0.05, color='gray', linestyle=':', alpha=0.7)
    ax.text(0.05, 0.5, 'α=0.05', fontsize=9, ha='left')
    ax.set_xlabel('Significance Level (α)', fontsize=11)
    ax.set_ylabel('Statistical Power', fontsize=11)
    ax.set_title('Power vs Significance Level\n(n=50, d=0.5)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    plt.suptitle('Factors Affecting Statistical Power (1 - β)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('hypothesis_power_analysis.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: hypothesis_power_analysis.png")


def plot_ab_testing():
    """Generate A/B testing example visualization."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    np.random.seed(42)

    # Simulate A/B test results
    control_rate = 0.10  # 10% conversion
    treatment_rate = 0.12  # 12% conversion (20% lift)
    n = 10000

    # Bootstrap distributions of conversion rates
    control_dist = np.random.binomial(n, control_rate, 10000) / n
    treatment_dist = np.random.binomial(n, treatment_rate, 10000) / n

    # Plot histograms
    ax.hist(control_dist, bins=50, alpha=0.6, color=COLORS['primary'],
            label=f'Control (A): {control_rate:.1%}', density=True)
    ax.hist(treatment_dist, bins=50, alpha=0.6, color=COLORS['tertiary'],
            label=f'Treatment (B): {treatment_rate:.1%}', density=True)

    # Mark means
    ax.axvline(x=control_rate, color=COLORS['primary'], linestyle='--', linewidth=2)
    ax.axvline(x=treatment_rate, color=COLORS['tertiary'], linestyle='--', linewidth=2)

    # Calculate p-value
    z = (treatment_rate - control_rate) / np.sqrt(control_rate*(1-control_rate)/n + treatment_rate*(1-treatment_rate)/n)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Add statistics box
    stats_text = f'Sample Size: {n:,} per group\n'
    stats_text += f'Absolute Lift: {(treatment_rate-control_rate):.1%}\n'
    stats_text += f'Relative Lift: {(treatment_rate/control_rate - 1):.1%}\n'
    stats_text += f'Z-statistic: {z:.2f}\n'
    stats_text += f'P-value: {p_value:.4f}\n'
    stats_text += f'Result: {"Significant! ✓" if p_value < 0.05 else "Not Significant"}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_xlabel('Conversion Rate', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('A/B Test: Comparing Conversion Rates\nControl vs Treatment Group',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('hypothesis_ab_testing.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: hypothesis_ab_testing.png")


def plot_decision_matrix():
    """Generate hypothesis testing decision matrix."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Headers
    ax.text(5, 7.5, 'Hypothesis Testing Decision Matrix', fontsize=14, fontweight='bold', ha='center')
    ax.text(6.5, 6.5, 'Reality', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 4.5, 'Decision', fontsize=12, fontweight='bold', ha='center', rotation=90)

    # Column headers
    ax.text(5, 6, 'H₀ True', fontsize=11, ha='center', fontweight='bold')
    ax.text(8, 6, 'H₀ False', fontsize=11, ha='center', fontweight='bold')

    # Row headers
    ax.text(2.5, 4.5, 'Reject H₀', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.5, 2.5, 'Fail to Reject H₀', fontsize=11, ha='center', fontweight='bold')

    # Cells
    cells = [
        (5, 4.5, 'Type I Error\n(False Positive)\nα', COLORS['fill2']),
        (8, 4.5, 'Correct!\n(True Positive)\n1-β (Power)', 'lightgreen'),
        (5, 2.5, 'Correct!\n(True Negative)\n1-α', 'lightgreen'),
        (8, 2.5, 'Type II Error\n(False Negative)\nβ', COLORS['fill2']),
    ]

    for x, y, text, color in cells:
        rect = FancyBboxPatch((x-1.3, y-0.8), 2.6, 1.6, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10)

    # Draw grid lines
    ax.plot([3.5, 9.5], [5.5, 5.5], 'k-', linewidth=1.5)
    ax.plot([3.5, 9.5], [3.5, 3.5], 'k-', linewidth=1.5)
    ax.plot([6.5, 6.5], [1.5, 5.5], 'k-', linewidth=1.5)

    # Add footer
    ax.text(5, 0.8, 'α = P(Type I Error) = Significance Level\n'
            'β = P(Type II Error)\nPower = 1 - β = P(Correctly detecting effect)',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('hypothesis_decision_matrix.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: hypothesis_decision_matrix.png")


if __name__ == '__main__':
    print("Generating Hypothesis Testing visualizations...")
    plot_type_i_ii_errors()
    plot_p_value()
    plot_one_two_tailed()
    plot_power_analysis()
    plot_ab_testing()
    plot_decision_matrix()
    print("Done!")
