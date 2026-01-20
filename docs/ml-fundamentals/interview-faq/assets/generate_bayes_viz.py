"""
Visualization generators for Probability and Bayes Theorem FAQ.
Creates matplotlib visualizations for key concepts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import numpy as np
from scipy import stats
from scipy.stats import norm, bernoulli, poisson, expon, beta, gamma
import os

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_bayes_venn_diagram():
    """Create a Venn diagram showing Bayes theorem visually."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Sample space rectangle
    rect = patches.Rectangle((0, 0), 10, 6, linewidth=2,
                              edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)

    # Circle A (Hypothesis)
    circle_a = Circle((3.5, 3), 2.2, alpha=0.5, facecolor='#3498db',
                       edgecolor='#2980b9', linewidth=2, label='A (Hypothesis)')
    ax.add_patch(circle_a)

    # Circle B (Evidence)
    circle_b = Circle((5.5, 3), 2.2, alpha=0.5, facecolor='#e74c3c',
                       edgecolor='#c0392b', linewidth=2, label='B (Evidence)')
    ax.add_patch(circle_b)

    # Labels
    ax.text(2.2, 3, 'A only', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(4.5, 3, 'A and B', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(6.8, 3, 'B only', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(9, 5.5, 'S (Sample Space)', fontsize=10, ha='right', va='top')

    # Add Bayes formula box
    formula_box = FancyBboxPatch((0.3, -1.8), 9.4, 1.4,
                                  boxstyle="round,pad=0.1,rounding_size=0.2",
                                  facecolor='white', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(formula_box)

    ax.text(5, -1.1, r'$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} = \frac{P(A \cap B)}{P(B)}$',
            fontsize=16, ha='center', va='center', fontweight='bold')

    # Legend for regions
    ax.text(0.5, -2.5, 'P(A|B) = Proportion of B that overlaps with A',
            fontsize=11, ha='left', va='top', color='#2c3e50')

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-3, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Bayes Theorem: Visual Interpretation", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bayes_venn_diagram.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: bayes_venn_diagram.png")


def create_prior_posterior_update():
    """Create animated GIF showing prior to posterior update."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    # Parameters
    x = np.linspace(0, 1, 200)

    # Prior: Beta(2, 2) - slight belief around 0.5
    prior_a, prior_b = 2, 2
    prior = beta.pdf(x, prior_a, prior_b)

    # Data: 7 successes out of 10 trials
    successes, trials = 7, 10

    # Likelihood (scaled for visualization)
    likelihood = x**successes * (1-x)**(trials-successes)
    likelihood = likelihood / np.max(likelihood) * np.max(prior)

    # Posterior: Beta(2+7, 2+3) = Beta(9, 5)
    post_a, post_b = prior_a + successes, prior_b + (trials - successes)
    posterior = beta.pdf(x, post_a, post_b)

    # Plot Prior
    axes[0].fill_between(x, prior, alpha=0.4, color='#3498db')
    axes[0].plot(x, prior, color='#2980b9', linewidth=2)
    axes[0].set_title('Prior: Beta(2, 2)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(r'$\theta$ (probability)', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].set_xlim(0, 1)
    axes[0].axvline(x=0.5, color='#2980b9', linestyle='--', alpha=0.7, label='Prior mode')
    axes[0].legend(fontsize=9)

    # Plot Likelihood
    axes[1].fill_between(x, likelihood, alpha=0.4, color='#27ae60')
    axes[1].plot(x, likelihood, color='#1e8449', linewidth=2)
    axes[1].set_title('Likelihood: 7/10 successes', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(r'$\theta$ (probability)', fontsize=10)
    axes[1].set_xlim(0, 1)
    axes[1].axvline(x=0.7, color='#1e8449', linestyle='--', alpha=0.7, label='MLE')
    axes[1].legend(fontsize=9)

    # Plot Posterior
    axes[2].fill_between(x, posterior, alpha=0.4, color='#9b59b6')
    axes[2].plot(x, posterior, color='#8e44ad', linewidth=2)
    axes[2].set_title('Posterior: Beta(9, 5)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel(r'$\theta$ (probability)', fontsize=10)
    axes[2].set_xlim(0, 1)
    posterior_mode = (post_a - 1) / (post_a + post_b - 2)
    axes[2].axvline(x=posterior_mode, color='#8e44ad', linestyle='--', alpha=0.7,
                    label=f'MAP: {posterior_mode:.2f}')
    axes[2].legend(fontsize=9)

    # Add arrows between plots
    fig.text(0.355, 0.5, r'$\times$', fontsize=24, ha='center', va='center', fontweight='bold')
    fig.text(0.645, 0.5, r'$\propto$', fontsize=24, ha='center', va='center', fontweight='bold')

    plt.suptitle('Bayesian Update: Prior x Likelihood = Posterior',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prior_posterior_update.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: prior_posterior_update.png")

    # Create animated version
    create_prior_posterior_animation()


def create_prior_posterior_animation():
    """Create animated GIF showing sequential Bayesian updates."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    x = np.linspace(0, 1, 200)
    true_theta = 0.7

    # Start with weak prior
    prior_a, prior_b = 1, 1

    # Generate sequence of observations
    np.random.seed(42)
    observations = np.random.binomial(1, true_theta, 20)

    def update(frame):
        ax.clear()

        # Calculate posterior after 'frame' observations
        successes = int(observations[:frame].sum()) if frame > 0 else 0
        failures = frame - successes

        post_a = prior_a + successes
        post_b = prior_b + failures

        posterior = beta.pdf(x, post_a, post_b)

        ax.fill_between(x, posterior, alpha=0.4, color='#9b59b6')
        ax.plot(x, posterior, color='#8e44ad', linewidth=2.5)
        ax.axvline(x=true_theta, color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'True value: {true_theta}')

        if frame > 0:
            posterior_mean = post_a / (post_a + post_b)
            ax.axvline(x=posterior_mean, color='#8e44ad', linestyle=':', linewidth=2,
                       label=f'Posterior mean: {posterior_mean:.3f}')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 12)
        ax.set_xlabel(r'$\theta$ (probability)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Bayesian Update: {frame} observations (Successes: {successes}, Failures: {failures})',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        return ax,

    anim = animation.FuncAnimation(fig, update, frames=21, interval=500, blit=False)
    anim.save(os.path.join(OUTPUT_DIR, 'bayesian_update_animation.gif'),
              writer='pillow', fps=2)
    plt.close()
    print("Created: bayesian_update_animation.gif")


def create_distribution_gallery():
    """Create 2x3 gallery of common probability distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)

    # Color palette
    colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

    # 1. Gaussian Distribution
    ax = axes[0, 0]
    x_gauss = np.linspace(-4, 4, 200)
    for mu, sigma, ls in [(0, 1, '-'), (0, 0.5, '--'), (1, 1.5, ':')]:
        y = norm.pdf(x_gauss, mu, sigma)
        ax.plot(x_gauss, y, ls, linewidth=2, label=rf'$\mu$={mu}, $\sigma$={sigma}')
    ax.fill_between(x_gauss, norm.pdf(x_gauss, 0, 1), alpha=0.3, color=colors[0])
    ax.set_title('Gaussian (Normal)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_xlim(-4, 4)

    # 2. Bernoulli Distribution
    ax = axes[0, 1]
    ps = [0.3, 0.5, 0.7]
    width = 0.25
    for i, p in enumerate(ps):
        x_pos = np.array([0, 1]) + (i - 1) * width
        heights = [1-p, p]
        ax.bar(x_pos, heights, width=width, alpha=0.7, label=f'p={p}', color=colors[i])
    ax.set_title('Bernoulli', fontsize=12, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('P(X=k)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0 (failure)', '1 (success)'])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # 3. Poisson Distribution
    ax = axes[0, 2]
    x_pois = np.arange(0, 20)
    for lam, c in [(2, colors[0]), (5, colors[1]), (10, colors[2])]:
        y = poisson.pmf(x_pois, lam)
        ax.bar(x_pois + (lam-5)*0.15, y, width=0.4, alpha=0.7, label=rf'$\lambda$={lam}', color=c)
    ax.set_title('Poisson', fontsize=12, fontweight='bold')
    ax.set_xlabel('k (count)')
    ax.set_ylabel('P(X=k)')
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 18)

    # 4. Exponential Distribution
    ax = axes[1, 0]
    x_exp = np.linspace(0, 5, 200)
    for lam, c in [(0.5, colors[0]), (1, colors[1]), (2, colors[2])]:
        y = expon.pdf(x_exp, scale=1/lam)
        ax.plot(x_exp, y, linewidth=2, label=rf'$\lambda$={lam}', color=c)
    ax.fill_between(x_exp, expon.pdf(x_exp, scale=1), alpha=0.3, color=colors[1])
    ax.set_title('Exponential', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2.2)

    # 5. Beta Distribution
    ax = axes[1, 1]
    x_beta = np.linspace(0.01, 0.99, 200)
    params = [(1, 1), (2, 5), (5, 2), (2, 2), (0.5, 0.5)]
    for (a, b), c in zip(params, colors):
        y = beta.pdf(x_beta, a, b)
        ax.plot(x_beta, y, linewidth=2, label=f'$\\alpha$={a}, $\\beta$={b}', color=c)
    ax.set_title('Beta', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, loc='upper center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4)

    # 6. Gamma Distribution
    ax = axes[1, 2]
    x_gamma = np.linspace(0.01, 15, 200)
    params = [(1, 2), (2, 2), (3, 2), (5, 1), (9, 0.5)]
    for (k, theta), c in zip(params, colors):
        y = gamma.pdf(x_gamma, k, scale=theta)
        ax.plot(x_gamma, y, linewidth=2, label=f'k={k}, $\\theta$={theta}', color=c)
    ax.set_title('Gamma', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 0.6)

    plt.suptitle('Common Probability Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_gallery.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: distribution_gallery.png")


def create_base_rate_fallacy_tree():
    """Create tree diagram for disease testing (base rate fallacy)."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # Colors
    disease_color = '#e74c3c'
    healthy_color = '#27ae60'
    positive_color = '#f39c12'
    negative_color = '#3498db'

    # Tree structure coordinates
    # Level 0: Population
    pop_x, pop_y = 0.5, 0.95

    # Level 1: Disease / No Disease
    disease_x, disease_y = 0.2, 0.7
    healthy_x, healthy_y = 0.8, 0.7

    # Level 2: Test results
    # Disease branch
    d_pos_x, d_pos_y = 0.1, 0.35
    d_neg_x, d_neg_y = 0.3, 0.35
    # Healthy branch
    h_pos_x, h_pos_y = 0.7, 0.35
    h_neg_x, h_neg_y = 0.9, 0.35

    # Draw nodes
    def draw_node(x, y, text, color, textcolor='white', size=0.08):
        circle = Circle((x, y), size, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                fontweight='bold', color=textcolor, wrap=True)

    def draw_edge(x1, y1, x2, y2, prob_text):
        ax.annotate('', xy=(x2, y2 + 0.08), xytext=(x1, y1 - 0.08),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x - 0.05, mid_y, prob_text, fontsize=10,
                fontweight='bold', color='#2c3e50')

    # Population node
    pop_box = FancyBboxPatch((pop_x - 0.12, pop_y - 0.04), 0.24, 0.08,
                              boxstyle="round,pad=0.02", facecolor='#34495e',
                              edgecolor='black', linewidth=2)
    ax.add_patch(pop_box)
    ax.text(pop_x, pop_y, 'Population\n100,000', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # Draw edges and nodes
    draw_edge(pop_x, pop_y, disease_x, disease_y, 'P(D)=0.1%')
    draw_edge(pop_x, pop_y, healthy_x, healthy_y, 'P(H)=99.9%')

    draw_node(disease_x, disease_y, 'Disease\n100', disease_color)
    draw_node(healthy_x, healthy_y, 'Healthy\n99,900', healthy_color)

    # Disease test results
    draw_edge(disease_x, disease_y, d_pos_x, d_pos_y, '99%')
    draw_edge(disease_x, disease_y, d_neg_x, d_neg_y, '1%')
    draw_node(d_pos_x, d_pos_y, 'Test +\n99', positive_color, 'black')
    draw_node(d_neg_x, d_neg_y, 'Test -\n1', negative_color)

    # Healthy test results
    draw_edge(healthy_x, healthy_y, h_pos_x, h_pos_y, '1%')
    draw_edge(healthy_x, healthy_y, h_neg_x, h_neg_y, '99%')
    draw_node(h_pos_x, h_pos_y, 'Test +\n999', positive_color, 'black')
    draw_node(h_neg_x, h_neg_y, 'Test -\n98,901', negative_color)

    # Add calculation box
    calc_box = FancyBboxPatch((0.25, 0.02), 0.5, 0.18,
                               boxstyle="round,pad=0.02", facecolor='#ecf0f1',
                               edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(calc_box)

    calc_text = (
        "P(Disease | Test+) = True Positives / All Positives\n"
        "                   = 99 / (99 + 999)\n"
        "                   = 99 / 1098 = 9.0%"
    )
    ax.text(0.5, 0.11, calc_text, ha='center', va='center', fontsize=11,
            fontfamily='monospace', color='#2c3e50')

    # Legend
    ax.text(0.5, 0.25, 'Despite 99% test accuracy, only 9% of positives have the disease!',
            ha='center', va='center', fontsize=12, fontweight='bold', color='#c0392b',
            bbox=dict(boxstyle='round', facecolor='#fadbd8', edgecolor='#c0392b'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Base Rate Fallacy: Disease Testing Example',
                 fontsize=16, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_rate_fallacy_tree.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: base_rate_fallacy_tree.png")


def create_mle_map_comparison():
    """Create 2D contour plot comparing MLE vs MAP estimation."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10
    true_theta = np.array([2, 3])
    X = np.random.randn(n_samples, 2) + true_theta

    # Create grid for contours
    theta1 = np.linspace(-1, 5, 100)
    theta2 = np.linspace(0, 6, 100)
    T1, T2 = np.meshgrid(theta1, theta2)

    # Calculate log-likelihood
    def log_likelihood(t1, t2):
        theta = np.array([t1, t2])
        ll = -0.5 * np.sum((X - theta)**2)
        return ll

    LL = np.vectorize(log_likelihood)(T1, T2)
    LL = LL - np.max(LL)  # Normalize

    # Calculate log-prior (Gaussian centered at origin)
    prior_sigma = 2
    log_prior = -0.5 * (T1**2 + T2**2) / prior_sigma**2

    # Calculate log-posterior
    log_posterior = LL + log_prior
    log_posterior = log_posterior - np.max(log_posterior)  # Normalize

    # MLE: Sample mean
    mle = X.mean(axis=0)

    # MAP: Regularized estimate (closed form for Gaussian-Gaussian)
    map_estimate = (n_samples * mle) / (n_samples + 1/prior_sigma**2)

    # Plot 1: Likelihood
    ax = axes[0]
    contour = ax.contourf(T1, T2, np.exp(LL), levels=20, cmap='Blues', alpha=0.8)
    ax.contour(T1, T2, np.exp(LL), levels=8, colors='#2980b9', linewidths=0.5)
    ax.scatter(X[:, 0], X[:, 1], c='black', s=50, marker='x', linewidths=2, label='Data')
    ax.scatter(*mle, c='#e74c3c', s=200, marker='*', edgecolors='black',
               linewidths=2, zorder=5, label=f'MLE ({mle[0]:.2f}, {mle[1]:.2f})')
    ax.scatter(*true_theta, c='#27ae60', s=150, marker='D', edgecolors='black',
               linewidths=2, zorder=5, label=f'True ({true_theta[0]}, {true_theta[1]})')
    ax.set_title('Likelihood P(Data|$\\theta$)', fontsize=12, fontweight='bold')
    ax.set_xlabel('$\\theta_1$', fontsize=11)
    ax.set_ylabel('$\\theta_2$', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-1, 5)
    ax.set_ylim(0, 6)

    # Plot 2: Prior
    ax = axes[1]
    contour = ax.contourf(T1, T2, np.exp(log_prior - np.max(log_prior)),
                          levels=20, cmap='Greens', alpha=0.8)
    ax.contour(T1, T2, np.exp(log_prior - np.max(log_prior)),
               levels=8, colors='#27ae60', linewidths=0.5)
    ax.scatter(0, 0, c='#27ae60', s=200, marker='o', edgecolors='black',
               linewidths=2, zorder=5, label=f'Prior mode (0, 0)')
    ax.set_title('Prior P($\\theta$) ~ N(0, $\\sigma^2$I)', fontsize=12, fontweight='bold')
    ax.set_xlabel('$\\theta_1$', fontsize=11)
    ax.set_ylabel('$\\theta_2$', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-1, 5)
    ax.set_ylim(0, 6)

    # Plot 3: Posterior
    ax = axes[2]
    contour = ax.contourf(T1, T2, np.exp(log_posterior), levels=20, cmap='Purples', alpha=0.8)
    ax.contour(T1, T2, np.exp(log_posterior), levels=8, colors='#8e44ad', linewidths=0.5)
    ax.scatter(X[:, 0], X[:, 1], c='black', s=50, marker='x', linewidths=2, label='Data')
    ax.scatter(*mle, c='#e74c3c', s=200, marker='*', edgecolors='black',
               linewidths=2, zorder=5, label=f'MLE ({mle[0]:.2f}, {mle[1]:.2f})')
    ax.scatter(*map_estimate, c='#9b59b6', s=200, marker='P', edgecolors='black',
               linewidths=2, zorder=5, label=f'MAP ({map_estimate[0]:.2f}, {map_estimate[1]:.2f})')
    ax.scatter(*true_theta, c='#27ae60', s=150, marker='D', edgecolors='black',
               linewidths=2, zorder=5, label=f'True ({true_theta[0]}, {true_theta[1]})')

    # Draw arrow from MLE to MAP
    ax.annotate('', xy=map_estimate, xytext=mle,
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))
    ax.text((mle[0] + map_estimate[0])/2 - 0.3, (mle[1] + map_estimate[1])/2 + 0.2,
            'Regularization\npulls toward 0', fontsize=9, color='#c0392b')

    ax.set_title('Posterior P($\\theta$|Data)', fontsize=12, fontweight='bold')
    ax.set_xlabel('$\\theta_1$', fontsize=11)
    ax.set_ylabel('$\\theta_2$', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-1, 5)
    ax.set_ylim(0, 6)

    plt.suptitle('MLE vs MAP: Prior Regularizes the Estimate',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mle_map_comparison.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: mle_map_comparison.png")


def main():
    """Generate all visualizations."""
    print("Generating Probability and Bayes visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    create_bayes_venn_diagram()
    create_prior_posterior_update()
    create_distribution_gallery()
    create_base_rate_fallacy_tree()
    create_mle_map_comparison()

    print("-" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
