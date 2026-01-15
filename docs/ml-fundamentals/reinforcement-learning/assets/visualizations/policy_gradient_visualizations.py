#!/usr/bin/env python3
"""
Policy Gradient Visualizations

Generates visualizations for the Policy Gradients module:
1. Policy probability evolution animation (GIF)
2. Gradient direction in parameter space (SVG)
3. Variance reduction with baseline comparison (SVG)

Author: ML Interview Playbook
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Color scheme - consistent green palette
GREEN_DARK = '#1B5E20'
GREEN_MEDIUM = '#43A047'
GREEN_LIGHT = '#81C784'
GREEN_PALE = '#C8E6C9'
GREEN_VERY_PALE = '#E8F5E9'
ACCENT_ORANGE = '#FF9800'
ACCENT_RED = '#D32F2F'
ACCENT_BLUE = '#1976D2'

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_policy_evolution_animation():
    """
    Create an animated GIF showing policy probability evolution during training.

    Shows a discrete action space with 5 actions, where the policy starts
    uniform and gradually concentrates on the optimal action.
    """
    print("Creating policy evolution animation...")

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Parameters
    n_actions = 5
    n_frames = 60
    actions = ['Left', 'Down', 'Stay', 'Up', 'Right']
    optimal_action = 3  # 'Up' is optimal

    # Simulate policy evolution (softmax of evolving logits)
    np.random.seed(42)
    initial_logits = np.zeros(n_actions)

    def get_policy(logits):
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    # Pre-compute all frames
    all_policies = []
    all_logits = []
    logits = initial_logits.copy()

    for frame in range(n_frames):
        # Gradient update: increase logit for optimal action
        # Simulate REINFORCE updates with positive returns
        gradient = np.zeros(n_actions)
        gradient[optimal_action] = 0.15  # Positive gradient for optimal
        gradient += np.random.randn(n_actions) * 0.02  # Noise

        logits = logits + gradient
        policy = get_policy(logits)

        all_policies.append(policy)
        all_logits.append(logits.copy())

    # Initialize bars
    bars = ax1.bar(actions, all_policies[0], color=[GREEN_LIGHT]*n_actions,
                   edgecolor=GREEN_DARK, linewidth=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Action', fontsize=12, fontweight='bold')
    title1 = ax1.set_title('Policy Distribution $\\pi_\\theta(a|s)$\nIteration: 0',
                           fontsize=14, fontweight='bold', color=GREEN_DARK)

    # Optimal action marker
    ax1.axhline(y=0.95, color=ACCENT_ORANGE, linestyle='--', alpha=0.7,
                label='Target probability')
    ax1.legend(loc='upper left')

    # Second subplot: logits over time
    lines = []
    ax2.set_xlim(0, n_frames)
    ax2.set_ylim(-2, 5)
    ax2.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Logit Value', fontsize=12, fontweight='bold')
    ax2.set_title('Logits Evolution Over Training', fontsize=14,
                  fontweight='bold', color=GREEN_DARK)

    colors = [GREEN_PALE, GREEN_LIGHT, GREEN_MEDIUM, ACCENT_ORANGE, GREEN_PALE]
    for i, action in enumerate(actions):
        line, = ax2.plot([], [], color=colors[i], linewidth=2.5 if i == optimal_action else 1.5,
                         label=action, alpha=1.0 if i == optimal_action else 0.6)
        lines.append(line)
    ax2.legend(loc='upper left', fontsize=9)

    # Storage for line data
    x_data = []
    y_data = [[] for _ in range(n_actions)]

    def animate(frame):
        # Update bar heights
        policy = all_policies[frame]
        for bar, height in zip(bars, policy):
            bar.set_height(height)

        # Color bars - highlight optimal action
        for i, bar in enumerate(bars):
            if i == optimal_action:
                bar.set_color(ACCENT_ORANGE)
            else:
                bar.set_color(GREEN_LIGHT)

        # Update title
        title1.set_text(f'Policy Distribution $\\pi_\\theta(a|s)$\nIteration: {frame}')

        # Update logit lines
        x_data.append(frame)
        for i in range(n_actions):
            y_data[i].append(all_logits[frame][i])
            lines[i].set_data(x_data, y_data[i])

        return list(bars) + lines + [title1]

    # Create animation
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=True)

    # Save as GIF
    output_path = OUTPUT_DIR / 'policy_evolution.gif'
    anim.save(output_path, writer=PillowWriter(fps=10), dpi=100)
    print(f"Saved: {output_path}")

    plt.close()


def create_gradient_direction_visualization():
    """
    Create a static SVG showing gradient directions in 2D parameter space.

    Visualizes the policy gradient landscape and shows how gradients
    point toward regions of higher expected return.
    """
    print("Creating gradient direction visualization...")

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a 2D parameter space
    theta1 = np.linspace(-3, 3, 50)
    theta2 = np.linspace(-3, 3, 50)
    T1, T2 = np.meshgrid(theta1, theta2)

    # Expected return surface (simplified model)
    # Peak at (1, 1) representing optimal policy parameters
    J = np.exp(-((T1 - 1)**2 + (T2 - 1)**2) / 2) - 0.3 * np.exp(-((T1 + 1)**2 + (T2 + 1)**2) / 1)
    J = J * 10  # Scale for visualization

    # Plot contours of expected return
    contour = ax.contourf(T1, T2, J, levels=20, cmap='Greens', alpha=0.8)
    ax.contour(T1, T2, J, levels=20, colors=GREEN_DARK, linewidths=0.5, alpha=0.5)
    cbar = plt.colorbar(contour, ax=ax, label='Expected Return $J(\\theta)$')
    cbar.ax.tick_params(labelsize=10)

    # Compute gradients for quiver plot
    dJ_dtheta1, dJ_dtheta2 = np.gradient(J, theta1[1] - theta1[0])

    # Subsample for cleaner arrows
    skip = 4
    T1_sub = T1[::skip, ::skip]
    T2_sub = T2[::skip, ::skip]
    dJ1_sub = dJ_dtheta1[::skip, ::skip]
    dJ2_sub = dJ_dtheta2[::skip, ::skip]

    # Normalize arrows for visibility
    magnitude = np.sqrt(dJ1_sub**2 + dJ2_sub**2) + 1e-8
    dJ1_norm = dJ1_sub / magnitude * 0.3
    dJ2_norm = dJ2_sub / magnitude * 0.3

    # Plot gradient arrows
    ax.quiver(T1_sub, T2_sub, dJ1_norm, dJ2_norm,
              color=ACCENT_RED, alpha=0.8, scale=5, width=0.008,
              headwidth=4, headlength=5)

    # Mark optimal point
    ax.scatter([1], [1], s=200, c=ACCENT_ORANGE, marker='*',
               edgecolors=GREEN_DARK, linewidths=2, zorder=5, label='Optimal $\\theta^*$')

    # Show a sample training trajectory
    trajectory_theta1 = [-2, -1.5, -0.8, 0, 0.5, 0.8, 1]
    trajectory_theta2 = [-2, -1.2, -0.5, 0.2, 0.6, 0.9, 1]
    ax.plot(trajectory_theta1, trajectory_theta2, 'o-', color=ACCENT_BLUE,
            markersize=8, linewidth=2.5, label='Training trajectory', zorder=4)
    ax.scatter(trajectory_theta1[0], trajectory_theta2[0], s=150, c=ACCENT_BLUE,
               marker='s', edgecolors=GREEN_DARK, linewidths=2, zorder=6)
    ax.annotate('Start', (trajectory_theta1[0], trajectory_theta2[0]),
                xytext=(-2.5, -2.8), fontsize=11, fontweight='bold', color=ACCENT_BLUE)

    # Labels and title
    ax.set_xlabel('$\\theta_1$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$\\theta_2$', fontsize=14, fontweight='bold')
    ax.set_title('Policy Gradient Ascent in Parameter Space\n'
                 '$\\nabla_\\theta J(\\theta)$ points toward higher expected return',
                 fontsize=14, fontweight='bold', color=GREEN_DARK)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    # Add annotation box
    textstr = 'Gradient ascent:\n$\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta J(\\theta)$'
    props = dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK, alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save as SVG
    output_path = OUTPUT_DIR / 'gradient_direction.svg'
    plt.savefig(output_path, format='svg', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close()


def create_variance_comparison_visualization():
    """
    Create a static SVG comparing gradient variance with and without baseline.

    Shows histograms of gradient estimates and demonstrates how
    baseline subtraction reduces variance while maintaining unbiasedness.
    """
    print("Creating variance comparison visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    np.random.seed(42)
    n_samples = 1000

    # Simulate gradient estimates
    # Without baseline: high variance, centered around true gradient
    true_gradient = 2.0
    returns = np.random.normal(100, 30, n_samples)  # High mean, high variance
    gradients_no_baseline = returns * np.random.randn(n_samples) * 0.1 + true_gradient

    # With baseline (subtracting mean return)
    baseline = np.mean(returns)
    returns_centered = returns - baseline
    gradients_with_baseline = returns_centered * np.random.randn(n_samples) * 0.1 + true_gradient

    # With advantage (even lower variance)
    advantage = np.random.normal(0, 5, n_samples)  # Much smaller scale
    gradients_with_advantage = advantage * np.random.randn(n_samples) * 0.1 + true_gradient

    # ===== Plot 1: Without Baseline =====
    ax1 = axes[0, 0]
    ax1.hist(gradients_no_baseline, bins=50, density=True, color=ACCENT_RED,
             edgecolor='white', alpha=0.8, label='Gradient estimates')
    ax1.axvline(true_gradient, color=GREEN_DARK, linewidth=3, linestyle='--',
                label=f'True gradient = {true_gradient}')
    ax1.axvline(np.mean(gradients_no_baseline), color=ACCENT_ORANGE, linewidth=2,
                linestyle='-', label=f'Mean = {np.mean(gradients_no_baseline):.2f}')

    var_no_baseline = np.var(gradients_no_baseline)
    ax1.set_xlabel('Gradient Estimate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'REINFORCE (No Baseline)\nVariance = {var_no_baseline:.1f}',
                  fontsize=14, fontweight='bold', color=ACCENT_RED)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(-50, 50)

    # ===== Plot 2: With Baseline =====
    ax2 = axes[0, 1]
    ax2.hist(gradients_with_baseline, bins=50, density=True, color=ACCENT_ORANGE,
             edgecolor='white', alpha=0.8, label='Gradient estimates')
    ax2.axvline(true_gradient, color=GREEN_DARK, linewidth=3, linestyle='--',
                label=f'True gradient = {true_gradient}')
    ax2.axvline(np.mean(gradients_with_baseline), color=GREEN_MEDIUM, linewidth=2,
                linestyle='-', label=f'Mean = {np.mean(gradients_with_baseline):.2f}')

    var_with_baseline = np.var(gradients_with_baseline)
    ax2.set_xlabel('Gradient Estimate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'With Baseline $b(s) = \\bar{{R}}$\nVariance = {var_with_baseline:.1f}',
                  fontsize=14, fontweight='bold', color=ACCENT_ORANGE)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(-50, 50)

    # ===== Plot 3: With Advantage =====
    ax3 = axes[1, 0]
    ax3.hist(gradients_with_advantage, bins=50, density=True, color=GREEN_MEDIUM,
             edgecolor='white', alpha=0.8, label='Gradient estimates')
    ax3.axvline(true_gradient, color=GREEN_DARK, linewidth=3, linestyle='--',
                label=f'True gradient = {true_gradient}')
    ax3.axvline(np.mean(gradients_with_advantage), color=ACCENT_BLUE, linewidth=2,
                linestyle='-', label=f'Mean = {np.mean(gradients_with_advantage):.2f}')

    var_with_advantage = np.var(gradients_with_advantage)
    ax3.set_xlabel('Gradient Estimate', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax3.set_title(f'With Advantage $A(s,a) = Q - V$\nVariance = {var_with_advantage:.1f}',
                  fontsize=14, fontweight='bold', color=GREEN_MEDIUM)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_xlim(-50, 50)

    # ===== Plot 4: Variance Comparison Bar Chart =====
    ax4 = axes[1, 1]
    methods = ['No Baseline', 'With Baseline', 'With Advantage']
    variances = [var_no_baseline, var_with_baseline, var_with_advantage]
    colors = [ACCENT_RED, ACCENT_ORANGE, GREEN_MEDIUM]

    bars = ax4.bar(methods, variances, color=colors, edgecolor=GREEN_DARK, linewidth=2)

    # Add variance reduction percentages
    for i, (bar, var) in enumerate(zip(bars, variances)):
        if i == 0:
            text = f'{var:.1f}'
        else:
            reduction = (1 - var / variances[0]) * 100
            text = f'{var:.1f}\n({reduction:.0f}% reduction)'
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, text,
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax4.set_ylabel('Variance of Gradient Estimate', fontsize=12, fontweight='bold')
    ax4.set_title('Variance Reduction Comparison', fontsize=14, fontweight='bold', color=GREEN_DARK)
    ax4.set_ylim(0, max(variances) * 1.3)

    # Add insight box
    insight_text = ('Key Insight:\n'
                    'All estimates are unbiased (same mean),\n'
                    'but variance differs dramatically.\n'
                    'Lower variance = faster, more stable learning.')
    props = dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK, alpha=0.9)
    ax4.text(0.98, 0.02, insight_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.suptitle('Policy Gradient Variance Reduction Techniques',
                 fontsize=16, fontweight='bold', color=GREEN_DARK, y=1.02)
    plt.tight_layout()

    # Save as SVG
    output_path = OUTPUT_DIR / 'variance_comparison.svg'
    plt.savefig(output_path, format='svg', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close()


def create_reinforce_update_animation():
    """
    Create an animated GIF showing how REINFORCE updates work.

    Visualizes the policy update rule and shows how positive/negative
    returns affect action probabilities.
    """
    print("Creating REINFORCE update animation...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plt.style.use('seaborn-v0_8-whitegrid')

    n_actions = 4
    n_frames = 40
    actions = ['Action A', 'Action B', 'Action C', 'Action D']

    # Simulate episodes with different returns
    np.random.seed(123)

    # Pre-compute trajectory: actions taken and returns received
    episodes = []
    for _ in range(n_frames):
        action_taken = np.random.randint(0, n_actions)
        # Action B (index 1) is optimal, gives high returns
        if action_taken == 1:
            return_val = np.random.normal(10, 2)
        elif action_taken == 2:
            return_val = np.random.normal(-5, 2)  # Action C is bad
        else:
            return_val = np.random.normal(2, 3)
        episodes.append((action_taken, return_val))

    # Simulate policy evolution
    logits = np.zeros(n_actions)
    all_policies = [np.ones(n_actions) / n_actions]  # Start uniform
    all_returns = [0]
    cumulative_returns = [0]

    lr = 0.1
    for action_taken, return_val in episodes:
        # REINFORCE update: increase logit for taken action proportional to return
        gradient = np.zeros(n_actions)
        gradient[action_taken] = return_val * lr
        logits = logits + gradient

        # Convert to policy via softmax
        exp_logits = np.exp(logits - np.max(logits))
        policy = exp_logits / np.sum(exp_logits)

        all_policies.append(policy)
        all_returns.append(return_val)
        cumulative_returns.append(cumulative_returns[-1] + return_val)

    # Initialize plots
    bars = ax1.bar(actions, all_policies[0], color=[GREEN_LIGHT]*n_actions,
                   edgecolor=GREEN_DARK, linewidth=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Probability $\\pi_\\theta(a|s)$', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Action', fontsize=12, fontweight='bold')
    title1 = ax1.set_title('Policy Update\nEpisode: 0', fontsize=14,
                           fontweight='bold', color=GREEN_DARK)

    # Return indicator
    return_text = ax1.text(0.5, 0.95, '', transform=ax1.transAxes, fontsize=12,
                           ha='center', va='top', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', edgecolor=GREEN_DARK))

    # Cumulative return plot
    line, = ax2.plot([], [], color=GREEN_MEDIUM, linewidth=2.5, label='Cumulative Return')
    ax2.set_xlim(0, n_frames)
    ax2.set_ylim(min(cumulative_returns) - 10, max(cumulative_returns) + 10)
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax2.set_title('Learning Progress', fontsize=14, fontweight='bold', color=GREEN_DARK)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')

    x_data = []
    y_data = []

    def animate(frame):
        # Update policy bars
        policy = all_policies[frame]
        for i, (bar, height) in enumerate(zip(bars, policy)):
            bar.set_height(height)
            # Color based on probability
            if height > 0.4:
                bar.set_color(GREEN_DARK)
            elif height > 0.25:
                bar.set_color(GREEN_MEDIUM)
            else:
                bar.set_color(GREEN_LIGHT)

        # Update return text
        if frame > 0:
            ret = all_returns[frame]
            color = GREEN_DARK if ret > 0 else ACCENT_RED
            return_text.set_text(f'Return: {ret:+.1f}')
            return_text.set_color(color)
        else:
            return_text.set_text('Starting...')

        title1.set_text(f'Policy Update\nEpisode: {frame}')

        # Update cumulative return line
        x_data.append(frame)
        y_data.append(cumulative_returns[frame])
        line.set_data(x_data, y_data)

        return list(bars) + [line, title1, return_text]

    # Create animation
    anim = FuncAnimation(fig, animate, frames=n_frames + 1, interval=200, blit=True)

    # Save as GIF
    output_path = OUTPUT_DIR / 'reinforce_update.gif'
    anim.save(output_path, writer=PillowWriter(fps=5), dpi=100)
    print(f"Saved: {output_path}")

    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Policy Gradient Visualizations Generator")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate all visualizations
    create_policy_evolution_animation()
    create_gradient_direction_visualization()
    create_variance_comparison_visualization()
    create_reinforce_update_animation()

    print()
    print("=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    for f in OUTPUT_DIR.glob("*"):
        if f.suffix in ['.gif', '.svg', '.png']:
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
