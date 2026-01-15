"""
PPO & Modern Policy Optimization Visualizations

This script generates visualizations for the PPO module:
1. PPO clipped objective function visualization (SVG)
2. Trust region concept animation (GIF)
3. PPO vs vanilla policy gradient stability comparison (GIF)
4. SAC entropy bonus effect (SVG)

All visualizations are designed for interview preparation content.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/reinforcement-learning/assets/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_ppo_clipped_objective():
    """
    Create visualization showing the PPO clipped objective function.
    Shows how the objective is clipped for both positive and negative advantages.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PPO Clipped Surrogate Objective', fontsize=14, fontweight='bold')

    epsilon = 0.2
    r = np.linspace(0, 2.5, 500)

    # Colors
    color_unclipped = '#1f77b4'
    color_clipped = '#2ca02c'
    color_final = '#d62728'

    # === Left plot: Positive Advantage (A > 0) ===
    ax1 = axes[0]
    A_pos = 1.0  # Positive advantage

    # Unclipped objective: r * A
    obj_unclipped = r * A_pos

    # Clipped ratio
    r_clipped = np.clip(r, 1 - epsilon, 1 + epsilon)
    obj_clipped = r_clipped * A_pos

    # Final objective: min(unclipped, clipped)
    obj_final = np.minimum(obj_unclipped, obj_clipped)

    ax1.plot(r, obj_unclipped, '--', color=color_unclipped, linewidth=2,
             label=r'$r_t(\theta) \cdot \hat{A}_t$ (unclipped)', alpha=0.7)
    ax1.plot(r, obj_clipped, '--', color=color_clipped, linewidth=2,
             label=r'$\text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t$', alpha=0.7)
    ax1.plot(r, obj_final, '-', color=color_final, linewidth=3,
             label=r'$L^{CLIP}$ (minimum)')

    # Mark the clip region
    ax1.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.5)
    ax1.axvspan(1-epsilon, 1+epsilon, alpha=0.1, color='green', label='Trust Region')

    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axvline(x=1, color='black', linewidth=0.5, linestyle='-', alpha=0.3)

    ax1.set_xlabel(r'Probability Ratio $r_t(\theta) = \pi_\theta / \pi_{\theta_{old}}$', fontsize=11)
    ax1.set_ylabel('Objective Value', fontsize=11)
    ax1.set_title(r'Positive Advantage ($\hat{A}_t > 0$)' + '\nGradient clipped when ratio too high', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(-0.5, 3)

    # Add annotation
    ax1.annotate('Clipped!\nNo gradient', xy=(1.8, 1.2), fontsize=9,
                 ha='center', color=color_final)

    # === Right plot: Negative Advantage (A < 0) ===
    ax2 = axes[1]
    A_neg = -1.0  # Negative advantage

    # Unclipped objective: r * A
    obj_unclipped = r * A_neg

    # Clipped ratio
    r_clipped = np.clip(r, 1 - epsilon, 1 + epsilon)
    obj_clipped = r_clipped * A_neg

    # Final objective: min(unclipped, clipped)
    # For negative A, min picks the more negative value
    obj_final = np.minimum(obj_unclipped, obj_clipped)

    ax2.plot(r, obj_unclipped, '--', color=color_unclipped, linewidth=2,
             label=r'$r_t(\theta) \cdot \hat{A}_t$ (unclipped)', alpha=0.7)
    ax2.plot(r, obj_clipped, '--', color=color_clipped, linewidth=2,
             label=r'$\text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t$', alpha=0.7)
    ax2.plot(r, obj_final, '-', color=color_final, linewidth=3,
             label=r'$L^{CLIP}$ (minimum)')

    # Mark the clip region
    ax2.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.5)
    ax2.axvspan(1-epsilon, 1+epsilon, alpha=0.1, color='green', label='Trust Region')

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axvline(x=1, color='black', linewidth=0.5, linestyle='-', alpha=0.3)

    ax2.set_xlabel(r'Probability Ratio $r_t(\theta) = \pi_\theta / \pi_{\theta_{old}}$', fontsize=11)
    ax2.set_ylabel('Objective Value', fontsize=11)
    ax2.set_title(r'Negative Advantage ($\hat{A}_t < 0$)' + '\nGradient clipped when ratio too low', fontsize=11)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(-3, 0.5)

    # Add annotation
    ax2.annotate('Clipped!\nNo gradient', xy=(0.4, -0.8), fontsize=9,
                 ha='center', color=color_final)

    plt.tight_layout()

    # Add formula box
    textstr = (r'$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,'
               r' \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$'
               '\n\n'
               r'$\epsilon = 0.2$ (typical value)')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    fig.text(0.5, 0.02, textstr, fontsize=10, ha='center',
             bbox=props, family='monospace')

    output_path = os.path.join(OUTPUT_DIR, 'ppo_clipped_objective.svg')
    plt.savefig(output_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def create_trust_region_animation():
    """
    Create animation showing trust region concept in policy space.
    Shows how updates are constrained to stay within a region.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a simple 2D "policy parameter" space with a reward landscape
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Create a reward landscape with multiple peaks
    Z = (np.exp(-((X-1)**2 + (Y-1)**2)) +
         0.8 * np.exp(-((X+1)**2 + (Y+0.5)**2)) +
         0.5 * np.exp(-((X-0.5)**2 + (Y+1.5)**2)))

    # Starting point and optimal trajectory
    trajectory_unconstrained = [
        (-2.0, -1.5),
        (-0.5, 0.8),  # Big jump - too far!
        (1.5, 2.5),   # Overshoots
        (-1.0, -0.5), # Bounces back
        (0.5, 1.2),
        (1.0, 1.0)    # Eventually reaches
    ]

    trajectory_trust_region = [
        (-2.0, -1.5),
        (-1.5, -1.0),
        (-1.0, -0.5),
        (-0.5, 0.0),
        (0.0, 0.4),
        (0.4, 0.7),
        (0.7, 0.9),
        (0.9, 1.0),
        (1.0, 1.0)
    ]

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()

        # Draw reward landscape
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)

        # Mark optimal point
        ax.plot(1, 1, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=1.5)
        ax.annotate('Optimal', (1, 1), (1.3, 1.3), fontsize=10, color='white',
                   fontweight='bold')

        # Draw trajectories up to current frame
        n_frames_per_trajectory = max(len(trajectory_unconstrained), len(trajectory_trust_region))

        # Unconstrained trajectory (red)
        if frame > 0:
            idx_u = min(frame, len(trajectory_unconstrained) - 1)
            traj_u = trajectory_unconstrained[:idx_u + 1]
            if len(traj_u) > 1:
                xs, ys = zip(*traj_u)
                ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.8, label='Unconstrained (unstable)')
                ax.plot(xs, ys, 'ro', markersize=8)

            # Current position for unconstrained
            curr_x, curr_y = traj_u[-1]
            ax.plot(curr_x, curr_y, 'r^', markersize=15, markeredgecolor='white',
                   markeredgewidth=2)

        # Trust region trajectory (green)
        if frame > 0:
            idx_t = min(frame, len(trajectory_trust_region) - 1)
            traj_t = trajectory_trust_region[:idx_t + 1]
            if len(traj_t) > 1:
                xs, ys = zip(*traj_t)
                ax.plot(xs, ys, 'g-', linewidth=2, alpha=0.8, label='Trust Region (stable)')
                ax.plot(xs, ys, 'go', markersize=8)

            # Current position for trust region
            curr_x, curr_y = traj_t[-1]
            ax.plot(curr_x, curr_y, 'g^', markersize=15, markeredgecolor='white',
                   markeredgewidth=2)

            # Draw trust region circle
            circle = Circle((curr_x, curr_y), 0.5, fill=False, color='lime',
                           linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
            ax.annotate('Trust\nRegion', (curr_x + 0.6, curr_y + 0.3), fontsize=9,
                       color='lime', fontweight='bold')

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('Policy Parameter 1', fontsize=11)
        ax.set_ylabel('Policy Parameter 2', fontsize=11)
        ax.set_title('Trust Region Optimization\nConstrained updates prevent overshooting',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)

        # Frame counter
        ax.text(2.5, -2.7, f'Step: {frame}', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        return []

    n_frames = max(len(trajectory_unconstrained), len(trajectory_trust_region)) + 2
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=800, blit=False)

    output_path = os.path.join(OUTPUT_DIR, 'trust_region_concept.gif')
    anim.save(output_path, writer='pillow', fps=1.5, dpi=100)
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def create_ppo_vs_pg_stability():
    """
    Create animation comparing PPO vs vanilla policy gradient training stability.
    Shows how PPO maintains stable learning while vanilla PG can collapse.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Stability: PPO vs Vanilla Policy Gradient',
                fontsize=14, fontweight='bold')

    np.random.seed(42)
    n_frames = 50

    # Generate training curves
    # Vanilla PG: unstable, can collapse
    x = np.arange(n_frames)

    # Vanilla PG reward curve - unstable with potential collapse
    pg_base = np.cumsum(np.random.randn(n_frames) * 0.3 + 0.05)
    pg_noise = np.random.randn(n_frames) * 2
    pg_rewards = pg_base + pg_noise
    # Add a collapse event
    collapse_start = 25
    pg_rewards[collapse_start:] = pg_rewards[collapse_start] - np.cumsum(
        np.abs(np.random.randn(n_frames - collapse_start)) * 0.8)

    # PPO reward curve - stable improvement
    ppo_base = np.cumsum(np.random.randn(n_frames) * 0.1 + 0.15)
    ppo_noise = np.random.randn(n_frames) * 0.5
    ppo_rewards = ppo_base + ppo_noise
    ppo_rewards = np.maximum.accumulate(ppo_rewards * 0.3 + np.linspace(0, 8, n_frames))

    # KL divergence during training
    pg_kl = np.abs(np.cumsum(np.random.randn(n_frames) * 0.1))
    pg_kl[collapse_start:] = pg_kl[collapse_start] + np.cumsum(
        np.abs(np.random.randn(n_frames - collapse_start)) * 0.5)

    ppo_kl = np.clip(np.abs(np.random.randn(n_frames) * 0.1) + 0.1, 0, 0.4)

    def init():
        return []

    def animate(frame):
        for ax in axes:
            ax.clear()

        # Left plot: Reward curves
        ax1 = axes[0]
        ax1.plot(x[:frame+1], pg_rewards[:frame+1], 'r-', linewidth=2,
                label='Vanilla PG', alpha=0.8)
        ax1.plot(x[:frame+1], ppo_rewards[:frame+1], 'g-', linewidth=2,
                label='PPO', alpha=0.8)

        ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax1.set_xlabel('Training Steps (x100)', fontsize=11)
        ax1.set_ylabel('Episode Reward', fontsize=11)
        ax1.set_title('Reward During Training', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.set_xlim(0, n_frames)
        ax1.set_ylim(-15, 15)

        # Add collapse annotation
        if frame > collapse_start + 5:
            ax1.annotate('Policy Collapse!',
                        xy=(collapse_start, pg_rewards[collapse_start]),
                        xytext=(collapse_start + 5, pg_rewards[collapse_start] + 5),
                        fontsize=10, color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))

        # Right plot: KL divergence
        ax2 = axes[1]
        ax2.plot(x[:frame+1], pg_kl[:frame+1], 'r-', linewidth=2,
                label='Vanilla PG', alpha=0.8)
        ax2.plot(x[:frame+1], ppo_kl[:frame+1], 'g-', linewidth=2,
                label='PPO', alpha=0.8)

        # Trust region boundary
        ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.5,
                   label='PPO Clip Threshold')
        ax2.fill_between([0, n_frames], 0, 0.2, alpha=0.1, color='green')

        ax2.set_xlabel('Training Steps (x100)', fontsize=11)
        ax2.set_ylabel('KL Divergence from Previous Policy', fontsize=11)
        ax2.set_title('Policy Update Magnitude', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.set_xlim(0, n_frames)
        ax2.set_ylim(0, 5)

        # Add annotation for large KL
        if frame > collapse_start + 5:
            ax2.annotate('Unbounded\nupdates',
                        xy=(frame, pg_kl[frame]),
                        xytext=(frame - 10, pg_kl[frame] - 1),
                        fontsize=10, color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))
            ax2.annotate('Constrained\nby clipping',
                        xy=(frame, ppo_kl[frame]),
                        xytext=(frame - 10, ppo_kl[frame] + 1),
                        fontsize=10, color='green',
                        arrowprops=dict(arrowstyle='->', color='green'))

        plt.tight_layout()
        return []

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=150, blit=False)

    output_path = os.path.join(OUTPUT_DIR, 'ppo_vs_pg_stability.gif')
    anim.save(output_path, writer='pillow', fps=8, dpi=100)
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def create_pg_instability_animation():
    """
    Create animation showing policy gradient instability / death spiral.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    np.random.seed(123)
    n_frames = 40

    # Simulate a training curve that collapses
    x = np.arange(n_frames)

    # Initial good progress
    rewards = np.zeros(n_frames)
    rewards[:15] = np.cumsum(np.random.randn(15) * 0.3 + 0.2)

    # Then collapse
    collapse_rate = np.abs(np.random.randn(n_frames - 15)) * 0.5 + 0.3
    rewards[15:] = rewards[14] - np.cumsum(collapse_rate)

    def animate(frame):
        ax.clear()

        ax.plot(x[:frame+1], rewards[:frame+1], 'b-', linewidth=2)
        ax.plot(x[frame], rewards[frame], 'ro', markersize=10)

        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('Episode Reward', fontsize=11)
        ax.set_title('Vanilla Policy Gradient: Training Collapse\n'
                    'Large updates cause death spiral', fontsize=13, fontweight='bold')
        ax.set_xlim(0, n_frames)
        ax.set_ylim(-10, 5)

        if frame > 15:
            ax.annotate('Bad policy\ncollects bad data',
                       xy=(frame, rewards[frame]),
                       xytext=(frame - 5, rewards[frame] + 2),
                       fontsize=10, color='red',
                       arrowprops=dict(arrowstyle='->', color='red'))

            # Add death spiral explanation - positioned lower right to avoid title overlap
            ax.text(n_frames * 0.65, -7.5, 'Death Spiral:\n'
                   '1. Large gradient step\n'
                   '2. Policy degrades\n'
                   '3. Worse data collected\n'
                   '4. Worse gradients\n'
                   '5. Repeat...', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                   verticalalignment='top')

        return []

    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=200, blit=False)

    output_path = os.path.join(OUTPUT_DIR, 'pg_instability.gif')
    anim.save(output_path, writer='pillow', fps=5, dpi=100)
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def create_sac_entropy_bonus():
    """
    Create visualization showing SAC's entropy bonus effect on policy learning.
    Compares policies with and without entropy regularization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('SAC Entropy Bonus: Effect on Policy Learning',
                fontsize=14, fontweight='bold')

    # Action space
    actions = np.linspace(-3, 3, 200)

    # === Left plot: Low entropy (deterministic-like) policy ===
    ax1 = axes[0]

    # Narrow Gaussian centered on "best" action
    mu_det = 1.0
    sigma_det = 0.2
    policy_det = np.exp(-0.5 * ((actions - mu_det) / sigma_det) ** 2) / (sigma_det * np.sqrt(2 * np.pi))

    ax1.fill_between(actions, policy_det, alpha=0.6, color='#d62728')
    ax1.plot(actions, policy_det, color='#d62728', linewidth=2)
    ax1.axvline(x=mu_det, color='#d62728', linestyle='--', alpha=0.5)

    # Show Q-values
    q_values = -0.5 * (actions - 0.8) ** 2 + 2 + 0.3 * np.sin(actions * 2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(actions, q_values, 'k--', linewidth=1.5, alpha=0.5, label='Q(s, a)')
    ax1_twin.set_ylabel('Q-value', fontsize=10, color='gray')
    ax1_twin.tick_params(axis='y', colors='gray')

    ax1.set_xlabel('Action', fontsize=11)
    ax1.set_ylabel('Policy Probability', fontsize=11)
    ax1.set_title('Low Entropy Policy\n(No entropy bonus)', fontsize=11)
    ax1.set_xlim(-3, 3)
    ax1.text(0, ax1.get_ylim()[1] * 0.9, f'H = {0.5:.2f}', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === Middle plot: High entropy (SAC) policy ===
    ax2 = axes[1]

    # Wider Gaussian - more exploration
    mu_sac = 0.8
    sigma_sac = 0.8
    policy_sac = np.exp(-0.5 * ((actions - mu_sac) / sigma_sac) ** 2) / (sigma_sac * np.sqrt(2 * np.pi))

    ax2.fill_between(actions, policy_sac, alpha=0.6, color='#2ca02c')
    ax2.plot(actions, policy_sac, color='#2ca02c', linewidth=2)
    ax2.axvline(x=mu_sac, color='#2ca02c', linestyle='--', alpha=0.5)

    # Show Q-values
    ax2_twin = ax2.twinx()
    ax2_twin.plot(actions, q_values, 'k--', linewidth=1.5, alpha=0.5, label='Q(s, a)')
    ax2_twin.set_ylabel('Q-value', fontsize=10, color='gray')
    ax2_twin.tick_params(axis='y', colors='gray')

    ax2.set_xlabel('Action', fontsize=11)
    ax2.set_ylabel('Policy Probability', fontsize=11)
    ax2.set_title('High Entropy Policy (SAC)\n(With entropy bonus)', fontsize=11)
    ax2.set_xlim(-3, 3)
    ax2.text(0, ax2.get_ylim()[1] * 0.9, f'H = {1.8:.2f}', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === Right plot: Multi-modal policy ===
    ax3 = axes[2]

    # Mixture of Gaussians - multiple good actions
    mu1, mu2 = -1.0, 1.5
    sigma_mm = 0.5
    policy_mm = (0.4 * np.exp(-0.5 * ((actions - mu1) / sigma_mm) ** 2) +
                 0.6 * np.exp(-0.5 * ((actions - mu2) / sigma_mm) ** 2))
    policy_mm = policy_mm / np.trapezoid(policy_mm, actions)  # Normalize

    ax3.fill_between(actions, policy_mm, alpha=0.6, color='#9467bd')
    ax3.plot(actions, policy_mm, color='#9467bd', linewidth=2)
    ax3.axvline(x=mu1, color='#9467bd', linestyle='--', alpha=0.5)
    ax3.axvline(x=mu2, color='#9467bd', linestyle='--', alpha=0.5)

    # Show Q-values with two modes
    q_values_mm = (np.exp(-0.5 * ((actions - mu1) / 1.0) ** 2) +
                   np.exp(-0.5 * ((actions - mu2) / 1.0) ** 2))
    ax3_twin = ax3.twinx()
    ax3_twin.plot(actions, q_values_mm, 'k--', linewidth=1.5, alpha=0.5)
    ax3_twin.set_ylabel('Q-value', fontsize=10, color='gray')
    ax3_twin.tick_params(axis='y', colors='gray')

    ax3.set_xlabel('Action', fontsize=11)
    ax3.set_ylabel('Policy Probability', fontsize=11)
    ax3.set_title('Multi-modal Policy (SAC)\nMaintains multiple good strategies', fontsize=11)
    ax3.set_xlim(-3, 3)
    ax3.text(-2.5, ax3.get_ylim()[1] * 0.9, 'Both modes\nare valid!', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Add formula box
    textstr = (r'SAC Objective: $J(\pi) = \sum_t \mathbb{E}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$'
               '\n'
               r'Entropy: $\mathcal{H}(\pi) = -\mathbb{E}[\log \pi(a|s)]$')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    fig.text(0.5, 0.02, textstr, fontsize=10, ha='center', bbox=props)

    output_path = os.path.join(OUTPUT_DIR, 'sac_entropy_bonus.svg')
    plt.savefig(output_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def main():
    """Generate all visualizations."""
    print("Generating PPO & Modern Methods visualizations...")
    print("=" * 50)

    # Create static visualizations (SVG)
    create_ppo_clipped_objective()
    create_sac_entropy_bonus()

    # Create animations (GIF)
    create_trust_region_animation()
    create_ppo_vs_pg_stability()
    create_pg_instability_animation()

    print("=" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
