#!/usr/bin/env python3
"""
Generate visualization images and animations for Exploration vs Exploitation module.

Outputs:
- bandit_selection.gif: Multi-armed bandit arm selection animation
- regret_comparison.gif: Animated regret curves comparing strategies
- decay_schedules.svg: Exploration decay schedule visualization (static)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os

# Set the output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'epsilon': '#e74c3c',
    'ucb': '#3498db',
    'thompson': '#2ecc71',
    'optimal': '#9b59b6',
    'highlight': '#f39c12',
    'background': '#ecf0f1',
    'text': '#2c3e50'
}


def save_fig(name, fig=None):
    """Save figure to the assets directory."""
    if fig is None:
        fig = plt.gcf()
    filepath = os.path.join(OUTPUT_DIR, name)
    if name.endswith('.svg'):
        fig.savefig(filepath, format='svg', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    else:
        fig.savefig(filepath, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Generated: {name}")


def save_animation(anim, name, fps=10):
    """Save animation as GIF."""
    filepath = os.path.join(OUTPUT_DIR, name)
    writer = PillowWriter(fps=fps)
    anim.save(filepath, writer=writer)
    plt.close()
    print(f"Generated: {name}")


class BanditEnvironment:
    """Simple multi-armed bandit environment."""

    def __init__(self, means, seed=42):
        self.means = np.array(means)
        self.k = len(means)
        self.rng = np.random.default_rng(seed)

    def pull(self, arm):
        """Pull an arm and get reward (Bernoulli)."""
        return self.rng.random() < self.means[arm]


class EpsilonGreedy:
    """Epsilon-greedy bandit algorithm."""

    def __init__(self, k, epsilon=0.1, seed=42):
        self.k = k
        self.epsilon = epsilon
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.rng = np.random.default_rng(seed)

    def select_arm(self):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.k)
        return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UCB:
    """Upper Confidence Bound algorithm."""

    def __init__(self, k, c=2.0, seed=42):
        self.k = k
        self.c = c
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def select_arm(self):
        self.t += 1
        # Initialize: pull each arm once
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSampling:
    """Thompson Sampling for Bernoulli bandits."""

    def __init__(self, k, seed=42):
        self.k = k
        self.alpha = np.ones(k)  # Successes + 1
        self.beta = np.ones(k)   # Failures + 1
        self.rng = np.random.default_rng(seed)

    def select_arm(self):
        samples = self.rng.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


def create_bandit_selection_animation():
    """Create animated visualization of bandit arm selection."""

    # Setup
    means = [0.3, 0.5, 0.7, 0.4, 0.6]
    k = len(means)
    n_steps = 100

    env = BanditEnvironment(means, seed=42)
    agent = UCB(k, c=2.0, seed=42)

    # Pre-compute all selections and rewards
    selections = []
    rewards = []
    cumulative_rewards = [0]
    counts_history = [np.zeros(k)]
    values_history = [np.zeros(k)]

    for t in range(n_steps):
        arm = agent.select_arm()
        reward = env.pull(arm)
        agent.update(arm, reward)

        selections.append(arm)
        rewards.append(reward)
        cumulative_rewards.append(cumulative_rewards[-1] + reward)
        counts_history.append(agent.counts.copy())
        values_history.append(agent.values.copy())

    # Create figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

    ax_arms = fig.add_subplot(gs[0, :])
    ax_counts = fig.add_subplot(gs[1, 0])
    ax_rewards = fig.add_subplot(gs[1, 1])

    # Arm visualization
    arm_positions = np.arange(k) * 2
    arm_colors = [COLORS['background']] * k

    def init():
        ax_arms.clear()
        ax_counts.clear()
        ax_rewards.clear()
        return []

    def animate(frame):
        ax_arms.clear()
        ax_counts.clear()
        ax_rewards.clear()

        t = frame

        # Draw slot machines
        for i in range(k):
            x = arm_positions[i]

            # Machine body
            rect = mpatches.FancyBboxPatch((x - 0.4, 0), 0.8, 1.5,
                                           boxstyle="round,pad=0.05",
                                           facecolor='#34495e',
                                           edgecolor='#2c3e50',
                                           linewidth=2)
            ax_arms.add_patch(rect)

            # Display showing estimated value
            if t > 0:
                val = values_history[t][i]
                display_color = plt.cm.RdYlGn(val)
            else:
                display_color = '#7f8c8d'

            display = mpatches.Rectangle((x - 0.3, 0.8), 0.6, 0.5,
                                         facecolor=display_color,
                                         edgecolor='#2c3e50',
                                         linewidth=1)
            ax_arms.add_patch(display)

            # Value text
            if t > 0 and counts_history[t][i] > 0:
                ax_arms.text(x, 1.05, f'{values_history[t][i]:.2f}',
                            ha='center', va='center', fontsize=10,
                            fontweight='bold', color='white')

            # Pull count
            count = int(counts_history[t][i]) if t > 0 else 0
            ax_arms.text(x, -0.3, f'n={count}', ha='center', va='center',
                        fontsize=9, color=COLORS['text'])

            # True mean (hidden from agent)
            ax_arms.text(x, -0.6, f'(mu={means[i]:.1f})', ha='center', va='center',
                        fontsize=8, color='#95a5a6', style='italic')

            # Arm label
            ax_arms.text(x, 1.7, f'Arm {i+1}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=COLORS['text'])

            # Highlight selected arm
            if t > 0 and selections[t-1] == i:
                highlight = mpatches.FancyBboxPatch((x - 0.5, -0.1), 1.0, 1.9,
                                                    boxstyle="round,pad=0.05",
                                                    facecolor='none',
                                                    edgecolor=COLORS['highlight'],
                                                    linewidth=3,
                                                    linestyle='--')
                ax_arms.add_patch(highlight)

                # Show reward
                reward_text = 'WIN!' if rewards[t-1] else 'LOSE'
                reward_color = COLORS['thompson'] if rewards[t-1] else COLORS['epsilon']
                ax_arms.text(x, 2.1, reward_text, ha='center', va='center',
                            fontsize=12, fontweight='bold', color=reward_color)

        ax_arms.set_xlim(-1, k * 2)
        ax_arms.set_ylim(-1, 2.5)
        ax_arms.set_aspect('equal')
        ax_arms.axis('off')
        ax_arms.set_title(f'UCB Bandit Selection - Step {t}/{n_steps}',
                         fontsize=14, fontweight='bold', color=COLORS['text'])

        # Pull counts bar chart
        if t > 0:
            colors = [COLORS['ucb'] if i == np.argmax(means) else '#95a5a6'
                     for i in range(k)]
            bars = ax_counts.bar(range(k), counts_history[t], color=colors,
                                edgecolor='#2c3e50', linewidth=1)
            ax_counts.set_xlabel('Arm', fontsize=10)
            ax_counts.set_ylabel('Pull Count', fontsize=10)
            ax_counts.set_title('Arm Pull Distribution', fontsize=12, fontweight='bold')
            ax_counts.set_xticks(range(k))
            ax_counts.set_xticklabels([f'{i+1}' for i in range(k)])

        # Cumulative reward
        ax_rewards.plot(range(t+1), cumulative_rewards[:t+1],
                       color=COLORS['ucb'], linewidth=2, label='UCB')

        # Optimal (always pulling best arm)
        optimal_rewards = np.arange(t+1) * max(means)
        ax_rewards.plot(range(t+1), optimal_rewards,
                       color=COLORS['optimal'], linewidth=2,
                       linestyle='--', label='Optimal', alpha=0.7)

        ax_rewards.set_xlabel('Time Step', fontsize=10)
        ax_rewards.set_ylabel('Cumulative Reward', fontsize=10)
        ax_rewards.set_title('Cumulative Reward Over Time', fontsize=12, fontweight='bold')
        ax_rewards.legend(loc='upper left', fontsize=9)
        ax_rewards.set_xlim(0, n_steps)
        ax_rewards.set_ylim(0, n_steps * max(means) * 1.1)

        return []

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=n_steps + 1, interval=150, blit=False)

    save_animation(anim, 'bandit_selection.gif', fps=8)


def create_regret_comparison_animation():
    """Create animated comparison of regret curves for different strategies."""

    # Setup
    means = [0.3, 0.5, 0.7, 0.4, 0.6]
    k = len(means)
    optimal_mean = max(means)
    n_steps = 200
    n_runs = 20  # Average over multiple runs

    # Run simulations
    def run_simulation(agent_class, **kwargs):
        all_regrets = []
        for run in range(n_runs):
            env = BanditEnvironment(means, seed=run)
            agent = agent_class(k, seed=run + 100, **kwargs)

            cumulative_regret = 0
            regrets = [0]

            for t in range(n_steps):
                arm = agent.select_arm()
                reward = env.pull(arm)
                agent.update(arm, reward)

                # Regret is the gap between optimal and chosen arm's true mean
                cumulative_regret += optimal_mean - means[arm]
                regrets.append(cumulative_regret)

            all_regrets.append(regrets)

        return np.mean(all_regrets, axis=0)

    # Pre-compute all regret curves
    regret_epsilon = run_simulation(EpsilonGreedy, epsilon=0.1)
    regret_ucb = run_simulation(UCB, c=2.0)
    regret_thompson = run_simulation(ThompsonSampling)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()

        t = frame * 2  # Skip frames for speed
        t = min(t, n_steps)

        times = np.arange(t + 1)

        # Plot regret curves up to current time
        ax.plot(times, regret_epsilon[:t+1],
               color=COLORS['epsilon'], linewidth=2.5,
               label=r'$\epsilon$-greedy ($\epsilon=0.1$)')

        ax.plot(times, regret_ucb[:t+1],
               color=COLORS['ucb'], linewidth=2.5,
               label='UCB (c=2)')

        ax.plot(times, regret_thompson[:t+1],
               color=COLORS['thompson'], linewidth=2.5,
               label='Thompson Sampling')

        # Theoretical bounds (for reference)
        if t > 10:
            sqrt_bound = 3 * np.sqrt(k * times * np.log(times + 1))
            ax.plot(times[10:], sqrt_bound[10:],
                   color='#95a5a6', linewidth=1.5,
                   linestyle=':', label=r'$O(\sqrt{KT \log T})$ bound', alpha=0.7)

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Cumulative Regret', fontsize=12)
        ax.set_title(f'Regret Comparison - Step {t}/{n_steps}\n(Averaged over {n_runs} runs)',
                    fontsize=14, fontweight='bold', color=COLORS['text'])
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(0, n_steps)
        ax.set_ylim(0, max(regret_epsilon) * 1.1)

        # Add current regret values
        if t > 0:
            text_x = n_steps * 0.7
            ax.text(text_x, regret_epsilon[t] + 1,
                   f'e-greedy: {regret_epsilon[t]:.1f}',
                   fontsize=10, color=COLORS['epsilon'], fontweight='bold')
            ax.text(text_x, regret_ucb[t] + 1,
                   f'UCB: {regret_ucb[t]:.1f}',
                   fontsize=10, color=COLORS['ucb'], fontweight='bold')
            ax.text(text_x, regret_thompson[t] + 1,
                   f'Thompson: {regret_thompson[t]:.1f}',
                   fontsize=10, color=COLORS['thompson'], fontweight='bold')

        ax.grid(True, alpha=0.3)

        return []

    frames = (n_steps // 2) + 1
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=frames, interval=100, blit=False)

    save_animation(anim, 'regret_comparison.gif', fps=10)


def create_decay_schedules_visualization():
    """Create static visualization of epsilon decay schedules."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    T = 1000
    t = np.arange(1, T + 1)

    eps_0 = 1.0
    eps_min = 0.01

    # Different decay schedules
    linear = np.maximum(eps_min, eps_0 - t / T)
    exponential = eps_0 * (0.995 ** t)
    exponential = np.maximum(eps_min, exponential)
    inverse = eps_0 / (1 + t / (T / 10))
    inverse = np.maximum(eps_min, inverse)

    # Left plot: Decay curves
    ax = axes[0]
    ax.plot(t, linear, color=COLORS['epsilon'], linewidth=2.5,
           label='Linear: $\\epsilon_0 - t/T$')
    ax.plot(t, exponential, color=COLORS['ucb'], linewidth=2.5,
           label='Exponential: $\\epsilon_0 \\cdot \\gamma^t$')
    ax.plot(t, inverse, color=COLORS['thompson'], linewidth=2.5,
           label='Inverse: $\\epsilon_0 / (1 + t/T)$')

    ax.axhline(y=eps_min, color='#95a5a6', linestyle='--', linewidth=1.5,
              label=f'$\\epsilon_{{min}}$ = {eps_min}')

    ax.set_xlabel('Time Step (t)', fontsize=12)
    ax.set_ylabel('Exploration Rate ($\\epsilon_t$)', fontsize=12)
    ax.set_title('Epsilon Decay Schedules', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Right plot: Cumulative exploration
    ax = axes[1]

    cumulative_linear = np.cumsum(linear)
    cumulative_exp = np.cumsum(exponential)
    cumulative_inverse = np.cumsum(inverse)

    ax.plot(t, cumulative_linear, color=COLORS['epsilon'], linewidth=2.5,
           label='Linear')
    ax.plot(t, cumulative_exp, color=COLORS['ucb'], linewidth=2.5,
           label='Exponential')
    ax.plot(t, cumulative_inverse, color=COLORS['thompson'], linewidth=2.5,
           label='Inverse')

    ax.set_xlabel('Time Step (t)', fontsize=12)
    ax.set_ylabel('Cumulative Exploration Budget', fontsize=12)
    ax.set_title('Total Exploration Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, T)
    ax.grid(True, alpha=0.3)

    # Add annotations
    final_budgets = [cumulative_linear[-1], cumulative_exp[-1], cumulative_inverse[-1]]
    labels = ['Linear', 'Exponential', 'Inverse']
    colors = [COLORS['epsilon'], COLORS['ucb'], COLORS['thompson']]

    for budget, label, color in zip(final_budgets, labels, colors):
        ax.annotate(f'{label}: {budget:.0f}',
                   xy=(T, budget), xytext=(T * 0.75, budget),
                   fontsize=9, color=color, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=color, lw=1))

    plt.tight_layout()
    save_fig('decay_schedules.svg', fig)


def create_ucb_bonus_visualization():
    """Create visualization showing UCB exploration bonus over time."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: UCB bonus as function of pull count
    ax = axes[0]

    t_values = [50, 100, 500, 1000]
    n_range = np.arange(1, 101)
    c = 2.0

    for t in t_values:
        bonus = c * np.sqrt(np.log(t) / n_range)
        ax.plot(n_range, bonus, linewidth=2, label=f't = {t}')

    ax.set_xlabel('Pull Count ($n_a$)', fontsize=12)
    ax.set_ylabel('Exploration Bonus', fontsize=12)
    ax.set_title('UCB Bonus: $c\\sqrt{\\ln(t)/n_a}$', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 4)
    ax.grid(True, alpha=0.3)

    # Right: UCB selection example
    ax = axes[1]

    arms = ['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4']
    q_values = [0.6, 0.5, 0.7, 0.4]
    n_values = [100, 10, 50, 5]
    t = 165  # sum of pulls

    bonuses = [c * np.sqrt(np.log(t) / n) for n in n_values]
    ucb_values = [q + b for q, b in zip(q_values, bonuses)]

    x = np.arange(len(arms))
    width = 0.35

    bars1 = ax.bar(x - width/2, q_values, width, label='Q(a)',
                   color=COLORS['ucb'], edgecolor='#2c3e50', linewidth=1)
    bars2 = ax.bar(x + width/2, bonuses, width, label='Bonus',
                   color=COLORS['highlight'], edgecolor='#2c3e50', linewidth=1)

    # Add UCB values on top
    for i, (q, b, ucb) in enumerate(zip(q_values, bonuses, ucb_values)):
        ax.text(i, q + b + 0.05, f'UCB={ucb:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(i - width/2, q + 0.02, f'{q:.2f}',
               ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, b + 0.02, f'{b:.2f}',
               ha='center', va='bottom', fontsize=8)

    # Highlight winner
    winner_idx = np.argmax(ucb_values)
    ax.annotate('Selected!', xy=(winner_idx, ucb_values[winner_idx] + 0.15),
               fontsize=11, fontweight='bold', color=COLORS['thompson'],
               ha='center')

    ax.set_xlabel('Arm', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('UCB Selection Example (t=165)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{arm}\n(n={n})' for arm, n in zip(arms, n_values)])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig('ucb_bonus.svg', fig)


def create_thompson_posterior_visualization():
    """Create visualization of Thompson Sampling posteriors."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Three stages: initial, after 10 pulls, after 100 pulls
    stages = [
        {'title': 'Initial Prior', 'alphas': [1, 1, 1], 'betas': [1, 1, 1]},
        {'title': 'After 20 Pulls', 'alphas': [3, 5, 8], 'betas': [5, 3, 4]},
        {'title': 'After 100 Pulls', 'alphas': [18, 35, 55], 'betas': [32, 15, 25]},
    ]

    true_means = [0.3, 0.7, 0.65]
    colors = [COLORS['epsilon'], COLORS['ucb'], COLORS['thompson']]

    x = np.linspace(0, 1, 200)

    for ax, stage in zip(axes, stages):
        from scipy import stats

        for i, (alpha, beta, color, true_mean) in enumerate(
            zip(stage['alphas'], stage['betas'], colors, true_means)):

            # Beta distribution
            y = stats.beta.pdf(x, alpha, beta)
            ax.plot(x, y, color=color, linewidth=2,
                   label=f'Arm {i+1} (true={true_mean})')
            ax.fill_between(x, y, alpha=0.2, color=color)

            # True mean vertical line
            ax.axvline(true_mean, color=color, linestyle='--',
                      linewidth=1, alpha=0.7)

        ax.set_xlabel('$\\mu$ (Success Probability)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(stage['title'], fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Thompson Sampling: Posterior Evolution',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig('thompson_posteriors.svg', fig)


if __name__ == '__main__':
    print("Generating Exploration vs Exploitation visualizations...")
    print("=" * 60)

    # Generate animated visualizations
    print("\n1. Creating bandit selection animation...")
    create_bandit_selection_animation()

    print("\n2. Creating regret comparison animation...")
    create_regret_comparison_animation()

    # Generate static visualizations
    print("\n3. Creating decay schedules visualization...")
    create_decay_schedules_visualization()

    print("\n4. Creating UCB bonus visualization...")
    create_ucb_bonus_visualization()

    print("\n5. Creating Thompson posterior visualization...")
    create_thompson_posterior_visualization()

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
