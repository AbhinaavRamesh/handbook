#!/usr/bin/env python3
"""
Temporal Difference Learning Visualizations

Generates animated GIFs and static SVGs for TD learning concepts:
1. Q-table evolution animation on GridWorld
2. SARSA vs Q-learning on cliff-walking (animated comparison)
3. TD error over training animation

Author: ML Interview Prep
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from PIL import Image
import io
import os

# Set up output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

# Color scheme
COLORS = {
    'primary': '#ff6b35',
    'secondary': '#2ecc71',
    'accent': '#3498db',
    'warning': '#e74c3c',
    'neutral': '#95a5a6',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}


class GridWorld:
    """Simple GridWorld environment for Q-learning demonstration."""

    def __init__(self, rows=4, cols=4):
        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)
        self.goal = (0, cols - 1)
        self.actions = ['up', 'down', 'left', 'right']
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

    def step(self, state, action):
        """Take action and return next_state, reward."""
        dr, dc = self.action_deltas[action]
        new_r = max(0, min(self.rows - 1, state[0] + dr))
        new_c = max(0, min(self.cols - 1, state[1] + dc))
        next_state = (new_r, new_c)

        if next_state == self.goal:
            return next_state, 10.0, True
        return next_state, -0.1, False

    def get_states(self):
        """Return all states."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]


class CliffWorld:
    """Cliff-walking environment for SARSA vs Q-learning comparison."""

    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        self.cliff = [(rows - 1, c) for c in range(1, cols - 1)]
        self.actions = ['up', 'down', 'left', 'right']
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

    def step(self, state, action):
        """Take action and return next_state, reward, done."""
        dr, dc = self.action_deltas[action]
        new_r = max(0, min(self.rows - 1, state[0] + dr))
        new_c = max(0, min(self.cols - 1, state[1] + dc))
        next_state = (new_r, new_c)

        if next_state == self.goal:
            return next_state, 0.0, True
        elif next_state in self.cliff:
            return self.start, -100.0, False
        return next_state, -1.0, False

    def get_states(self):
        """Return all states."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]


def epsilon_greedy(Q, state, epsilon, actions):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    q_values = [Q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)
    best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
    return np.random.choice(best_actions)


def q_learning_episode(env, Q, alpha=0.1, gamma=0.95, epsilon=0.1):
    """Run one episode of Q-learning."""
    state = env.start
    total_reward = 0
    steps = 0
    td_errors = []

    while steps < 500:
        action = epsilon_greedy(Q, state, epsilon, env.actions)
        next_state, reward, done = env.step(state, action)

        # Q-learning update
        current_q = Q.get((state, action), 0.0)
        max_next_q = max([Q.get((next_state, a), 0.0) for a in env.actions])
        td_error = reward + gamma * max_next_q - current_q
        Q[(state, action)] = current_q + alpha * td_error

        td_errors.append(abs(td_error))
        total_reward += reward
        steps += 1

        if done:
            break
        state = next_state

    return total_reward, td_errors


def sarsa_episode(env, Q, alpha=0.1, gamma=0.95, epsilon=0.1):
    """Run one episode of SARSA."""
    state = env.start
    action = epsilon_greedy(Q, state, epsilon, env.actions)
    total_reward = 0
    steps = 0
    td_errors = []

    while steps < 500:
        next_state, reward, done = env.step(state, action)
        next_action = epsilon_greedy(Q, next_state, epsilon, env.actions)

        # SARSA update
        current_q = Q.get((state, action), 0.0)
        next_q = Q.get((next_state, next_action), 0.0)
        td_error = reward + gamma * next_q - current_q
        Q[(state, action)] = current_q + alpha * td_error

        td_errors.append(abs(td_error))
        total_reward += reward
        steps += 1

        if done:
            break
        state = next_state
        action = next_action

    return total_reward, td_errors


def generate_qtable_evolution():
    """Generate Q-table evolution animation on simple GridWorld."""
    print("Generating Q-table evolution animation...")

    env = GridWorld(rows=4, cols=4)
    Q = {}

    # Train and capture snapshots
    snapshots = []
    episodes_per_snapshot = 50
    total_snapshots = 12

    for snap in range(total_snapshots):
        for _ in range(episodes_per_snapshot):
            q_learning_episode(env, Q, alpha=0.2, gamma=0.9, epsilon=0.3)
        snapshots.append((snap * episodes_per_snapshot, Q.copy()))

    # Generate frames
    frames = []

    for episode_num, Q_snap in snapshots:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create Q-value heatmap (max Q per state)
        q_grid = np.zeros((env.rows, env.cols))
        for r in range(env.rows):
            for c in range(env.cols):
                q_values = [Q_snap.get(((r, c), a), 0.0) for a in env.actions]
                q_grid[r, c] = max(q_values) if q_values else 0

        # Normalize for visualization
        vmin, vmax = -1, 10

        # Draw cells
        cmap = plt.cm.RdYlGn
        for r in range(env.rows):
            for c in range(env.cols):
                q_val = q_grid[r, c]
                norm_val = (q_val - vmin) / (vmax - vmin)
                norm_val = max(0, min(1, norm_val))
                color = cmap(norm_val)

                rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                                  facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)

                # Draw Q-value
                ax.text(c, r, f'{q_val:.1f}', ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       color='white' if norm_val > 0.5 or norm_val < 0.2 else 'black')

                # Draw arrows for best action
                best_action = max(env.actions,
                                  key=lambda a: Q_snap.get(((r, c), a), 0.0))
                dx, dy = env.action_deltas[best_action]
                if (r, c) != env.goal:
                    ax.annotate('', xy=(c + dy * 0.3, r + dx * 0.3),
                               xytext=(c, r),
                               arrowprops=dict(arrowstyle='->', color='darkblue',
                                             lw=2, alpha=0.7))

        # Mark start and goal
        ax.plot(env.start[1], env.start[0], 'o', markersize=25,
                markerfacecolor=COLORS['accent'], markeredgecolor='black',
                markeredgewidth=2, label='Start')
        ax.plot(env.goal[1], env.goal[0], '*', markersize=30,
                markerfacecolor=COLORS['secondary'], markeredgecolor='black',
                markeredgewidth=2, label='Goal')

        ax.set_xlim(-0.6, env.cols - 0.4)
        ax.set_ylim(env.rows - 0.4, -0.6)
        ax.set_aspect('equal')
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.grid(False)

        ax.set_title(f'Q-Table Evolution: Episode {episode_num}\n'
                    f'Values propagate backward from goal',
                    fontsize=14, fontweight='bold', color=COLORS['dark'])
        ax.legend(loc='upper left', fontsize=10)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Max Q-Value', fontsize=11)

        plt.tight_layout()

        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close()

    # Save as GIF
    output_path = os.path.join(output_dir, 'qtable_evolution.gif')
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=800,
        loop=0
    )
    print(f"Saved Q-table evolution to: {output_path}")


def generate_sarsa_vs_qlearning():
    """Generate SARSA vs Q-learning comparison on cliff-walking."""
    print("Generating SARSA vs Q-learning comparison...")

    env = CliffWorld(rows=4, cols=12)

    # Train both algorithms
    Q_sarsa = {}
    Q_qlearn = {}

    sarsa_rewards = []
    qlearn_rewards = []

    n_episodes = 500

    for ep in range(n_episodes):
        epsilon = max(0.05, 1.0 - ep / 200)

        r_sarsa, _ = sarsa_episode(env, Q_sarsa, alpha=0.1, gamma=0.95, epsilon=epsilon)
        r_qlearn, _ = q_learning_episode(env, Q_qlearn, alpha=0.1, gamma=0.95, epsilon=epsilon)

        sarsa_rewards.append(r_sarsa)
        qlearn_rewards.append(r_qlearn)

    # Generate comparison frames
    frames = []

    def get_greedy_path(Q, env, max_steps=50):
        """Extract greedy path from Q-table."""
        path = [env.start]
        state = env.start
        steps = 0
        while state != env.goal and steps < max_steps:
            q_values = [Q.get((state, a), 0.0) for a in env.actions]
            best_idx = np.argmax(q_values)
            action = env.actions[best_idx]
            next_state, _, done = env.step(state, action)
            if next_state in path:  # Avoid loops
                break
            path.append(next_state)
            state = next_state
            steps += 1
            if done:
                break
        return path

    # Create frames showing paths evolving
    checkpoints = [0, 50, 100, 200, 300, 400, 500]

    for checkpoint in checkpoints:
        # Train up to checkpoint
        Q_s = {}
        Q_q = {}
        for ep in range(checkpoint):
            epsilon = max(0.05, 1.0 - ep / 200)
            sarsa_episode(env, Q_s, alpha=0.1, gamma=0.95, epsilon=epsilon)
            q_learning_episode(env, Q_q, alpha=0.1, gamma=0.95, epsilon=epsilon)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        for ax, Q, title, color, algo in [
            (axes[0], Q_s, 'SARSA (On-Policy)', COLORS['accent'], 'SARSA'),
            (axes[1], Q_q, 'Q-Learning (Off-Policy)', COLORS['secondary'], 'Q-Learn')
        ]:
            # Draw grid
            for r in range(env.rows):
                for c in range(env.cols):
                    if (r, c) in env.cliff:
                        rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor=COLORS['warning'], alpha=0.8,
                                        edgecolor='black', linewidth=1)
                    elif (r, c) == env.goal:
                        rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor=COLORS['secondary'], alpha=0.6,
                                        edgecolor='black', linewidth=1)
                    elif (r, c) == env.start:
                        rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor=COLORS['accent'], alpha=0.6,
                                        edgecolor='black', linewidth=1)
                    else:
                        rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor=COLORS['light'], alpha=0.5,
                                        edgecolor='black', linewidth=1)
                    ax.add_patch(rect)

            # Get and draw path
            if checkpoint > 0:
                path = get_greedy_path(Q, env)
                path_x = [p[1] for p in path]
                path_y = [p[0] for p in path]
                ax.plot(path_x, path_y, '-o', color=color, linewidth=3,
                       markersize=10, markeredgecolor='black', markeredgewidth=1,
                       label=f'{algo} path', zorder=5)

            # Labels
            ax.text(env.start[1], env.start[0], 'S', ha='center', va='center',
                   fontsize=14, fontweight='bold')
            ax.text(env.goal[1], env.goal[0], 'G', ha='center', va='center',
                   fontsize=14, fontweight='bold')
            ax.text(env.cols / 2, env.rows - 1, 'CLIFF', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')

            ax.set_xlim(-0.6, env.cols - 0.4)
            ax.set_ylim(env.rows - 0.4, -0.6)
            ax.set_aspect('equal')
            ax.set_title(f'{title}\nEpisode {checkpoint}', fontsize=13, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            if checkpoint > 0:
                ax.legend(loc='upper right', fontsize=10)

        plt.suptitle('Cliff Walking: SARSA learns safer path, Q-Learning learns optimal path',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close()

    # Save as GIF
    output_path = os.path.join(output_dir, 'sarsa_vs_qlearning.gif')
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1200,
        loop=0
    )
    print(f"Saved SARSA vs Q-learning comparison to: {output_path}")


def generate_td_error_animation():
    """Generate TD error over training animation."""
    print("Generating TD error animation...")

    env = GridWorld(rows=4, cols=4)
    Q = {}

    # Collect TD errors over training
    all_td_errors = []
    episode_rewards = []

    for ep in range(200):
        epsilon = max(0.05, 1.0 - ep / 100)
        reward, td_errors = q_learning_episode(env, Q, alpha=0.2, gamma=0.9, epsilon=epsilon)
        avg_td = np.mean(td_errors) if td_errors else 0
        all_td_errors.append(avg_td)
        episode_rewards.append(reward)

    # Smooth the data
    window = 10
    smoothed_td = np.convolve(all_td_errors, np.ones(window)/window, mode='valid')
    smoothed_reward = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

    # Generate animation frames
    frames = []
    frame_points = list(range(10, len(smoothed_td), 5)) + [len(smoothed_td)]

    for n_points in frame_points:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # TD Error plot
        ax1 = axes[0]
        x = np.arange(window - 1, window - 1 + n_points)
        ax1.fill_between(x, 0, smoothed_td[:n_points], alpha=0.3, color=COLORS['primary'])
        ax1.plot(x, smoothed_td[:n_points], color=COLORS['primary'], linewidth=2)
        ax1.scatter([x[-1]], [smoothed_td[n_points-1]], s=100, c=COLORS['primary'],
                   edgecolors='black', linewidth=2, zorder=5)

        ax1.set_xlim(0, 200)
        ax1.set_ylim(0, max(smoothed_td) * 1.1)
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Average |TD Error|', fontsize=12)
        ax1.set_title('TD Error Decreases as Learning Converges',
                     fontsize=13, fontweight='bold', color=COLORS['dark'])
        ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.7, label='Low error threshold')
        ax1.legend(loc='upper right')

        # Reward plot
        ax2 = axes[1]
        ax2.fill_between(x, min(smoothed_reward) - 1, smoothed_reward[:n_points],
                        alpha=0.3, color=COLORS['secondary'])
        ax2.plot(x, smoothed_reward[:n_points], color=COLORS['secondary'], linewidth=2)
        ax2.scatter([x[-1]], [smoothed_reward[n_points-1]], s=100, c=COLORS['secondary'],
                   edgecolors='black', linewidth=2, zorder=5)

        ax2.set_xlim(0, 200)
        ax2.set_ylim(min(smoothed_reward) - 0.5, max(smoothed_reward) + 0.5)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Episode Reward (smoothed)', fontsize=12)
        ax2.set_title('Episode Reward Increases as Policy Improves',
                     fontsize=13, fontweight='bold', color=COLORS['dark'])
        ax2.axhline(y=9.5, color='gray', linestyle='--', alpha=0.7, label='Near-optimal reward')
        ax2.legend(loc='lower right')

        plt.tight_layout()

        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close()

    # Save as GIF
    output_path = os.path.join(output_dir, 'td_error_training.gif')
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )
    print(f"Saved TD error animation to: {output_path}")


def generate_td_concept_diagram():
    """Generate static SVG showing TD learning concept."""
    print("Generating TD concept diagram...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw timeline
    timeline_y = 0.5
    ax.axhline(y=timeline_y, color=COLORS['dark'], linewidth=3, xmin=0.05, xmax=0.95)

    # States
    states = [
        (0.1, 'S_t', 'Current\nState'),
        (0.35, 'R_{t+1}', 'Reward'),
        (0.6, 'S_{t+1}', 'Next\nState'),
        (0.85, '...', 'Continue')
    ]

    for x, label, desc in states:
        circle = Circle((x, timeline_y), 0.05, facecolor=COLORS['accent'],
                        edgecolor='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, timeline_y + 0.15, f'${label}$', ha='center', va='bottom',
               fontsize=14, fontweight='bold')
        ax.text(x, timeline_y - 0.12, desc, ha='center', va='top',
               fontsize=10, style='italic')

    # Arrows between states
    ax.annotate('', xy=(0.3, timeline_y), xytext=(0.15, timeline_y),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(0.55, timeline_y), xytext=(0.4, timeline_y),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(0.8, timeline_y), xytext=(0.65, timeline_y),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # TD Update box
    box_y = 0.1
    update_box = Rectangle((0.2, box_y - 0.08), 0.6, 0.16,
                           facecolor=COLORS['light'], edgecolor=COLORS['primary'],
                           linewidth=3, zorder=3)
    ax.add_patch(update_box)

    ax.text(0.5, box_y, r'$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$',
           ha='center', va='center', fontsize=14, fontweight='bold')

    # Annotations
    ax.annotate('TD Target\n(Bootstrapped)', xy=(0.58, box_y + 0.08),
               xytext=(0.75, box_y + 0.25),
               arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2),
               fontsize=10, ha='center', color=COLORS['secondary'])

    ax.annotate('TD Error\n(Learning Signal)', xy=(0.5, box_y - 0.08),
               xytext=(0.5, box_y - 0.25),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2),
               fontsize=10, ha='center', color=COLORS['warning'])

    # Title
    ax.text(0.5, 0.9, 'Temporal Difference Learning: Update After Each Step',
           ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['dark'])

    ax.text(0.5, 0.8, 'Bootstrap from estimate V(S_{t+1}) instead of waiting for episode end',
           ha='center', va='center', fontsize=12, style='italic', color=COLORS['neutral'])

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'td_concept.svg')
    plt.savefig(output_path, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved TD concept diagram to: {output_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating TD Learning Visualizations")
    print("=" * 60)

    np.random.seed(42)

    generate_qtable_evolution()
    generate_sarsa_vs_qlearning()
    generate_td_error_animation()
    generate_td_concept_diagram()

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
