#!/usr/bin/env python3
"""
Value Functions Visualizations for Reinforcement Learning

Generates:
1. V(s) heatmap on GridWorld (animated convergence) - GIF
2. Bellman backup diagram - SVG
3. Value iteration convergence animation - GIF

Color palette: Green-based to match project theme
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# Green color palette (matching project theme)
GREEN_DARK = '#1B5E20'
GREEN_MEDIUM = '#43A047'
GREEN_LIGHT = '#81C784'
GREEN_PALE = '#C8E6C9'
GREEN_VERY_PALE = '#E8F5E9'
ACCENT_RED = '#D32F2F'
ACCENT_ORANGE = '#FF9800'
ACCENT_BLUE = '#1976D2'

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GridWorld Environment
# =============================================================================

class GridWorld:
    """Simple 4x4 GridWorld environment for visualization."""

    def __init__(self, size=4, gamma=0.9):
        self.size = size
        self.gamma = gamma
        self.n_states = size * size

        # Terminal states: top-left (0,0) = +1, bottom-right (3,3) = +1
        self.terminal_states = [(0, 0), (size-1, size-1)]
        self.terminal_rewards = {(0, 0): 1.0, (size-1, size-1): 1.0}

        # Actions: up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['Up', 'Down', 'Left', 'Right']

        # Reward for non-terminal transitions
        self.step_reward = -0.04

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_next_state(self, state, action_idx):
        """Get next state given current state and action."""
        if self.is_terminal(state):
            return state

        action = self.actions[action_idx]
        next_row = state[0] + action[0]
        next_col = state[1] + action[1]

        # Stay in bounds
        if 0 <= next_row < self.size and 0 <= next_col < self.size:
            return (next_row, next_col)
        return state

    def get_reward(self, state, action_idx, next_state):
        """Get reward for transition."""
        if next_state in self.terminal_rewards:
            return self.terminal_rewards[next_state]
        return self.step_reward


def value_iteration_step(env, V):
    """Perform one step of value iteration."""
    V_new = np.zeros_like(V)

    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)

            if env.is_terminal(state):
                V_new[i, j] = env.terminal_rewards.get(state, 0)
                continue

            # Compute max over actions
            action_values = []
            for a in range(len(env.actions)):
                next_state = env.get_next_state(state, a)
                reward = env.get_reward(state, a, next_state)
                action_values.append(reward + env.gamma * V[next_state[0], next_state[1]])

            V_new[i, j] = max(action_values)

    return V_new


def run_value_iteration(env, n_iterations=20):
    """Run value iteration and store all intermediate value functions."""
    V = np.zeros((env.size, env.size))
    V_history = [V.copy()]

    for _ in range(n_iterations):
        V = value_iteration_step(env, V)
        V_history.append(V.copy())

    return V_history


# =============================================================================
# Visualization 1: V(s) Heatmap Convergence Animation
# =============================================================================

def create_value_function_animation():
    """Create animated GIF showing V(s) convergence on GridWorld."""
    print("Creating V(s) convergence animation...")

    env = GridWorld(size=4, gamma=0.9)
    V_history = run_value_iteration(env, n_iterations=25)

    # Create figure with space for colorbar
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create initial plot to get colorbar (will be updated in animation)
    V_init = V_history[0]
    im = ax.imshow(V_init, cmap='RdYlGn', vmin=-1, vmax=1.5)
    cbar = plt.colorbar(im, ax=ax, label='State Value V(s)', shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    # Add legend as figure text (outside the axes, bottom right)
    legend_text = f'gamma = {env.gamma} | Terminal states: corners | Goal reward: +1'
    fig.text(0.5, 0.02, legend_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK, alpha=0.9))

    def animate(frame):
        ax.clear()
        V = V_history[min(frame, len(V_history)-1)]

        # Create heatmap
        im = ax.imshow(V, cmap='RdYlGn', vmin=-1, vmax=1.5)

        # Add value annotations
        for i in range(env.size):
            for j in range(env.size):
                value = V[i, j]
                text_color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       fontsize=14, fontweight='bold', color=text_color)

        # Mark terminal states
        for term_state in env.terminal_states:
            rect = mpatches.Rectangle((term_state[1]-0.5, term_state[0]-0.5), 1, 1,
                                      linewidth=3, edgecolor=GREEN_DARK,
                                      facecolor='none', linestyle='--')
            ax.add_patch(rect)

        # Labels and title
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        ax.set_xticklabels([f'Col {i}' for i in range(env.size)])
        ax.set_yticklabels([f'Row {i}' for i in range(env.size)])
        ax.set_title(f'Value Iteration: V(s) Convergence\nIteration {frame}',
                    fontsize=16, fontweight='bold', color=GREEN_DARK)

        return [im]

    # Adjust layout to make room for legend at bottom and title at top
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(V_history), interval=400, blit=False)

    # Save as GIF
    output_path = OUTPUT_DIR / 'value_function_convergence.gif'
    anim.save(output_path, writer=PillowWriter(fps=2.5))
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# Visualization 2: Bellman Backup Diagram
# =============================================================================

def create_bellman_backup_diagram():
    """Create static SVG showing Bellman backup structure."""
    print("Creating Bellman backup diagram...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: V(s) backup diagram
    ax1 = axes[0]
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-3, 1.5)
    ax1.axis('off')
    ax1.set_title('Bellman Backup for V(s)', fontsize=14, fontweight='bold', color=GREEN_DARK)

    # Current state (top)
    state_circle = plt.Circle((0, 1), 0.3, color=GREEN_MEDIUM, ec=GREEN_DARK, linewidth=2)
    ax1.add_patch(state_circle)
    ax1.text(0, 1, 's', fontsize=14, ha='center', va='center', fontweight='bold', color='white')

    # Actions (middle row) - small black dots
    action_positions = [(-1.2, 0), (-0.4, 0), (0.4, 0), (1.2, 0)]
    action_labels = ['a1', 'a2', 'a3', 'a4']
    for pos, label in zip(action_positions, action_labels):
        action_dot = plt.Circle(pos, 0.15, color=GREEN_DARK)
        ax1.add_patch(action_dot)
        ax1.text(pos[0], pos[1], label, fontsize=8, ha='center', va='center', color='white')
        # Arrow from state to action
        ax1.annotate('', xy=(pos[0], pos[1]+0.15), xytext=(0, 0.7),
                    arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=1.5))
        # Policy probability
        ax1.text(pos[0]-0.3, 0.5, r'$\pi$', fontsize=10, color=ACCENT_BLUE)

    # Next states (bottom row)
    next_state_positions = [(-1.5, -1.5), (-0.5, -1.5), (0.5, -1.5), (1.5, -1.5)]
    next_state_labels = ["s'1", "s'2", "s'3", "s'4"]
    for i, (pos, label) in enumerate(zip(next_state_positions, next_state_labels)):
        ns_circle = plt.Circle(pos, 0.25, color=GREEN_LIGHT, ec=GREEN_DARK, linewidth=2)
        ax1.add_patch(ns_circle)
        ax1.text(pos[0], pos[1], label, fontsize=10, ha='center', va='center', fontweight='bold')
        # Arrow from action to next state
        action_idx = i
        ax1.annotate('', xy=(pos[0], pos[1]+0.25),
                    xytext=(action_positions[action_idx][0], action_positions[action_idx][1]-0.15),
                    arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=1.5))
        # Transition probability
        ax1.text(pos[0]+0.3, -0.75, 'P', fontsize=9, color=ACCENT_ORANGE)

    # Add reward annotation
    ax1.text(-1.8, -0.75, 'R', fontsize=10, fontweight='bold', color=ACCENT_RED)

    # Formula
    ax1.text(0, -2.7, r'$V^\pi(s) = \sum_a \pi(a|s) \sum_{s^\prime} P(s^\prime|s,a)[R + \gamma V^\pi(s^\prime)]$',
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

    # Right: Q(s,a) backup diagram
    ax2 = axes[1]
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-3, 1.5)
    ax2.axis('off')
    ax2.set_title('Bellman Backup for Q(s,a)', fontsize=14, fontweight='bold', color=GREEN_DARK)

    # Current state-action pair (top)
    sa_rect = mpatches.FancyBboxPatch((-0.4, 0.7), 0.8, 0.5,
                                       boxstyle='round,pad=0.05',
                                       facecolor=GREEN_DARK, edgecolor=GREEN_DARK)
    ax2.add_patch(sa_rect)
    ax2.text(0, 0.95, '(s, a)', fontsize=12, ha='center', va='center', fontweight='bold', color='white')

    # Next states (middle row)
    ns_positions = [(-1.2, -0.3), (0, -0.3), (1.2, -0.3)]
    for i, pos in enumerate(ns_positions):
        ns_circle = plt.Circle(pos, 0.25, color=GREEN_MEDIUM, ec=GREEN_DARK, linewidth=2)
        ax2.add_patch(ns_circle)
        ax2.text(pos[0], pos[1], f"s'{i+1}", fontsize=10, ha='center', va='center',
                fontweight='bold', color='white')
        # Arrow from (s,a) to s'
        ax2.annotate('', xy=(pos[0], pos[1]+0.25), xytext=(0, 0.7),
                    arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=1.5))
        ax2.text(pos[0]-0.4, 0.2, 'P', fontsize=9, color=ACCENT_ORANGE)

    # Next actions (bottom row)
    for i, ns_pos in enumerate(ns_positions):
        action_xs = [ns_pos[0] - 0.3, ns_pos[0] + 0.3]
        for j, ax_pos in enumerate(action_xs):
            na_dot = plt.Circle((ax_pos, -1.3), 0.12, color=GREEN_DARK)
            ax2.add_patch(na_dot)
            ax2.text(ax_pos, -1.3, f'a{j+1}', fontsize=7, ha='center', va='center', color='white')
            # Arrow
            ax2.annotate('', xy=(ax_pos, -1.18), xytext=(ns_pos[0], ns_pos[1]-0.25),
                        arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=1))
        ax2.text(ns_pos[0]-0.5, -0.8, r'$\pi$', fontsize=9, color=ACCENT_BLUE)

    # Reward annotation
    ax2.text(-1.7, 0.2, 'R', fontsize=10, fontweight='bold', color=ACCENT_RED)

    # Formula
    ax2.text(0, -2.7, r"$Q^\pi(s,a) = \sum_{s^\prime} P(s^\prime|s,a)[R + \gamma \sum_{a^\prime} \pi(a^\prime|s^\prime) Q^\pi(s^\prime,a^\prime)]$",
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

    # Legend
    fig.text(0.5, 0.02,
             'White/Light circles = States    Dark dots = Actions    Arrows = Transitions\n'
             'P = Transition probability    R = Reward    pi = Policy probability',
             fontsize=10, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor=GREEN_PALE, edgecolor=GREEN_DARK))

    plt.suptitle('Bellman Backup Diagrams', fontsize=16, fontweight='bold', color=GREEN_DARK, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save as SVG
    output_path = OUTPUT_DIR / 'bellman_backup_diagram.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# Visualization 3: Value Iteration Convergence Animation
# =============================================================================

def create_value_iteration_convergence_animation():
    """Create animation showing convergence metrics during value iteration."""
    print("Creating value iteration convergence animation...")

    env = GridWorld(size=4, gamma=0.9)
    V_history = run_value_iteration(env, n_iterations=30)

    # Compute convergence metrics
    max_changes = []
    mean_values = []
    for i in range(1, len(V_history)):
        max_change = np.max(np.abs(V_history[i] - V_history[i-1]))
        max_changes.append(max_change)
        mean_values.append(np.mean(V_history[i]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def animate(frame):
        for ax in axes:
            ax.clear()

        # Left: Current value function heatmap
        ax1 = axes[0]
        V = V_history[min(frame, len(V_history)-1)]
        im = ax1.imshow(V, cmap='RdYlGn', vmin=-1, vmax=1.5)
        for i in range(env.size):
            for j in range(env.size):
                value = V[i, j]
                text_color = 'white' if abs(value) > 0.5 else 'black'
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=text_color)
        ax1.set_title(f'V(s) at Iteration {frame}', fontsize=12, fontweight='bold', color=GREEN_DARK)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Middle: Max change over iterations
        ax2 = axes[1]
        if frame > 0:
            ax2.plot(range(1, min(frame+1, len(max_changes)+1)),
                    max_changes[:min(frame, len(max_changes))],
                    color=GREEN_DARK, linewidth=2, marker='o', markersize=4)
        ax2.set_xlim(0, len(V_history))
        ax2.set_ylim(0, max(max_changes) * 1.1 if max_changes else 1)
        ax2.axhline(y=0.01, color=ACCENT_RED, linestyle='--', label='Convergence threshold')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Max |V_new - V_old|', fontsize=11)
        ax2.set_title('Convergence: Max Change', fontsize=12, fontweight='bold', color=GREEN_DARK)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Right: Mean value over iterations
        ax3 = axes[2]
        if frame > 0:
            ax3.plot(range(1, min(frame+1, len(mean_values)+1)),
                    mean_values[:min(frame, len(mean_values))],
                    color=GREEN_MEDIUM, linewidth=2, marker='s', markersize=4)
        ax3.set_xlim(0, len(V_history))
        if mean_values:
            ax3.set_ylim(min(mean_values) * 0.9, max(mean_values) * 1.1)
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Mean V(s)', fontsize=11)
        ax3.set_title('Mean State Value', fontsize=12, fontweight='bold', color=GREEN_DARK)
        ax3.grid(True, alpha=0.3)

        # Add convergence info
        if frame > 0 and frame <= len(max_changes):
            current_change = max_changes[frame-1]
            converged = current_change < 0.01
            status = 'CONVERGED' if converged else 'Iterating...'
            status_color = GREEN_DARK if converged else ACCENT_ORANGE
            ax2.text(0.5, 0.95, status, transform=ax2.transAxes, fontsize=10,
                    ha='center', va='top', fontweight='bold', color=status_color,
                    bbox=dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=status_color))

        return axes

    plt.suptitle('Value Iteration Convergence Analysis', fontsize=14, fontweight='bold',
                color=GREEN_DARK, y=1.02)
    plt.tight_layout()

    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(V_history), interval=300, blit=False)

    # Save as GIF
    output_path = OUTPUT_DIR / 'value_iteration_convergence.gif'
    anim.save(output_path, writer=PillowWriter(fps=3))
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Value Functions Visualizations Generator")
    print("=" * 60)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Generate all visualizations
    paths = []

    try:
        paths.append(create_value_function_animation())
    except Exception as e:
        print(f"Error creating value function animation: {e}")

    try:
        paths.append(create_bellman_backup_diagram())
    except Exception as e:
        print(f"Error creating Bellman backup diagram: {e}")

    try:
        paths.append(create_value_iteration_convergence_animation())
    except Exception as e:
        print(f"Error creating convergence animation: {e}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    for p in paths:
        print(f"  - {p}")

    return paths


if __name__ == "__main__":
    main()
