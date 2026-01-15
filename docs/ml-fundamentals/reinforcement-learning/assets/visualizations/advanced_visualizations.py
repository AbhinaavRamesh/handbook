#!/usr/bin/env python3
"""
Generate visualization images for Advanced RL Topics.
Outputs: GIF for animations, SVG for static images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os

# Set the output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

# Color palette
COLORS = {
    'agent1': '#4ECDC4',
    'agent2': '#FF6B6B',
    'agent3': '#45B7D1',
    'environment': '#95A5A6',
    'reward': '#F39C12',
    'state': '#9B59B6',
    'action': '#2ECC71',
    'policy': '#3498DB',
    'value': '#E74C3C',
    'data': '#1ABC9C',
    'model': '#F1C40F',
    'background': '#FAFAFA',
    'text': '#2C3E50',
    'highlight': '#E74C3C',
    'lowlight': '#BDC3C7',
}


def save_svg(name):
    """Save figure as SVG."""
    filepath = os.path.join(OUTPUT_DIR, name)
    plt.savefig(filepath, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Generated: {name}")


def save_gif(ani, name, fps=10):
    """Save animation as GIF."""
    filepath = os.path.join(OUTPUT_DIR, name)
    writer = PillowWriter(fps=fps)
    ani.save(filepath, writer=writer)
    plt.close()
    print(f"Generated: {name}")


# =============================================================================
# 1. MARL Interactions Visualization (Static SVG)
# =============================================================================

def marl_interactions_visualization():
    """Generate multi-agent interaction types visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Common setup
    def draw_agents(ax, positions, colors, labels):
        for pos, color, label in zip(positions, colors, labels):
            circle = Circle(pos, 0.3, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], label, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')

    # Cooperative Setting
    ax = axes[0]
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)

    # Draw shared goal
    goal = Circle((1.5, 3.2), 0.4, facecolor=COLORS['reward'],
                  edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(goal)
    ax.text(1.5, 3.2, 'R', ha='center', va='center', fontsize=14, fontweight='bold')

    # Draw agents
    agent_positions = [(0.5, 1), (1.5, 0.5), (2.5, 1)]
    agent_colors = [COLORS['agent1'], COLORS['agent2'], COLORS['agent3']]
    draw_agents(ax, agent_positions, agent_colors, ['A1', 'A2', 'A3'])

    # Draw arrows to shared goal
    for pos in agent_positions:
        ax.annotate('', xy=(1.5, 2.8), xytext=(pos[0], pos[1] + 0.3),
                   arrowprops=dict(arrowstyle='->', color=COLORS['action'], lw=2))

    ax.set_title('Cooperative\n(Shared Reward)', fontsize=12, fontweight='bold')
    ax.text(1.5, -0.7, 'r1 = r2 = r3', ha='center', fontsize=10, style='italic')
    ax.axis('off')
    ax.set_aspect('equal')

    # Competitive Setting
    ax = axes[1]
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)

    # Draw agents facing each other
    agent_positions = [(0.8, 1.5), (2.2, 1.5)]
    agent_colors = [COLORS['agent1'], COLORS['agent2']]
    draw_agents(ax, agent_positions, agent_colors, ['A1', 'A2'])

    # Draw opposing arrows
    ax.annotate('', xy=(1.8, 1.5), xytext=(1.1, 1.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['agent1'], lw=3))
    ax.annotate('', xy=(1.2, 1.5), xytext=(1.9, 1.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['agent2'], lw=3))

    # Draw reward indicators
    ax.text(0.8, 2.5, '+R', ha='center', fontsize=14, fontweight='bold', color=COLORS['reward'])
    ax.text(2.2, 2.5, '-R', ha='center', fontsize=14, fontweight='bold', color=COLORS['value'])

    ax.set_title('Competitive\n(Zero-Sum)', fontsize=12, fontweight='bold')
    ax.text(1.5, -0.7, 'r1 + r2 = 0', ha='center', fontsize=10, style='italic')
    ax.axis('off')
    ax.set_aspect('equal')

    # Mixed Setting
    ax = axes[2]
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)

    # Draw agents with individual goals
    agent_positions = [(0.7, 1.5), (2.3, 1.5)]
    goal_positions = [(0.7, 3), (2.3, 3)]
    agent_colors = [COLORS['agent1'], COLORS['agent2']]

    draw_agents(ax, agent_positions, agent_colors, ['A1', 'A2'])

    # Draw individual goals
    for i, (pos, color) in enumerate(zip(goal_positions, agent_colors)):
        goal = Circle(pos, 0.25, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.5)
        ax.add_patch(goal)
        ax.text(pos[0], pos[1], f'G{i+1}', ha='center', va='center', fontsize=10)

    # Draw paths
    for agent_pos, goal_pos, color in zip(agent_positions, goal_positions, agent_colors):
        ax.annotate('', xy=(goal_pos[0], goal_pos[1] - 0.25),
                   xytext=(agent_pos[0], agent_pos[1] + 0.3),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Draw interaction
    ax.annotate('', xy=(1.5, 2), xytext=(1.0, 1.5),
               arrowprops=dict(arrowstyle='<->', color=COLORS['lowlight'], lw=1.5, ls='--'))
    ax.text(1.5, 2.2, '?', ha='center', fontsize=12, color=COLORS['text'])

    ax.set_title('Mixed\n(General-Sum)', fontsize=12, fontweight='bold')
    ax.text(1.5, -0.7, 'Independent rewards', ha='center', fontsize=10, style='italic')
    ax.axis('off')
    ax.set_aspect('equal')

    plt.tight_layout()
    save_svg('marl_interactions.svg')


# =============================================================================
# 2. CTDE Architecture Visualization (Static SVG)
# =============================================================================

def ctde_architecture_visualization():
    """Generate Centralized Training Decentralized Execution visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training Phase (Centralized)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    # Draw central critic
    critic_box = FancyBboxPatch((3.5, 5.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['policy'], edgecolor='black', linewidth=2)
    ax.add_patch(critic_box)
    ax.text(5, 6.25, 'Central Critic\nV(s) or Q(s, a1, a2, ...)', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

    # Draw agents
    agent_positions = [(1.5, 2), (5, 2), (8.5, 2)]
    for i, pos in enumerate(agent_positions):
        agent_box = FancyBboxPatch((pos[0]-0.8, pos[1]-0.5), 1.6, 1.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS[f'agent{i+1}' if i < 3 else 'agent1'],
                                    edgecolor='black', linewidth=2)
        ax.add_patch(agent_box)
        ax.text(pos[0], pos[1], f'Agent {i+1}\n(o{i+1}, a{i+1})', ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')

    # Draw global state
    state_box = FancyBboxPatch((3.5, 0), 3, 1, boxstyle="round,pad=0.1",
                                facecolor=COLORS['state'], edgecolor='black', linewidth=2)
    ax.add_patch(state_box)
    ax.text(5, 0.5, 'Global State s', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

    # Draw arrows (all info flows to critic)
    for pos in agent_positions:
        ax.annotate('', xy=(5, 5.5), xytext=(pos[0], 2.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))
    ax.annotate('', xy=(5, 5.5), xytext=(5, 1),
               arrowprops=dict(arrowstyle='->', color=COLORS['state'], lw=2))

    ax.set_title('Training Phase\n(Centralized)', fontsize=14, fontweight='bold', color=COLORS['policy'])
    ax.text(5, 7.5, 'Full observability during learning', ha='center', fontsize=10, style='italic')
    ax.axis('off')

    # Execution Phase (Decentralized)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    # Draw agents with local observations
    agent_positions = [(1.5, 4), (5, 4), (8.5, 4)]
    obs_positions = [(1.5, 2), (5, 2), (8.5, 2)]

    for i, (agent_pos, obs_pos) in enumerate(zip(agent_positions, obs_positions)):
        # Agent
        agent_box = FancyBboxPatch((agent_pos[0]-0.8, agent_pos[1]-0.5), 1.6, 1.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS[f'agent{i+1}' if i < 3 else 'agent1'],
                                    edgecolor='black', linewidth=2)
        ax.add_patch(agent_box)
        ax.text(agent_pos[0], agent_pos[1], f'Agent {i+1}\npi(a|o{i+1})', ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')

        # Local observation
        obs_box = FancyBboxPatch((obs_pos[0]-0.6, obs_pos[1]-0.4), 1.2, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLORS['environment'], edgecolor='black', linewidth=1.5)
        ax.add_patch(obs_box)
        ax.text(obs_pos[0], obs_pos[1], f'o{i+1}', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        # Arrow from observation to agent
        ax.annotate('', xy=(agent_pos[0], agent_pos[1]-0.5), xytext=(obs_pos[0], obs_pos[1]+0.4),
                   arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))

        # Action arrow
        ax.annotate('', xy=(agent_pos[0], 6), xytext=(agent_pos[0], agent_pos[1]+0.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['action'], lw=2))
        ax.text(agent_pos[0], 6.3, f'a{i+1}', ha='center', fontsize=10, fontweight='bold')

    # Draw "no communication" indicators
    ax.plot([3.2, 3.8], [4, 4], 'r-', linewidth=3)
    ax.plot([3.5, 3.5], [3.7, 4.3], 'r-', linewidth=3)
    ax.plot([6.2, 6.8], [4, 4], 'r-', linewidth=3)
    ax.plot([6.5, 6.5], [3.7, 4.3], 'r-', linewidth=3)

    ax.set_title('Execution Phase\n(Decentralized)', fontsize=14, fontweight='bold', color=COLORS['agent1'])
    ax.text(5, 7.5, 'Local observations only, no communication', ha='center', fontsize=10, style='italic')
    ax.axis('off')

    plt.tight_layout()
    save_svg('ctde_architecture.svg')


# =============================================================================
# 3. Multi-Agent Coordination Animation (GIF)
# =============================================================================

def multi_agent_coordination_animation():
    """Generate animated visualization of multi-agent coordination."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Environment setup
    grid_size = 8

    # Agent starting and goal positions
    agents_start = [(1, 1), (6, 1), (1, 6)]
    agents_goal = [(6, 6), (1, 6), (6, 1)]
    agent_colors = [COLORS['agent1'], COLORS['agent2'], COLORS['agent3']]

    # Pre-compute paths (simple waypoints for animation)
    paths = [
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)],
        [(6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6)],
        [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)]
    ]

    # Make paths same length
    max_len = max(len(p) for p in paths)
    for i, path in enumerate(paths):
        while len(paths[i]) < max_len:
            paths[i].append(paths[i][-1])

    num_frames = len(paths[0]) + 5

    def init():
        ax.clear()
        ax.set_xlim(-0.5, grid_size + 0.5)
        ax.set_ylim(-0.5, grid_size + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        return []

    def animate(frame):
        ax.clear()
        ax.set_xlim(-0.5, grid_size + 0.5)
        ax.set_ylim(-0.5, grid_size + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw grid
        for i in range(grid_size + 1):
            ax.axhline(y=i, color='lightgray', linewidth=0.5)
            ax.axvline(x=i, color='lightgray', linewidth=0.5)

        # Draw goals
        for goal, color in zip(agents_goal, agent_colors):
            goal_marker = Circle(goal, 0.35, facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=2, linestyle='--')
            ax.add_patch(goal_marker)

        # Draw agents
        step = min(frame, len(paths[0]) - 1)

        for i, (path, color) in enumerate(zip(paths, agent_colors)):
            pos = path[step]

            # Draw trail
            for j in range(max(0, step - 3), step):
                trail_pos = path[j]
                alpha = 0.3 * (j - max(0, step - 3) + 1) / 3
                trail = Circle(trail_pos, 0.15, facecolor=color, alpha=alpha, edgecolor='none')
                ax.add_patch(trail)

            # Draw agent
            agent = Circle(pos, 0.35, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(agent)
            ax.text(pos[0], pos[1], f'A{i+1}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

        # Title and info
        ax.set_title('Multi-Agent Coordination', fontsize=14, fontweight='bold')

        if step < len(paths[0]) - 1:
            ax.text(grid_size / 2, -0.3, f'Step {step + 1}: Agents coordinate to avoid collisions',
                   ha='center', fontsize=10)
        else:
            ax.text(grid_size / 2, -0.3, 'All agents reached their goals!',
                   ha='center', fontsize=11, fontweight='bold', color=COLORS['action'])

        return []

    ani = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=500, blit=True)
    save_gif(ani, 'multi_agent_coordination.gif', fps=2)


# =============================================================================
# 4. Offline RL Distribution Shift Visualization (GIF)
# =============================================================================

def distribution_shift_visualization():
    """Generate animated visualization of distribution shift problem in offline RL."""
    fig, ax = plt.subplots(figsize=(12, 8))

    num_frames = 30

    # Generate behavior policy distribution (data)
    np.random.seed(42)
    behavior_states = np.random.multivariate_normal([2, 2], [[0.8, 0.3], [0.3, 0.8]], 200)

    # True Q-function (ground truth)
    def true_q(s, a):
        return -((s[0] - 5)**2 + (s[1] - 5)**2) / 10 + 5

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 8)

        phase = frame // 10
        sub_frame = frame % 10

        if phase == 0:
            # Phase 1: Show data distribution
            alpha = min(1.0, sub_frame / 5)
            ax.scatter(behavior_states[:, 0], behavior_states[:, 1],
                      c=COLORS['data'], alpha=alpha * 0.6, s=30, label='Training Data')

            ax.set_title('Offline RL: Distribution Shift Problem\nPhase 1: Dataset from Behavior Policy',
                        fontsize=12, fontweight='bold')
            ax.text(2, -0.5, 'Data collected by behavior policy', ha='center', fontsize=10)

        elif phase == 1:
            # Phase 2: Show learned policy trying to go elsewhere
            ax.scatter(behavior_states[:, 0], behavior_states[:, 1],
                      c=COLORS['data'], alpha=0.3, s=30, label='Training Data')

            # Learned policy wants to go to high-value region
            target_x = 2 + sub_frame * 0.3
            target_y = 2 + sub_frame * 0.3

            ax.scatter([target_x], [target_y], c=COLORS['highlight'], s=200,
                      marker='*', label='Learned Policy', zorder=5)

            # Draw arrow showing policy direction
            ax.annotate('', xy=(5, 5), xytext=(target_x, target_y),
                       arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=2))

            # Shade OOD region
            if sub_frame > 3:
                ood_region = plt.Rectangle((3.5, 3.5), 4, 4, fill=True,
                                           facecolor=COLORS['highlight'], alpha=0.1,
                                           edgecolor=COLORS['highlight'], linewidth=2, linestyle='--')
                ax.add_patch(ood_region)
                ax.text(5.5, 7, 'OOD Region\n(No Data)', ha='center', fontsize=10,
                       color=COLORS['highlight'], fontweight='bold')

            ax.set_title('Offline RL: Distribution Shift Problem\nPhase 2: Policy Extrapolates Beyond Data',
                        fontsize=12, fontweight='bold')
            ax.text(4, -0.5, 'Learned policy queries unseen state-action pairs!',
                   ha='center', fontsize=10, color=COLORS['highlight'])

        else:
            # Phase 3: Show the problem
            ax.scatter(behavior_states[:, 0], behavior_states[:, 1],
                      c=COLORS['data'], alpha=0.3, s=30, label='Training Data (known)')

            # OOD region
            ood_region = plt.Rectangle((3.5, 3.5), 4, 4, fill=True,
                                       facecolor=COLORS['highlight'], alpha=0.1,
                                       edgecolor=COLORS['highlight'], linewidth=2, linestyle='--')
            ax.add_patch(ood_region)

            # Show true vs estimated Q
            ax.scatter([5], [5], c=COLORS['highlight'], s=200, marker='*', zorder=5)

            # Annotations
            if sub_frame > 2:
                ax.annotate('True Q: 5.0', xy=(5, 5), xytext=(6.5, 6.5),
                           fontsize=10, ha='left',
                           arrowprops=dict(arrowstyle='->', color='black'))
            if sub_frame > 5:
                ax.annotate('Estimated Q: 15.0\n(Overestimation!)', xy=(5, 5), xytext=(6.5, 4),
                           fontsize=10, ha='left', color=COLORS['highlight'], fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color=COLORS['highlight']))

            ax.set_title('Offline RL: Distribution Shift Problem\nPhase 3: Extrapolation Error',
                        fontsize=12, fontweight='bold')
            ax.text(4, -0.5, 'Q-function overestimates value in unseen regions',
                   ha='center', fontsize=10, color=COLORS['highlight'], fontweight='bold')

        ax.legend(loc='upper left')
        ax.set_xlabel('State Dimension 1')
        ax.set_ylabel('State Dimension 2')
        ax.grid(True, alpha=0.3)

        return []

    ani = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=300, blit=True)
    save_gif(ani, 'distribution_shift.gif', fps=3)


# =============================================================================
# 5. World Model Prediction vs Reality (GIF)
# =============================================================================

def world_model_prediction_visualization():
    """Generate animated visualization of world model predictions vs reality."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    num_frames = 40

    # Generate trajectories
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 100)

    # True trajectory (sinusoidal)
    true_x = t / 2
    true_y = 2 * np.sin(t / 2) + 3

    # Predicted trajectory (drifts over time due to model error)
    pred_x = t / 2
    pred_y = 2 * np.sin(t / 2) + 3 + 0.02 * t**1.5  # Compounding error

    def init():
        for ax in axes:
            ax.clear()
        return []

    def animate(frame):
        for ax in axes:
            ax.clear()

        step = int(frame * 2.5)  # Convert frame to trajectory step
        step = min(step, len(t) - 1)

        # Left plot: Trajectories
        ax = axes[0]
        ax.set_xlim(-0.5, 7)
        ax.set_ylim(0, 7)

        # Plot true trajectory (so far)
        ax.plot(true_x[:step+1], true_y[:step+1], 'b-', linewidth=3,
               label='True Trajectory', alpha=0.8)
        ax.scatter([true_x[step]], [true_y[step]], c='blue', s=150,
                  marker='o', zorder=5, edgecolors='black', linewidth=2)

        # Plot predicted trajectory (so far)
        ax.plot(pred_x[:step+1], pred_y[:step+1], 'r--', linewidth=2.5,
               label='Model Prediction', alpha=0.8)
        ax.scatter([pred_x[step]], [pred_y[step]], c='red', s=150,
                  marker='s', zorder=5, edgecolors='black', linewidth=2)

        # Draw error line
        if step > 10:
            ax.plot([true_x[step], pred_x[step]], [true_y[step], pred_y[step]],
                   'g-', linewidth=2, label='Prediction Error')

        ax.set_xlabel('Time / Position')
        ax.set_ylabel('State Value')
        ax.set_title('World Model: Prediction vs Reality', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Right plot: Error over time
        ax = axes[1]
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 3)

        # Calculate errors
        errors = np.abs(pred_y[:step+1] - true_y[:step+1])
        times = pred_x[:step+1]

        ax.fill_between(times, 0, errors, color=COLORS['highlight'], alpha=0.3)
        ax.plot(times, errors, color=COLORS['highlight'], linewidth=2.5)

        # Add annotations
        if step > 20:
            ax.text(2, 2.5, 'Error compounds\nover horizon', fontsize=10,
                   ha='center', color=COLORS['highlight'], fontweight='bold')

        ax.set_xlabel('Planning Horizon (timesteps)')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Model Error Compounding', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add formula
        if step > 30:
            ax.text(3.5, 0.5, r'Total Error $\approx$ H $\times$ $\epsilon_{model}$',
                   fontsize=11, ha='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        plt.tight_layout()
        return []

    ani = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=150, blit=True)
    save_gif(ani, 'world_model_prediction.gif', fps=6)


# =============================================================================
# Main execution
# =============================================================================

if __name__ == '__main__':
    print("Generating Advanced RL visualization images...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    # Static visualizations (SVG)
    print("\n[1/5] Generating MARL interactions diagram...")
    marl_interactions_visualization()

    print("\n[2/5] Generating CTDE architecture diagram...")
    ctde_architecture_visualization()

    # Animated visualizations (GIF)
    print("\n[3/5] Generating multi-agent coordination animation...")
    multi_agent_coordination_animation()

    print("\n[4/5] Generating distribution shift animation...")
    distribution_shift_visualization()

    print("\n[5/5] Generating world model prediction animation...")
    world_model_prediction_visualization()

    print("\n" + "=" * 50)
    print("All visualizations generated successfully!")
    print("=" * 50)
