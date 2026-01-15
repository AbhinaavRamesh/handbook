#!/usr/bin/env python3
"""
Actor-Critic Methods Visualizations

Generates:
1. Actor-Critic architecture diagram with animated data flow (GIF)
2. A3C parallel workers visualization (SVG)
3. Variance comparison: REINFORCE vs A2C (animated GIF)

Output directory: ../images/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import os

# Set style for clean, professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# Color palette
COLORS = {
    'actor': '#48bb78',       # Green
    'critic': '#4299e1',      # Blue
    'environment': '#ed8936', # Orange
    'reward': '#e53e3e',      # Red
    'state': '#9f7aea',       # Purple
    'action': '#38b2ac',      # Teal
    'gradient': '#f6e05e',    # Yellow
    'background': '#f7fafc',  # Light gray
    'text': '#2d3748',        # Dark gray
    'reinforce': '#e53e3e',   # Red for REINFORCE
    'a2c': '#48bb78',         # Green for A2C
}

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_rounded_box(ax, x, y, width, height, color, label, fontsize=12):
    """Create a rounded rectangle with a centered label."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='black', linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')
    return box


def draw_arrow(ax, start, end, color='black', style='->', lw=2):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=lw,
        mutation_scale=15,
        connectionstyle="arc3,rad=0"
    )
    ax.add_patch(arrow)
    return arrow


def actor_critic_architecture_animated():
    """
    Generate animated Actor-Critic architecture diagram showing data flow.
    Saves as GIF.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Actor-Critic Architecture: Data Flow', fontsize=16, fontweight='bold', pad=20)

    # Static elements positions
    env_pos = (7, 8.5)
    state_pos = (3, 6)
    actor_pos = (3, 3.5)
    critic_pos = (7, 3.5)
    action_pos = (10.5, 6)
    reward_pos = (10.5, 3.5)
    td_error_pos = (5, 1.5)

    # Draw static boxes
    env_box = create_rounded_box(ax, *env_pos, 3, 1.2, COLORS['environment'], 'Environment')
    state_box = create_rounded_box(ax, *state_pos, 2.5, 1, COLORS['state'], 'State s')
    actor_box = create_rounded_box(ax, *actor_pos, 2.5, 1.2, COLORS['actor'], 'Actor\npi(a|s)')
    critic_box = create_rounded_box(ax, *critic_pos, 2.5, 1.2, COLORS['critic'], 'Critic\nV(s)')
    action_box = create_rounded_box(ax, *action_pos, 2, 1, COLORS['action'], 'Action a')
    reward_box = create_rounded_box(ax, *reward_pos, 2.2, 1, COLORS['reward'], 'Reward r')

    # TD Error box
    td_box = FancyBboxPatch(
        (td_error_pos[0] - 2, td_error_pos[1] - 0.5), 4, 1,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['gradient'], edgecolor='black', linewidth=2
    )
    ax.add_patch(td_box)
    ax.text(td_error_pos[0], td_error_pos[1], r'TD Error: $\delta = r + \gamma V(s\') - V(s)$',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['text'])

    # Static arrows
    arrows_static = [
        (env_pos, (env_pos[0] - 2.5, env_pos[1] - 1), COLORS['state']),  # Env -> State area
        ((state_pos[0], state_pos[1] - 0.5), (actor_pos[0], actor_pos[1] + 0.6), COLORS['state']),  # State -> Actor
        ((state_pos[0] + 1.5, state_pos[1] - 0.5), (critic_pos[0] - 1, critic_pos[1] + 0.6), COLORS['state']),  # State -> Critic
        ((actor_pos[0] + 1.5, actor_pos[1] + 0.3), (action_pos[0] - 1.2, action_pos[1] - 0.3), COLORS['action']),  # Actor -> Action
        ((action_pos[0], action_pos[1] + 0.5), (env_pos[0] + 1.5, env_pos[1] - 0.5), COLORS['action']),  # Action -> Env
        ((env_pos[0] + 2, env_pos[1] - 0.8), (reward_pos[0], reward_pos[1] + 0.6), COLORS['reward']),  # Env -> Reward
        ((reward_pos[0] - 1.5, reward_pos[1]), (critic_pos[0] + 1.5, critic_pos[1]), COLORS['reward']),  # Reward -> Critic
        ((critic_pos[0] - 1, critic_pos[1] - 0.6), (td_error_pos[0] + 1, td_error_pos[1] + 0.5), COLORS['gradient']),  # Critic -> TD
        ((td_error_pos[0] - 1.5, td_error_pos[1] + 0.5), (actor_pos[0], actor_pos[1] - 0.6), COLORS['gradient']),  # TD -> Actor
    ]

    for start, end, color in arrows_static:
        draw_arrow(ax, start, end, color=color)

    # Labels for arrows
    ax.text(2, 7.3, 'observe', fontsize=9, color=COLORS['text'], style='italic')
    ax.text(5.5, 4.8, 'evaluate', fontsize=9, color=COLORS['text'], style='italic')
    ax.text(7.5, 6.5, 'sample', fontsize=9, color=COLORS['text'], style='italic')
    ax.text(11, 5, 'execute', fontsize=9, color=COLORS['text'], style='italic')
    ax.text(11.5, 5.8, 'receive', fontsize=9, color=COLORS['text'], style='italic', rotation=60)
    ax.text(8.8, 3.2, 'compute', fontsize=9, color=COLORS['text'], style='italic')
    ax.text(3.5, 2.2, 'update policy', fontsize=9, color=COLORS['text'], style='italic')

    # Animated data flow circles
    flow_points = []

    # Animation data - define paths for data flow
    paths = {
        'state_to_actor': {
            'points': [(state_pos[0], state_pos[1] - 0.5), (actor_pos[0], actor_pos[1] + 0.6)],
            'color': COLORS['state'],
            'frames': (0, 15)
        },
        'state_to_critic': {
            'points': [(state_pos[0] + 1.5, state_pos[1] - 0.5), (critic_pos[0] - 1, critic_pos[1] + 0.6)],
            'color': COLORS['state'],
            'frames': (0, 15)
        },
        'actor_to_action': {
            'points': [(actor_pos[0] + 1.5, actor_pos[1] + 0.3), (action_pos[0] - 1.2, action_pos[1] - 0.3)],
            'color': COLORS['action'],
            'frames': (15, 30)
        },
        'action_to_env': {
            'points': [(action_pos[0], action_pos[1] + 0.5), (env_pos[0] + 1.5, env_pos[1] - 0.5)],
            'color': COLORS['action'],
            'frames': (30, 45)
        },
        'env_to_reward': {
            'points': [(env_pos[0] + 2, env_pos[1] - 0.8), (reward_pos[0], reward_pos[1] + 0.6)],
            'color': COLORS['reward'],
            'frames': (45, 60)
        },
        'reward_to_critic': {
            'points': [(reward_pos[0] - 1.5, reward_pos[1]), (critic_pos[0] + 1.5, critic_pos[1])],
            'color': COLORS['reward'],
            'frames': (60, 75)
        },
        'critic_to_td': {
            'points': [(critic_pos[0] - 1, critic_pos[1] - 0.6), (td_error_pos[0] + 1, td_error_pos[1] + 0.5)],
            'color': COLORS['gradient'],
            'frames': (75, 90)
        },
        'td_to_actor': {
            'points': [(td_error_pos[0] - 1.5, td_error_pos[1] + 0.5), (actor_pos[0], actor_pos[1] - 0.6)],
            'color': COLORS['gradient'],
            'frames': (90, 105)
        },
    }

    # Create animated circles
    circles = []
    for name, path_data in paths.items():
        circle = Circle((0, 0), 0.15, color=path_data['color'], zorder=10)
        circle.set_visible(False)
        ax.add_patch(circle)
        circles.append((circle, path_data))

    def interpolate(p1, p2, t):
        """Linear interpolation between two points."""
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    def animate(frame):
        """Animation function."""
        frame = frame % 120  # Loop every 120 frames

        for circle, path_data in circles:
            start_frame, end_frame = path_data['frames']
            if start_frame <= frame < end_frame:
                circle.set_visible(True)
                t = (frame - start_frame) / (end_frame - start_frame)
                pos = interpolate(path_data['points'][0], path_data['points'][1], t)
                circle.center = pos
            else:
                circle.set_visible(False)

        return [c[0] for c in circles]

    # Create animation
    anim = FuncAnimation(fig, animate, frames=120, interval=50, blit=True)

    # Save as GIF
    output_path = OUTPUT_DIR / 'actor_critic_architecture.gif'
    writer = PillowWriter(fps=20)
    anim.save(str(output_path), writer=writer, dpi=100)
    plt.close()
    print(f"Saved: {output_path}")


def a3c_parallel_workers():
    """
    Generate A3C parallel workers visualization.
    Saves as SVG (static).
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A3C: Asynchronous Parallel Workers Architecture', fontsize=16, fontweight='bold', pad=20)

    # Global network (top center)
    global_box = FancyBboxPatch(
        (5, 7.5), 4, 1.5,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        facecolor=COLORS['environment'], edgecolor='black', linewidth=3
    )
    ax.add_patch(global_box)
    ax.text(7, 8.5, 'Global Network', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(7, 7.9, r'$\theta_{global}, \phi_{global}$', ha='center', va='center',
            fontsize=12, color='white')

    # Worker environments and networks
    num_workers = 4
    worker_colors = ['#48bb78', '#4299e1', '#9f7aea', '#ed64a6']
    worker_x_positions = [1.5, 4.5, 9.5, 12.5]

    for i, (x, color) in enumerate(zip(worker_x_positions, worker_colors)):
        # Environment box
        env_box = FancyBboxPatch(
            (x - 1, 4.5), 2, 1,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='black', linewidth=2, alpha=0.8
        )
        ax.add_patch(env_box)
        ax.text(x, 5, f'Env {i+1}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Local network box
        local_box = FancyBboxPatch(
            (x - 1.2, 2.5), 2.4, 1.2,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='black', linewidth=2
        )
        ax.add_patch(local_box)
        ax.text(x, 3.4, f'Worker {i+1}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax.text(x, 2.9, r'$\theta_{local}$', ha='center', va='center',
                fontsize=9, color='white')

        # Arrow from env to local network
        ax.annotate('', xy=(x, 3.7), xytext=(x, 4.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Bidirectional arrows to global network (pull/push)
        # Pull arrow (down)
        ax.annotate('', xy=(x - 0.3, 6.2), xytext=(x - 0.3 + (7 - x) * 0.2, 7.4),
                   arrowprops=dict(arrowstyle='->', color=COLORS['actor'], lw=2))

        # Push arrow (up)
        ax.annotate('', xy=(x + 0.3 + (7 - x) * 0.2, 7.4), xytext=(x + 0.3, 6.2),
                   arrowprops=dict(arrowstyle='->', color=COLORS['reward'], lw=2))

        # Labels for arrows
        if i == 0:
            ax.text(x + 1.5, 6.8, 'Pull\nparams', fontsize=8, color=COLORS['actor'],
                   ha='center', va='center')
            ax.text(x + 2.5, 6.8, 'Push\ngrads', fontsize=8, color=COLORS['reward'],
                   ha='center', va='center')

    # Add gradient computation boxes
    for i, x in enumerate(worker_x_positions):
        grad_box = FancyBboxPatch(
            (x - 0.8, 1), 1.6, 0.8,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=COLORS['gradient'], edgecolor='black', linewidth=1.5
        )
        ax.add_patch(grad_box)
        ax.text(x, 1.4, r'$\nabla_\theta L$', ha='center', va='center',
                fontsize=9, fontweight='bold', color=COLORS['text'])

        # Arrow from local to gradient
        ax.annotate('', xy=(x, 1.8), xytext=(x, 2.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Legend box
    legend_box = FancyBboxPatch(
        (0.3, 0), 4, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor='white', edgecolor='black', linewidth=1
    )
    ax.add_patch(legend_box)
    ax.text(2.3, 0.4, 'Asynchronous: Workers update independently',
            fontsize=9, ha='center', va='center', style='italic')

    # Process description
    process_text = """
    1. Worker pulls latest global parameters
    2. Interacts with local environment
    3. Computes local gradients
    4. Pushes gradients to global network
    5. Repeat asynchronously
    """
    ax.text(11.5, 1.5, process_text, fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'a3c_workers.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def variance_comparison_animated():
    """
    Generate animated variance comparison between REINFORCE and A2C.
    Shows gradient estimates over training steps.
    Saves as GIF.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Variance Comparison: REINFORCE vs Actor-Critic (A2C)',
                 fontsize=16, fontweight='bold')

    # Generate synthetic gradient estimate data
    n_steps = 200
    true_gradient = 0.5

    # REINFORCE: High variance due to Monte Carlo returns
    reinforce_variance = 2.0
    reinforce_gradients = np.random.normal(true_gradient, reinforce_variance, n_steps)
    reinforce_cumsum = np.cumsum(reinforce_gradients) / (np.arange(n_steps) + 1)

    # A2C: Lower variance due to TD error baseline
    a2c_variance = 0.5
    a2c_gradients = np.random.normal(true_gradient, a2c_variance, n_steps)
    a2c_cumsum = np.cumsum(a2c_gradients) / (np.arange(n_steps) + 1)

    # Left plot: Gradient estimates over time
    ax1 = axes[0]
    ax1.set_xlim(0, n_steps)
    ax1.set_ylim(-4, 5)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Gradient Estimate', fontsize=12)
    ax1.set_title('Gradient Estimates Over Time', fontsize=13, fontweight='bold')
    ax1.axhline(y=true_gradient, color='black', linestyle='--', linewidth=2,
                label=f'True Gradient = {true_gradient}')

    line_reinforce, = ax1.plot([], [], color=COLORS['reinforce'], alpha=0.7,
                                linewidth=1, label='REINFORCE')
    line_a2c, = ax1.plot([], [], color=COLORS['a2c'], alpha=0.7,
                         linewidth=1, label='A2C')
    ax1.legend(loc='upper right')

    # Right plot: Running average
    ax2 = axes[1]
    ax2.set_xlim(0, n_steps)
    ax2.set_ylim(-1, 2)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Running Average', fontsize=12)
    ax2.set_title('Convergence of Gradient Estimate', fontsize=13, fontweight='bold')
    ax2.axhline(y=true_gradient, color='black', linestyle='--', linewidth=2,
                label=f'True Gradient = {true_gradient}')

    line_reinforce_avg, = ax2.plot([], [], color=COLORS['reinforce'],
                                    linewidth=2, label='REINFORCE (avg)')
    line_a2c_avg, = ax2.plot([], [], color=COLORS['a2c'],
                              linewidth=2, label='A2C (avg)')

    # Shade variance regions
    steps = np.arange(n_steps)
    ax2.fill_between(steps, reinforce_cumsum - 0.5, reinforce_cumsum + 0.5,
                     alpha=0.1, color=COLORS['reinforce'])
    ax2.fill_between(steps, a2c_cumsum - 0.15, a2c_cumsum + 0.15,
                     alpha=0.1, color=COLORS['a2c'])

    ax2.legend(loc='upper right')

    # Text annotations
    variance_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def init():
        line_reinforce.set_data([], [])
        line_a2c.set_data([], [])
        line_reinforce_avg.set_data([], [])
        line_a2c_avg.set_data([], [])
        return line_reinforce, line_a2c, line_reinforce_avg, line_a2c_avg

    def animate(frame):
        step = min(frame * 2, n_steps - 1)  # Speed up animation

        # Update gradient lines
        line_reinforce.set_data(range(step + 1), reinforce_gradients[:step + 1])
        line_a2c.set_data(range(step + 1), a2c_gradients[:step + 1])

        # Update running average lines
        line_reinforce_avg.set_data(range(step + 1), reinforce_cumsum[:step + 1])
        line_a2c_avg.set_data(range(step + 1), a2c_cumsum[:step + 1])

        # Update variance text
        if step > 10:
            r_var = np.var(reinforce_gradients[:step + 1])
            a_var = np.var(a2c_gradients[:step + 1])
            variance_text.set_text(
                f'Variance (step {step}):\n'
                f'REINFORCE: {r_var:.2f}\n'
                f'A2C: {a_var:.2f}\n'
                f'Reduction: {(1 - a_var/r_var)*100:.1f}%'
            )

        return line_reinforce, line_a2c, line_reinforce_avg, line_a2c_avg

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=n_steps // 2 + 1, interval=50, blit=True)

    plt.tight_layout()

    # Save as GIF
    output_path = OUTPUT_DIR / 'variance_comparison.gif'
    writer = PillowWriter(fps=20)
    anim.save(str(output_path), writer=writer, dpi=100)
    plt.close()
    print(f"Saved: {output_path}")


def variance_histogram_comparison():
    """
    Generate static comparison of gradient distributions.
    Complementary to the animated version.
    Saves as SVG.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Gradient Estimate Distribution: REINFORCE vs A2C',
                 fontsize=14, fontweight='bold')

    true_gradient = 0.5
    n_samples = 1000

    # REINFORCE samples
    reinforce_samples = np.random.normal(true_gradient, 2.0, n_samples)

    # A2C samples
    a2c_samples = np.random.normal(true_gradient, 0.5, n_samples)

    # Left: REINFORCE histogram
    ax1 = axes[0]
    ax1.hist(reinforce_samples, bins=50, density=True, color=COLORS['reinforce'],
             alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(true_gradient, color='black', linestyle='--', linewidth=2,
                label=f'True = {true_gradient}')
    ax1.axvline(np.mean(reinforce_samples), color=COLORS['reinforce'], linewidth=2,
                label=f'Mean = {np.mean(reinforce_samples):.2f}')
    ax1.set_xlabel('Gradient Estimate', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'REINFORCE\nVariance = {np.var(reinforce_samples):.2f}',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(-6, 7)
    ax1.legend(loc='upper right', fontsize=9)

    # Right: A2C histogram
    ax2 = axes[1]
    ax2.hist(a2c_samples, bins=50, density=True, color=COLORS['a2c'],
             alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(true_gradient, color='black', linestyle='--', linewidth=2,
                label=f'True = {true_gradient}')
    ax2.axvline(np.mean(a2c_samples), color=COLORS['a2c'], linewidth=2,
                label=f'Mean = {np.mean(a2c_samples):.2f}')
    ax2.set_xlabel('Gradient Estimate', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'A2C (Actor-Critic)\nVariance = {np.var(a2c_samples):.2f}',
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(-6, 7)
    ax2.legend(loc='upper right', fontsize=9)

    # Add explanation
    fig.text(0.5, 0.01,
             'Lower variance in A2C leads to more stable gradient updates and faster convergence',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = OUTPUT_DIR / 'variance_histograms.svg'
    plt.savefig(str(output_path), format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def td_error_explanation():
    """
    Generate visualization explaining TD error as advantage estimate.
    Saves as SVG.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('TD Error as Advantage Estimate', fontsize=16, fontweight='bold', pad=20)

    # Timeline
    ax.plot([1, 11], [6, 6], 'k-', linewidth=3)

    # State circles
    state_positions = [(2, 6), (6, 6), (10, 6)]
    state_labels = [r'$s_t$', r'$s_{t+1}$', r'$s_{t+2}$']

    for (x, y), label in zip(state_positions, state_labels):
        circle = Circle((x, y), 0.4, color=COLORS['state'], zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12,
                fontweight='bold', color='white')

    # Action and reward
    ax.text(4, 6.8, r'$a_t$', ha='center', fontsize=12, color=COLORS['action'], fontweight='bold')
    ax.annotate('', xy=(5.5, 6), xytext=(2.5, 6),
               arrowprops=dict(arrowstyle='->', color=COLORS['action'], lw=2))

    ax.text(4, 5.2, r'$r_t$', ha='center', fontsize=12, color=COLORS['reward'], fontweight='bold')

    # Value function estimates
    value_positions = [(2, 4), (6, 4)]
    value_labels = [r'$V(s_t)$', r'$V(s_{t+1})$']

    for (x, y), label in zip(value_positions, value_labels):
        box = FancyBboxPatch(
            (x - 0.8, y - 0.4), 1.6, 0.8,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=COLORS['critic'], edgecolor='black', linewidth=2
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')

    # TD Error formula
    td_box = FancyBboxPatch(
        (2, 1.5), 8, 1.5,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        facecolor=COLORS['gradient'], edgecolor='black', linewidth=2
    )
    ax.add_patch(td_box)
    ax.text(6, 2.5, 'TD Error (Advantage Estimate):', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.text(6, 1.9, r'$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$', ha='center', va='center',
            fontsize=14, color=COLORS['text'])

    # Explanation boxes
    explanations = [
        (2.5, 8.5, r'Observed reward', COLORS['reward']),
        (6, 8.5, r'Bootstrapped future value', COLORS['critic']),
        (9.5, 8.5, r'Expected value', COLORS['critic']),
    ]

    ax.text(2.5, 7.8, r'$r_t$', ha='center', fontsize=14, fontweight='bold', color=COLORS['reward'])
    ax.text(4.5, 7.8, r'$+\ \gamma$', ha='center', fontsize=14, fontweight='bold')
    ax.text(6.5, 7.8, r'$V(s_{t+1})$', ha='center', fontsize=14, fontweight='bold', color=COLORS['critic'])
    ax.text(8.5, 7.8, r'$-$', ha='center', fontsize=14, fontweight='bold')
    ax.text(10, 7.8, r'$V(s_t)$', ha='center', fontsize=14, fontweight='bold', color=COLORS['critic'])

    # Interpretation
    ax.text(6, 0.5,
            r'If $\delta_t > 0$: Action was better than expected (increase probability)',
            ha='center', fontsize=10, color=COLORS['actor'])
    ax.text(6, 0.1,
            r'If $\delta_t < 0$: Action was worse than expected (decrease probability)',
            ha='center', fontsize=10, color=COLORS['reward'])

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'td_error_explanation.svg'
    plt.savefig(str(output_path), format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    print("Generating Actor-Critic visualizations...")
    print("-" * 50)

    # Generate all visualizations
    actor_critic_architecture_animated()
    a3c_parallel_workers()
    variance_comparison_animated()
    variance_histogram_comparison()
    td_error_explanation()

    print("-" * 50)
    print("All visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
