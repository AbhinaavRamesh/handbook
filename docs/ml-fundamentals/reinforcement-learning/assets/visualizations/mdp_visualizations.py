#!/usr/bin/env python3
"""
Generate visualization images/animations for MDP Foundations module.

Outputs:
1. agent_environment_loop.gif - Animated agent-environment interaction loop
2. gridworld_mdp.svg - GridWorld MDP with state transitions
3. discount_effect.gif - Animated discount factor effect on returns

Style: Professional, clean design with dark-mode friendly colors.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os

# Import shared style
from _style import COLORS, apply_style, save_figure, get_box_style, get_sequential_cmap

# Apply professional style
apply_style()

# Set output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_animation(anim, name, fps=2):
    """Save animation as GIF."""
    filepath = os.path.join(OUTPUT_DIR, name)
    writer = PillowWriter(fps=fps)
    anim.save(filepath, writer=writer, dpi=100)
    plt.close()
    print(f"Generated: {name}")


# =============================================================================
# 1. Agent-Environment Loop Animation
# =============================================================================

def create_agent_environment_loop():
    """Create animated GIF showing the agent-environment interaction loop."""

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')

    # Component positions
    agent_pos = (2, 4)
    env_pos = (10, 4)

    # Static elements (drawn every frame)
    def draw_static_elements():
        ax.clear()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('white')
        ax.set_title('Agent-Environment Interaction Loop', fontsize=14, fontweight='bold',
                    pad=20, color=COLORS['text'])

        # Agent box - clean white with border
        agent_box = FancyBboxPatch((agent_pos[0]-1, agent_pos[1]-1), 2, 2,
                                    boxstyle="round,pad=0.1",
                                    facecolor='white',
                                    edgecolor=COLORS['primary'],
                                    linewidth=2)
        ax.add_patch(agent_box)
        ax.text(agent_pos[0], agent_pos[1], 'Agent', ha='center', va='center',
               fontsize=13, fontweight='bold', color=COLORS['primary'])

        # Environment box - clean white with border
        env_box = FancyBboxPatch((env_pos[0]-1.5, env_pos[1]-1), 3, 2,
                                  boxstyle="round,pad=0.1",
                                  facecolor='white',
                                  edgecolor=COLORS['secondary'],
                                  linewidth=2)
        ax.add_patch(env_box)
        ax.text(env_pos[0], env_pos[1], 'Environment', ha='center', va='center',
               fontsize=13, fontweight='bold', color=COLORS['secondary'])

    # Animation frames
    frames_data = [
        # Frame 0: State observation
        {
            'active': 'state',
            'arrow_start': (env_pos[0]-1.5, env_pos[1]+0.5),
            'arrow_end': (agent_pos[0]+1, agent_pos[1]+0.5),
            'label': 'State $s_t$',
            'label_pos': (6, 5.5),
            'description': 'Agent observes current state',
            'color': COLORS['state']
        },
        # Frame 1: Action selection
        {
            'active': 'action',
            'arrow_start': (agent_pos[0]+1, agent_pos[1]-0.5),
            'arrow_end': (env_pos[0]-1.5, env_pos[1]-0.5),
            'label': 'Action $a_t$',
            'label_pos': (6, 2.5),
            'description': 'Agent selects action based on policy',
            'color': COLORS['action']
        },
        # Frame 2: State transition
        {
            'active': 'transition',
            'arrow_start': None,
            'arrow_end': None,
            'label': '$s_t \\rightarrow s_{t+1}$',
            'label_pos': (10, 6.5),
            'description': 'Environment transitions to new state',
            'color': COLORS['secondary']
        },
        # Frame 3: Reward
        {
            'active': 'reward',
            'arrow_start': (env_pos[0]-1.5, env_pos[1]),
            'arrow_end': (agent_pos[0]+1, agent_pos[1]),
            'label': 'Reward $r_{t+1}$',
            'label_pos': (6, 4.5),
            'description': 'Agent receives reward signal',
            'color': COLORS['reward']
        },
    ]

    def animate(frame_idx):
        draw_static_elements()
        frame = frames_data[frame_idx % len(frames_data)]

        # Draw arrow if present
        if frame['arrow_start'] is not None:
            color = frame['color']

            ax.annotate('', xy=frame['arrow_end'], xytext=frame['arrow_start'],
                       arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                      connectionstyle='arc3,rad=0.1'))

            # Label on arrow - clean white background
            ax.text(frame['label_pos'][0], frame['label_pos'][1], frame['label'],
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color=COLORS['text'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=color, linewidth=1.5, alpha=0.95))

        # Transition effect (internal to environment)
        if frame['active'] == 'transition':
            ax.text(env_pos[0], env_pos[1]+1.8, frame['label'],
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color=COLORS['text'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=COLORS['secondary'], linewidth=1.5))

        # Description text - clean white box
        ax.text(6, 1, frame['description'], ha='center', va='center',
               fontsize=11, style='italic', color=COLORS['text_light'],
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor=COLORS['border'], linewidth=1))

        # Step indicator
        ax.text(6, 7.3, f'Step {frame_idx % len(frames_data) + 1} of {len(frames_data)}',
               ha='center', va='center', fontsize=10, fontweight='bold',
               color=COLORS['text_light'])

        return []

    anim = FuncAnimation(fig, animate, frames=len(frames_data)*2, interval=1500, blit=False)
    save_animation(anim, 'agent_environment_loop.gif', fps=1)


# =============================================================================
# 2. GridWorld MDP Visualization (Static SVG)
# =============================================================================

def create_gridworld_mdp():
    """Create static visualization of a GridWorld MDP with transitions."""

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    grid_size = 4
    cell_size = 2
    offset_x = 1
    offset_y = 1

    # Define special cells
    goal_cell = (3, 3)
    trap_cell = (1, 2)
    agent_start = (0, 0)

    # Cell colors - subtle, professional
    cell_colors = {
        'default': 'white',
        'goal': '#dcfce7',      # Very light green
        'trap': '#fee2e2',      # Very light red
        'agent': '#dbeafe',     # Very light blue
    }

    # Draw grid
    for i in range(grid_size):
        for j in range(grid_size):
            x = offset_x + j * cell_size
            y = offset_y + (grid_size - 1 - i) * cell_size

            cell = (i, j)
            if cell == goal_cell:
                color = cell_colors['goal']
                border_color = COLORS['accent']
            elif cell == trap_cell:
                color = cell_colors['trap']
                border_color = COLORS['error']
            elif cell == agent_start:
                color = cell_colors['agent']
                border_color = COLORS['primary']
            else:
                color = cell_colors['default']
                border_color = COLORS['border']

            rect = Rectangle((x, y), cell_size, cell_size,
                            facecolor=color, edgecolor=border_color, linewidth=1.5)
            ax.add_patch(rect)

            # State label
            ax.text(x + cell_size/2, y + cell_size/2 + 0.3, f'$s_{{{i},{j}}}$',
                   ha='center', va='center', fontsize=9, color=COLORS['text_light'])

            # Special labels
            if cell == goal_cell:
                ax.text(x + cell_size/2, y + cell_size/2 - 0.3, 'GOAL\n+10',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color=COLORS['accent_dark'])
            elif cell == trap_cell:
                ax.text(x + cell_size/2, y + cell_size/2 - 0.3, 'TRAP\n-10',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color=COLORS['error'])
            elif cell == agent_start:
                ax.text(x + cell_size/2, y + cell_size/2 - 0.3, 'START',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color=COLORS['primary'])

    # Draw sample transitions with probabilities
    x1, y1 = offset_x + 1 * cell_size + cell_size/2, offset_y + (grid_size - 1 - 1) * cell_size + cell_size/2
    x2, y2 = offset_x + 2 * cell_size + cell_size/2, offset_y + (grid_size - 1 - 1) * cell_size + cell_size/2

    ax.annotate('', xy=(x2-0.3, y2), xytext=(x1+0.3, y1),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))
    ax.text((x1+x2)/2, y1+0.5, 'a=Right', ha='center', va='center', fontsize=8,
           color=COLORS['text'],
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor=COLORS['primary'], linewidth=1))

    # Action legend
    legend_x = 10.5
    legend_y = 7
    ax.text(legend_x, legend_y + 1.5, 'Actions:', fontsize=11, fontweight='bold',
           color=COLORS['text'])
    actions = ['Up', 'Down', 'Left', 'Right']
    arrows = [(0, 0.3), (0, -0.3), (-0.3, 0), (0.3, 0)]
    for idx, (action, (dx, dy)) in enumerate(zip(actions, arrows)):
        ax.annotate('', xy=(legend_x + dx + 0.5, legend_y - idx*0.6 + dy),
                   xytext=(legend_x + 0.5, legend_y - idx*0.6),
                   arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))
        ax.text(legend_x + 1.2, legend_y - idx*0.6, action, ha='left', va='center',
               fontsize=9, color=COLORS['text'])

    # MDP components box - clean white with border
    ax.add_patch(FancyBboxPatch((10.2, 2.5), 3.5, 2.8, boxstyle='round,pad=0.1',
                                facecolor='white', edgecolor=COLORS['border'], linewidth=1))
    ax.text(10.5, 4.8, 'MDP Components:', fontsize=11, fontweight='bold', color=COLORS['text'])
    components = [
        '$\\mathcal{S}$: 16 grid states',
        '$\\mathcal{A}$: {Up, Down, Left, Right}',
        '$\\mathcal{P}$: Deterministic transitions',
        '$\\mathcal{R}$: -1/step, +10 goal, -10 trap',
        '$\\gamma = 0.9$'
    ]
    for idx, comp in enumerate(components):
        ax.text(10.5, 4.2 - idx*0.45, comp, ha='left', va='center', fontsize=9,
               color=COLORS['text_light'])

    # Legend
    ax.text(10.5, 1.5, 'Legend:', fontsize=11, fontweight='bold', color=COLORS['text'])
    legend_items = [
        (cell_colors['agent'], COLORS['primary'], 'Start'),
        (cell_colors['goal'], COLORS['accent'], 'Goal (+10)'),
        (cell_colors['trap'], COLORS['error'], 'Trap (-10)'),
        (cell_colors['default'], COLORS['border'], 'Empty (-1/step)'),
    ]
    for idx, (color, border, label) in enumerate(legend_items):
        rect = Rectangle((10.5, 0.8 - idx*0.5), 0.4, 0.35, facecolor=color,
                         edgecolor=border, linewidth=1)
        ax.add_patch(rect)
        ax.text(11.1, 0.95 - idx*0.5, label, ha='left', va='center', fontsize=9,
               color=COLORS['text'])

    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('GridWorld MDP Example', fontsize=14, fontweight='bold',
                pad=20, color=COLORS['text'])

    # Save as SVG
    filepath = os.path.join(OUTPUT_DIR, 'gridworld_mdp.svg')
    plt.savefig(filepath, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: gridworld_mdp.svg")


# =============================================================================
# 3. Discount Factor Effect Animation
# =============================================================================

def create_discount_effect():
    """Create animated visualization showing how discount factor affects returns."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # Reward sequence (same for all gamma values)
    rewards = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    timesteps = list(range(len(rewards)))

    gamma_values = [0.0, 0.5, 0.9, 0.99, 1.0]

    def calculate_discounted_values(gamma):
        """Calculate discounted value at each timestep."""
        values = []
        for t, r in enumerate(rewards):
            values.append(r * (gamma ** t))
        return values

    def calculate_return(gamma):
        """Calculate total discounted return."""
        return sum(r * (gamma ** t) for t, r in enumerate(rewards))

    def animate(frame_idx):
        gamma = gamma_values[frame_idx % len(gamma_values)]

        for ax in axes:
            ax.clear()
            ax.set_facecolor('white')

        # Left plot: Bar chart of discounted rewards
        ax1 = axes[0]
        discounted = calculate_discounted_values(gamma)

        # Use sequential colormap
        cmap = get_sequential_cmap()
        colors = [cmap(v / max(rewards)) for v in discounted]

        bars = ax1.bar(timesteps, discounted, color=colors, edgecolor=COLORS['border'], linewidth=1)
        ax1.axhline(y=0, color=COLORS['border'], linewidth=0.5)

        # Original rewards as outline
        ax1.bar(timesteps, rewards, fill=False, edgecolor=COLORS['text_muted'],
               linewidth=1, linestyle='--', alpha=0.5)

        ax1.set_xlabel('Timestep $t$', fontsize=11, color=COLORS['text'])
        ax1.set_ylabel('Discounted Reward $\\gamma^t R_t$', fontsize=11, color=COLORS['text'])
        ax1.set_title(f'Discounted Rewards (R=10 at each step)', fontsize=12, fontweight='bold',
                     color=COLORS['text'])
        ax1.set_ylim(0, 12)
        ax1.set_xticks(timesteps)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Annotation - clean white box
        ax1.text(4.5, 11, f'$\\gamma = {gamma}$', ha='center', va='center',
                fontsize=14, fontweight='bold', color=COLORS['text'],
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor=COLORS['primary'], linewidth=1.5))

        # Right plot: Cumulative return comparison
        ax2 = axes[1]

        # Professional color palette for bars
        bar_colors = [COLORS['chart_1'], COLORS['chart_2'], COLORS['chart_3'],
                     COLORS['chart_4'], COLORS['chart_5']]
        returns = [calculate_return(g) for g in gamma_values]
        bar_positions = range(len(gamma_values))

        bars2 = ax2.bar(bar_positions, returns, color=bar_colors,
                       edgecolor=COLORS['border'], linewidth=1)

        # Highlight current gamma
        bars2[frame_idx % len(gamma_values)].set_edgecolor(COLORS['text'])
        bars2[frame_idx % len(gamma_values)].set_linewidth(2.5)

        ax2.set_xlabel('Discount Factor $\\gamma$', fontsize=11, color=COLORS['text'])
        ax2.set_ylabel('Total Return $G_0$', fontsize=11, color=COLORS['text'])
        ax2.set_title('Total Return for Different $\\gamma$', fontsize=12, fontweight='bold',
                     color=COLORS['text'])
        ax2.set_xticks(bar_positions)
        ax2.set_xticklabels([str(g) for g in gamma_values])
        ax2.set_ylim(0, 110)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Add value labels on bars
        for i, (bar, ret) in enumerate(zip(bars2, returns)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{ret:.1f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=COLORS['text'])

        # Current return highlight
        current_return = calculate_return(gamma)
        ax2.text(2, 95, f'$G_0 = \\sum_{{t=0}}^{{9}} \\gamma^t R_t = {current_return:.1f}$',
                ha='center', va='center', fontsize=11, color=COLORS['text'],
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor=COLORS['border'], linewidth=1))

        fig.suptitle('Effect of Discount Factor on Returns', fontsize=14, fontweight='bold',
                    y=1.02, color=COLORS['text'])
        plt.tight_layout()

        return []

    anim = FuncAnimation(fig, animate, frames=len(gamma_values)*2, interval=2000, blit=False)
    save_animation(anim, 'discount_effect.gif', fps=1)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    print("Generating MDP Foundations visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Generate all visualizations
    create_agent_environment_loop()
    create_gridworld_mdp()
    create_discount_effect()

    print("\nAll MDP visualizations generated successfully!")
