"""
Generate Deep RL visualizations for the Deep RL documentation page.
Creates both static SVG and animated GIF visualizations:
1. DQN architecture diagram (SVG)
2. Experience replay buffer animation (GIF)
3. Target network update animation (GIF)
4. Q-value overestimation problem visualization (SVG)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Set style for clean visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images/deep-rl'

# Color palette
COLORS = {
    'primary': '#4299e1',      # Blue
    'secondary': '#48bb78',    # Green
    'accent': '#ed8936',       # Orange
    'highlight': '#9f7aea',    # Purple
    'danger': '#fc8181',       # Red
    'neutral': '#718096',      # Gray
    'dark': '#2d3748',         # Dark gray
    'light': '#edf2f7',        # Light gray
    'white': '#ffffff',
}


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_rounded_box(ax, x, y, width, height, color, text='', text_color='white',
                     fontsize=10, alpha=1.0, linewidth=2, edgecolor=None):
    """Helper to draw a rounded rectangle with centered text."""
    if edgecolor is None:
        edgecolor = color
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha
    )
    ax.add_patch(box)
    if text:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold')
    return box


def draw_arrow(ax, start, end, color=COLORS['dark'], style='simple',
               connectionstyle='arc3,rad=0', linewidth=2):
    """Helper to draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=f'->,head_length=0.4,head_width=0.2',
        color=color, linewidth=linewidth,
        connectionstyle=connectionstyle
    )
    ax.add_patch(arrow)
    return arrow


def generate_dqn_architecture():
    """
    Visualization 1: DQN Architecture Diagram
    Shows the flow from input frames through conv layers to Q-value output.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 8.3, 'DQN Architecture', fontsize=16, fontweight='bold',
            ha='center', color=COLORS['dark'])

    # Input: Stacked frames
    input_x = 0.5
    for i in range(4):
        offset = i * 0.15
        rect = Rectangle((input_x - 0.6 + offset, 2.5 - offset), 1.2, 3,
                         facecolor=COLORS['light'], edgecolor=COLORS['neutral'],
                         linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
    ax.text(input_x + 0.2, 1.3, 'Input\n84x84x4\nframes', ha='center', va='top',
            fontsize=9, color=COLORS['dark'])

    # Conv Layer 1
    conv1_x = 3
    draw_rounded_box(ax, conv1_x, 4, 1.5, 2.5, COLORS['primary'],
                     'Conv1\n32@8x8\nstride 4', fontsize=8)
    draw_arrow(ax, (1.3, 4), (conv1_x - 0.9, 4), COLORS['dark'])

    # Conv Layer 2
    conv2_x = 5.5
    draw_rounded_box(ax, conv2_x, 4, 1.5, 2.2, COLORS['primary'],
                     'Conv2\n64@4x4\nstride 2', fontsize=8)
    draw_arrow(ax, (conv1_x + 0.9, 4), (conv2_x - 0.9, 4), COLORS['dark'])

    # Conv Layer 3
    conv3_x = 8
    draw_rounded_box(ax, conv3_x, 4, 1.5, 1.9, COLORS['primary'],
                     'Conv3\n64@3x3\nstride 1', fontsize=8)
    draw_arrow(ax, (conv2_x + 0.9, 4), (conv3_x - 0.9, 4), COLORS['dark'])

    # Flatten + FC
    fc_x = 10.5
    draw_rounded_box(ax, fc_x, 4, 1.4, 1.6, COLORS['secondary'],
                     'FC\n512', fontsize=9)
    draw_arrow(ax, (conv3_x + 0.9, 4), (fc_x - 0.85, 4), COLORS['dark'])

    # Output Q-values
    output_x = 13.5
    q_height = 4.5
    draw_rounded_box(ax, output_x, 4, 1.2, q_height, COLORS['accent'], '', fontsize=9)

    # Individual Q-value labels
    n_actions = 6
    for i in range(n_actions):
        y_pos = 4 + (q_height/2 - 0.4) - i * (q_height - 0.8) / (n_actions - 1)
        ax.text(output_x, y_pos, f'Q(s,a{i+1})', ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')

    draw_arrow(ax, (fc_x + 0.85, 4), (output_x - 0.75, 4), COLORS['dark'])
    ax.text(output_x, 1.3, 'Output\nQ-values\nfor all actions', ha='center', va='top',
            fontsize=9, color=COLORS['dark'])

    # ReLU annotations
    for x in [conv1_x, conv2_x, conv3_x, fc_x]:
        ax.text(x, 2.3, 'ReLU', ha='center', fontsize=7, color=COLORS['neutral'],
                style='italic')

    # Add legend box
    legend_box = FancyBboxPatch(
        (0.2, 6.5), 5, 1.3,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['white'], edgecolor=COLORS['neutral'],
        linewidth=1, alpha=0.9
    )
    ax.add_patch(legend_box)

    # Legend items
    draw_rounded_box(ax, 1, 7.3, 0.6, 0.4, COLORS['primary'], '', fontsize=7)
    ax.text(1.5, 7.3, 'Conv Layer', ha='left', va='center', fontsize=8)
    draw_rounded_box(ax, 3.2, 7.3, 0.6, 0.4, COLORS['secondary'], '', fontsize=7)
    ax.text(3.7, 7.3, 'FC Layer', ha='left', va='center', fontsize=8)
    draw_rounded_box(ax, 1, 6.8, 0.6, 0.4, COLORS['accent'], '', fontsize=7)
    ax.text(1.5, 6.8, 'Output', ha='left', va='center', fontsize=8)

    # Argmax annotation
    ax.annotate('', xy=(output_x + 0.8, 5.5), xytext=(output_x + 1.8, 6.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=2))
    ax.text(output_x + 2.5, 6.5, r'$\arg\max_a Q(s,a)$', fontsize=10,
            color=COLORS['highlight'], fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dqn-architecture.svg'), format='svg',
                bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print("Generated: dqn-architecture.svg")


def generate_experience_replay_animation():
    """
    Visualization 2: Experience Replay Buffer Animation (GIF)
    Shows transitions being added to buffer and random sampling.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    buffer_size = 10
    batch_size = 3

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()
        ax.set_xlim(-1, 13)
        ax.set_ylim(-0.5, 6)
        ax.axis('off')

        # Title
        ax.text(6, 5.5, 'Experience Replay Buffer', fontsize=14, fontweight='bold',
                ha='center', color=COLORS['dark'])

        # Buffer slots
        buffer_y = 3
        slot_width = 1.0
        slot_height = 0.8

        # Calculate filled slots (circular buffer behavior)
        total_added = frame
        filled_slots = min(total_added, buffer_size)
        current_idx = total_added % buffer_size

        for i in range(buffer_size):
            x = 1.5 + i * 1.1

            # Determine if slot is filled and if it's the current write position
            is_filled = i < filled_slots or (total_added >= buffer_size)
            is_current = (i == current_idx) and (frame % 5 < 3)  # Blinking effect for new

            if is_current and frame > 0:
                color = COLORS['highlight']
                text_color = 'white'
                label = f'NEW'
            elif is_filled:
                color = COLORS['primary']
                text_color = 'white'
                transition_num = (total_added - filled_slots + i) % buffer_size + 1
                if total_added >= buffer_size:
                    transition_num = ((current_idx - buffer_size + i) % buffer_size) + 1
                label = f't{i}'
            else:
                color = COLORS['light']
                text_color = COLORS['neutral']
                label = 'empty'

            draw_rounded_box(ax, x, buffer_y, slot_width, slot_height, color,
                           label, text_color, fontsize=8)

        # Buffer label
        ax.text(0.3, buffer_y, 'Buffer\n(size=10)', ha='center', va='center',
                fontsize=9, color=COLORS['dark'])

        # Incoming experience arrow (appears every few frames)
        if frame % 5 < 3 and frame > 0:
            ax.annotate('', xy=(1.5 + current_idx * 1.1, buffer_y + 0.7),
                       xytext=(1.5 + current_idx * 1.1, buffer_y + 1.5),
                       arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2))
            ax.text(1.5 + current_idx * 1.1, buffer_y + 1.8,
                   '$(s_t, a_t, r_t, s_{t+1})$', ha='center', fontsize=9,
                   color=COLORS['secondary'])

        # Random sampling visualization (every 5 frames when buffer has data)
        if frame >= 5 and frame % 5 >= 3:
            # Select random indices for batch
            np.random.seed(frame)
            available = min(filled_slots, buffer_size)
            if available >= batch_size:
                sampled_indices = np.random.choice(available, batch_size, replace=False)

                # Highlight sampled slots
                for idx in sampled_indices:
                    x = 1.5 + idx * 1.1
                    circle = Circle((x, buffer_y), 0.55, fill=False,
                                  edgecolor=COLORS['accent'], linewidth=3)
                    ax.add_patch(circle)

                # Draw arrows to mini-batch
                batch_y = 1
                ax.text(6, 0.3, 'Random Mini-batch', ha='center', fontsize=10,
                       color=COLORS['accent'], fontweight='bold')

                for i, idx in enumerate(sampled_indices):
                    src_x = 1.5 + idx * 1.1
                    dst_x = 4.5 + i * 1.5

                    # Curved arrow
                    ax.annotate('', xy=(dst_x, batch_y + 0.5),
                               xytext=(src_x, buffer_y - 0.5),
                               arrowprops=dict(arrowstyle='->', color=COLORS['accent'],
                                             connectionstyle='arc3,rad=0.2', lw=1.5))

                    # Mini-batch slot
                    draw_rounded_box(ax, dst_x, batch_y, slot_width, slot_height,
                                   COLORS['accent'], f't{idx}', 'white', fontsize=8)

        # Info text
        info_text = f'Step: {frame} | Buffer: {filled_slots}/{buffer_size}'
        ax.text(6, -0.2, info_text, ha='center', fontsize=10, color=COLORS['neutral'])

        return []

    anim = FuncAnimation(fig, animate, init_func=init, frames=30,
                        interval=400, blit=False)

    writer = PillowWriter(fps=2.5)
    anim.save(os.path.join(OUTPUT_DIR, 'experience-replay.gif'), writer=writer, dpi=100)
    plt.close()
    print("Generated: experience-replay.gif")


def generate_target_network_animation():
    """
    Visualization 3: Target Network Update Animation (GIF)
    Shows online network updating while target stays frozen, then periodic sync.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    update_interval = 5  # Sync every 5 frames

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()
        ax.set_xlim(-1, 13)
        ax.set_ylim(-0.5, 6.5)
        ax.axis('off')

        # Title
        ax.text(6, 6, 'Target Network Updates', fontsize=14, fontweight='bold',
                ha='center', color=COLORS['dark'])

        # Online network (left)
        online_x = 3
        online_y = 3.5

        # Pulsing effect for online network (always updating)
        pulse = 0.1 * np.sin(frame * 0.5)

        draw_rounded_box(ax, online_x, online_y, 2.5 + pulse, 2 + pulse,
                        COLORS['primary'], 'Online\nNetwork\n$\\theta$',
                        fontsize=10)
        ax.text(online_x, 1.8, 'Updates every step', ha='center',
                fontsize=9, color=COLORS['primary'], style='italic')

        # Target network (right)
        target_x = 9
        target_y = 3.5

        # Check if this is a sync frame
        is_sync_frame = (frame % update_interval == 0) and frame > 0

        if is_sync_frame:
            target_color = COLORS['secondary']
            border_color = COLORS['highlight']
        else:
            target_color = COLORS['neutral']
            border_color = COLORS['neutral']

        draw_rounded_box(ax, target_x, target_y, 2.5, 2, target_color,
                        'Target\nNetwork\n$\\theta^-$', fontsize=10,
                        edgecolor=border_color, linewidth=3 if is_sync_frame else 2)

        sync_count = frame // update_interval
        ax.text(target_x, 1.8, f'Synced {sync_count}x (every {update_interval} steps)',
                ha='center', fontsize=9, color=target_color, style='italic')

        # Sync arrow (appears during sync)
        if is_sync_frame:
            ax.annotate('', xy=(target_x - 1.5, target_y),
                       xytext=(online_x + 1.5, online_y),
                       arrowprops=dict(arrowstyle='->', color=COLORS['highlight'],
                                      lw=4, connectionstyle='arc3,rad=0.2'))
            ax.text(6, 4.8, r'$\theta^- \leftarrow \theta$', ha='center',
                   fontsize=14, color=COLORS['highlight'], fontweight='bold')

        # Loss computation visualization
        ax.text(6, 0.8, 'Loss Computation', ha='center', fontsize=11,
                fontweight='bold', color=COLORS['dark'])

        # Show loss equation (simplified to avoid LaTeX parsing issues)
        loss_eq = r'$L = (r + \gamma \max Q_{\theta^-} - Q_\theta)^2$'
        ax.text(6, 0.2, loss_eq, ha='center', fontsize=10, color=COLORS['dark'])

        # Arrows showing which network contributes what
        ax.annotate('', xy=(3.8, 0.5), xytext=(online_x, 2.3),
                   arrowprops=dict(arrowstyle='->', color=COLORS['primary'],
                                  lw=1.5, ls='--'))
        ax.annotate('', xy=(8.2, 0.5), xytext=(target_x, 2.3),
                   arrowprops=dict(arrowstyle='->', color=target_color,
                                  lw=1.5, ls='--'))

        # Step counter
        ax.text(11.5, 5.5, f'Step: {frame}', ha='center', fontsize=12,
                color=COLORS['dark'],
                bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8))

        return []

    anim = FuncAnimation(fig, animate, init_func=init, frames=25,
                        interval=500, blit=False)

    writer = PillowWriter(fps=2)
    anim.save(os.path.join(OUTPUT_DIR, 'target-network-update.gif'), writer=writer, dpi=100)
    plt.close()
    print("Generated: target-network-update.gif")


def generate_overestimation_visualization():
    """
    Visualization 4: Q-value Overestimation Problem (SVG)
    Shows how max operator leads to systematic overestimation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    np.random.seed(42)

    # Left plot: Single state overestimation
    ax1 = axes[0]
    n_actions = 5
    true_q = np.array([1.0, 0.8, 0.9, 0.7, 0.85])
    noise_std = 0.3
    n_samples = 1000

    # Simulate noisy estimates
    estimated_max_values = []
    for _ in range(n_samples):
        noise = np.random.normal(0, noise_std, n_actions)
        estimated_q = true_q + noise
        estimated_max_values.append(np.max(estimated_q))

    true_max = np.max(true_q)
    estimated_max_mean = np.mean(estimated_max_values)

    # Bar plot
    x_pos = np.arange(n_actions)
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, true_q, width, label='True Q-values',
                   color=COLORS['primary'], alpha=0.8)

    # Show one noisy estimate
    noisy_estimate = true_q + np.random.normal(0, noise_std, n_actions)
    bars2 = ax1.bar(x_pos + width/2, noisy_estimate, width, label='Noisy Estimate',
                   color=COLORS['accent'], alpha=0.8)

    # Highlight max values
    true_max_idx = np.argmax(true_q)
    noisy_max_idx = np.argmax(noisy_estimate)

    ax1.bar(true_max_idx - width/2, true_q[true_max_idx], width,
           color=COLORS['primary'], edgecolor=COLORS['dark'], linewidth=3)
    ax1.bar(noisy_max_idx + width/2, noisy_estimate[noisy_max_idx], width,
           color=COLORS['accent'], edgecolor=COLORS['dark'], linewidth=3)

    # Reference lines
    ax1.axhline(y=true_max, color=COLORS['secondary'], linestyle='--', linewidth=2,
               label=f'True max = {true_max:.2f}')
    ax1.axhline(y=estimated_max_mean, color=COLORS['danger'], linestyle='--', linewidth=2,
               label=f'E[estimated max] = {estimated_max_mean:.2f}')

    ax1.set_xlabel('Action')
    ax1.set_ylabel('Q-value')
    ax1.set_title('Max Operator Bias (Single State)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'$a_{i+1}$' for i in range(n_actions)])
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim(0, 1.8)

    # Annotation for bias
    bias = estimated_max_mean - true_max
    ax1.annotate('', xy=(4.3, estimated_max_mean), xytext=(4.3, true_max),
                arrowprops=dict(arrowstyle='<->', color=COLORS['danger'], lw=2))
    ax1.text(4.5, (estimated_max_mean + true_max) / 2, f'Bias\n+{bias:.2f}',
            fontsize=9, color=COLORS['danger'], fontweight='bold')

    # Right plot: Overestimation over training
    ax2 = axes[1]

    steps = np.arange(0, 1000, 10)
    true_values = 10 * (1 - np.exp(-steps / 300))  # True Q converges to ~10

    # Standard Q-learning (overestimates)
    q_learning_values = true_values * (1 + 0.3 * (1 - np.exp(-steps / 200)))
    q_learning_values += np.random.normal(0, 0.3, len(steps)).cumsum() * 0.02

    # Double Q-learning (less bias)
    double_q_values = true_values * (1 + 0.05 * (1 - np.exp(-steps / 300)))
    double_q_values += np.random.normal(0, 0.2, len(steps)).cumsum() * 0.01

    ax2.fill_between(steps, true_values, q_learning_values, alpha=0.3,
                    color=COLORS['danger'], label='Overestimation')
    ax2.plot(steps, true_values, color=COLORS['secondary'], linewidth=2.5,
            label='True Q-value', linestyle='-')
    ax2.plot(steps, q_learning_values, color=COLORS['danger'], linewidth=2.5,
            label='Standard DQN', linestyle='--')
    ax2.plot(steps, double_q_values, color=COLORS['primary'], linewidth=2.5,
            label='Double DQN', linestyle='-.')

    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Q-value Estimate')
    ax2.set_title('Overestimation During Training', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 18)

    # Add grid
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'overestimation.svg'), format='svg',
                bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print("Generated: overestimation.svg")


if __name__ == '__main__':
    print("Generating Deep RL visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    ensure_output_dir()

    generate_dqn_architecture()
    generate_experience_replay_animation()
    generate_target_network_animation()
    generate_overestimation_visualization()

    print("-" * 50)
    print("All visualizations generated successfully!")
