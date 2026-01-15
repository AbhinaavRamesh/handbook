"""
ReAct Loop Animation Generator

Creates an animated GIF showing the Thought-Action-Observation cycle
used in ReAct agents.

Output: /docs/genai/assets/images/agents/react_loop.gif
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image
import io
import os

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "agents"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_react_frame(step: int, total_steps: int = 12) -> Image.Image:
    """Create a single frame of the ReAct animation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 7.5, "ReAct: Reasoning + Acting", fontsize=16,
            ha='center', fontweight='bold', color='#1a237e')

    # Define node positions (circular layout)
    center = (5, 4)
    radius = 2.5
    angles = [90, 330, 210]  # Thought, Action, Observation

    nodes = {
        'Thought': {
            'angle': 90,
            'color': '#e3f2fd',
            'border': '#1976d2',
            'icon': '?',
            'desc': 'Reason about\nwhat to do next'
        },
        'Action': {
            'angle': 330,
            'color': '#fff3e0',
            'border': '#f57c00',
            'icon': '!',
            'desc': 'Execute tool\nor generate response'
        },
        'Observation': {
            'angle': 210,
            'color': '#e8f5e9',
            'border': '#388e3c',
            'icon': 'i',
            'desc': 'Receive result\nfrom environment'
        }
    }

    # Calculate positions
    positions = {}
    for name, info in nodes.items():
        angle_rad = np.radians(info['angle'])
        x = center[0] + radius * np.cos(angle_rad)
        y = center[1] + radius * np.sin(angle_rad)
        positions[name] = (x, y)

    # Determine which element is active based on step
    cycle_step = step % 3
    active_node = list(nodes.keys())[cycle_step]

    # Draw arrows between nodes
    arrow_order = [('Thought', 'Action'), ('Action', 'Observation'), ('Observation', 'Thought')]
    for i, (start, end) in enumerate(arrow_order):
        start_pos = positions[start]
        end_pos = positions[end]

        # Calculate arrow start/end points (offset from node centers)
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/dist, dy/dist

        arrow_start = (start_pos[0] + dx * 0.8, start_pos[1] + dy * 0.8)
        arrow_end = (end_pos[0] - dx * 0.8, end_pos[1] - dy * 0.8)

        # Highlight active arrow
        is_active = (i == cycle_step)
        arrow_color = '#d32f2f' if is_active else '#9e9e9e'
        arrow_width = 3 if is_active else 1.5

        ax.annotate(
            '', xy=arrow_end, xytext=arrow_start,
            arrowprops=dict(
                arrowstyle='-|>',
                color=arrow_color,
                lw=arrow_width,
                mutation_scale=15
            )
        )

    # Draw nodes
    for name, info in nodes.items():
        x, y = positions[name]
        is_active = (name == active_node)

        # Node circle
        circle = patches.Circle(
            (x, y), 0.7,
            facecolor=info['color'],
            edgecolor=info['border'],
            linewidth=4 if is_active else 2,
            zorder=2
        )
        ax.add_patch(circle)

        # Pulsing effect for active node
        if is_active:
            pulse_circle = patches.Circle(
                (x, y), 0.85,
                facecolor='none',
                edgecolor=info['border'],
                linewidth=2,
                alpha=0.5,
                linestyle='--',
                zorder=1
            )
            ax.add_patch(pulse_circle)

        # Node label
        ax.text(x, y + 0.1, name, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#333')

        # Description (below node)
        ax.text(x, y - 1.3, info['desc'], ha='center', va='top',
                fontsize=8, color='#666', style='italic')

    # Add iteration counter
    iteration = (step // 3) + 1
    ax.text(5, 1, f"Iteration {iteration}", ha='center',
            fontsize=12, color='#666')

    # Add current step indicator
    step_texts = {
        0: "Thinking about the next step...",
        1: "Executing action...",
        2: "Processing observation..."
    }
    ax.text(5, 0.5, step_texts[cycle_step], ha='center',
            fontsize=10, color='#d32f2f', fontweight='bold')

    # Add legend
    legend_y = 7
    ax.add_patch(patches.Rectangle((0.3, legend_y - 0.5), 2.5, 0.8,
                                   facecolor='#f5f5f5', edgecolor='#ccc'))
    ax.text(0.5, legend_y, "Active step", fontsize=9, color='#666')
    ax.add_patch(patches.Circle((2.3, legend_y), 0.15, facecolor='#d32f2f'))

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return img


def create_static_diagram() -> None:
    """Create a static version of the ReAct diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6, 7.5, "ReAct: Reasoning + Acting Loop", fontsize=18,
            ha='center', fontweight='bold', color='#1a237e')

    # Subtitle
    ax.text(6, 7, "Thought - Action - Observation Cycle", fontsize=12,
            ha='center', color='#666', style='italic')

    # Node positions
    positions = {
        'Thought': (2, 4),
        'Action': (6, 4),
        'Observation': (10, 4)
    }

    node_info = {
        'Thought': {
            'color': '#e3f2fd',
            'border': '#1976d2',
            'desc': 'Reason about\ncurrent state\nand next steps'
        },
        'Action': {
            'color': '#fff3e0',
            'border': '#f57c00',
            'desc': 'Execute tool\ncall or generate\nfinal response'
        },
        'Observation': {
            'color': '#e8f5e9',
            'border': '#388e3c',
            'desc': 'Receive and\nprocess result\nfrom action'
        }
    }

    # Draw nodes
    for name, pos in positions.items():
        info = node_info[name]
        x, y = pos

        # Box
        rect = patches.FancyBboxPatch(
            (x - 1.2, y - 0.8), 2.4, 1.6,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=info['color'],
            edgecolor=info['border'],
            linewidth=2
        )
        ax.add_patch(rect)

        # Label
        ax.text(x, y + 0.3, name, ha='center', va='center',
                fontsize=14, fontweight='bold', color='#333')

        # Description
        ax.text(x, y - 1.5, info['desc'], ha='center', va='top',
                fontsize=9, color='#666')

    # Draw arrows
    arrow_style = dict(arrowstyle='-|>', color='#333', lw=2, mutation_scale=20)

    # Thought -> Action
    ax.annotate('', xy=(4.8, 4), xytext=(3.2, 4), arrowprops=arrow_style)
    ax.text(4, 4.3, '1', fontsize=10, ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#333'))

    # Action -> Observation
    ax.annotate('', xy=(8.8, 4), xytext=(7.2, 4), arrowprops=arrow_style)
    ax.text(8, 4.3, '2', fontsize=10, ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#333'))

    # Observation -> Thought (loop back)
    # Draw curved arrow at top
    loop_arrow = FancyArrowPatch(
        (9.5, 4.8), (2.5, 4.8),
        connectionstyle="arc3,rad=0.3",
        arrowstyle='-|>',
        color='#d32f2f',
        lw=2,
        mutation_scale=20
    )
    ax.add_patch(loop_arrow)
    ax.text(6, 6, '3 (loop)', fontsize=10, ha='center', color='#d32f2f',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#d32f2f'))

    # Exit condition
    ax.annotate('', xy=(6, 2), xytext=(6, 3.2),
                arrowprops=dict(arrowstyle='-|>', color='#388e3c', lw=2))
    ax.text(6, 1.5, 'Final Answer\n(when task complete)', ha='center',
            fontsize=10, color='#388e3c', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'react_loop_static.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_animation():
    """Create the animated GIF."""
    frames = []
    total_frames = 12  # 4 full cycles

    for step in range(total_frames):
        frame = create_react_frame(step, total_frames)
        frames.append(frame)

    # Save as GIF
    output_path = os.path.join(OUTPUT_DIR, 'react_loop.gif')
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=800,  # ms per frame
        loop=0  # loop forever
    )
    print(f"Animation saved to: {output_path}")


if __name__ == "__main__":
    print("Creating ReAct loop visualizations...")
    create_static_diagram()
    print(f"Static diagram saved to: {OUTPUT_DIR}/react_loop_static.png")
    create_animation()
    print("Done!")
