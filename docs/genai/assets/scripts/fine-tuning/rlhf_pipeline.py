"""
RLHF Pipeline Visualization

This script creates visualizations of the RLHF training pipeline including:
- Three-phase RLHF overview
- Reward model training
- PPO training loop

Output: rlhf_pipeline.png, reward_model_training.png, ppo_loop.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "fine-tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, width, height, text, color='lightblue',
             fontsize=10, alpha=0.8, edgecolor='black'):
    """Draw a rounded box with text"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', wrap=True)


def draw_arrow(ax, start, end, color='black', style='->', lw=2, connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=lw,
        connectionstyle=connectionstyle,
        mutation_scale=15
    )
    ax.add_patch(arrow)


def create_rlhf_pipeline():
    """Create the main RLHF pipeline diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(8, 9.5, 'RLHF: Reinforcement Learning from Human Feedback',
            ha='center', fontsize=16, fontweight='bold')
    ax.text(8, 9, 'Three-phase training pipeline',
            ha='center', fontsize=12, style='italic', color='gray')

    # Phase 1: SFT
    phase1_x = 0.5
    ax.text(phase1_x + 2, 8.2, 'Phase 1: Supervised Fine-Tuning',
            fontsize=12, fontweight='bold', color='#1565c0')

    draw_box(ax, phase1_x, 6, 1.8, 1.2, 'Base\nModel', color='#e3f2fd')
    draw_box(ax, phase1_x + 2.5, 6, 2, 1.2, 'Demonstration\nData', color='#fff3e0')
    draw_box(ax, phase1_x + 2.5, 4, 2, 1.2, 'SFT\nTraining', color='#bbdefb')
    draw_box(ax, phase1_x + 0.5, 2, 2, 1.2, 'SFT Model\n(π_SFT)', color='#c8e6c9')

    # Arrows for phase 1
    draw_arrow(ax, (phase1_x + 1.8, 6.6), (phase1_x + 2.5, 6.6))
    draw_arrow(ax, (phase1_x + 3.5, 6), (phase1_x + 3.5, 5.2))
    draw_arrow(ax, (phase1_x + 3.5, 4), (phase1_x + 2, 3.2),
               connectionstyle='arc3,rad=0.3')

    # Phase 2: Reward Model
    phase2_x = 5.5
    ax.text(phase2_x + 2, 8.2, 'Phase 2: Reward Modeling',
            fontsize=12, fontweight='bold', color='#c62828')

    draw_box(ax, phase2_x, 6, 1.8, 1.2, 'Prompts', color='#e3f2fd')
    draw_box(ax, phase2_x + 2.2, 6, 2.2, 1.2, 'Generate\nResponses', color='#e8f5e9')
    draw_box(ax, phase2_x + 2.2, 4, 2.2, 1.2, 'Human\nRankings', color='#fff9c4')
    draw_box(ax, phase2_x + 0.5, 2, 2.2, 1.2, 'Reward\nModel (R)', color='#ffccbc')

    # Arrows for phase 2
    draw_arrow(ax, (phase2_x + 1.8, 6.6), (phase2_x + 2.2, 6.6))
    draw_arrow(ax, (phase2_x + 3.3, 6), (phase2_x + 3.3, 5.2))
    draw_arrow(ax, (phase2_x + 3.3, 4), (phase2_x + 2.2, 3.2),
               connectionstyle='arc3,rad=0.3')

    # Human icon
    ax.text(phase2_x + 4.6, 4.5, '👤', fontsize=20)

    # Phase 3: RL Fine-tuning
    phase3_x = 10.5
    ax.text(phase3_x + 2.2, 8.2, 'Phase 3: RL Fine-Tuning',
            fontsize=12, fontweight='bold', color='#2e7d32')

    draw_box(ax, phase3_x, 6, 1.8, 1.2, 'SFT\nModel', color='#c8e6c9')
    draw_box(ax, phase3_x + 2.2, 6, 1.8, 1.2, 'Reward\nModel', color='#ffccbc')
    draw_box(ax, phase3_x + 1.1, 4, 2, 1.2, 'PPO\nTraining', color='#e1bee7')
    draw_box(ax, phase3_x + 0.5, 2, 2.2, 1.2, 'Aligned\nModel', color='#a5d6a7')

    # Arrows for phase 3
    draw_arrow(ax, (phase3_x + 0.9, 6), (phase3_x + 1.6, 5.2),
               connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (phase3_x + 3.1, 6), (phase3_x + 2.6, 5.2),
               connectionstyle='arc3,rad=0.2')
    draw_arrow(ax, (phase3_x + 2.1, 4), (phase3_x + 1.6, 3.2))

    # Reference model arrow
    draw_box(ax, phase3_x + 4.2, 4.5, 1.5, 0.8, 'Reference\n(frozen)', color='#e0e0e0', fontsize=9)
    draw_arrow(ax, (phase3_x + 4.2, 4.9), (phase3_x + 3.1, 4.5),
               style='->', connectionstyle='arc3,rad=-0.2', color='gray')
    ax.text(phase3_x + 3.7, 4.3, 'KL\npenalty', fontsize=8, color='gray')

    # Connecting arrows between phases
    draw_arrow(ax, (phase1_x + 2.5, 2.6), (phase2_x + 0.5, 2.6),
               style='->', color='#1565c0', lw=2)
    draw_arrow(ax, (phase2_x + 2.7, 2.6), (phase3_x + 0.5, 2.6),
               style='->', color='#c62828', lw=2)

    # Final output arrow
    draw_arrow(ax, (phase3_x + 2.7, 2.6), (phase3_x + 4.5, 2.6),
               style='->', color='#2e7d32', lw=3)
    ax.text(phase3_x + 5, 2.6, '✓', fontsize=24, color='#2e7d32', va='center')

    # Legend
    legend_y = 0.5
    ax.add_patch(patches.Rectangle((0.5, legend_y), 0.4, 0.4, facecolor='#e3f2fd', edgecolor='black'))
    ax.text(1.1, legend_y + 0.2, 'Data', fontsize=9, va='center')

    ax.add_patch(patches.Rectangle((2.5, legend_y), 0.4, 0.4, facecolor='#bbdefb', edgecolor='black'))
    ax.text(3.1, legend_y + 0.2, 'Process', fontsize=9, va='center')

    ax.add_patch(patches.Rectangle((4.5, legend_y), 0.4, 0.4, facecolor='#c8e6c9', edgecolor='black'))
    ax.text(5.1, legend_y + 0.2, 'Model', fontsize=9, va='center')

    ax.add_patch(patches.Rectangle((6.5, legend_y), 0.4, 0.4, facecolor='#ffccbc', edgecolor='black'))
    ax.text(7.1, legend_y + 0.2, 'Reward', fontsize=9, va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rlhf_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'rlhf_pipeline.png'}")


def create_reward_model_diagram():
    """Create detailed reward model training diagram"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'Reward Model Training',
            ha='center', fontsize=14, fontweight='bold')

    # Prompt
    draw_box(ax, 0.5, 4.5, 2, 1.5, 'Prompt\n"Explain X"', color='#e3f2fd')

    # Two responses
    draw_box(ax, 3.5, 5.5, 2.5, 1.2, 'Response A\n(Better)', color='#c8e6c9')
    draw_box(ax, 3.5, 3, 2.5, 1.2, 'Response B\n(Worse)', color='#ffcdd2')

    # Human comparison
    draw_box(ax, 7, 4.2, 2, 1.5, 'Human\nComparison\n👤', color='#fff9c4')

    # Reward model
    draw_box(ax, 10, 4.2, 2.5, 1.5, 'Reward\nModel', color='#e1bee7')

    # Scores
    ax.text(13, 5.5, 'R(A) = 0.8', fontsize=11, color='#2e7d32', fontweight='bold')
    ax.text(13, 4, 'R(B) = 0.3', fontsize=11, color='#c62828', fontweight='bold')

    # Arrows
    draw_arrow(ax, (2.5, 5.25), (3.5, 6.1))
    draw_arrow(ax, (2.5, 5.25), (3.5, 3.6))
    draw_arrow(ax, (6, 6.1), (7, 5.5))
    draw_arrow(ax, (6, 3.6), (7, 4.4))
    draw_arrow(ax, (9, 5), (10, 5))
    draw_arrow(ax, (12.5, 5), (12.8, 5.5))
    draw_arrow(ax, (12.5, 4.7), (12.8, 4))

    # Loss function
    ax.text(7, 2, 'Loss = -log σ(R(A) - R(B))',
            ha='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='gray'))
    ax.text(7, 1.2, 'Train reward model to predict human preferences',
            ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'reward_model_training.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'reward_model_training.png'}")


def create_ppo_loop():
    """Create PPO training loop diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'PPO Training Loop',
            ha='center', fontsize=14, fontweight='bold')

    # Center positions for circular layout
    cx, cy = 6, 5
    radius = 3

    # Boxes in circular arrangement
    positions = [
        (cx - 1, cy + radius - 0.5, 'Sample\nPrompts', '#e3f2fd'),
        (cx + radius, cy + 1, 'Generate\nResponses', '#c8e6c9'),
        (cx + radius, cy - 1.5, 'Compute\nRewards', '#ffccbc'),
        (cx - 1, cy - radius, 'PPO\nUpdate', '#e1bee7'),
        (cx - radius - 0.5, cy - 0.5, 'KL\nPenalty', '#fff9c4'),
    ]

    boxes = []
    for x, y, text, color in positions:
        draw_box(ax, x, y, 2, 1.2, text, color=color, fontsize=10)
        boxes.append((x + 1, y + 0.6))

    # Arrows connecting the loop
    arrow_style = '->'
    # Prompts to Generate
    draw_arrow(ax, (boxes[0][0] + 1, boxes[0][1]), (boxes[1][0] - 1, boxes[1][1]),
               connectionstyle='arc3,rad=0.3')
    # Generate to Rewards
    draw_arrow(ax, (boxes[1][0], boxes[1][1] - 0.8), (boxes[2][0], boxes[2][1] + 0.5))
    # Rewards to PPO
    draw_arrow(ax, (boxes[2][0] - 1, boxes[2][1] - 0.5), (boxes[3][0] + 1, boxes[3][1] + 0.8),
               connectionstyle='arc3,rad=0.3')
    # PPO to KL
    draw_arrow(ax, (boxes[3][0] - 1, boxes[3][1] + 0.3), (boxes[4][0] + 1, boxes[4][1]),
               connectionstyle='arc3,rad=0.3')
    # KL to Prompts (loop back)
    draw_arrow(ax, (boxes[4][0], boxes[4][1] + 0.8), (boxes[0][0] - 1, boxes[0][1]),
               connectionstyle='arc3,rad=0.3')

    # Policy and Reference models
    draw_box(ax, 4.5, 4.5, 3, 1.2, 'Policy Model (π_θ)\nBeing trained', color='#a5d6a7')
    draw_box(ax, 0.5, 6, 2.5, 1, 'Reference (π_ref)\nFrozen', color='#e0e0e0', fontsize=9)
    draw_box(ax, 8.5, 3, 2.5, 1, 'Reward Model\nFrozen', color='#ffccbc', fontsize=9)

    # Annotation for KL
    ax.text(2, 3.5, 'KL(π_θ || π_ref)\nPrevents drift',
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='#fffde7', edgecolor='#ffc107'))

    # Iteration counter
    ax.text(6, 1, 'Repeat for N iterations until convergence',
            ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ppo_loop.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'ppo_loop.png'}")


if __name__ == "__main__":
    create_rlhf_pipeline()
    create_reward_model_diagram()
    create_ppo_loop()
    print("\nAll RLHF visualizations created successfully!")
