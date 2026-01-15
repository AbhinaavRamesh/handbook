"""
DPO vs PPO Comparison Visualization

This script creates visualizations comparing DPO and PPO/RLHF approaches,
including loss landscapes and training dynamics.

Output: dpo_vs_ppo.png, dpo_loss_curve.png, preference_optimization.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "fine-tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def draw_box(ax, x, y, width, height, text, color='lightblue',
             fontsize=10, alpha=0.8):
    """Draw a rounded box with text"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        alpha=alpha,
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', wrap=True)


def draw_arrow(ax, start, end, color='black', style='->', lw=2):
    """Draw an arrow"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=lw,
        mutation_scale=15
    )
    ax.add_patch(arrow)


def create_dpo_vs_ppo_pipeline():
    """Create side-by-side comparison of DPO and PPO pipelines"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # PPO Pipeline (left)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('RLHF/PPO Pipeline', fontsize=14, fontweight='bold', color='#c62828')

    # PPO boxes
    draw_box(ax, 1, 8, 2.5, 1.2, 'Preferences', color='#e3f2fd')
    draw_box(ax, 5.5, 8, 2.5, 1.2, 'Train Reward\nModel', color='#ffccbc')
    draw_box(ax, 1, 5.5, 2.5, 1.2, 'SFT Model', color='#c8e6c9')
    draw_box(ax, 5.5, 5.5, 2.5, 1.2, 'Reward\nModel', color='#ffccbc')
    draw_box(ax, 3.25, 3, 2.5, 1.2, 'PPO\nTraining', color='#e1bee7')
    draw_box(ax, 3.25, 0.5, 2.5, 1.2, 'Aligned\nModel', color='#a5d6a7')

    # PPO arrows
    draw_arrow(ax, (3.5, 8.6), (5.5, 8.6))
    draw_arrow(ax, (6.75, 8), (6.75, 6.7))
    draw_arrow(ax, (2.25, 5.5), (3.5, 4.2))
    draw_arrow(ax, (6.75, 5.5), (5.5, 4.2))
    draw_arrow(ax, (4.5, 3), (4.5, 1.7))

    # Complexity indicators
    ax.text(5, 0.8, 'Complexity: HIGH', fontsize=11, color='#c62828',
            ha='center', fontweight='bold')
    ax.text(5, 0.3, '• 3 training phases\n• 4 models in memory\n• RL instabilities',
            fontsize=9, ha='center', color='gray')

    # DPO Pipeline (right)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('DPO Pipeline', fontsize=14, fontweight='bold', color='#2e7d32')

    # DPO boxes
    draw_box(ax, 3.25, 8, 2.5, 1.2, 'Preferences', color='#e3f2fd')
    draw_box(ax, 1, 5.5, 2.5, 1.2, 'Policy\nModel', color='#c8e6c9')
    draw_box(ax, 5.5, 5.5, 2.5, 1.2, 'Reference\n(frozen)', color='#e0e0e0')
    draw_box(ax, 3.25, 3, 2.5, 1.2, 'DPO\nTraining', color='#c5e1a5')
    draw_box(ax, 3.25, 0.5, 2.5, 1.2, 'Aligned\nModel', color='#a5d6a7')

    # DPO arrows
    draw_arrow(ax, (4.5, 8), (4.5, 6.7))
    draw_arrow(ax, (2.25, 5.5), (3.5, 4.2))
    draw_arrow(ax, (6.75, 5.5), (5.5, 4.2), color='gray')
    draw_arrow(ax, (4.5, 3), (4.5, 1.7))

    # Complexity indicators
    ax.text(5, 0.8, 'Complexity: LOW', fontsize=11, color='#2e7d32',
            ha='center', fontweight='bold')
    ax.text(5, 0.3, '• 1 training phase\n• 2 models in memory\n• Stable optimization',
            fontsize=9, ha='center', color='gray')

    # Add "NO REWARD MODEL" crossed out
    ax.text(7.5, 8.5, 'No Reward\nModel!', fontsize=10, color='#2e7d32',
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='#4caf50'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dpo_vs_ppo.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dpo_vs_ppo.png'}")


def create_dpo_loss_visualization():
    """Create visualization of DPO loss function behavior"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: DPO loss curve
    ax = axes[0]
    x = np.linspace(-4, 4, 100)
    # DPO loss: -log(sigmoid(beta * reward_margin))
    beta = 0.1
    reward_margin = x  # r_chosen - r_rejected

    # Loss for different beta values
    for beta, color, label in [(0.05, '#4caf50', 'β=0.05 (loose)'),
                                (0.1, '#1976d2', 'β=0.1 (default)'),
                                (0.5, '#f44336', 'β=0.5 (strict)')]:
        loss = -np.log(1 / (1 + np.exp(-beta * reward_margin * 10)))
        ax.plot(x, loss, linewidth=2.5, color=color, label=label)

    ax.set_xlabel('Implicit Reward Margin (chosen - rejected)', fontsize=11)
    ax.set_ylabel('DPO Loss', fontsize=11)
    ax.set_title('DPO Loss Function', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.693, color='gray', linestyle=':', alpha=0.5)  # log(2)

    # Annotations
    ax.annotate('Low loss:\nChosen >> Rejected',
               xy=(3, 0.1), fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))
    ax.annotate('High loss:\nRejected > Chosen',
               xy=(-3, 2.5), fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.8))

    # Right: Training dynamics comparison
    ax = axes[1]
    steps = 500
    x_steps = np.arange(steps)

    # Simulated training curves
    np.random.seed(42)
    ppo_loss = 2.5 * np.exp(-0.005 * x_steps) + 0.5 + np.random.normal(0, 0.15, steps)
    dpo_loss = 2.5 * np.exp(-0.006 * x_steps) + 0.4 + np.random.normal(0, 0.05, steps)

    ax.plot(x_steps, ppo_loss, linewidth=2, color='#f44336', label='PPO (more variance)', alpha=0.8)
    ax.plot(x_steps, dpo_loss, linewidth=2, color='#4caf50', label='DPO (stable)')

    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Stability Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Add annotations
    ax.annotate('PPO can be\nunstable',
               xy=(200, 2.8), fontsize=9, ha='center', color='#c62828')
    ax.annotate('DPO is\nmore stable',
               xy=(350, 0.5), fontsize=9, ha='center', color='#2e7d32')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dpo_loss_curve.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dpo_loss_curve.png'}")


def create_preference_optimization_diagram():
    """Create diagram showing how DPO optimizes preferences"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'DPO: Direct Preference Optimization',
            ha='center', fontsize=14, fontweight='bold')

    # Input section
    draw_box(ax, 0.5, 4.5, 2, 1.5, 'Prompt\n"How to..."', color='#e3f2fd')

    # Chosen response
    draw_box(ax, 3.5, 5.5, 3, 1.3, 'Chosen Response\n(Preferred)', color='#c8e6c9')
    ax.text(5, 5, 'log π_θ(y_w|x)', fontsize=9, style='italic')

    # Rejected response
    draw_box(ax, 3.5, 2.5, 3, 1.3, 'Rejected Response\n(Less Preferred)', color='#ffcdd2')
    ax.text(5, 2, 'log π_θ(y_l|x)', fontsize=9, style='italic')

    # Reference probabilities
    draw_box(ax, 7.5, 5.5, 2.5, 1.3, 'Reference\nπ_ref(y_w|x)', color='#e0e0e0')
    draw_box(ax, 7.5, 2.5, 2.5, 1.3, 'Reference\nπ_ref(y_l|x)', color='#e0e0e0')

    # Computation
    draw_box(ax, 10.5, 4, 2.5, 2, 'Compute\nMargin', color='#fff9c4')

    # Loss
    ax.add_patch(patches.Circle((13, 5), 0.4, facecolor='#e1bee7', edgecolor='black', lw=2))
    ax.text(13, 5, 'L', fontsize=12, ha='center', va='center', fontweight='bold')

    # Arrows
    draw_arrow(ax, (2.5, 5.25), (3.5, 6.1))
    draw_arrow(ax, (2.5, 5.25), (3.5, 3.1))
    draw_arrow(ax, (6.5, 6.1), (7.5, 6.1))
    draw_arrow(ax, (6.5, 3.1), (7.5, 3.1))
    draw_arrow(ax, (10, 6.1), (10.5, 5.3))
    draw_arrow(ax, (10, 3.1), (10.5, 4.7))
    draw_arrow(ax, (13, 4.6), (13, 3.5))

    # Formula
    formula_box = FancyBboxPatch(
        (1, 0.3), 12, 1.5,
        boxstyle="round,pad=0.02",
        facecolor='#f5f5f5',
        edgecolor='gray',
        linewidth=1
    )
    ax.add_patch(formula_box)

    ax.text(7, 1.3, r'$\mathcal{L}_{DPO} = -\log\sigma\left(\beta \cdot \left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)$',
            fontsize=11, ha='center', va='center')
    ax.text(7, 0.6, 'Increase probability of preferred response while staying close to reference',
            fontsize=10, ha='center', va='center', style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'preference_optimization.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'preference_optimization.png'}")


def create_method_comparison_table():
    """Create a visual comparison table of alignment methods"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Table data
    headers = ['Aspect', 'PPO/RLHF', 'DPO', 'Winner']
    rows = [
        ['Training Phases', '3 (SFT→RM→RL)', '1 (Direct)', 'DPO'],
        ['Models in Memory', '4 (Policy, Ref, RM, Value)', '2 (Policy, Ref)', 'DPO'],
        ['Stability', 'Can be unstable', 'Very stable', 'DPO'],
        ['Hyperparameters', 'Many (KL coef, PPO params)', 'Few (mainly β)', 'DPO'],
        ['Online Learning', 'Yes', 'No (offline only)', 'PPO'],
        ['Performance at Scale', 'Slightly better', 'Comparable', 'Tie'],
        ['Implementation', 'Complex', 'Simple', 'DPO'],
        ['Compute Cost', 'High', 'Medium', 'DPO'],
    ]

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.2, 0.3, 0.3, 0.1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style headers
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#1976d2')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style winner column
    for i, row in enumerate(rows, 1):
        winner = row[-1]
        if winner == 'DPO':
            table[(i, 3)].set_facecolor('#c8e6c9')
        elif winner == 'PPO':
            table[(i, 3)].set_facecolor('#ffccbc')
        else:
            table[(i, 3)].set_facecolor('#fff9c4')

    ax.set_title('DPO vs PPO/RLHF Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dpo_ppo_comparison_table.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dpo_ppo_comparison_table.png'}")


if __name__ == "__main__":
    create_dpo_vs_ppo_pipeline()
    create_dpo_loss_visualization()
    create_preference_optimization_diagram()
    create_method_comparison_table()
    print("\nAll DPO visualizations created successfully!")
