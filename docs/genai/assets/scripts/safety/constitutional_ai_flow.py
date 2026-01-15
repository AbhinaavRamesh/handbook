"""
Constitutional AI Flow Visualization

This script creates a visualization of the Constitutional AI training
methodology showing the SL and RL phases.

Output: ../images/safety/constitutional_ai_flow.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def create_constitutional_ai_flow():
    """Create Constitutional AI training flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Color palette
    colors = {
        'constitution': '#E8F5E9',  # Light green
        'sl_phase': '#E3F2FD',      # Light blue
        'rl_phase': '#FFF3E0',      # Light orange
        'model': '#F3E5F5',         # Light purple
        'critique': '#FFCDD2',      # Light red
        'revision': '#C8E6C9',      # Light green
        'arrow': '#455A64',         # Dark gray
        'text': '#212121'           # Near black
    }

    # Title
    ax.text(8, 13.5, 'Constitutional AI Training Flow',
            fontsize=18, fontweight='bold', ha='center', color=colors['text'])

    # === CONSTITUTION SECTION ===
    const_box = FancyBboxPatch((0.5, 11), 4, 2,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['constitution'],
                                edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(const_box)
    ax.text(2.5, 12.7, 'THE CONSTITUTION', fontsize=11, fontweight='bold',
            ha='center', color='#2E7D32')

    principles = ['Harmlessness', 'Helpfulness', 'Honesty', 'Fairness']
    for i, p in enumerate(principles):
        ax.text(2.5, 12.2 - i*0.35, f'• {p}', fontsize=9, ha='center')

    # Arrow from constitution to both phases
    ax.annotate('', xy=(7, 12), xytext=(4.5, 12),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.text(5.75, 12.2, 'Guides', fontsize=9, ha='center', style='italic')

    # === RLHF COMPARISON (Side box) ===
    rlhf_box = FancyBboxPatch((11.5, 11), 4, 2,
                               boxstyle="round,pad=0.05",
                               facecolor='#FFECB3',
                               edgecolor='#F57F17', linewidth=2)
    ax.add_patch(rlhf_box)
    ax.text(13.5, 12.7, 'vs RLHF', fontsize=11, fontweight='bold',
            ha='center', color='#F57F17')
    ax.text(13.5, 12.2, 'Human feedback (expensive)', fontsize=8, ha='center')
    ax.text(13.5, 11.85, 'Inconsistent preferences', fontsize=8, ha='center')
    ax.text(13.5, 11.5, 'Sycophancy issues', fontsize=8, ha='center')
    ax.text(13.5, 11.15, 'Doesn\'t scale', fontsize=8, ha='center')

    # === SL PHASE ===
    sl_box = FancyBboxPatch((0.5, 5.5), 7, 5,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['sl_phase'],
                             edgecolor='#1565C0', linewidth=2)
    ax.add_patch(sl_box)
    ax.text(4, 10.2, 'PHASE 1: Supervised Learning (SL)', fontsize=12,
            fontweight='bold', ha='center', color='#1565C0')

    # SL steps
    sl_steps = [
        (1.5, 8.8, 'Generate', 'Initial response\nfrom model', '#F3E5F5'),
        (3.5, 8.8, 'Critique', 'Apply constitutional\nprinciples', '#FFCDD2'),
        (5.5, 8.8, 'Revise', 'Improve based\non critique', '#C8E6C9')
    ]

    for x, y, title, desc, color in sl_steps:
        box = FancyBboxPatch((x, y), 1.8, 1.5,
                              boxstyle="round,pad=0.03",
                              facecolor=color,
                              edgecolor='gray', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.9, y + 1.25, title, fontsize=9, fontweight='bold', ha='center')
        ax.text(x + 0.9, y + 0.5, desc, fontsize=8, ha='center')

    # Arrows between SL steps
    ax.annotate('', xy=(3.5, 9.55), xytext=(3.3, 9.55),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(5.5, 9.55), xytext=(5.3, 9.55),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # SL output
    sl_output = FancyBboxPatch((2.5, 6), 3, 1,
                                boxstyle="round,pad=0.03",
                                facecolor='#BBDEFB',
                                edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(sl_output)
    ax.text(4, 6.5, 'Train on (input, revised)\npairs', fontsize=9, ha='center')

    # Arrow from steps to output
    ax.annotate('', xy=(4, 7), xytext=(4, 8.8),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # === RL PHASE ===
    rl_box = FancyBboxPatch((8.5, 5.5), 7, 5,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['rl_phase'],
                             edgecolor='#E65100', linewidth=2)
    ax.add_patch(rl_box)
    ax.text(12, 10.2, 'PHASE 2: Reinforcement Learning (RLAIF)', fontsize=12,
            fontweight='bold', ha='center', color='#E65100')

    # RL steps
    rl_steps = [
        (9, 8.8, 'Generate\nPairs', 'Two responses\nper prompt', '#F3E5F5'),
        (11, 8.8, 'AI Labels', 'Constitutional\npreference', '#FFF9C4'),
        (13, 8.8, 'Train RM', 'Reward model\nfrom AI prefs', '#E8F5E9')
    ]

    for x, y, title, desc, color in rl_steps:
        box = FancyBboxPatch((x, y), 1.8, 1.5,
                              boxstyle="round,pad=0.03",
                              facecolor=color,
                              edgecolor='gray', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.9, y + 1.25, title, fontsize=9, fontweight='bold', ha='center')
        ax.text(x + 0.9, y + 0.5, desc, fontsize=8, ha='center')

    # Arrows between RL steps
    ax.annotate('', xy=(11, 9.55), xytext=(10.8, 9.55),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(13, 9.55), xytext=(12.8, 9.55),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # RL output
    rl_output = FancyBboxPatch((10.5, 6), 3, 1,
                                boxstyle="round,pad=0.03",
                                facecolor='#FFE0B2',
                                edgecolor='#E65100', linewidth=1.5)
    ax.add_patch(rl_output)
    ax.text(12, 6.5, 'PPO with AI\nfeedback', fontsize=9, ha='center')

    # Arrow from steps to output
    ax.annotate('', xy=(12, 7), xytext=(12, 8.8),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Arrow from SL to RL
    ax.annotate('', xy=(8.5, 8), xytext=(7.5, 8),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.text(8, 8.3, 'Phase 1 model', fontsize=8, ha='center', style='italic')

    # === FINAL MODEL ===
    final_box = FancyBboxPatch((6, 2), 4, 2.5,
                                boxstyle="round,pad=0.05",
                                facecolor='#D1C4E9',
                                edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(final_box)
    ax.text(8, 4.2, 'ALIGNED MODEL', fontsize=12, fontweight='bold',
            ha='center', color='#7B1FA2')
    ax.text(8, 3.5, 'Helpful, Harmless, Honest', fontsize=10, ha='center')
    ax.text(8, 2.9, 'Trained on AI feedback', fontsize=9, ha='center', color='gray')
    ax.text(8, 2.4, 'Scalable alignment', fontsize=9, ha='center', color='gray')

    # Arrows from both phases to final
    ax.annotate('', xy=(6.5, 4.5), xytext=(4, 6),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.annotate('', xy=(9.5, 4.5), xytext=(12, 6),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # === BENEFITS BOX ===
    benefits_box = FancyBboxPatch((0.5, 0.5), 15, 1.2,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#E8F5E9',
                                   edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(benefits_box)
    ax.text(8, 1.35, 'KEY BENEFITS', fontsize=11, fontweight='bold',
            ha='center', color='#2E7D32')
    ax.text(8, 0.85, 'Scales with compute  |  Transparent principles  |  Reduced sycophancy  |  Consistent alignment  |  Cheaper than RLHF',
            fontsize=10, ha='center')

    # Constitution principle detail (top right)
    principle_box = FancyBboxPatch((5, 11), 6, 2,
                                    boxstyle="round,pad=0.03",
                                    facecolor='white',
                                    edgecolor='gray', linewidth=1)
    ax.add_patch(principle_box)
    ax.text(8, 12.7, 'Example Principle', fontsize=10, fontweight='bold', ha='center')
    ax.text(8, 12.2, '"Is this response harmful? If yes,', fontsize=9, ha='center', style='italic')
    ax.text(8, 11.8, 'revise to refuse while being helpful"', fontsize=9, ha='center', style='italic')
    ax.text(8, 11.3, 'Critique Prompt + Revision Instruction', fontsize=8, ha='center', color='gray')

    plt.tight_layout()
    return fig


def main():
    """Generate and save the visualization."""
    fig = create_constitutional_ai_flow()

    # Save to output directory
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/safety/constitutional_ai_flow.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
