#!/usr/bin/env python3
"""
RLHF & LLM Alignment Visualization Generator
Generates visualizations for the RLHF alignment documentation.

Outputs:
- rlhf_pipeline.svg: Static flowchart of the RLHF pipeline
- reward_model_training.svg: Reward model training visualization
- kl_constraint_effect.gif: Animated visualization of KL divergence constraint
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from scipy.special import softmax
from scipy.stats import norm
import os

# Set the directory for output
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style configuration
COLORS = {
    'primary': '#4299e1',      # Blue
    'secondary': '#48bb78',    # Green
    'accent': '#ed8936',       # Orange
    'danger': '#f56565',       # Red
    'purple': '#9f7aea',       # Purple
    'gray': '#718096',         # Gray
    'light_gray': '#e2e8f0',   # Light gray
    'dark': '#2d3748',         # Dark gray
    'white': '#ffffff',
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def save_fig(fig, name, fmt='svg'):
    """Save figure to the assets directory."""
    path = os.path.join(OUTPUT_DIR, name)
    if fmt == 'svg':
        fig.savefig(path, format='svg', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    else:
        fig.savefig(path, dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {path}")


def save_animation(anim, name, fps=10):
    """Save animation as GIF."""
    path = os.path.join(OUTPUT_DIR, name)
    anim.save(path, writer='pillow', fps=fps)
    print(f"Saved: {path}")


# =============================================================================
# 1. RLHF Pipeline Flowchart (Static SVG)
# =============================================================================
def create_rlhf_pipeline():
    """Create a detailed RLHF pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 11)
    ax.axis('off')

    # Title
    ax.text(8, 10.5, 'RLHF Pipeline: From Pretrained LLM to Aligned Model',
            ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['dark'])

    # Stage 1: SFT
    stage1_box = FancyBboxPatch((0, 6), 4.5, 3.5, boxstyle='round,pad=0.1',
                                 facecolor='#ebf8ff', edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(2.25, 9, 'Stage 1: Supervised Fine-Tuning', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLORS['primary'])

    # SFT components
    ax.add_patch(FancyBboxPatch((0.3, 7.8), 1.6, 0.8, boxstyle='round,pad=0.05',
                                facecolor=COLORS['primary'], edgecolor='none'))
    ax.text(1.1, 8.2, 'Pretrained\nLLM', ha='center', va='center', fontsize=8, color='white')

    ax.add_patch(FancyBboxPatch((2.6, 7.8), 1.6, 0.8, boxstyle='round,pad=0.05',
                                facecolor=COLORS['secondary'], edgecolor='none'))
    ax.text(3.4, 8.2, 'Human\nDemos', ha='center', va='center', fontsize=8, color='white')

    # Arrow down
    ax.annotate('', xy=(2.25, 7), xytext=(2.25, 7.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

    ax.add_patch(FancyBboxPatch((0.8, 6.2), 2.9, 0.7, boxstyle='round,pad=0.05',
                                facecolor=COLORS['accent'], edgecolor='none'))
    ax.text(2.25, 6.55, 'SFT Model (pi_sft)', ha='center', va='center', fontsize=9,
            color='white', fontweight='bold')

    # Stage 2: Reward Model
    stage2_box = FancyBboxPatch((5.75, 6), 4.5, 3.5, boxstyle='round,pad=0.1',
                                 facecolor='#f0fff4', edgecolor=COLORS['secondary'], linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(8, 9, 'Stage 2: Reward Model', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLORS['secondary'])

    # RM components
    ax.add_patch(FancyBboxPatch((6.05, 7.8), 1.6, 0.8, boxstyle='round,pad=0.05',
                                facecolor=COLORS['primary'], edgecolor='none'))
    ax.text(6.85, 8.2, 'SFT\nResponses', ha='center', va='center', fontsize=8, color='white')

    ax.add_patch(FancyBboxPatch((8.35, 7.8), 1.6, 0.8, boxstyle='round,pad=0.05',
                                facecolor=COLORS['purple'], edgecolor='none'))
    ax.text(9.15, 8.2, 'Human\nRankings', ha='center', va='center', fontsize=8, color='white')

    ax.annotate('', xy=(8, 7), xytext=(8, 7.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

    ax.add_patch(FancyBboxPatch((6.55, 6.2), 2.9, 0.7, boxstyle='round,pad=0.05',
                                facecolor=COLORS['secondary'], edgecolor='none'))
    ax.text(8, 6.55, 'Reward Model (r_phi)', ha='center', va='center', fontsize=9,
            color='white', fontweight='bold')

    # Stage 3: PPO
    stage3_box = FancyBboxPatch((11.5, 6), 4.5, 3.5, boxstyle='round,pad=0.1',
                                 facecolor='#fffaf0', edgecolor=COLORS['accent'], linewidth=2)
    ax.add_patch(stage3_box)
    ax.text(13.75, 9, 'Stage 3: PPO Fine-Tuning', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLORS['accent'])

    # PPO components
    ax.add_patch(FancyBboxPatch((11.8, 7.8), 1.6, 0.8, boxstyle='round,pad=0.05',
                                facecolor=COLORS['accent'], edgecolor='none'))
    ax.text(12.6, 8.2, 'SFT Model\n(init)', ha='center', va='center', fontsize=8, color='white')

    ax.add_patch(FancyBboxPatch((14.1, 7.8), 1.6, 0.8, boxstyle='round,pad=0.05',
                                facecolor=COLORS['secondary'], edgecolor='none'))
    ax.text(14.9, 8.2, 'Reward\nModel', ha='center', va='center', fontsize=8, color='white')

    ax.annotate('', xy=(13.75, 7), xytext=(13.75, 7.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

    ax.add_patch(FancyBboxPatch((12.3, 6.2), 2.9, 0.7, boxstyle='round,pad=0.05',
                                facecolor=COLORS['danger'], edgecolor='none'))
    ax.text(13.75, 6.55, 'Aligned Model', ha='center', va='center', fontsize=9,
            color='white', fontweight='bold')

    # Arrows between stages
    ax.annotate('', xy=(5.5, 7.5), xytext=(4.7, 7.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2.5))
    ax.annotate('', xy=(11.25, 7.5), xytext=(10.45, 7.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2.5))

    # DPO Alternative box
    dpo_box = FancyBboxPatch((5.75, 1), 4.5, 3.5, boxstyle='round,pad=0.1',
                              facecolor='#faf5ff', edgecolor=COLORS['purple'], linewidth=2)
    ax.add_patch(dpo_box)
    ax.text(8, 4, 'Alternative: DPO', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLORS['purple'])

    ax.add_patch(FancyBboxPatch((6.05, 2.8), 1.6, 0.8, boxstyle='round,pad=0.05',
                                facecolor=COLORS['accent'], edgecolor='none'))
    ax.text(6.85, 3.2, 'SFT\nModel', ha='center', va='center', fontsize=8, color='white')

    ax.add_patch(FancyBboxPatch((8.35, 2.8), 1.6, 0.8, boxstyle='round,pad=0.05',
                                facecolor=COLORS['purple'], edgecolor='none'))
    ax.text(9.15, 3.2, 'Preference\nData', ha='center', va='center', fontsize=8, color='white')

    ax.annotate('', xy=(8, 2), xytext=(8, 2.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

    ax.add_patch(FancyBboxPatch((6.55, 1.2), 2.9, 0.7, boxstyle='round,pad=0.05',
                                facecolor=COLORS['purple'], edgecolor='none'))
    ax.text(8, 1.55, 'Aligned Model (DPO)', ha='center', va='center', fontsize=9,
            color='white', fontweight='bold')

    # "OR" indicator
    ax.text(8, 5.3, 'OR', ha='center', va='center', fontsize=12, fontweight='bold',
            color=COLORS['gray'], bbox=dict(boxstyle='circle,pad=0.3',
                                             facecolor=COLORS['light_gray'], edgecolor=COLORS['gray']))

    # Annotations
    ax.text(2.25, 5.5, 'Cross-entropy loss\non demonstrations', ha='center', va='center',
            fontsize=8, style='italic', color=COLORS['gray'])
    ax.text(8, 5.5, 'Bradley-Terry loss\non comparisons', ha='center', va='center',
            fontsize=8, style='italic', color=COLORS['gray'])
    ax.text(13.75, 5.5, 'Reward - beta * KL\nobjective', ha='center', va='center',
            fontsize=8, style='italic', color=COLORS['gray'])
    ax.text(8, 0.5, 'Direct optimization\nwithout reward model', ha='center', va='center',
            fontsize=8, style='italic', color=COLORS['gray'])

    plt.tight_layout()
    save_fig(fig, 'rlhf_pipeline.svg')


# =============================================================================
# 2. Reward Model Training Visualization (Static SVG)
# =============================================================================
def create_reward_model_training():
    """Visualize reward model training with Bradley-Terry model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Pairwise Comparison
    ax1 = axes[0]
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-0.5, 4)
    ax1.axis('off')
    ax1.set_title('Step 1: Human Comparison', fontsize=12, fontweight='bold',
                  color=COLORS['primary'], pad=10)

    # Prompt
    ax1.add_patch(FancyBboxPatch((0.25, 3.2), 2.5, 0.6, boxstyle='round,pad=0.05',
                                  facecolor=COLORS['light_gray'], edgecolor=COLORS['gray']))
    ax1.text(1.5, 3.5, 'Prompt: "Explain gravity"', ha='center', va='center', fontsize=9)

    # Response A (preferred)
    ax1.add_patch(FancyBboxPatch((0, 1.5), 1.4, 1.4, boxstyle='round,pad=0.05',
                                  facecolor='#c6f6d5', edgecolor=COLORS['secondary'], linewidth=2))
    ax1.text(0.7, 2.5, 'Response A', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(0.7, 2.1, '"Gravity is the\nforce that..."', ha='center', va='center', fontsize=7)
    ax1.text(0.7, 1.65, 'PREFERRED', ha='center', va='center', fontsize=8,
             fontweight='bold', color=COLORS['secondary'])

    # Response B (rejected)
    ax1.add_patch(FancyBboxPatch((1.6, 1.5), 1.4, 1.4, boxstyle='round,pad=0.05',
                                  facecolor='#fed7d7', edgecolor=COLORS['danger'], linewidth=2))
    ax1.text(2.3, 2.5, 'Response B', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(2.3, 2.1, '"IDK its like\nstuff falls..."', ha='center', va='center', fontsize=7)
    ax1.text(2.3, 1.65, 'REJECTED', ha='center', va='center', fontsize=8,
             fontweight='bold', color=COLORS['danger'])

    # Human annotator
    ax1.add_patch(Circle((1.5, 0.5), 0.35, facecolor=COLORS['purple'], edgecolor='none'))
    ax1.text(1.5, 0.5, 'H', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax1.text(1.5, -0.1, 'Human Annotator', ha='center', va='center', fontsize=8, color=COLORS['gray'])

    # Panel 2: Bradley-Terry Model
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 4)
    ax2.set_ylim(-0.5, 4)
    ax2.axis('off')
    ax2.set_title('Step 2: Bradley-Terry Scoring', fontsize=12, fontweight='bold',
                  color=COLORS['secondary'], pad=10)

    # Reward model box
    ax2.add_patch(FancyBboxPatch((0.5, 2), 2.5, 1.5, boxstyle='round,pad=0.1',
                                  facecolor=COLORS['secondary'], edgecolor='none'))
    ax2.text(1.75, 3.1, 'Reward Model', ha='center', va='center', fontsize=11,
             fontweight='bold', color='white')
    ax2.text(1.75, 2.6, 'r(prompt, response)', ha='center', va='center', fontsize=9, color='white')

    # Inputs
    ax2.annotate('', xy=(0.5, 2.5), xytext=(0, 2.5),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
    ax2.text(-0.1, 2.8, 'y_w', ha='right', va='center', fontsize=10, color=COLORS['secondary'])
    ax2.text(-0.1, 2.2, 'y_l', ha='right', va='center', fontsize=10, color=COLORS['danger'])

    # Outputs
    ax2.annotate('', xy=(3.5, 2.8), xytext=(3, 2.8),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
    ax2.annotate('', xy=(3.5, 2.2), xytext=(3, 2.2),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
    ax2.text(3.6, 2.8, 'r_w = 2.3', ha='left', va='center', fontsize=10, color=COLORS['secondary'])
    ax2.text(3.6, 2.2, 'r_l = 0.8', ha='left', va='center', fontsize=10, color=COLORS['danger'])

    # Formula
    ax2.text(1.75, 1.2, r'$P(y_w \succ y_l) = \sigma(r_w - r_l)$', ha='center', va='center',
             fontsize=11, style='italic')
    ax2.text(1.75, 0.7, r'$= \sigma(2.3 - 0.8) = 0.82$', ha='center', va='center', fontsize=10)
    ax2.text(1.75, 0.2, '82% confident A is better', ha='center', va='center',
             fontsize=9, color=COLORS['gray'])

    # Panel 3: Loss Computation
    ax3 = axes[2]
    ax3.set_xlim(-0.5, 4)
    ax3.set_ylim(-0.5, 4)
    ax3.axis('off')
    ax3.set_title('Step 3: Training Loss', fontsize=12, fontweight='bold',
                  color=COLORS['accent'], pad=10)

    # Loss formula box
    ax3.add_patch(FancyBboxPatch((0.25, 2.2), 3, 1.3, boxstyle='round,pad=0.1',
                                  facecolor=COLORS['light_gray'], edgecolor=COLORS['accent'], linewidth=2))
    ax3.text(1.75, 3.1, 'Reward Model Loss', ha='center', va='center', fontsize=10, fontweight='bold')
    ax3.text(1.75, 2.6, r'$\mathcal{L} = -\log\sigma(r_w - r_l)$', ha='center', va='center', fontsize=11)

    # Gradient descent visualization
    ax3.add_patch(FancyBboxPatch((0.5, 0.5), 2.5, 1.2, boxstyle='round,pad=0.05',
                                  facecolor='#fff5f5', edgecolor=COLORS['danger']))
    ax3.text(1.75, 1.4, 'Backprop:', ha='center', va='center', fontsize=9, fontweight='bold')
    ax3.text(1.75, 1.0, r'$\uparrow r_w$  and  $\downarrow r_l$', ha='center', va='center', fontsize=11)
    ax3.text(1.75, 0.65, 'Push rewards apart', ha='center', va='center', fontsize=8, color=COLORS['gray'])

    plt.tight_layout()
    save_fig(fig, 'reward_model_training.svg')


# =============================================================================
# 3. KL Divergence Constraint Effect Animation (GIF)
# =============================================================================
def create_kl_constraint_animation():
    """Create animated visualization of KL constraint effect during PPO training."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Parameters
    n_frames = 60
    x = np.linspace(-4, 6, 500)

    # Reference distribution (SFT policy)
    ref_mean = 0
    ref_std = 1
    ref_dist = norm.pdf(x, ref_mean, ref_std)

    def animate(frame):
        for ax in axes:
            ax.clear()

        progress = frame / n_frames

        # Panel 1: Without KL constraint (reward hacking)
        ax1 = axes[0]
        # Policy drifts dramatically toward high reward region
        no_kl_mean = ref_mean + 5 * progress  # Drifts far
        no_kl_std = ref_std * (1 - 0.7 * progress)  # Collapses
        no_kl_dist = norm.pdf(x, no_kl_mean, max(no_kl_std, 0.2))

        ax1.fill_between(x, ref_dist, alpha=0.3, color=COLORS['primary'], label='Reference (SFT)')
        ax1.fill_between(x, no_kl_dist, alpha=0.5, color=COLORS['danger'], label='Policy (no KL)')
        ax1.axvline(x=4, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Reward peak')
        ax1.set_xlim(-4, 6)
        ax1.set_ylim(0, 1.5)
        ax1.set_xlabel('Response Space')
        ax1.set_ylabel('Probability Density')
        ax1.set_title(f'Without KL Constraint\n(Reward Hacking)', fontsize=11, fontweight='bold', color=COLORS['danger'])
        ax1.legend(loc='upper right', fontsize=7)

        # Add KL divergence value
        kl_no_constraint = 0.5 * ((no_kl_mean - ref_mean)**2 / ref_std**2 +
                                   (no_kl_std / ref_std)**2 - 1 -
                                   2 * np.log(max(no_kl_std / ref_std, 0.01)))
        ax1.text(0.05, 0.95, f'KL = {kl_no_constraint:.2f}', transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=COLORS['danger'], alpha=0.3))

        # Panel 2: With KL constraint (proper learning)
        ax2 = axes[1]
        # Policy moves toward reward but stays close to reference
        kl_mean = ref_mean + 1.5 * progress  # Moderate drift
        kl_std = ref_std * (1 - 0.2 * progress)  # Slight tightening
        kl_dist = norm.pdf(x, kl_mean, max(kl_std, 0.5))

        ax2.fill_between(x, ref_dist, alpha=0.3, color=COLORS['primary'], label='Reference (SFT)')
        ax2.fill_between(x, kl_dist, alpha=0.5, color=COLORS['secondary'], label='Policy (with KL)')
        ax2.axvline(x=4, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Reward peak')
        ax2.set_xlim(-4, 6)
        ax2.set_ylim(0, 1.5)
        ax2.set_xlabel('Response Space')
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'With KL Constraint (beta=0.1)\n(Balanced Learning)', fontsize=11, fontweight='bold', color=COLORS['secondary'])
        ax2.legend(loc='upper right', fontsize=7)

        # KL divergence value
        kl_with_constraint = 0.5 * ((kl_mean - ref_mean)**2 / ref_std**2 +
                                     (kl_std / ref_std)**2 - 1 -
                                     2 * np.log(max(kl_std / ref_std, 0.01)))
        ax2.text(0.05, 0.95, f'KL = {kl_with_constraint:.2f}', transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=COLORS['secondary'], alpha=0.3))

        # Panel 3: Objective comparison
        ax3 = axes[2]
        steps = np.arange(0, frame + 1)

        # Reward curves
        reward_no_kl = 3 * (1 - np.exp(-steps / 15))  # Quickly maximizes
        reward_with_kl = 2.5 * (1 - np.exp(-steps / 25))  # Slower but steady

        # Quality curves (what we actually want)
        quality_no_kl = reward_no_kl * np.exp(-0.05 * steps)  # Degrades due to hacking
        quality_with_kl = reward_with_kl  # Stays aligned

        ax3.plot(steps, reward_no_kl, '--', color=COLORS['danger'], label='Reward (no KL)', linewidth=2)
        ax3.plot(steps, quality_no_kl, '-', color=COLORS['danger'], label='Quality (no KL)', linewidth=2, alpha=0.7)
        ax3.plot(steps, reward_with_kl, '--', color=COLORS['secondary'], label='Reward (KL)', linewidth=2)
        ax3.plot(steps, quality_with_kl, '-', color=COLORS['secondary'], label='Quality (KL)', linewidth=2, alpha=0.7)

        ax3.set_xlim(0, n_frames)
        ax3.set_ylim(0, 4)
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Score')
        ax3.set_title('Reward vs Actual Quality', fontsize=11, fontweight='bold', color=COLORS['dark'])
        ax3.legend(loc='upper left', fontsize=7)
        ax3.axhline(y=0, color='gray', linewidth=0.5)
        ax3.grid(True, alpha=0.3)

        # Progress indicator
        fig.suptitle(f'KL Divergence Constraint Effect (Step {frame}/{n_frames})',
                    fontsize=13, fontweight='bold', y=1.02)

        plt.tight_layout()
        return axes

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=False)
    save_animation(anim, 'kl_constraint_effect.gif', fps=10)
    plt.close(fig)


# =============================================================================
# Main: Generate all visualizations
# =============================================================================
if __name__ == '__main__':
    print("Generating RLHF visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Create static visualizations
    print("Creating RLHF pipeline flowchart...")
    create_rlhf_pipeline()

    print("Creating reward model training visualization...")
    create_reward_model_training()

    print("Creating KL constraint effect animation...")
    create_kl_constraint_animation()

    print()
    print("All RLHF visualizations generated successfully!")
