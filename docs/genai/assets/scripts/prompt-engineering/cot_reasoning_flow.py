"""
Chain-of-Thought Reasoning Flow Visualization

This script creates visualizations demonstrating how CoT prompting works,
comparing direct answering vs step-by-step reasoning, and showing the
self-consistency aggregation process.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def create_cot_flow_diagram():
    """Create a comprehensive CoT visualization."""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # ============== Panel 1: Direct vs CoT Comparison ==============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Direct Answering vs Chain-of-Thought', fontsize=14, fontweight='bold', pad=20)

    # Direct path (top)
    ax1.text(0.5, 9, 'Direct:', fontsize=12, fontweight='bold', color='#c53030')
    direct_box = FancyBboxPatch((0.3, 7.5), 3, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#4299e1', edgecolor='#2b6cb0', linewidth=2)
    ax1.add_patch(direct_box)
    ax1.text(1.8, 8.1, 'Question', fontsize=10, color='white', ha='center', va='center')

    ax1.annotate('', xy=(4.5, 8.1), xytext=(3.5, 8.1),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))

    # Black box
    black_box = FancyBboxPatch((4.5, 7.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                               facecolor='#2d3748', edgecolor='#1a202c', linewidth=2)
    ax1.add_patch(black_box)
    ax1.text(5.75, 8.1, '???', fontsize=14, color='white', ha='center', va='center')

    ax1.annotate('', xy=(8.2, 8.1), xytext=(7.2, 8.1),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))

    answer_box1 = FancyBboxPatch((8.2, 7.5), 1.5, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='#fc8181', edgecolor='#c53030', linewidth=2)
    ax1.add_patch(answer_box1)
    ax1.text(8.95, 8.1, '???', fontsize=10, color='white', ha='center', va='center')

    # CoT path (bottom)
    ax1.text(0.5, 5.5, 'Chain-of-Thought:', fontsize=12, fontweight='bold', color='#276749')
    cot_q = FancyBboxPatch((0.3, 3.8), 2, 1.2, boxstyle="round,pad=0.05",
                           facecolor='#4299e1', edgecolor='#2b6cb0', linewidth=2)
    ax1.add_patch(cot_q)
    ax1.text(1.3, 4.4, 'Question', fontsize=9, color='white', ha='center', va='center')

    # Steps
    steps = [('Step 1', '#48bb78', 2.8), ('Step 2', '#38a169', 4.6), ('Step 3', '#276749', 6.4)]
    for label, color, x in steps:
        box = FancyBboxPatch((x, 3.8), 1.5, 1.2, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='#1a202c', linewidth=2)
        ax1.add_patch(box)
        ax1.text(x + 0.75, 4.4, label, fontsize=9, color='white', ha='center', va='center')

    # Arrows between steps
    for i in range(3):
        start_x = 2.3 + i * 1.8 if i == 0 else 2.8 + 1.5 + (i-1) * 1.8
        ax1.annotate('', xy=(2.8 + i * 1.8, 4.4), xytext=(start_x, 4.4),
                    arrowprops=dict(arrowstyle='->', color='#718096', lw=1.5))

    ax1.annotate('', xy=(8.4, 4.4), xytext=(7.9, 4.4),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=1.5))

    answer_box2 = FancyBboxPatch((8.4, 3.8), 1.3, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='#48bb78', edgecolor='#276749', linewidth=2)
    ax1.add_patch(answer_box2)
    ax1.text(9.05, 4.4, '11', fontsize=12, color='white', ha='center', va='center',
            fontweight='bold')

    # Labels
    ax1.text(5.75, 6.8, 'Single forward pass\nMay fail on complex problems',
            fontsize=9, ha='center', color='#c53030')
    ax1.text(5, 2.8, 'Multiple tokens = More compute\nExplicit reasoning steps',
            fontsize=9, ha='center', color='#276749')

    # ============== Panel 2: Self-Consistency ==============
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Self-Consistency: Multiple Paths', fontsize=14, fontweight='bold', pad=20)

    # Question
    q_box = FancyBboxPatch((3.5, 8.5), 3, 1),
    ax2.add_patch(FancyBboxPatch((3.5, 8.5), 3, 1, boxstyle="round,pad=0.05",
                                  facecolor='#4299e1', edgecolor='#2b6cb0', linewidth=2))
    ax2.text(5, 9, 'Question', fontsize=11, color='white', ha='center', va='center')

    # Multiple paths
    paths = [
        ("Path 1: 5+6=11", "42", '#48bb78', 1.5),
        ("Path 2: 5+6=11", "42", '#48bb78', 3.5),
        ("Path 3: error", "37", '#fc8181', 5.5),
        ("Path 4: 5+6=11", "42", '#48bb78', 7.5),
    ]

    for desc, answer, color, x in paths:
        # Path line
        ax2.plot([x + 0.5, 5], [6.5, 8.5], color='#718096', linewidth=1.5, alpha=0.5)

        # Path box
        box = FancyBboxPatch((x, 4.5), 1.5, 2, boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor='#718096', linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(x + 0.75, 5.8, desc.split(':')[0], fontsize=8, ha='center')
        ax2.text(x + 0.75, 5.2, desc.split(':')[1] if ':' in desc else '', fontsize=7, ha='center')

        # Answer circle
        circle = Circle((x + 0.75, 3.5), 0.4, facecolor=color, edgecolor='#1a202c', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x + 0.75, 3.5, answer, fontsize=10, ha='center', va='center',
                color='white', fontweight='bold')

        # Arrow down
        ax2.annotate('', xy=(x + 0.75, 3.9), xytext=(x + 0.75, 4.5),
                    arrowprops=dict(arrowstyle='->', color='#718096', lw=1.5))

    # Aggregation
    ax2.annotate('', xy=(5, 1.8), xytext=(2, 3.1),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=1.5, connectionstyle="arc3,rad=-0.2"))
    ax2.annotate('', xy=(5, 1.8), xytext=(4, 3.1),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=1.5, connectionstyle="arc3,rad=-0.1"))
    ax2.annotate('', xy=(5, 1.8), xytext=(6, 3.1),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=1.5, connectionstyle="arc3,rad=0.1"))
    ax2.annotate('', xy=(5, 1.8), xytext=(8, 3.1),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=1.5, connectionstyle="arc3,rad=0.2"))

    # Vote box
    vote_box = FancyBboxPatch((3.5, 0.8), 3, 1, boxstyle="round,pad=0.05",
                              facecolor='#9f7aea', edgecolor='#6b46c1', linewidth=2)
    ax2.add_patch(vote_box)
    ax2.text(5, 1.3, 'Majority Vote: 42', fontsize=11, color='white',
            ha='center', va='center', fontweight='bold')
    ax2.text(5, 0.4, '(3/4 agreement)', fontsize=9, ha='center', color='#6b46c1')

    # ============== Panel 3: Performance Comparison ==============
    ax3 = fig.add_subplot(gs[1, 0])

    techniques = ['Direct\nAnswer', 'Zero-Shot\nCoT', 'Few-Shot\nCoT', 'Self-\nConsistency']
    gsm8k = [17.7, 78.7, 56.5, 74.4]
    svamp = [30.0, 75.0, 68.9, 86.6]

    x = np.arange(len(techniques))
    width = 0.35

    bars1 = ax3.bar(x - width/2, gsm8k, width, label='GSM8K', color='#4299e1', edgecolor='#2b6cb0')
    bars2 = ax3.bar(x + width/2, svamp, width, label='SVAMP', color='#48bb78', edgecolor='#276749')

    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Chain-of-Thought Performance Gains', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(techniques, fontsize=10)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 100)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Highlight the gains
    ax3.annotate('+61%', xy=(1, 85), fontsize=12, color='#276749', fontweight='bold', ha='center')

    # ============== Panel 4: Technique Selection Guide ==============
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('When to Use Each Technique', fontsize=14, fontweight='bold', pad=20)

    techniques_info = [
        ('Zero-Shot CoT', '"Let\'s think step by step"', 'Quick, no examples needed', '#4299e1', 8.5),
        ('Few-Shot CoT', 'Demonstrated reasoning', 'Reliable, controlled format', '#48bb78', 6.5),
        ('Self-Consistency', 'Multiple paths + voting', 'High stakes, verifiable', '#9f7aea', 4.5),
        ('Tree-of-Thought', 'Branching exploration', 'Creative, search problems', '#ed8936', 2.5),
    ]

    for name, method, use_case, color, y in techniques_info:
        # Box
        box = FancyBboxPatch((0.3, y - 0.4), 9.4, 1.5, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='#1a202c', linewidth=2, alpha=0.85)
        ax4.add_patch(box)

        # Name
        ax4.text(0.6, y + 0.35, name, fontsize=11, fontweight='bold', color='white')
        # Method
        ax4.text(0.6, y - 0.1, method, fontsize=9, color='white', alpha=0.9)
        # Use case
        ax4.text(9.5, y + 0.15, use_case, fontsize=10, color='white', ha='right')

    # Cost indicator
    ax4.text(5, 0.8, 'Cost: Low → High (top to bottom)', fontsize=10,
            ha='center', style='italic', color='#718096')

    plt.suptitle('Chain-of-Thought Prompting: Techniques and Performance',
                fontsize=16, fontweight='bold', y=0.98)

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/prompt-engineering/cot_reasoning_flow.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_cot_flow_diagram()
