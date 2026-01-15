"""
Prompt Anatomy Visualization

This script creates a visualization showing the structure of modern LLM prompts,
including system, user, and assistant roles, delimiters, and attention patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_prompt_structure_diagram():
    """Create a visual diagram of prompt anatomy."""

    fig = plt.figure(figsize=(16, 12))

    # Create grid for subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ============== Panel 1: Message Structure ==============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Message-Based Prompt Structure', fontsize=14, fontweight='bold', pad=20)

    # Draw message boxes
    messages = [
        ('System', 'Sets persona, rules,\nand constraints', '#ed8936', 8.5),
        ('User', 'First query\nfrom human', '#4299e1', 6.5),
        ('Assistant', 'Model response', '#48bb78', 4.5),
        ('User', 'Follow-up query', '#4299e1', 2.5),
        ('Assistant', 'Current generation\n(being produced)', '#48bb78', 0.5),
    ]

    for i, (role, desc, color, y) in enumerate(messages):
        # Message box
        box = FancyBboxPatch((1, y), 8, 1.5, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='#2d3748', linewidth=2,
                             alpha=0.8)
        ax1.add_patch(box)

        # Role label
        ax1.text(2, y + 0.75, role, fontsize=12, fontweight='bold',
                color='white', va='center')

        # Description
        ax1.text(5.5, y + 0.75, desc, fontsize=10, color='white',
                va='center', ha='center')

        # Arrow to next
        if i < len(messages) - 1:
            ax1.annotate('', xy=(5, y), xytext=(5, y + 0.2),
                        arrowprops=dict(arrowstyle='->', color='#718096', lw=2))

    # ============== Panel 2: Delimiter Patterns ==============
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Delimiter Patterns', fontsize=14, fontweight='bold', pad=20)

    delimiters = [
        ('```...```', 'Code blocks,\ntext sections', '#805ad5'),
        ('<tag>...</tag>', 'XML-style,\nclear boundaries', '#3182ce'),
        ('## Header', 'Markdown,\nsection organization', '#38a169'),
        ('"""...."""', 'Triple quotes,\ntext embedding', '#dd6b20'),
    ]

    for i, (pattern, use, color) in enumerate(delimiters):
        y = 8 - i * 2.2
        # Pattern box
        box = FancyBboxPatch((0.5, y), 4, 1.5, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='#2d3748', linewidth=2,
                             alpha=0.85)
        ax2.add_patch(box)
        ax2.text(2.5, y + 0.75, pattern, fontsize=11, fontweight='bold',
                color='white', va='center', ha='center', family='monospace')

        # Use case
        ax2.text(7, y + 0.75, use, fontsize=10, va='center', ha='center')

        # Arrow
        ax2.annotate('', xy=(5, y + 0.75), xytext=(4.7, y + 0.75),
                    arrowprops=dict(arrowstyle='->', color='#718096', lw=1.5))

    # ============== Panel 3: Position Attention Pattern ==============
    ax3 = fig.add_subplot(gs[1, 0])

    # Simulate attention scores across prompt positions
    positions = np.arange(100)
    # U-shaped attention: high at start and end, lower in middle
    attention = 0.4 + 0.3 * np.exp(-positions/10) + 0.3 * np.exp(-(100-positions)/10)
    attention += 0.05 * np.random.randn(100)  # Add noise
    attention = np.clip(attention, 0.2, 1.0)

    ax3.fill_between(positions, attention, alpha=0.3, color='#4299e1')
    ax3.plot(positions, attention, color='#2b6cb0', linewidth=2)

    # Mark regions
    ax3.axvspan(0, 15, alpha=0.2, color='#48bb78', label='High attention (start)')
    ax3.axvspan(85, 100, alpha=0.2, color='#48bb78', label='High attention (end)')
    ax3.axvspan(35, 65, alpha=0.2, color='#fc8181', label='Lower attention (middle)')

    ax3.set_xlabel('Position in Prompt', fontsize=11)
    ax3.set_ylabel('Relative Attention', fontsize=11)
    ax3.set_title('Position Effects: "Lost in the Middle"', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(0, 1.1)

    # Annotations
    ax3.annotate('Place critical\ninstructions here', xy=(7, 0.85), fontsize=9,
                ha='center', color='#276749')
    ax3.annotate('May be\noverlooked', xy=(50, 0.35), fontsize=9,
                ha='center', color='#c53030')
    ax3.annotate('Place format\nspecs here', xy=(93, 0.85), fontsize=9,
                ha='center', color='#276749')

    # ============== Panel 4: Prompt Template Components ==============
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('Effective Prompt Components', fontsize=14, fontweight='bold', pad=20)

    components = [
        ('Task Definition', 'Clear, specific goal statement', '#e53e3e', 9),
        ('Context', 'Background information needed', '#dd6b20', 7.2),
        ('Input Data', 'The specific data to process', '#38a169', 5.4),
        ('Constraints', 'Rules, limitations, requirements', '#3182ce', 3.6),
        ('Output Format', 'Expected structure of response', '#805ad5', 1.8),
    ]

    # Draw stacked boxes
    for i, (name, desc, color, y) in enumerate(components):
        # Main box
        box = FancyBboxPatch((0.5, y - 0.6), 9, 1.2, boxstyle="round,pad=0.03",
                             facecolor=color, edgecolor='#1a202c', linewidth=2,
                             alpha=0.85)
        ax4.add_patch(box)

        # Component name
        ax4.text(1.5, y, name, fontsize=11, fontweight='bold',
                color='white', va='center')

        # Description
        ax4.text(6, y, desc, fontsize=10, color='white', va='center', ha='center')

    # Add connecting lines
    for i in range(len(components) - 1):
        ax4.plot([5, 5], [components[i][3] - 0.6, components[i+1][3] + 0.6],
                color='#718096', linewidth=2, linestyle='--', alpha=0.5)

    plt.suptitle('Prompt Anatomy: Structure and Components',
                fontsize=16, fontweight='bold', y=0.98)

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/prompt-engineering/prompt_anatomy_diagram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_prompt_structure_diagram()
