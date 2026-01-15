"""
Content Filter Flow Visualization

This script creates a visualization of the content filtering pipeline
for LLM safety systems.

Output: ../images/safety/content_filter_flow.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def create_content_filter_flow():
    """Create content filtering pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color palette
    colors = {
        'stage': '#E3F2FD',       # Light blue
        'decision': '#FFF9C4',    # Light yellow
        'block': '#FFCDD2',       # Light red
        'pass': '#C8E6C9',        # Light green
        'arrow': '#455A64',       # Dark gray
        'text': '#212121'         # Near black
    }

    # Title
    ax.text(8, 9.5, 'Content Filtering Pipeline',
            fontsize=16, fontweight='bold', ha='center', color=colors['text'])

    # === INPUT PIPELINE (Top) ===
    ax.text(2, 8.7, 'INPUT FILTERING', fontsize=12, fontweight='bold',
            ha='center', color='#1565C0')

    # Stage boxes for input
    input_stages = [
        ('Preprocessing', 'Normalize\nUnicode\nLang Detect'),
        ('Blocklist', 'Pattern\nMatching\n<1ms'),
        ('PII Detection', 'Regex\nNER Model\n10-30ms'),
        ('Toxicity', 'ML Classifier\nMulti-label\n20-50ms'),
        ('Injection', 'Pattern +\nML Detect\n5-15ms')
    ]

    x_start = 0.5
    box_width = 2.5
    box_height = 1.8

    for i, (name, details) in enumerate(input_stages):
        x = x_start + i * 3
        box = FancyBboxPatch((x, 6.5), box_width, box_height,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['stage'],
                              edgecolor='#1565C0', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + box_width/2, 8.0, name, fontsize=9, fontweight='bold',
                ha='center', va='center')
        ax.text(x + box_width/2, 7.2, details, fontsize=8,
                ha='center', va='center')

        # Arrows between stages
        if i < len(input_stages) - 1:
            ax.annotate('', xy=(x + box_width + 0.5, 7.4),
                        xytext=(x + box_width, 7.4),
                        arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Decision diamond for input
    decision_x, decision_y = 14.5, 7.4
    diamond = plt.Polygon([(decision_x, decision_y + 0.6),
                           (decision_x + 0.6, decision_y),
                           (decision_x, decision_y - 0.6),
                           (decision_x - 0.6, decision_y)],
                          facecolor=colors['decision'], edgecolor='#F57F17', linewidth=2)
    ax.add_patch(diamond)
    ax.text(decision_x, decision_y, 'Pass?', fontsize=8, ha='center', va='center')

    # Arrow to decision
    ax.annotate('', xy=(decision_x - 0.6, decision_y),
                xytext=(x_start + len(input_stages) * 3 - 0.5, 7.4),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Block path
    block_box = FancyBboxPatch((14, 8.2), 1.5, 0.8,
                                boxstyle="round,pad=0.03",
                                facecolor=colors['block'],
                                edgecolor='#B71C1C', linewidth=1.5)
    ax.add_patch(block_box)
    ax.text(14.75, 8.6, 'BLOCK', fontsize=9, fontweight='bold',
            ha='center', va='center', color='#B71C1C')
    ax.annotate('', xy=(14.5, 8.2), xytext=(decision_x, decision_y + 0.6),
                arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5))

    # === LLM PROCESSING (Middle) ===
    llm_box = FancyBboxPatch((6.5, 4.5), 3, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#E8F5E9',
                              edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(llm_box)
    ax.text(8, 5.5, 'LLM', fontsize=14, fontweight='bold', ha='center', color='#2E7D32')
    ax.text(8, 4.9, 'Generate Response', fontsize=9, ha='center')

    # Arrow from input to LLM
    ax.annotate('', xy=(8, 6.5), xytext=(decision_x, decision_y - 0.6),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2,
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(12, 5.8, 'Pass', fontsize=9, color='#2E7D32', fontweight='bold')

    # === OUTPUT PIPELINE (Bottom) ===
    ax.text(8, 3.8, 'OUTPUT FILTERING', fontsize=12, fontweight='bold',
            ha='center', color='#2E7D32')

    output_stages = [
        ('Toxicity\nScan', '20-50ms'),
        ('Hallucination\nCheck', '50-100ms'),
        ('PII\nLeakage', '10-30ms'),
        ('Policy\nCompliance', '5-10ms')
    ]

    y_output = 2
    out_box_width = 2.8
    out_box_height = 1.4

    for i, (name, latency) in enumerate(output_stages):
        x = 1 + i * 3.5
        box = FancyBboxPatch((x, y_output), out_box_width, out_box_height,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['stage'],
                              edgecolor='#2E7D32', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + out_box_width/2, y_output + 1.0, name, fontsize=9,
                fontweight='bold', ha='center', va='center')
        ax.text(x + out_box_width/2, y_output + 0.3, latency, fontsize=8,
                ha='center', va='center', color='gray')

        # Arrows between stages
        if i < len(output_stages) - 1:
            ax.annotate('', xy=(x + out_box_width + 0.7, y_output + 0.7),
                        xytext=(x + out_box_width, y_output + 0.7),
                        arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Arrow from LLM to output pipeline
    ax.annotate('', xy=(2.5, y_output + out_box_height),
                xytext=(6.5, 4.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2,
                               connectionstyle='arc3,rad=0.2'))

    # Output decision
    out_decision_x = 14.5
    out_decision_y = y_output + 0.7
    diamond2 = plt.Polygon([(out_decision_x, out_decision_y + 0.5),
                            (out_decision_x + 0.5, out_decision_y),
                            (out_decision_x, out_decision_y - 0.5),
                            (out_decision_x - 0.5, out_decision_y)],
                           facecolor=colors['decision'], edgecolor='#F57F17', linewidth=2)
    ax.add_patch(diamond2)
    ax.text(out_decision_x, out_decision_y, 'Pass?', fontsize=8, ha='center', va='center')

    # Arrow to output decision
    ax.annotate('', xy=(out_decision_x - 0.5, out_decision_y),
                xytext=(1 + len(output_stages) * 3.5 - 0.7, y_output + 0.7),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Success path
    success_box = FancyBboxPatch((14, 0.3), 1.5, 0.8,
                                  boxstyle="round,pad=0.03",
                                  facecolor=colors['pass'],
                                  edgecolor='#2E7D32', linewidth=1.5)
    ax.add_patch(success_box)
    ax.text(14.75, 0.7, 'RETURN', fontsize=9, fontweight='bold',
            ha='center', va='center', color='#2E7D32')
    ax.annotate('', xy=(14.75, 1.1), xytext=(out_decision_x, out_decision_y - 0.5),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

    # Regenerate path
    ax.annotate('', xy=(8, 4.5), xytext=(out_decision_x + 0.5, out_decision_y),
                arrowprops=dict(arrowstyle='->', color='#F57F17', lw=1.5,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(12.5, 3.5, 'Regenerate', fontsize=9, color='#F57F17', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['stage'], edgecolor='gray', label='Processing Stage'),
        mpatches.Patch(facecolor=colors['decision'], edgecolor='#F57F17', label='Decision Point'),
        mpatches.Patch(facecolor=colors['block'], edgecolor='#B71C1C', label='Block Action'),
        mpatches.Patch(facecolor=colors['pass'], edgecolor='#2E7D32', label='Pass/Return'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

    plt.tight_layout()
    return fig


def main():
    """Generate and save the visualization."""
    fig = create_content_filter_flow()

    # Save to output directory
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/safety/content_filter_flow.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
