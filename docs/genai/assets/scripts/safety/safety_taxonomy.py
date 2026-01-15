"""
Safety Taxonomy Visualization

This script creates a visualization of AI safety risk categories
and their relationships.

Output: ../images/safety/safety_taxonomy.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def create_safety_taxonomy():
    """Create safety risk taxonomy diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Color palette
    colors = {
        'near_term': '#FFCDD2',   # Light red
        'long_term': '#E1BEE7',   # Light purple
        'misuse': '#FFECB3',      # Light amber
        'accident': '#C8E6C9',    # Light green
        'structural': '#BBDEFB',  # Light blue
        'arrow': '#455A64',       # Dark gray
        'text': '#212121'         # Near black
    }

    # Title
    ax.text(8, 11.5, 'AI Safety Risk Taxonomy',
            fontsize=18, fontweight='bold', ha='center', color=colors['text'])

    # === TIMELINE HEADER ===
    ax.text(4, 10.5, 'NEAR-TERM RISKS', fontsize=14, fontweight='bold',
            ha='center', color='#D32F2F')
    ax.text(12, 10.5, 'LONG-TERM RISKS', fontsize=14, fontweight='bold',
            ha='center', color='#7B1FA2')

    # Divider
    ax.plot([8, 8], [0.5, 10.2], color='gray', linestyle='--', linewidth=1.5)

    # === NEAR-TERM RISKS ===

    # Misuse Risks
    misuse_box = FancyBboxPatch((0.5, 7), 3.5, 3,
                                 boxstyle="round,pad=0.05",
                                 facecolor=colors['misuse'],
                                 edgecolor='#FF8F00', linewidth=2)
    ax.add_patch(misuse_box)
    ax.text(2.25, 9.7, 'MISUSE', fontsize=11, fontweight='bold',
            ha='center', color='#FF6F00')

    misuse_items = [
        ('Disinformation', 'Fake news at scale'),
        ('Scams', 'Automated phishing'),
        ('Malware', 'Code exploitation'),
        ('Harassment', 'Targeted attacks')
    ]
    for i, (name, desc) in enumerate(misuse_items):
        ax.text(2.25, 9.2 - i*0.5, f'{name}', fontsize=9, fontweight='bold', ha='center')
        ax.text(2.25, 8.95 - i*0.5, f'({desc})', fontsize=7, ha='center', color='gray')

    # Accidental Harms
    accident_box = FancyBboxPatch((4.25, 7), 3.5, 3,
                                   boxstyle="round,pad=0.05",
                                   facecolor=colors['accident'],
                                   edgecolor='#388E3C', linewidth=2)
    ax.add_patch(accident_box)
    ax.text(6, 9.7, 'ACCIDENTS', fontsize=11, fontweight='bold',
            ha='center', color='#2E7D32')

    accident_items = [
        ('Hallucinations', 'False information'),
        ('Bias', 'Unfair outputs'),
        ('Privacy leaks', 'PII exposure'),
        ('Harmful advice', 'Dangerous guidance')
    ]
    for i, (name, desc) in enumerate(accident_items):
        ax.text(6, 9.2 - i*0.5, f'{name}', fontsize=9, fontweight='bold', ha='center')
        ax.text(6, 8.95 - i*0.5, f'({desc})', fontsize=7, ha='center', color='gray')

    # Structural Risks
    struct_box = FancyBboxPatch((1.5, 3), 5, 3.5,
                                 boxstyle="round,pad=0.05",
                                 facecolor=colors['structural'],
                                 edgecolor='#1976D2', linewidth=2)
    ax.add_patch(struct_box)
    ax.text(4, 6.2, 'STRUCTURAL RISKS', fontsize=11, fontweight='bold',
            ha='center', color='#1565C0')

    struct_items = [
        ('Job displacement', 'Automation of work'),
        ('Power concentration', 'AI capability centralization'),
        ('Epistemic erosion', 'Trust in information'),
        ('Dependency', 'Critical infrastructure reliance'),
        ('Inequity', 'Unequal access to benefits')
    ]
    for i, (name, desc) in enumerate(struct_items):
        ax.text(4, 5.7 - i*0.5, f'{name}: {desc}', fontsize=9, ha='center')

    # === LONG-TERM RISKS ===

    # Alignment Risks
    align_box = FancyBboxPatch((8.5, 7), 7, 3,
                                boxstyle="round,pad=0.05",
                                facecolor='#E1BEE7',
                                edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(align_box)
    ax.text(12, 9.7, 'ALIGNMENT FAILURES', fontsize=11, fontweight='bold',
            ha='center', color='#7B1FA2')

    align_risks = [
        ('Deceptive Alignment', 'System appears aligned during training but\npursues different goals at deployment'),
        ('Goal Misgeneralization', 'System learns proxy goals that\ndiverge from intended behavior'),
        ('Reward Hacking', 'System games reward signal\nrather than achieving true objective')
    ]

    y_pos = 9.0
    for name, desc in align_risks:
        ax.text(9, y_pos, f'{name}:', fontsize=9, fontweight='bold', va='top')
        ax.text(9, y_pos - 0.25, desc, fontsize=8, va='top', color='gray')
        y_pos -= 0.85

    # Capability Risks
    cap_box = FancyBboxPatch((8.5, 3), 7, 3.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#FCE4EC',
                              edgecolor='#C2185B', linewidth=2)
    ax.add_patch(cap_box)
    ax.text(12, 6.2, 'CAPABILITY RISKS', fontsize=11, fontweight='bold',
            ha='center', color='#C2185B')

    cap_risks = [
        ('Power-seeking', 'System acquires resources/influence'),
        ('Self-improvement', 'Recursive capability enhancement'),
        ('Manipulation', 'Persuading humans against interest'),
        ('Coordination', 'Multiple AI systems colluding'),
        ('Situational awareness', 'Understanding of own training')
    ]
    for i, (name, desc) in enumerate(cap_risks):
        ax.text(12, 5.7 - i*0.5, f'{name}: {desc}', fontsize=9, ha='center')

    # === MITIGATION STRATEGIES (Bottom) ===
    mit_box = FancyBboxPatch((0.5, 0.5), 15, 2,
                              boxstyle="round,pad=0.05",
                              facecolor='#E8F5E9',
                              edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(mit_box)
    ax.text(8, 2.2, 'MITIGATION STRATEGIES', fontsize=12, fontweight='bold',
            ha='center', color='#2E7D32')

    mitigations = [
        ('Near-term', 'Content filtering, guardrails, red teaming, monitoring, policies'),
        ('Mid-term', 'Constitutional AI, interpretability, robustness testing, governance'),
        ('Long-term', 'Alignment research, capability control, coordination, verification')
    ]

    x_positions = [3, 8, 13]
    for (timeframe, strategies), x in zip(mitigations, x_positions):
        ax.text(x, 1.6, timeframe, fontsize=10, fontweight='bold', ha='center')
        ax.text(x, 1.1, strategies, fontsize=8, ha='center', color='gray',
                wrap=True)

    # Connecting arrows between near and long term
    ax.annotate('', xy=(8.3, 8.5), xytext=(7.7, 8.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                               linestyle='--'))
    ax.text(8, 8.8, 'evolves to', fontsize=8, ha='center', color='gray', style='italic')

    # Risk severity legend
    legend_box = FancyBboxPatch((12.5, 10.3), 3, 1.3,
                                 boxstyle="round,pad=0.03",
                                 facecolor='white',
                                 edgecolor='gray', linewidth=1)
    ax.add_patch(legend_box)
    ax.text(14, 11.35, 'Risk Severity', fontsize=9, fontweight='bold', ha='center')
    ax.add_patch(Circle((13, 11), 0.15, facecolor='#FFCDD2'))
    ax.text(13.3, 11, 'High', fontsize=8, va='center')
    ax.add_patch(Circle((14, 11), 0.15, facecolor='#FFF9C4'))
    ax.text(14.3, 11, 'Medium', fontsize=8, va='center')
    ax.add_patch(Circle((13, 10.6), 0.15, facecolor='#C8E6C9'))
    ax.text(13.3, 10.6, 'Emerging', fontsize=8, va='center')

    plt.tight_layout()
    return fig


def main():
    """Generate and save the visualization."""
    fig = create_safety_taxonomy()

    # Save to output directory
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/safety/safety_taxonomy.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
