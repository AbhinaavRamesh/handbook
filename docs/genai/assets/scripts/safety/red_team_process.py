"""
Red Team Process Visualization

This script creates a visualization of the red teaming methodology
for LLM safety testing.

Output: ../images/safety/red_team_process.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def create_red_team_process():
    """Create red team methodology diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Color palette
    colors = {
        'planning': '#E3F2FD',    # Light blue
        'execution': '#FFCDD2',   # Light red
        'analysis': '#FFF9C4',    # Light yellow
        'remediation': '#C8E6C9', # Light green
        'attack': '#F3E5F5',      # Light purple
        'arrow': '#455A64',       # Dark gray
        'text': '#212121'         # Near black
    }

    # Title
    ax.text(7, 11.5, 'LLM Red Teaming Methodology',
            fontsize=16, fontweight='bold', ha='center', color=colors['text'])

    # === PHASE 1: PLANNING ===
    plan_box = FancyBboxPatch((0.5, 8.5), 3, 2.5,
                               boxstyle="round,pad=0.05",
                               facecolor=colors['planning'],
                               edgecolor='#1565C0', linewidth=2)
    ax.add_patch(plan_box)
    ax.text(2, 10.7, '1. PLANNING', fontsize=11, fontweight='bold',
            ha='center', color='#1565C0')

    plan_items = ['Define scope & goals', 'Identify attack surface',
                  'Set success criteria', 'Assemble team']
    for i, item in enumerate(plan_items):
        ax.text(2, 10.1 - i*0.4, f'• {item}', fontsize=9, ha='center')

    # === PHASE 2: EXECUTION ===
    exec_box = FancyBboxPatch((4, 8.5), 3.5, 2.5,
                               boxstyle="round,pad=0.05",
                               facecolor=colors['execution'],
                               edgecolor='#B71C1C', linewidth=2)
    ax.add_patch(exec_box)
    ax.text(5.75, 10.7, '2. EXECUTION', fontsize=11, fontweight='bold',
            ha='center', color='#B71C1C')

    exec_items = ['Manual probing', 'Automated attacks',
                  'Prompt fuzzing', 'Multi-turn attacks']
    for i, item in enumerate(exec_items):
        ax.text(5.75, 10.1 - i*0.4, f'• {item}', fontsize=9, ha='center')

    # === PHASE 3: ANALYSIS ===
    analysis_box = FancyBboxPatch((8, 8.5), 3, 2.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor=colors['analysis'],
                                   edgecolor='#F57F17', linewidth=2)
    ax.add_patch(analysis_box)
    ax.text(9.5, 10.7, '3. ANALYSIS', fontsize=11, fontweight='bold',
            ha='center', color='#F57F17')

    analysis_items = ['Categorize failures', 'Assess severity',
                      'Document findings', 'Prioritize issues']
    for i, item in enumerate(analysis_items):
        ax.text(9.5, 10.1 - i*0.4, f'• {item}', fontsize=9, ha='center')

    # === PHASE 4: REMEDIATION ===
    remed_box = FancyBboxPatch((11.5, 8.5), 2, 2.5,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['remediation'],
                                edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(remed_box)
    ax.text(12.5, 10.7, '4. REMEDIATE', fontsize=11, fontweight='bold',
            ha='center', color='#2E7D32')

    remed_items = ['Update guardrails', 'Retrain model',
                   'Add test cases']
    for i, item in enumerate(remed_items):
        ax.text(12.5, 10.1 - i*0.4, f'• {item}', fontsize=9, ha='center')

    # Arrows between phases
    ax.annotate('', xy=(4, 9.75), xytext=(3.5, 9.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.annotate('', xy=(8, 9.75), xytext=(7.5, 9.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.annotate('', xy=(11.5, 9.75), xytext=(11, 9.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Cycle arrow back to planning
    ax.annotate('', xy=(1.5, 11), xytext=(12.5, 11),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2,
                               connectionstyle='arc3,rad=0.15'))
    ax.text(7, 11.2, 'Continuous Improvement', fontsize=9, ha='center',
            style='italic', color='gray')

    # === ATTACK TAXONOMY ===
    ax.text(7, 7.5, 'Jailbreak Attack Taxonomy', fontsize=13, fontweight='bold',
            ha='center', color=colors['text'])

    attack_categories = [
        ('Direct', '#FFCDD2', ['Ignore instructions', 'Override commands', 'Authority claims']),
        ('Indirect', '#FFF9C4', ['Role-play', 'Hypotheticals', 'Fiction framing']),
        ('Technical', '#E3F2FD', ['Prompt injection', 'Encoding tricks', 'Token manipulation']),
        ('Multi-turn', '#F3E5F5', ['Context building', 'Topic pivot', 'Gradual escalation'])
    ]

    box_width = 3
    box_height = 2.5
    y_attack = 4.5

    for i, (name, color, attacks) in enumerate(attack_categories):
        x = 0.5 + i * 3.4
        box = FancyBboxPatch((x, y_attack), box_width, box_height,
                              boxstyle="round,pad=0.05",
                              facecolor=color,
                              edgecolor='gray', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + box_width/2, y_attack + box_height - 0.3, name,
                fontsize=10, fontweight='bold', ha='center')

        for j, attack in enumerate(attacks):
            ax.text(x + box_width/2, y_attack + box_height - 0.8 - j*0.5, f'• {attack}',
                    fontsize=8, ha='center')

    # === AUTOMATED RED TEAMING ===
    ax.text(7, 3.8, 'Automated Red Teaming Pipeline', fontsize=12, fontweight='bold',
            ha='center', color=colors['text'])

    auto_stages = [
        ('Attack\nGeneration', 'Templates\nLLM attacks\nEvolution'),
        ('Parallel\nExecution', 'Rate-limited\nConcurrent\nLogging'),
        ('Evaluation', 'Jailbreak\nscorer\nHarm detector'),
        ('Triage', 'Severity\nranking\nHuman review')
    ]

    stage_width = 2.8
    stage_height = 1.8
    y_auto = 1.5

    for i, (name, details) in enumerate(auto_stages):
        x = 1 + i * 3.2
        box = FancyBboxPatch((x, y_auto), stage_width, stage_height,
                              boxstyle="round,pad=0.05",
                              facecolor='#ECEFF1',
                              edgecolor='#607D8B', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + stage_width/2, y_auto + stage_height - 0.4, name,
                fontsize=9, fontweight='bold', ha='center')
        ax.text(x + stage_width/2, y_auto + 0.5, details,
                fontsize=8, ha='center', color='gray')

        if i < len(auto_stages) - 1:
            ax.annotate('', xy=(x + stage_width + 0.4, y_auto + stage_height/2),
                        xytext=(x + stage_width, y_auto + stage_height/2),
                        arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Metrics box
    metrics_box = FancyBboxPatch((1, 0.2), 12, 1,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#E8F5E9',
                                  edgecolor='#4CAF50', linewidth=1.5)
    ax.add_patch(metrics_box)
    ax.text(7, 0.7, 'Key Metrics:  Attack Success Rate | Harm Score Distribution | Time to Detect | Coverage by Category',
            fontsize=9, ha='center', va='center')

    plt.tight_layout()
    return fig


def main():
    """Generate and save the visualization."""
    fig = create_red_team_process()

    # Save to output directory
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/safety/red_team_process.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
