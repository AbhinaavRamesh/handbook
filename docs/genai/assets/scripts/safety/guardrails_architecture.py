"""
Guardrails Architecture Visualization

This script creates a visualization of the guardrails architecture
for LLM safety systems.

Output: ../images/safety/guardrails_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def create_guardrails_architecture():
    """Create guardrails architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color palette
    colors = {
        'input': '#FFCDD2',      # Light red
        'processing': '#E3F2FD',  # Light blue
        'output': '#C8E6C9',      # Light green
        'guardrails': '#FFF9C4',  # Light yellow
        'arrow': '#455A64',       # Dark gray
        'text': '#212121'         # Near black
    }

    # Title
    ax.text(7, 9.5, 'LLM Guardrails Architecture',
            fontsize=16, fontweight='bold', ha='center', color=colors['text'])

    # Input Section
    input_box = FancyBboxPatch((0.5, 6), 3, 2.5,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['input'],
                                edgecolor='#B71C1C', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 8.1, 'Input Guardrails', fontsize=11, fontweight='bold', ha='center')

    input_items = ['PII Detection', 'Toxicity Filter', 'Injection Detection', 'Rate Limiting']
    for i, item in enumerate(input_items):
        ax.text(2, 7.5 - i*0.4, f'• {item}', fontsize=9, ha='center')

    # Processing Section
    proc_box = FancyBboxPatch((5, 6), 4, 2.5,
                               boxstyle="round,pad=0.05",
                               facecolor=colors['processing'],
                               edgecolor='#1565C0', linewidth=2)
    ax.add_patch(proc_box)
    ax.text(7, 8.1, 'Core Processing', fontsize=11, fontweight='bold', ha='center')

    proc_items = ['System Prompt', 'LLM Inference', 'Context Management', 'Token Limits']
    for i, item in enumerate(proc_items):
        ax.text(7, 7.5 - i*0.4, f'• {item}', fontsize=9, ha='center')

    # Output Section
    output_box = FancyBboxPatch((10.5, 6), 3, 2.5,
                                 boxstyle="round,pad=0.05",
                                 facecolor=colors['output'],
                                 edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(output_box)
    ax.text(12, 8.1, 'Output Guardrails', fontsize=11, fontweight='bold', ha='center')

    output_items = ['Content Filter', 'Hallucination Check', 'Policy Compliance', 'Schema Validation']
    for i, item in enumerate(output_items):
        ax.text(12, 7.5 - i*0.4, f'• {item}', fontsize=9, ha='center')

    # Arrows
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=6"
    kw = dict(arrowstyle=arrow_style, color=colors['arrow'])

    # Input to Processing
    ax.annotate('', xy=(5, 7.25), xytext=(3.5, 7.25),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Processing to Output
    ax.annotate('', xy=(10.5, 7.25), xytext=(9, 7.25),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # User and Response
    ax.text(0.3, 7.25, 'User\nInput', fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.text(13.7, 7.25, 'Safe\nResponse', fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # Guardrails Framework Box
    framework_box = FancyBboxPatch((0.5, 2), 13, 3,
                                    boxstyle="round,pad=0.05",
                                    facecolor=colors['guardrails'],
                                    edgecolor='#F57F17', linewidth=2)
    ax.add_patch(framework_box)
    ax.text(7, 4.7, 'Guardrails Framework Options', fontsize=11, fontweight='bold', ha='center')

    # Framework columns
    frameworks = [
        ('NeMo Guardrails', ['Colang DSL', 'Dialog Flows', 'Topic Control', 'Best for: Chatbots']),
        ('Guardrails AI', ['Pydantic Schemas', 'Auto-Retry', 'Validators', 'Best for: Structured Output']),
        ('Custom', ['Full Control', 'Domain Logic', 'Integration', 'Best for: Complex Needs'])
    ]

    x_positions = [2.5, 7, 11.5]
    for (name, items), x in zip(frameworks, x_positions):
        ax.text(x, 4.2, name, fontsize=10, fontweight='bold', ha='center')
        for i, item in enumerate(items):
            style = 'italic' if item.startswith('Best') else 'normal'
            ax.text(x, 3.7 - i*0.4, f'• {item}', fontsize=9, ha='center', style=style)

    # Monitoring box
    monitor_box = FancyBboxPatch((4.5, 0.5), 5, 1,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#E8EAF6',
                                  edgecolor='#3F51B5', linewidth=2)
    ax.add_patch(monitor_box)
    ax.text(7, 1, 'Monitoring: Metrics | Alerts | Audit Logs | Human Review',
            fontsize=9, ha='center', va='center')

    # Connector from processing to monitoring
    ax.annotate('', xy=(7, 1.5), xytext=(7, 6),
                arrowprops=dict(arrowstyle='->', color='#3F51B5', lw=1.5, ls='--'))

    plt.tight_layout()
    return fig


def main():
    """Generate and save the visualization."""
    fig = create_guardrails_architecture()

    # Save to output directory
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/safety/guardrails_architecture.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
