#!/usr/bin/env python3
"""
Hallucination Detection Flow Visualization

Creates diagrams illustrating hallucination detection pipelines,
types of hallucinations, and detection method comparison.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_hallucination_types_chart():
    """
    Create a visual breakdown of hallucination types with examples.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define hallucination types with their characteristics
    types = [
        {
            'name': 'Intrinsic\nHallucination',
            'color': '#e74c3c',
            'description': 'Contradicts\nsource text',
            'example': '"Revenue was $10B"\n(source says $5B)',
            'detection': 'NLI-based',
            'severity': 0.9,
            'x': 0.2
        },
        {
            'name': 'Extrinsic\nHallucination',
            'color': '#f39c12',
            'description': 'Info not in\nsource',
            'example': 'Adds details\nnot mentioned',
            'detection': 'Entailment',
            'severity': 0.6,
            'x': 0.4
        },
        {
            'name': 'Factual\nHallucination',
            'color': '#9b59b6',
            'description': 'Wrong world\nknowledge',
            'example': '"Paris is capital\nof Germany"',
            'detection': 'KB Retrieval',
            'severity': 0.85,
            'x': 0.6
        },
        {
            'name': 'Fabricated\nEntities',
            'color': '#3498db',
            'description': 'Made-up\nreferences',
            'example': 'Cites non-existent\npapers/people',
            'detection': 'Entity Linking',
            'severity': 0.95,
            'x': 0.8
        }
    ]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Taxonomy of LLM Hallucinations',
            fontsize=18, fontweight='bold', ha='center', va='top')

    # Draw each type
    for t in types:
        y_base = 0.7

        # Main box
        rect = mpatches.FancyBboxPatch(
            (t['x'] - 0.08, y_base - 0.15), 0.16, 0.25,
            boxstyle="round,pad=0.02",
            facecolor=t['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)

        # Type name
        ax.text(t['x'], y_base + 0.05, t['name'],
                fontsize=11, fontweight='bold', ha='center', va='center', color='white')

        # Description box
        desc_rect = mpatches.FancyBboxPatch(
            (t['x'] - 0.07, y_base - 0.35), 0.14, 0.12,
            boxstyle="round,pad=0.01",
            facecolor='white',
            edgecolor=t['color'],
            linewidth=1.5
        )
        ax.add_patch(desc_rect)
        ax.text(t['x'], y_base - 0.29, t['description'],
                fontsize=9, ha='center', va='center')

        # Example box
        example_rect = mpatches.FancyBboxPatch(
            (t['x'] - 0.07, y_base - 0.52), 0.14, 0.12,
            boxstyle="round,pad=0.01",
            facecolor='#f8f9fa',
            edgecolor='gray',
            linewidth=1
        )
        ax.add_patch(example_rect)
        ax.text(t['x'], y_base - 0.46, t['example'],
                fontsize=8, ha='center', va='center', style='italic')

        # Detection method
        ax.text(t['x'], y_base - 0.62, f'Detection:\n{t["detection"]}',
                fontsize=9, ha='center', va='top', color='darkgreen', fontweight='bold')

        # Severity bar
        bar_width = 0.12
        bar_height = 0.015
        severity_color = plt.cm.RdYlGn_r(t['severity'])

        ax.add_patch(mpatches.Rectangle(
            (t['x'] - bar_width/2, y_base - 0.78),
            bar_width, bar_height,
            facecolor='lightgray'
        ))
        ax.add_patch(mpatches.Rectangle(
            (t['x'] - bar_width/2, y_base - 0.78),
            bar_width * t['severity'], bar_height,
            facecolor=severity_color
        ))
        ax.text(t['x'], y_base - 0.82, f'Severity: {t["severity"]:.0%}',
                fontsize=8, ha='center', va='top')

    # Add legend
    legend_y = 0.05
    ax.text(0.1, legend_y, 'Legend:', fontsize=10, fontweight='bold')
    ax.text(0.25, legend_y, 'Higher severity = more harmful to user trust',
            fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hallucination_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'hallucination_types.png'}")


def create_detection_pipeline():
    """
    Create a flowchart-style diagram of the hallucination detection pipeline.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Hallucination Detection Pipeline', fontsize=16, fontweight='bold', ha='center')

    # Define pipeline stages
    stages = [
        {'x': 1, 'y': 3, 'label': 'LLM\nResponse', 'color': '#3498db'},
        {'x': 3, 'y': 3, 'label': 'Claim\nExtraction', 'color': '#2ecc71'},
        {'x': 5, 'y': 4.2, 'label': 'NLI Check\n(vs Source)', 'color': '#e74c3c'},
        {'x': 5, 'y': 3, 'label': 'KB\nRetrieval', 'color': '#9b59b6'},
        {'x': 5, 'y': 1.8, 'label': 'Self-\nConsistency', 'color': '#f39c12'},
        {'x': 7, 'y': 3, 'label': 'Verdict\nAggregation', 'color': '#1abc9c'},
        {'x': 9, 'y': 3, 'label': 'Factuality\nScore', 'color': '#34495e'},
    ]

    # Draw boxes
    box_width, box_height = 1.2, 0.9

    for stage in stages:
        rect = mpatches.FancyBboxPatch(
            (stage['x'] - box_width/2, stage['y'] - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=stage['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(stage['x'], stage['y'], stage['label'],
                fontsize=10, fontweight='bold', ha='center', va='center', color='white')

    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=2)

    # Main flow
    ax.annotate('', xy=(2.4, 3), xytext=(1.6, 3), arrowprops=arrow_style)
    ax.annotate('', xy=(4.4, 4.2), xytext=(3.6, 3.2), arrowprops=arrow_style)
    ax.annotate('', xy=(4.4, 3), xytext=(3.6, 3), arrowprops=arrow_style)
    ax.annotate('', xy=(4.4, 1.8), xytext=(3.6, 2.8), arrowprops=arrow_style)
    ax.annotate('', xy=(6.4, 3.4), xytext=(5.6, 4.0), arrowprops=arrow_style)
    ax.annotate('', xy=(6.4, 3), xytext=(5.6, 3), arrowprops=arrow_style)
    ax.annotate('', xy=(6.4, 2.6), xytext=(5.6, 2.0), arrowprops=arrow_style)
    ax.annotate('', xy=(8.4, 3), xytext=(7.6, 3), arrowprops=arrow_style)

    # Add annotations
    annotations = [
        (2, 3.4, 'Extract factual claims\nfrom response'),
        (4, 4.8, 'Check against\nprovided context'),
        (4, 2.2, 'Verify against\nknowledge base'),
        (4, 1.2, 'Sample multiple\nresponses'),
        (8, 3.8, 'Supported / Refuted\n/ Unverifiable'),
    ]

    for x, y, text in annotations:
        ax.text(x, y, text, fontsize=8, ha='center', va='center',
                style='italic', color='gray')

    # Add stats box
    stats_text = """Detection Performance (typical):
    - NLI: 85% precision on intrinsic
    - KB Retrieval: 70% recall on factual
    - Self-consistency: Good for uncertainty"""

    ax.text(8, 1.2, stats_text, fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hallucination_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'hallucination_pipeline.png'}")


def create_detection_method_comparison():
    """
    Create a comparison chart of different hallucination detection methods.
    """
    methods = ['NLI-based', 'Self-Consistency', 'KB Retrieval', 'LLM Self-Check', 'Human Eval']

    # Metrics for each method (0-1 scale)
    metrics = {
        'Precision': [0.85, 0.70, 0.80, 0.75, 0.95],
        'Recall': [0.70, 0.85, 0.60, 0.80, 0.90],
        'Speed': [0.90, 0.30, 0.50, 0.40, 0.05],
        'Cost': [0.95, 0.60, 0.70, 0.50, 0.10],  # Higher = cheaper
        'Coverage': [0.50, 0.80, 0.70, 0.85, 0.95],
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(methods))
    width = 0.15
    multiplier = 0

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for i, (metric, scores) in enumerate(metrics.items()):
        offset = width * multiplier
        bars = ax.bar(x + offset, scores, width, label=metric, color=colors[i], alpha=0.8)
        multiplier += 1

    ax.set_ylabel('Score (higher is better)', fontsize=12)
    ax.set_xlabel('Detection Method', fontsize=12)
    ax.set_title('Hallucination Detection Methods Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(loc='upper right', ncols=5)
    ax.set_ylim(0, 1.1)

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detection_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'detection_method_comparison.png'}")


def create_hallucination_rate_by_task():
    """
    Create a bar chart showing typical hallucination rates by task type.
    """
    tasks = [
        'Open QA',
        'Summarization',
        'Creative Writing',
        'Code Generation',
        'RAG-based QA',
        'Closed-book QA',
        'Math Reasoning'
    ]

    # Hallucination rates (approximate, based on literature)
    rates = [0.35, 0.20, 0.45, 0.15, 0.10, 0.40, 0.25]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn_r(np.array(rates))
    bars = ax.barh(tasks, rates, color=colors, edgecolor='black', linewidth=1)

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(rate + 0.02, bar.get_y() + bar.get_height()/2,
                f'{rate:.0%}', va='center', fontsize=10)

    ax.set_xlabel('Hallucination Rate', fontsize=12)
    ax.set_title('Typical Hallucination Rates by Task Type', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.6)

    # Add reference lines
    ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Acceptable threshold')
    ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.7, label='Good threshold')

    ax.legend(loc='lower right')

    # Add note
    ax.text(0.55, 0.5, 'Note: Rates vary\nsignificantly by\nmodel and prompt',
            fontsize=9, transform=ax.transAxes, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hallucination_rate_by_task.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'hallucination_rate_by_task.png'}")


def main():
    """Generate all hallucination detection visualizations."""
    print("Generating hallucination detection visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    create_hallucination_types_chart()
    create_detection_pipeline()
    create_detection_method_comparison()
    create_hallucination_rate_by_task()

    print()
    print("All hallucination detection visualizations generated successfully!")


if __name__ == "__main__":
    main()
