#!/usr/bin/env python3
"""
Model Family Lineage Diagram

This script creates visualizations showing the evolution and relationships
between major LLM families (GPT, Claude, LLaMA, Mistral, Gemini).

Output: ../images/llm-foundations/model_family_*.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection

# Ensure output directory exists
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "llm-foundations"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_timeline():
    """Create a timeline of major LLM releases."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Timeline data: (year, month, name, family, size_B, notable)
    models = [
        (2018, 6, 'GPT-1', 'GPT', 0.117, 'First GPT'),
        (2018, 10, 'BERT', 'Google', 0.340, 'Bidirectional'),
        (2019, 2, 'GPT-2', 'GPT', 1.5, 'Zero-shot'),
        (2020, 5, 'GPT-3', 'GPT', 175, 'Few-shot learning'),
        (2022, 3, 'Chinchilla', 'DeepMind', 70, 'Scaling laws'),
        (2022, 3, 'Claude', 'Anthropic', None, 'Constitutional AI'),
        (2022, 11, 'ChatGPT', 'GPT', 175, 'RLHF chat'),
        (2023, 2, 'LLaMA', 'Meta', 65, 'Open weights'),
        (2023, 3, 'GPT-4', 'GPT', None, 'Multimodal'),
        (2023, 7, 'Claude 2', 'Anthropic', None, '100K context'),
        (2023, 7, 'LLaMA 2', 'Meta', 70, 'Commercial license'),
        (2023, 9, 'Mistral 7B', 'Mistral', 7, 'Efficient open'),
        (2023, 12, 'Mixtral', 'Mistral', 47, 'MoE architecture'),
        (2023, 12, 'Gemini', 'Google', None, 'Native multimodal'),
        (2024, 3, 'Claude 3', 'Anthropic', None, 'Opus/Sonnet/Haiku'),
        (2024, 4, 'LLaMA 3', 'Meta', 405, '405B flagship'),
        (2024, 5, 'GPT-4o', 'GPT', None, 'Omni model'),
        (2024, 6, 'Claude 3.5', 'Anthropic', None, 'Sonnet'),
    ]

    # Family colors
    family_colors = {
        'GPT': '#74b9ff',
        'Google': '#fd79a8',
        'DeepMind': '#fd79a8',
        'Anthropic': '#a29bfe',
        'Meta': '#55efc4',
        'Mistral': '#ffeaa7',
    }

    # Create timeline
    years = range(2018, 2025)
    ax.set_xlim(2017.5, 2025)
    ax.set_ylim(-1, 12)

    # Draw year axis
    for year in years:
        ax.axvline(x=year, color='lightgray', linestyle='-', linewidth=0.5)
        ax.text(year, -0.5, str(year), ha='center', fontsize=11)

    # Position models
    family_rows = {'GPT': 10, 'Google': 8, 'DeepMind': 8, 'Anthropic': 6, 'Meta': 4, 'Mistral': 2}

    for year, month, name, family, size, notable in models:
        x = year + (month - 1) / 12
        y = family_rows[family]

        # Draw model box
        color = family_colors[family]
        box = FancyBboxPatch((x - 0.15, y - 0.35), 0.3, 0.7,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)

        # Model name
        ax.text(x, y + 0.8, name, ha='center', va='bottom', fontsize=9,
                fontweight='bold', rotation=45)

        # Size annotation if known
        if size:
            ax.text(x, y - 0.5, f'{size}B' if size >= 1 else f'{int(size * 1000)}M',
                    ha='center', fontsize=7, color='gray')

    # Family labels
    for family, y in family_rows.items():
        if family != 'DeepMind':  # Skip duplicate
            ax.text(2017.7, y, family, ha='right', va='center', fontsize=11,
                    fontweight='bold', color=family_colors[family])

    ax.axis('off')
    ax.set_title('Timeline of Major LLM Releases', fontsize=14, fontweight='bold', pad=20)

    # Legend for notable features
    legend_text = (
        "Key Innovations:\n"
        "- GPT-3: Few-shot learning\n"
        "- Chinchilla: Scaling laws\n"
        "- LLaMA: Open weights movement\n"
        "- Mixtral: MoE for open models\n"
        "- Gemini/GPT-4: Native multimodal"
    )
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "model_family_timeline.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved model timeline to: {output_path}")


def create_capability_comparison():
    """Create radar chart comparing model capabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Capability dimensions (normalized 0-10)
    categories = ['Reasoning', 'Coding', 'Creative\nWriting', 'Math',
                  'Factual\nKnowledge', 'Long\nContext', 'Speed']

    # Model scores (illustrative, based on general benchmarks)
    models = {
        'GPT-4': [9.5, 9.0, 9.0, 9.0, 9.0, 8.0, 6.0],
        'Claude 3 Opus': [9.5, 8.5, 9.5, 8.5, 9.0, 9.5, 5.0],
        'Gemini Ultra': [9.0, 8.0, 8.5, 9.5, 9.0, 9.0, 7.0],
        'LLaMA 3 70B': [8.0, 8.0, 7.5, 7.5, 8.0, 7.0, 8.0],
        'Mistral Large': [8.0, 8.0, 7.5, 7.5, 7.5, 6.0, 8.5],
    }

    colors = ['#74b9ff', '#a29bfe', '#fd79a8', '#55efc4', '#ffeaa7']

    # Left: Radar chart
    ax = axes[0]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    ax = plt.subplot(121, polar=True)

    for (name, values), color in zip(models.items(), colors):
        values_plot = values + values[:1]  # Complete the circle
        ax.plot(angles, values_plot, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values_plot, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_title('Model Capability Comparison', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

    # Right: Bar chart for specific metrics
    ax = axes[1]

    metrics = ['MMLU', 'HumanEval', 'GSM8K', 'MATH']
    x = np.arange(len(metrics))
    width = 0.15

    # Approximate benchmark scores
    scores = {
        'GPT-4': [86.4, 67.0, 92.0, 52.9],
        'Claude 3 Opus': [86.8, 84.9, 95.0, 60.1],
        'Gemini Ultra': [90.0, 74.4, 94.4, 53.2],
        'LLaMA 3 70B': [82.0, 81.7, 93.0, 50.4],
        'Mistral Large': [81.2, 45.1, 91.2, 45.0],
    }

    for i, (name, vals) in enumerate(scores.items()):
        offset = (i - 2) * width
        ax.bar(x + offset, vals, width, label=name, color=colors[i], alpha=0.8)

    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title('Benchmark Scores', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "model_family_capabilities.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved capability comparison to: {output_path}")


def create_architecture_comparison():
    """Create comparison of architectural choices."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Architecture features for each family
    features = {
        'GPT-4': {
            'Attention': 'MHA',
            'Position': 'Unknown',
            'Normalization': 'LayerNorm',
            'Activation': 'Unknown',
            'Architecture': 'MoE (rumored)',
            'Context': '128K',
            'Multimodal': 'Yes',
        },
        'Claude 3': {
            'Attention': 'Unknown',
            'Position': 'Unknown',
            'Normalization': 'Unknown',
            'Activation': 'Unknown',
            'Architecture': 'Dense',
            'Context': '200K',
            'Multimodal': 'Yes',
        },
        'LLaMA 3': {
            'Attention': 'GQA',
            'Position': 'RoPE',
            'Normalization': 'RMSNorm',
            'Activation': 'SwiGLU',
            'Architecture': 'Dense',
            'Context': '8K-128K',
            'Multimodal': 'No',
        },
        'Mistral': {
            'Attention': 'GQA + Sliding Window',
            'Position': 'RoPE',
            'Normalization': 'RMSNorm',
            'Activation': 'SwiGLU',
            'Architecture': 'Dense/MoE',
            'Context': 'Infinite*',
            'Multimodal': 'No',
        },
        'Gemini': {
            'Attention': 'Unknown',
            'Position': 'Unknown',
            'Normalization': 'Unknown',
            'Activation': 'Unknown',
            'Architecture': 'MoE',
            'Context': '2M',
            'Multimodal': 'Native',
        },
    }

    # Create table
    models = list(features.keys())
    feature_names = list(features['LLaMA 3'].keys())

    # Build data matrix
    cell_text = []
    for feature in feature_names:
        row = [features[model][feature] for model in models]
        cell_text.append(row)

    # Create table
    table = ax.table(
        cellText=cell_text,
        rowLabels=feature_names,
        colLabels=models,
        cellLoc='center',
        loc='center',
        colColours=['#74b9ff', '#a29bfe', '#55efc4', '#ffeaa7', '#fd79a8'],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    ax.axis('off')
    ax.set_title('LLM Architecture Comparison', fontsize=14, fontweight='bold', pad=20)

    # Add footnote
    ax.text(0.5, -0.05, '* Sliding window enables theoretically infinite context generation',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "model_family_architecture.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved architecture comparison to: {output_path}")


def create_open_vs_proprietary():
    """Visualize the open vs proprietary model landscape."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Model data: (capability score, openness score, size_for_marker, name)
    models = [
        # Proprietary
        (9.5, 1, 1500, 'GPT-4', '#74b9ff'),
        (9.0, 1, 1200, 'Claude 3 Opus', '#a29bfe'),
        (9.0, 1, 1400, 'Gemini Ultra', '#fd79a8'),
        (8.5, 1, 800, 'GPT-4 Turbo', '#74b9ff'),
        (8.5, 1, 600, 'Claude 3.5 Sonnet', '#a29bfe'),

        # Open weights
        (8.0, 8, 1000, 'LLaMA 3 70B', '#55efc4'),
        (8.5, 8, 1200, 'LLaMA 3 405B', '#55efc4'),
        (7.0, 9, 300, 'Mistral 7B', '#ffeaa7'),
        (8.0, 8, 600, 'Mixtral 8x7B', '#ffeaa7'),
        (7.5, 8, 400, 'LLaMA 3 8B', '#55efc4'),

        # Fully open
        (6.5, 10, 200, 'Falcon 7B', '#81ecec'),
        (7.0, 10, 400, 'MPT 30B', '#81ecec'),
    ]

    for cap, openness, size, name, color in models:
        ax.scatter(openness, cap, s=size, c=color, alpha=0.7, edgecolor='black')
        ax.annotate(name, (openness, cap), xytext=(5, 5), textcoords='offset points',
                    fontsize=9)

    ax.set_xlim(0, 11)
    ax.set_ylim(5, 10)
    ax.set_xlabel('Openness Score', fontsize=12)
    ax.set_ylabel('Capability Score', fontsize=12)
    ax.set_title('Open vs Proprietary LLM Landscape', fontsize=14, fontweight='bold')

    # Add region labels
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.3)
    ax.text(2.5, 9.7, 'Proprietary\n(API only)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    ax.text(8, 9.7, 'Open Weights\n(Self-hostable)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Legend for size
    ax.scatter([], [], s=200, c='gray', alpha=0.5, label='Small model')
    ax.scatter([], [], s=600, c='gray', alpha=0.5, label='Medium model')
    ax.scatter([], [], s=1200, c='gray', alpha=0.5, label='Large model')
    ax.legend(title='Model Size', loc='lower right', fontsize=9)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "model_family_open_proprietary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved open vs proprietary chart to: {output_path}")


if __name__ == "__main__":
    create_timeline()
    create_capability_comparison()
    create_architecture_comparison()
    create_open_vs_proprietary()
    print("All model family visualizations created successfully!")
