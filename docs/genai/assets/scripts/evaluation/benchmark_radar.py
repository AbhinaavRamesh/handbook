#!/usr/bin/env python3
"""
Benchmark Radar Chart Visualization

Creates radar/spider charts comparing LLM performance across different benchmarks,
showing capability profiles for different models.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_radar_chart(ax, categories, values, label, color, alpha=0.25):
    """
    Create a radar chart on the given axes.

    Args:
        ax: Matplotlib axes
        categories: List of category names
        values: List of values (0-1 scale)
        label: Legend label
        color: Line/fill color
        alpha: Fill transparency
    """
    N = len(categories)

    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    # Values must also complete the loop
    values = list(values) + [values[0]]

    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
    ax.fill(angles, values, alpha=alpha, color=color)


def create_model_comparison_radar():
    """
    Create a radar chart comparing different LLM models across benchmarks.
    """
    # Benchmark categories
    categories = ['MMLU', 'HumanEval', 'GSM8K', 'TruthfulQA', 'MT-Bench', 'HellaSwag']

    # Model performance (normalized 0-1)
    models = {
        'GPT-4': [0.86, 0.67, 0.92, 0.59, 0.94, 0.95],
        'Claude 3 Opus': [0.86, 0.84, 0.95, 0.65, 0.91, 0.95],
        'Llama 3 70B': [0.79, 0.48, 0.83, 0.52, 0.82, 0.87],
        'Gemini Pro': [0.79, 0.67, 0.86, 0.55, 0.79, 0.84],
    }

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Set up the angles
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    # Set category labels
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11)

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)

    # Plot each model
    for (model_name, values), color in zip(models.items(), colors):
        create_radar_chart(ax, categories, values, model_name, color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    plt.title('LLM Capability Profiles Across Benchmarks', fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'benchmark_radar_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'benchmark_radar_models.png'}")


def create_capability_dimension_radar():
    """
    Create a radar chart showing capability dimensions rather than specific benchmarks.
    """
    # Capability categories
    categories = ['Reasoning', 'Knowledge', 'Coding', 'Math', 'Safety', 'Instruction\nFollowing', 'Creativity']

    # Model profiles (0-1 scale)
    models = {
        'Foundation Model': [0.7, 0.8, 0.5, 0.6, 0.4, 0.5, 0.7],
        'Instruction-Tuned': [0.75, 0.78, 0.55, 0.65, 0.7, 0.85, 0.6],
        'RLHF-Aligned': [0.8, 0.75, 0.6, 0.7, 0.9, 0.9, 0.55],
    }

    colors = ['#95a5a6', '#3498db', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)

    for (model_name, values), color in zip(models.items(), colors):
        create_radar_chart(ax, categories, values, model_name, color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    plt.title('Impact of Training Stages on Capability Profile', fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'capability_dimension_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'capability_dimension_radar.png'}")


def create_benchmark_coverage_chart():
    """
    Create a bar chart showing what each major benchmark covers.
    """
    benchmarks = ['MMLU', 'HumanEval', 'GSM8K', 'TruthfulQA', 'MT-Bench', 'HellaSwag']

    # What each benchmark tests (0-1 relevance to each capability)
    capabilities = ['Knowledge', 'Reasoning', 'Coding', 'Math', 'Honesty', 'Conv.']

    coverage = {
        'MMLU':       [0.9, 0.4, 0.1, 0.3, 0.2, 0.1],
        'HumanEval':  [0.2, 0.5, 1.0, 0.3, 0.1, 0.1],
        'GSM8K':      [0.2, 0.9, 0.1, 1.0, 0.1, 0.1],
        'TruthfulQA': [0.5, 0.3, 0.0, 0.0, 1.0, 0.2],
        'MT-Bench':   [0.6, 0.7, 0.3, 0.3, 0.5, 1.0],
        'HellaSwag':  [0.4, 0.8, 0.0, 0.1, 0.1, 0.2],
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(capabilities))
    width = 0.12
    multiplier = 0

    colors = plt.cm.Set2(np.linspace(0, 1, len(benchmarks)))

    for i, (benchmark, scores) in enumerate(coverage.items()):
        offset = width * multiplier
        bars = ax.bar(x + offset, scores, width, label=benchmark, color=colors[i], alpha=0.85)
        multiplier += 1

    ax.set_ylabel('Coverage / Relevance', fontsize=12)
    ax.set_xlabel('Capability Dimension', fontsize=12)
    ax.set_title('What Each Benchmark Actually Measures', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(capabilities, fontsize=11)
    ax.legend(loc='upper right', ncols=3, fontsize=9)
    ax.set_ylim(0, 1.15)

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add note
    ax.text(0.02, 0.98, 'Note: No single benchmark\ncovers all capabilities',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'benchmark_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'benchmark_coverage.png'}")


def create_benchmark_timeline():
    """
    Create a timeline showing benchmark score progression over model generations.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Timeline data (approximate scores over time)
    years = ['2020', '2021', '2022', '2023', '2024']
    x = np.arange(len(years))

    benchmarks = {
        'MMLU': [0.43, 0.52, 0.70, 0.86, 0.90],
        'HumanEval': [0.28, 0.35, 0.47, 0.67, 0.90],
        'GSM8K': [0.18, 0.35, 0.58, 0.92, 0.95],
        'HellaSwag': [0.79, 0.84, 0.87, 0.95, 0.96],
    }

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    markers = ['o', 's', '^', 'D']

    for (benchmark, scores), color, marker in zip(benchmarks.items(), colors, markers):
        ax.plot(x, scores, marker=marker, markersize=10, linewidth=2.5,
                label=benchmark, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11)
    ax.set_xlabel('Year (Best Model)', fontsize=12)
    ax.set_ylabel('Benchmark Score', fontsize=12)
    ax.set_title('Benchmark Score Progression Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.05)

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add "saturation" annotation
    ax.annotate('Approaching\nsaturation?',
                xy=(4, 0.95), xytext=(3.2, 0.75),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'benchmark_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'benchmark_timeline.png'}")


def main():
    """Generate all benchmark radar visualizations."""
    print("Generating benchmark radar visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    create_model_comparison_radar()
    create_capability_dimension_radar()
    create_benchmark_coverage_chart()
    create_benchmark_timeline()

    print()
    print("All benchmark radar visualizations generated successfully!")


if __name__ == "__main__":
    main()
