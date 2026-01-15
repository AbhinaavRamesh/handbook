#!/usr/bin/env python3
"""
Metric Comparison Chart Visualization

Creates a comparison visualization of different automated metrics for LLM evaluation,
showing their correlation with human judgment across different tasks.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_metric_correlation_heatmap():
    """
    Create a heatmap showing correlation between automated metrics and human judgment.
    """
    # Metrics and tasks
    metrics = ['BLEU', 'ROUGE-L', 'BERTScore', 'Perplexity', 'Semantic Sim']
    tasks = ['Translation', 'Summarization', 'QA', 'Dialogue', 'Code Gen']

    # Correlation data (simulated based on typical findings from literature)
    correlations = np.array([
        [0.75, 0.55, 0.45, 0.35, 0.30],  # BLEU
        [0.60, 0.72, 0.50, 0.40, 0.25],  # ROUGE-L
        [0.82, 0.78, 0.70, 0.65, 0.55],  # BERTScore
        [0.45, 0.40, 0.35, 0.50, 0.60],  # Perplexity (inverted)
        [0.78, 0.75, 0.68, 0.72, 0.50],  # Semantic Similarity
    ])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(correlations, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation with Human Judgment', rotation=-90, va="bottom", fontsize=12)

    # Set ticks
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_yticklabels(metrics, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values as text
    for i in range(len(metrics)):
        for j in range(len(tasks)):
            text_color = "white" if correlations[i, j] < 0.5 else "black"
            ax.text(j, i, f'{correlations[i, j]:.2f}',
                   ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')

    ax.set_title('Automated Metrics vs Human Judgment Correlation\nby Task Type',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metric_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'metric_correlation_heatmap.png'}")


def create_metric_tradeoff_chart():
    """
    Create a scatter plot showing speed vs quality tradeoff for different metrics.
    """
    # Metric data: (name, compute_time_ms, correlation_with_human, annotation)
    metrics_data = [
        ('Exact Match', 0.1, 0.45, 'Fastest\nbut limited'),
        ('BLEU', 1, 0.55, 'Fast\nn-gram based'),
        ('ROUGE', 2, 0.58, 'Good for\nsummarization'),
        ('METEOR', 5, 0.62, 'Better than\nBLEU'),
        ('BERTScore', 50, 0.78, 'Semantic\nmatching'),
        ('BLEURT', 100, 0.82, 'Learned\nmetric'),
        ('GPT-4 Judge', 1000, 0.88, 'LLM-based\nevaluation'),
        ('Human Eval', 60000, 0.95, 'Gold standard\nbut expensive'),
    ]

    names = [m[0] for m in metrics_data]
    times = [m[1] for m in metrics_data]
    correlations = [m[2] for m in metrics_data]
    annotations = [m[3] for m in metrics_data]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot with size based on correlation
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics_data)))
    sizes = [c * 400 for c in correlations]

    scatter = ax.scatter(times, correlations, s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Add labels
    for i, (name, time, corr, annot) in enumerate(metrics_data):
        offset = (10, 10) if i % 2 == 0 else (-10, -10)
        ax.annotate(f'{name}\n({annot})', (time, corr),
                   xytext=offset, textcoords='offset points',
                   fontsize=9, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax.set_xscale('log')
    ax.set_xlabel('Compute Time per Sample (ms)', fontsize=12)
    ax.set_ylabel('Correlation with Human Judgment', fontsize=12)
    ax.set_title('Evaluation Metrics: Speed vs Quality Tradeoff', fontsize=14, fontweight='bold')

    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(0.05, 200000)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add reference lines
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good correlation threshold')
    ax.axvline(x=100, color='orange', linestyle='--', alpha=0.5, label='Real-time threshold')

    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metric_tradeoff_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'metric_tradeoff_chart.png'}")


def create_metric_coverage_bar():
    """
    Create a grouped bar chart showing what each metric captures.
    """
    categories = ['Fluency', 'Adequacy', 'Factuality', 'Coherence', 'Relevance']
    metrics = ['BLEU', 'ROUGE', 'BERTScore', 'LLM Judge']

    # Coverage scores (0-1 scale)
    data = {
        'BLEU':      [0.6, 0.5, 0.2, 0.3, 0.4],
        'ROUGE':     [0.4, 0.7, 0.3, 0.4, 0.6],
        'BERTScore': [0.7, 0.8, 0.5, 0.6, 0.7],
        'LLM Judge': [0.9, 0.85, 0.75, 0.85, 0.9],
    }

    x = np.arange(len(categories))
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    for i, (metric, scores) in enumerate(data.items()):
        offset = width * multiplier
        bars = ax.bar(x + offset, scores, width, label=metric, color=colors[i], alpha=0.8)
        multiplier += 1

    ax.set_xlabel('Quality Dimension', fontsize=12)
    ax.set_ylabel('Coverage Score', fontsize=12)
    ax.set_title('What Different Metrics Capture', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 1.1)

    # Add horizontal line for "good coverage"
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good coverage')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metric_coverage_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'metric_coverage_bar.png'}")


def main():
    """Generate all metric comparison visualizations."""
    print("Generating metric comparison visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    create_metric_correlation_heatmap()
    create_metric_tradeoff_chart()
    create_metric_coverage_bar()

    print()
    print("All metric comparison visualizations generated successfully!")


if __name__ == "__main__":
    main()
