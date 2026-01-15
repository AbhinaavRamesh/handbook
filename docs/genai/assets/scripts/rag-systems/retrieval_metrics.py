#!/usr/bin/env python3
"""
Retrieval Metrics Visualization
Charts for MRR, NDCG, Precision, Recall, and other RAG evaluation metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'images', 'rag-systems')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_mrr_visualization():
    """Visualize MRR calculation with examples."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MRR calculation example
    ax1 = axes[0]

    queries = ['Query 1', 'Query 2', 'Query 3', 'Query 4']
    first_relevant_pos = [1, 3, 2, 5]
    reciprocal_ranks = [1/p for p in first_relevant_pos]

    # Create bar chart showing positions
    colors = ['#4CAF50' if p == 1 else '#FFC107' if p <= 3 else '#F44336'
              for p in first_relevant_pos]

    bars = ax1.bar(queries, first_relevant_pos, color=colors, edgecolor='black', alpha=0.8)

    # Add RR values
    for bar, rr, pos in zip(bars, reciprocal_ranks, first_relevant_pos):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'RR = 1/{pos} = {rr:.2f}', ha='center', fontsize=10)

    ax1.set_ylabel('Position of First Relevant Result')
    ax1.set_xlabel('Query')
    ax1.set_title('MRR Calculation Example', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 7)
    ax1.invert_yaxis()  # Lower position is better
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Position 1 (Best)')

    # Add MRR calculation
    mrr = np.mean(reciprocal_ranks)
    ax1.text(0.5, 0.95, f'MRR = ({" + ".join([f"{rr:.2f}" for rr in reciprocal_ranks])}) / {len(queries)} = {mrr:.3f}',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right: MRR comparison across systems
    ax2 = axes[1]

    systems = ['Baseline\nRAG', 'With\nReranking', 'Hybrid\nSearch', 'Multi-stage\nRetrieval']
    mrr_scores = [0.52, 0.68, 0.72, 0.81]
    colors = ['#9E9E9E', '#4CAF50', '#2196F3', '#9C27B0']

    bars = ax2.bar(systems, mrr_scores, color=colors, edgecolor='black', alpha=0.8)

    for bar, score in zip(bars, mrr_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', fontsize=11, fontweight='bold')

    ax2.set_ylabel('MRR Score')
    ax2.set_title('MRR Comparison Across Retrieval Strategies', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Target')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'mrr_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_ndcg_visualization():
    """Visualize NDCG calculation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: DCG calculation
    ax1 = axes[0]

    # Relevance scores (graded: 0-3)
    positions = [1, 2, 3, 4, 5]
    retrieved_relevance = [2, 3, 1, 0, 2]  # Actual retrieval order
    ideal_relevance = [3, 2, 2, 1, 0]  # Sorted by relevance

    # Calculate gains and discounts
    gains = retrieved_relevance
    discounts = [np.log2(p + 1) for p in positions]
    dcg_terms = [g / d for g, d in zip(gains, discounts)]

    x = np.arange(len(positions))
    width = 0.3

    bars1 = ax1.bar(x - width, gains, width, label='Relevance', color='#4CAF50', alpha=0.8)
    bars2 = ax1.bar(x, discounts, width, label='Discount log2(i+1)', color='#FF9800', alpha=0.8)
    bars3 = ax1.bar(x + width, dcg_terms, width, label='DCG Term', color='#9C27B0', alpha=0.8)

    ax1.set_xlabel('Position')
    ax1.set_ylabel('Value')
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.legend()
    ax1.set_title('DCG Calculation: Relevance / log2(pos+1)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add DCG sum
    dcg = sum(dcg_terms)
    ax1.text(0.5, 0.95, f'DCG = {" + ".join([f"{t:.2f}" for t in dcg_terms])} = {dcg:.2f}',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right: NDCG@k for different k values
    ax2 = axes[1]

    k_values = [1, 3, 5, 10, 20]
    baseline_ndcg = [0.65, 0.72, 0.75, 0.78, 0.80]
    optimized_ndcg = [0.85, 0.88, 0.90, 0.91, 0.92]

    ax2.plot(k_values, baseline_ndcg, 'o-', label='Baseline RAG',
             linewidth=2, markersize=10, color='#9E9E9E')
    ax2.plot(k_values, optimized_ndcg, 's-', label='Optimized RAG',
             linewidth=2, markersize=10, color='#4CAF50')

    ax2.set_xlabel('k (number of results)')
    ax2.set_ylabel('NDCG@k')
    ax2.set_title('NDCG@k: Ranking Quality vs Cutoff', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.0)
    ax2.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Target')

    # Add improvement annotation
    ax2.annotate('+23% at k=5', xy=(5, 0.90), xytext=(7, 0.82),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'ndcg_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_precision_recall_curve():
    """Create precision-recall curves for retrieval systems."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Precision-Recall curve
    ax1 = axes[0]

    # Simulated PR curves for different systems
    recall_points = np.linspace(0, 1, 100)

    # Different systems with different trade-offs
    precision_baseline = 0.9 * np.exp(-2 * recall_points) + 0.1
    precision_optimized = 0.95 * np.exp(-1.5 * recall_points) + 0.15
    precision_hybrid = 0.92 * np.exp(-1.2 * recall_points) + 0.2

    ax1.plot(recall_points, precision_baseline, label='Baseline RAG',
             linewidth=2, color='#9E9E9E')
    ax1.plot(recall_points, precision_optimized, label='With Reranking',
             linewidth=2, color='#4CAF50')
    ax1.plot(recall_points, precision_hybrid, label='Hybrid + Reranking',
             linewidth=2, color='#2196F3')

    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Add F1 iso-lines
    for f1 in [0.2, 0.4, 0.6, 0.8]:
        r = np.linspace(0.01, 1, 100)
        p = f1 * r / (2 * r - f1)
        p = np.clip(p, 0, 1)
        ax1.plot(r, p, '--', color='gray', alpha=0.3, linewidth=1)
        # Add F1 label
        idx = np.argmin(np.abs(p - 0.5))
        if p[idx] > 0 and r[idx] < 1:
            ax1.text(r[idx], p[idx], f'F1={f1}', fontsize=8, color='gray')

    # Right: Metrics at different k values
    ax2 = axes[1]

    k_values = [1, 3, 5, 10, 20]
    precision_at_k = [0.95, 0.85, 0.78, 0.65, 0.52]
    recall_at_k = [0.15, 0.38, 0.55, 0.78, 0.92]
    f1_at_k = [2 * p * r / (p + r) for p, r in zip(precision_at_k, recall_at_k)]

    x = np.arange(len(k_values))
    width = 0.25

    bars1 = ax2.bar(x - width, precision_at_k, width, label='Precision@k',
                    color='#4CAF50', alpha=0.8)
    bars2 = ax2.bar(x, recall_at_k, width, label='Recall@k',
                    color='#2196F3', alpha=0.8)
    bars3 = ax2.bar(x + width, f1_at_k, width, label='F1@k',
                    color='#FF9800', alpha=0.8)

    ax2.set_xlabel('k')
    ax2.set_ylabel('Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_values)
    ax2.set_title('Precision, Recall, F1 at Different k', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'precision_recall_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_ragas_metrics():
    """Create RAGAS framework metrics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sample data for different RAG configurations
    configs = ['Baseline', 'Better Chunking', 'With Reranking', 'Full Pipeline']

    metrics = {
        'Faithfulness': [0.72, 0.75, 0.82, 0.89],
        'Answer Relevance': [0.68, 0.72, 0.78, 0.85],
        'Context Precision': [0.65, 0.78, 0.82, 0.88],
        'Context Recall': [0.70, 0.75, 0.80, 0.86]
    }

    colors = ['#E53935', '#FB8C00', '#43A047', '#1E88E5']

    # Plot 1: Faithfulness over iterations
    ax1 = axes[0, 0]
    bars = ax1.bar(configs, metrics['Faithfulness'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Score')
    ax1.set_title('Faithfulness (Groundedness)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.5)
    ax1.text(3.5, 0.86, 'Target', fontsize=9, color='red')
    for bar, val in zip(bars, metrics['Faithfulness']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10)
    ax1.tick_params(axis='x', rotation=15)

    # Plot 2: Answer Relevance
    ax2 = axes[0, 1]
    bars = ax2.bar(configs, metrics['Answer Relevance'], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Score')
    ax2.set_title('Answer Relevance', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.80, color='red', linestyle='--', alpha=0.5)
    ax2.text(3.5, 0.81, 'Target', fontsize=9, color='red')
    for bar, val in zip(bars, metrics['Answer Relevance']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10)
    ax2.tick_params(axis='x', rotation=15)

    # Plot 3: Context Precision
    ax3 = axes[1, 0]
    bars = ax3.bar(configs, metrics['Context Precision'], color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Score')
    ax3.set_title('Context Precision', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.axhline(y=0.80, color='red', linestyle='--', alpha=0.5)
    ax3.text(3.5, 0.81, 'Target', fontsize=9, color='red')
    for bar, val in zip(bars, metrics['Context Precision']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10)
    ax3.tick_params(axis='x', rotation=15)

    # Plot 4: Radar chart summary
    ax4 = axes[1, 1]

    # Convert to radar chart
    categories = list(metrics.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax4 = plt.subplot(224, projection='polar')

    for i, config in enumerate(configs):
        values = [metrics[cat][i] for cat in categories]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=config, color=colors[i])
        ax4.fill(angles, values, alpha=0.1, color=colors[i])

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.set_ylim(0, 1)
    ax4.set_title('RAGAS Metrics Overview', fontsize=12, fontweight='bold', pad=20)
    ax4.legend(loc='lower right', bbox_to_anchor=(1.3, 0))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'ragas_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    create_mrr_visualization()
    create_ndcg_visualization()
    create_precision_recall_curve()
    create_ragas_metrics()
    print("Retrieval metrics visualizations complete!")
