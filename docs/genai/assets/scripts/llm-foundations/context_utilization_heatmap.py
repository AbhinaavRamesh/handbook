#!/usr/bin/env python3
"""
Context Utilization Heatmap - Lost in the Middle Visualization

This script creates heatmap visualizations demonstrating the
"lost in the middle" phenomenon in LLM context utilization.

Output: ../images/llm-foundations/context_*.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Ensure output directory exists
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "llm-foundations"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def simulate_recall_curve(
    num_positions: int,
    primacy_strength: float = 0.9,
    recency_strength: float = 0.85,
    middle_floor: float = 0.5
) -> np.ndarray:
    """
    Simulate the U-shaped recall curve observed in LLMs.

    The curve shows high recall at the beginning (primacy)
    and end (recency), with a dip in the middle.
    """
    positions = np.arange(num_positions)
    normalized = positions / (num_positions - 1)  # 0 to 1

    # Primacy effect (exponential decay from start)
    primacy = primacy_strength * np.exp(-5 * normalized)

    # Recency effect (exponential rise toward end)
    recency = recency_strength * np.exp(-5 * (1 - normalized))

    # Combine with a floor
    recall = np.maximum(primacy, recency)
    recall = np.maximum(recall, middle_floor)

    # Add some noise
    noise = np.random.normal(0, 0.03, num_positions)
    recall = np.clip(recall + noise, 0, 1)

    return recall


def create_lost_in_middle_curve():
    """Create the classic U-shaped recall curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Single curve with annotations
    ax = axes[0]

    num_positions = 20
    recall = simulate_recall_curve(num_positions)

    ax.plot(range(1, num_positions + 1), recall, 'b-o', linewidth=2, markersize=8)
    ax.fill_between(range(1, num_positions + 1), recall, alpha=0.3)

    # Annotate regions
    ax.axvspan(1, 5, alpha=0.2, color='green', label='Primacy zone')
    ax.axvspan(num_positions - 4, num_positions, alpha=0.2, color='orange', label='Recency zone')
    ax.axvspan(6, num_positions - 5, alpha=0.2, color='red', label='Lost middle')

    ax.set_xlabel('Document Position in Context', fontsize=11)
    ax.set_ylabel('Recall Accuracy', fontsize=11)
    ax.set_title('"Lost in the Middle" Effect', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower center', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add key insight
    ax.annotate('~30-40% drop\nin accuracy',
                xy=(10, 0.55), xytext=(14, 0.45),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Right: Multiple context lengths
    ax = axes[1]

    context_lengths = [10, 20, 40]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for ctx_len, color in zip(context_lengths, colors):
        recall = simulate_recall_curve(ctx_len)
        positions = np.linspace(0, 1, ctx_len)  # Normalize to 0-1
        ax.plot(positions * 100, recall, '-o', linewidth=2, markersize=6,
                color=color, label=f'{ctx_len} documents', alpha=0.8)

    ax.set_xlabel('Relative Position in Context (%)', fontsize=11)
    ax.set_ylabel('Recall Accuracy', fontsize=11)
    ax.set_title('Effect Across Context Lengths', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "context_lost_in_middle.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved lost in middle curve to: {output_path}")


def create_needle_haystack_heatmap():
    """Create a needle-in-haystack style heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Simulate NIAH results
    # X-axis: context length, Y-axis: needle depth
    context_lengths = [4, 8, 16, 32, 64, 128]  # in K tokens
    depths = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Generate accuracy matrix
    np.random.seed(42)
    accuracy_matrix = np.zeros((len(depths), len(context_lengths)))

    for i, depth in enumerate(depths):
        for j, length in enumerate(context_lengths):
            # Base accuracy affected by depth (U-shape)
            depth_normalized = depth / 100
            u_shape = 1 - 0.4 * np.sin(np.pi * depth_normalized)

            # Longer contexts make retrieval harder
            length_penalty = 0.95 ** (j * 0.5)

            # More variance at middle depths
            variance = 0.1 if 30 <= depth <= 70 else 0.05
            noise = np.random.normal(0, variance)

            accuracy = np.clip(u_shape * length_penalty + noise, 0.3, 1.0)
            accuracy_matrix[i, j] = accuracy

    # Create heatmap
    cmap = LinearSegmentedColormap.from_list('custom',
                                              ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c'])

    im = ax.imshow(accuracy_matrix, cmap=cmap, aspect='auto', vmin=0.3, vmax=1.0)

    # Labels
    ax.set_xticks(range(len(context_lengths)))
    ax.set_xticklabels([f'{l}K' for l in context_lengths], fontsize=10)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f'{d}%' for d in depths], fontsize=10)

    ax.set_xlabel('Context Length (tokens)', fontsize=12)
    ax.set_ylabel('Needle Depth (% into context)', fontsize=12)
    ax.set_title('Needle in a Haystack: Retrieval Accuracy Heatmap', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Retrieval Accuracy', fontsize=11)

    # Add annotations for key regions
    ax.annotate('High accuracy\n(near beginning)', xy=(0, 0), xytext=(2, 0.5),
                fontsize=9, color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white'))

    ax.annotate('Degraded accuracy\n(middle depths)', xy=(3, 5), xytext=(4.5, 3),
                fontsize=9, color='black', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "context_niah_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved NIAH heatmap to: {output_path}")


def create_attention_pattern_visualization():
    """Visualize attention patterns that cause lost-in-middle."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    seq_len = 50

    # Simulated attention patterns
    def create_attention_matrix(pattern_type):
        """Create different attention pattern types."""
        matrix = np.zeros((seq_len, seq_len))

        for i in range(seq_len):
            if pattern_type == 'uniform':
                # Ideal: uniform attention to relevant tokens
                weights = np.ones(i + 1)
                weights = weights / weights.sum()

            elif pattern_type == 'recency_bias':
                # Bias toward recent tokens
                positions = np.arange(i + 1)
                weights = np.exp((positions - i) / 10)
                weights = weights / weights.sum()

            elif pattern_type == 'combined_bias':
                # Both primacy and recency
                positions = np.arange(i + 1)
                primacy = np.exp(-positions / 10)
                recency = np.exp((positions - i) / 10)
                weights = primacy + recency
                weights = weights / weights.sum()

            matrix[i, :i + 1] = weights

        return matrix

    patterns = ['uniform', 'recency_bias', 'combined_bias']
    titles = ['Ideal: Uniform Attention', 'Recency Bias', 'Primacy + Recency Bias']
    descriptions = [
        'All positions attended equally',
        'Recent tokens dominate attention',
        'Start and end get most attention'
    ]

    for ax, pattern, title, desc in zip(axes, patterns, titles, descriptions):
        matrix = create_attention_matrix(pattern)

        im = ax.imshow(matrix, cmap='Blues', aspect='auto')

        ax.set_xlabel('Key Position', fontsize=10)
        ax.set_ylabel('Query Position', fontsize=10)
        ax.set_title(f'{title}', fontsize=11, fontweight='bold')

        # Add description
        ax.text(0.5, -0.12, desc, transform=ax.transAxes, ha='center',
                fontsize=9, style='italic')

        # Highlight diagonal
        ax.plot([0, seq_len - 1], [0, seq_len - 1], 'r--', alpha=0.5, linewidth=1)

    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Attention Weight')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_path = os.path.join(OUTPUT_DIR, "context_attention_patterns.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved attention patterns to: {output_path}")


def create_placement_strategy_comparison():
    """Compare different document placement strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    num_docs = 10
    doc_relevances = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # Ranked by relevance

    strategies = {
        'Chronological': list(range(10)),
        'Most Relevant First': list(range(10)),
        'Edges Strategy': [0, 2, 4, 6, 8, 9, 7, 5, 3, 1],
        'Reverse (Anti-pattern)': list(range(9, -1, -1)),
    }

    strategy_descriptions = {
        'Chronological': 'Natural order (random placement)',
        'Most Relevant First': 'Highest relevance at the start',
        'Edges Strategy': 'Alternate: start, end, start, end...',
        'Reverse (Anti-pattern)': 'Least relevant first (worst)',
    }

    for ax, (strategy_name, order) in zip(axes.flat, strategies.items()):
        # Reorder documents
        if strategy_name == 'Most Relevant First':
            ordered_relevances = sorted(doc_relevances, reverse=True)
        elif strategy_name == 'Edges Strategy':
            ordered_relevances = []
            sorted_docs = sorted(enumerate(doc_relevances), key=lambda x: -x[1])
            for i, (idx, rel) in enumerate(sorted_docs):
                ordered_relevances.append(rel)
            # Interleave: best at 0, 2nd best at end, 3rd at 1, 4th at end-1...
            result = [None] * len(doc_relevances)
            left, right = 0, len(doc_relevances) - 1
            for i, rel in enumerate(sorted(doc_relevances, reverse=True)):
                if i % 2 == 0:
                    result[left] = rel
                    left += 1
                else:
                    result[right] = rel
                    right -= 1
            ordered_relevances = result
        elif strategy_name == 'Reverse (Anti-pattern)':
            ordered_relevances = doc_relevances[::-1]
        else:
            ordered_relevances = doc_relevances

        # Simulate recall based on position and relevance
        position_weights = simulate_recall_curve(num_docs, primacy_strength=1.0,
                                                  recency_strength=0.9, middle_floor=0.6)

        effective_scores = [rel * weight for rel, weight in zip(ordered_relevances, position_weights)]

        # Create bar chart
        positions = np.arange(num_docs)
        colors = plt.cm.RdYlGn(np.array(ordered_relevances) / 10)

        bars = ax.bar(positions, ordered_relevances, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)

        # Overlay position weight line
        ax2 = ax.twinx()
        ax2.plot(positions, position_weights, 'b--', linewidth=2, marker='o',
                 markersize=6, label='Position Weight')
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel('Position Weight', fontsize=9, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Calculate and show total effective score
        total_score = sum(effective_scores)
        ax.set_title(f'{strategy_name}\nEffective Score: {total_score:.1f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Position in Context', fontsize=10)
        ax.set_ylabel('Document Relevance', fontsize=10)
        ax.set_ylim(0, 12)
        ax.set_xticks(positions)

        # Add strategy description
        ax.text(0.5, -0.18, strategy_descriptions[strategy_name],
                transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "context_placement_strategies.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved placement strategies to: {output_path}")


def create_model_comparison():
    """Compare lost-in-middle effect across different models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    num_positions = 20
    positions = np.arange(1, num_positions + 1)

    # Simulated model behaviors
    models = {
        'GPT-4': {'primacy': 0.92, 'recency': 0.88, 'floor': 0.55, 'color': '#74b9ff'},
        'Claude 3': {'primacy': 0.95, 'recency': 0.90, 'floor': 0.60, 'color': '#a29bfe'},
        'LLaMA 3 70B': {'primacy': 0.88, 'recency': 0.82, 'floor': 0.50, 'color': '#55efc4'},
        'Gemini 1.5': {'primacy': 0.93, 'recency': 0.91, 'floor': 0.65, 'color': '#fd79a8'},
    }

    for model_name, params in models.items():
        recall = simulate_recall_curve(
            num_positions,
            primacy_strength=params['primacy'],
            recency_strength=params['recency'],
            middle_floor=params['floor']
        )
        ax.plot(positions, recall, '-o', linewidth=2, markersize=6,
                color=params['color'], label=model_name, alpha=0.8)

    ax.set_xlabel('Document Position', fontsize=11)
    ax.set_ylabel('Retrieval Accuracy', fontsize=11)
    ax.set_title('Lost-in-the-Middle Effect Across Models\n(Illustrative Comparison)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add insight
    ax.text(0.5, 0.05, 'Note: Models with longer native context windows tend to show less severe middle degradation',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "context_model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved model comparison to: {output_path}")


if __name__ == "__main__":
    create_lost_in_middle_curve()
    create_needle_haystack_heatmap()
    create_attention_pattern_visualization()
    create_placement_strategy_comparison()
    create_model_comparison()
    print("All context utilization visualizations created successfully!")
