"""
LLM Caching Performance Visualization

Creates visualizations for cache hit rates, cache types comparison,
and the impact of caching on performance and cost.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Output directory
OUTPUT_DIR = "/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/llmops"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_cache_hit_rate_chart():
    """Create a chart showing cache hit rates over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Simulate 24 hours of cache data
    hours = np.arange(24)
    np.random.seed(42)

    # Different cache types
    exact_hits = 10 + 5 * np.sin(hours * np.pi / 12) + np.random.normal(0, 1, 24)
    semantic_hits = 25 + 8 * np.sin(hours * np.pi / 12) + np.random.normal(0, 2, 24)
    kv_cache_hits = 35 + 10 * np.sin(hours * np.pi / 12) + np.random.normal(0, 2, 24)
    misses = 100 - exact_hits - semantic_hits - kv_cache_hits

    ax.stackplot(hours, exact_hits, semantic_hits, kv_cache_hits, misses,
                 labels=['Exact Match', 'Semantic Cache', 'KV-Cache', 'Cache Miss'],
                 colors=['#4CAF50', '#2196F3', '#FF9800', '#F44336'],
                 alpha=0.8)

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Percentage of Requests', fontsize=12)
    ax.set_title('Cache Hit Rate Distribution Over 24 Hours', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add average hit rate annotation
    total_hit_rate = (exact_hits + semantic_hits + kv_cache_hits).mean()
    ax.axhline(y=total_hit_rate, color='white', linestyle='--', linewidth=2)
    ax.text(12, total_hit_rate + 3, f'Avg Total Hit Rate: {total_hit_rate:.1f}%',
            ha='center', fontsize=10, fontweight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor='#333', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cache_hit_rate.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cache hit rate chart to {OUTPUT_DIR}")


def create_cache_type_comparison():
    """Create a comparison of different caching strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cache_types = ['Exact\nMatch', 'Semantic\nCache', 'KV-Cache\nSharing', 'Prompt\nCache']

    # Left: Hit Rate
    ax1 = axes[0]
    hit_rates = [15, 35, 50, 60]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

    bars1 = ax1.bar(cache_types, hit_rates, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Hit Rate (%)', fontsize=12)
    ax1.set_title('Cache Hit Rate', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 80)

    for bar, val in zip(bars1, hit_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Middle: Latency Savings
    ax2 = axes[1]
    latency_savings = [95, 90, 40, 50]

    bars2 = ax2.bar(cache_types, latency_savings, color=colors, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Latency Reduction (%)', fontsize=12)
    ax2.set_title('Latency Savings', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)

    for bar, val in zip(bars2, latency_savings):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Right: Implementation Complexity
    ax3 = axes[2]
    complexity = [1, 3, 2, 1.5]  # 1=Easy, 5=Hard
    complexity_labels = ['Low', 'Medium', 'Medium', 'Low-Med']

    bars3 = ax3.bar(cache_types, complexity, color=colors, edgecolor='white', linewidth=2)
    ax3.set_ylabel('Complexity (1-5)', fontsize=12)
    ax3.set_title('Implementation Complexity', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 5)

    for bar, val, label in zip(bars3, complexity, complexity_labels):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.1, label,
                ha='center', va='bottom', fontsize=9)

    plt.suptitle('Caching Strategy Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cache_type_comparison.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cache type comparison chart to {OUTPUT_DIR}")


def create_semantic_similarity_distribution():
    """Create a histogram of semantic similarity scores for cache lookups."""
    fig, ax = plt.subplots(figsize=(12, 6))

    np.random.seed(42)

    # Simulate similarity scores
    # Most queries are somewhat similar (bell curve around 0.7)
    # Some are very similar (near 1.0) - these are cache hits
    similarities = np.concatenate([
        np.random.beta(2, 5, 3000) * 0.6,  # Low similarity (0-0.6)
        np.random.beta(5, 2, 2000) * 0.3 + 0.6,  # Medium similarity (0.6-0.9)
        np.random.beta(5, 1.5, 1000) * 0.1 + 0.9,  # High similarity (0.9-1.0)
    ])

    # Create histogram
    n, bins, patches = ax.hist(similarities, bins=50, edgecolor='white', alpha=0.8)

    # Color by threshold
    threshold = 0.95
    for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
        if left_edge >= threshold:
            patch.set_facecolor('#4CAF50')  # Hit (green)
        else:
            patch.set_facecolor('#2196F3')  # Miss (blue)

    # Add threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
    ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.9, f'Threshold: {threshold}',
            fontsize=11, fontweight='bold', color='red')

    # Calculate hit rate
    hit_rate = (similarities >= threshold).mean() * 100
    ax.text(0.98, 0.95, f'Hit Rate: {hit_rate:.1f}%',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#4CAF50'))

    ax.set_xlabel('Semantic Similarity Score', fontsize=12)
    ax.set_ylabel('Query Count', fontsize=12)
    ax.set_title('Distribution of Semantic Similarity Scores\nfor Cache Lookups', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label='Cache Hit'),
        mpatches.Patch(color='#2196F3', label='Cache Miss'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'semantic_similarity_distribution.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved semantic similarity distribution chart to {OUTPUT_DIR}")


def create_cache_impact_analysis():
    """Create a chart showing the overall impact of caching."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Before/After comparison
    ax1 = axes[0]
    metrics = ['Avg Latency\n(ms)', 'Monthly Cost\n($K)', 'GPU\nUtilization (%)']
    before = [2500, 50, 85]
    after = [800, 30, 60]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, before, width, label='Without Caching',
                    color='#F44336', edgecolor='white')
    bars2 = ax1.bar(x + width/2, after, width, label='With Caching',
                    color='#4CAF50', edgecolor='white')

    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Impact of Caching Strategy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.legend(loc='upper right', fontsize=10)

    # Add improvement annotations
    improvements = [(before[i] - after[i]) / before[i] * 100 for i in range(len(before))]
    for i, (b1, b2, imp) in enumerate(zip(bars1, bars2, improvements)):
        ax1.text(i, max(before[i], after[i]) + 100, f'-{imp:.0f}%',
                ha='center', fontsize=10, fontweight='bold', color='#388E3C')

    # Right: ROI over time
    ax2 = axes[1]
    months = np.arange(1, 13)
    cache_cost = 5  # $5K/month for cache infrastructure
    savings_per_month = 20  # $20K/month in LLM cost savings
    cumulative_savings = months * savings_per_month - months * cache_cost
    cumulative_cost = months * cache_cost

    ax2.fill_between(months, cumulative_savings, 0,
                     where=(cumulative_savings > 0),
                     alpha=0.3, color='#4CAF50', label='Net Savings')
    ax2.plot(months, cumulative_savings, 'g-', linewidth=2.5, label='Cumulative Savings')
    ax2.plot(months, cumulative_cost, 'r--', linewidth=2, label='Cache Infrastructure Cost')

    # Break-even point
    breakeven = cache_cost / (savings_per_month - cache_cost)
    ax2.axvline(x=breakeven, color='orange', linestyle=':', linewidth=2)
    ax2.scatter([breakeven], [0], color='orange', s=100, zorder=5)
    ax2.text(breakeven + 0.5, 10, f'Break-even:\n{breakeven:.1f} months',
            fontsize=10, color='orange')

    ax2.set_xlabel('Months', fontsize=12)
    ax2.set_ylabel('Cumulative Value ($K)', fontsize=12)
    ax2.set_title('Caching ROI Over Time', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_xlim(0.5, 12.5)
    ax2.grid(alpha=0.3)

    plt.suptitle('Caching Strategy Business Impact', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cache_impact_analysis.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cache impact analysis chart to {OUTPUT_DIR}")


def create_cache_architecture_flow():
    """Create a flow diagram showing cache lookup process."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'LLM Cache Lookup Flow', fontsize=16, fontweight='bold',
            ha='center', va='center')

    # Define boxes
    boxes = [
        ('Request', 1, 5.5, '#E3F2FD'),
        ('Exact\nMatch', 3.5, 5.5, '#C8E6C9'),
        ('Semantic\nSearch', 6, 5.5, '#BBDEFB'),
        ('KV-Cache\nLookup', 8.5, 5.5, '#FFE0B2'),
        ('Model\nInference', 11, 5.5, '#FFCDD2'),
        ('Response', 13, 5.5, '#E1BEE7'),
    ]

    for name, x, y, color in boxes:
        rect = mpatches.FancyBboxPatch((x-0.8, y-0.5), 1.6, 1,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='#666',
                                        linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='#666', lw=1.5, mutation_scale=15)

    # Main flow
    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + 0.8
        x2 = boxes[i+1][1] - 0.8
        y = boxes[i][2]
        ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=arrow_style)

    # Cache hit paths (bypass to response)
    hit_style = dict(arrowstyle='->', color='#4CAF50', lw=2,
                     connectionstyle='arc3,rad=-0.5', mutation_scale=12)

    # Exact hit
    ax.annotate('', xy=(13-0.8, 5.5+0.3), xytext=(3.5+0.8, 5.5+0.3),
                arrowprops=hit_style)
    ax.text(8, 6.5, 'Exact Hit', fontsize=8, color='#4CAF50', ha='center')

    # Semantic hit
    ax.annotate('', xy=(13-0.8, 5.5+0.1), xytext=(6+0.8, 5.5+0.1),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2,
                               connectionstyle='arc3,rad=-0.3', mutation_scale=12))
    ax.text(9.5, 6.2, 'Semantic Hit', fontsize=8, color='#2196F3', ha='center')

    # KV-cache partial hit
    ax.annotate('', xy=(11-0.8, 5.5-0.1), xytext=(8.5+0.8, 5.5-0.1),
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2, mutation_scale=12))
    ax.text(9.75, 4.8, 'Partial KV', fontsize=8, color='#FF9800', ha='center')

    # Add hit rate annotations
    annotations = [
        ('~15%', 3.5, 4.5, '#4CAF50'),
        ('~25%', 6, 4.5, '#2196F3'),
        ('~30%', 8.5, 4.5, '#FF9800'),
        ('~30%', 11, 4.5, '#F44336'),
    ]

    for text, x, y, color in annotations:
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                fontweight='bold', color=color)

    # Legend at bottom
    ax.text(7, 3.5, 'Cache Hit Rates (Typical)', fontsize=11,
            ha='center', fontweight='bold')

    # Add latency info
    latencies = [
        ('5ms', 3.5),
        ('20ms', 6),
        ('50ms', 8.5),
        ('500ms+', 11),
    ]

    ax.text(7, 2.5, 'Typical Latency:', fontsize=10, ha='center')
    for lat, x in latencies:
        ax.text(x, 2, lat, ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cache_architecture_flow.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cache architecture flow chart to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_cache_hit_rate_chart()
    create_cache_type_comparison()
    create_semantic_similarity_distribution()
    create_cache_impact_analysis()
    create_cache_architecture_flow()
    print("All caching visualization charts created successfully!")
