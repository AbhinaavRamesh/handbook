"""
LLM Cost Analysis Visualization

Creates visualizations for LLM inference cost breakdown, model comparisons,
and cost optimization impact analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Output directory
OUTPUT_DIR = "/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/llmops"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_cost_breakdown_chart():
    """Create a cost breakdown chart showing components of LLM inference costs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Pie chart of cost components
    ax1 = axes[0]
    components = ['Input Tokens', 'Output Tokens', 'GPU Compute', 'Infrastructure\nOverhead']
    sizes = [25, 45, 20, 10]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    explode = (0, 0.05, 0, 0)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=components,
                                        colors=colors, autopct='%1.0f%%',
                                        startangle=90, pctdistance=0.75)

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax1.set_title('LLM Inference Cost Breakdown', fontsize=14, fontweight='bold', pad=20)

    # Right: Bar chart of cost by model
    ax2 = axes[1]
    models = ['GPT-4\nTurbo', 'GPT-4o', 'Claude 3\nSonnet', 'Llama 3\n70B*', 'Llama 3\n8B*']
    input_costs = [10, 2.5, 3, 0.8, 0.1]
    output_costs = [30, 10, 15, 0.8, 0.1]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax2.bar(x - width/2, input_costs, width, label='Input ($/1M tokens)',
                    color='#2196F3', edgecolor='white')
    bars2 = ax2.bar(x + width/2, output_costs, width, label='Output ($/1M tokens)',
                    color='#FF9800', edgecolor='white')

    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Cost per 1M Tokens ($)', fontsize=12)
    ax2.set_title('Cost Comparison by Model', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_yscale('log')
    ax2.set_ylim(0.05, 100)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'${height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    ax2.text(0.02, 0.02, '*Self-hosted pricing estimate',
             transform=ax2.transAxes, fontsize=8, style='italic', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cost_breakdown.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cost breakdown chart to {OUTPUT_DIR}")


def create_cost_optimization_impact():
    """Create a chart showing impact of various cost optimization strategies."""
    fig, ax = plt.subplots(figsize=(12, 7))

    strategies = [
        'Baseline\n(No Optimization)',
        'Prompt\nCompression',
        'Semantic\nCaching',
        'Model\nRouting',
        'Batch\nProcessing',
        'Self-Hosting\n(High Volume)',
        'Full\nOptimization'
    ]

    # Monthly costs for 10M tokens/month workload
    costs = [10000, 8500, 6500, 5500, 5000, 3500, 2500]
    savings_pct = [0, 15, 35, 45, 50, 65, 75]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(strategies)))

    bars = ax.bar(strategies, costs, color=colors, edgecolor='white', linewidth=2)

    # Add cost labels
    for i, (bar, cost, saving) in enumerate(zip(bars, costs, savings_pct)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'${cost:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        if saving > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'-{saving}%', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')

    ax.set_xlabel('Optimization Strategy', fontsize=12)
    ax.set_ylabel('Monthly Cost ($)', fontsize=12)
    ax.set_title('Cost Optimization Impact\n(10M tokens/month workload)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 12000)
    ax.grid(axis='y', alpha=0.3)

    # Add savings annotation
    ax.annotate('', xy=(6, 2500), xytext=(0, 10000),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=3))
    ax.text(3, 7500, 'Up to 75%\nCost Reduction',
            fontsize=12, fontweight='bold', color='#388E3C',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cost_optimization_impact.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cost optimization impact chart to {OUTPUT_DIR}")


def create_self_hosted_breakeven():
    """Create a chart showing self-hosted vs API break-even analysis."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # X-axis: Monthly token volume (millions)
    volumes = np.linspace(0, 500, 100)

    # API costs (using $10/1M as average)
    api_cost = volumes * 10

    # Self-hosted costs (fixed + variable)
    # Fixed: ~$15K/month for 4x A100 cluster
    # Variable: minimal compute costs
    fixed_cost = 15
    self_hosted_cost = np.full_like(volumes, fixed_cost)

    # Find break-even point
    breakeven_idx = np.argmin(np.abs(api_cost - self_hosted_cost))
    breakeven_volume = volumes[breakeven_idx]

    ax.plot(volumes, api_cost, 'b-', linewidth=2.5, label='API Provider')
    ax.plot(volumes, self_hosted_cost, 'g-', linewidth=2.5, label='Self-Hosted (4x A100)')
    ax.fill_between(volumes, api_cost, self_hosted_cost,
                    where=(api_cost > self_hosted_cost),
                    alpha=0.3, color='green', label='Self-Host Savings')
    ax.fill_between(volumes, api_cost, self_hosted_cost,
                    where=(api_cost <= self_hosted_cost),
                    alpha=0.3, color='blue', label='API Savings')

    # Mark break-even point
    ax.axvline(x=breakeven_volume, color='red', linestyle='--', linewidth=1.5)
    ax.scatter([breakeven_volume], [fixed_cost], color='red', s=100, zorder=5)
    ax.annotate(f'Break-even:\n~{breakeven_volume:.0f}M tokens/mo',
                xy=(breakeven_volume, fixed_cost),
                xytext=(breakeven_volume + 50, fixed_cost + 20),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Monthly Token Volume (Millions)', fontsize=12)
    ax.set_ylabel('Monthly Cost ($K)', fontsize=12)
    ax.set_title('API vs Self-Hosted Cost Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    # Add zones annotation
    ax.text(breakeven_volume/2, 5, 'Use APIs', fontsize=11,
            ha='center', fontweight='bold', color='#1976D2')
    ax.text(breakeven_volume + 150, 5, 'Self-Host', fontsize=11,
            ha='center', fontweight='bold', color='#388E3C')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'self_hosted_breakeven.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved self-hosted break-even chart to {OUTPUT_DIR}")


def create_token_distribution():
    """Create a histogram showing typical token distribution in requests."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    np.random.seed(42)

    # Left: Input token distribution
    ax1 = axes[0]
    input_tokens = np.concatenate([
        np.random.lognormal(5.5, 0.8, 5000),  # General queries
        np.random.lognormal(7, 0.5, 2000),     # RAG with context
        np.random.lognormal(8, 0.3, 500),      # Long documents
    ])
    input_tokens = np.clip(input_tokens, 10, 10000)

    ax1.hist(input_tokens, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
    ax1.axvline(x=np.median(input_tokens), color='red', linestyle='--',
                linewidth=2, label=f'Median: {np.median(input_tokens):.0f}')
    ax1.axvline(x=np.percentile(input_tokens, 95), color='orange', linestyle='--',
                linewidth=2, label=f'P95: {np.percentile(input_tokens, 95):.0f}')
    ax1.set_xlabel('Input Tokens', fontsize=12)
    ax1.set_ylabel('Request Count', fontsize=12)
    ax1.set_title('Input Token Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 5000)

    # Right: Output token distribution
    ax2 = axes[1]
    output_tokens = np.concatenate([
        np.random.lognormal(4, 0.6, 4000),     # Short answers
        np.random.lognormal(5.5, 0.7, 3000),   # Medium responses
        np.random.lognormal(6.5, 0.5, 500),    # Long responses
    ])
    output_tokens = np.clip(output_tokens, 5, 4000)

    ax2.hist(output_tokens, bins=50, color='#FF9800', edgecolor='white', alpha=0.8)
    ax2.axvline(x=np.median(output_tokens), color='red', linestyle='--',
                linewidth=2, label=f'Median: {np.median(output_tokens):.0f}')
    ax2.axvline(x=np.percentile(output_tokens, 95), color='purple', linestyle='--',
                linewidth=2, label=f'P95: {np.percentile(output_tokens, 95):.0f}')
    ax2.set_xlabel('Output Tokens', fontsize=12)
    ax2.set_ylabel('Request Count', fontsize=12)
    ax2.set_title('Output Token Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 2000)

    plt.suptitle('Typical LLM Request Token Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'token_distribution.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved token distribution chart to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_cost_breakdown_chart()
    create_cost_optimization_impact()
    create_self_hosted_breakeven()
    create_token_distribution()
    print("All cost visualization charts created successfully!")
