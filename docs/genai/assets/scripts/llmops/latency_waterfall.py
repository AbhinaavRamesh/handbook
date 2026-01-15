"""
LLM Latency Waterfall Visualization

Creates visualizations showing the breakdown of latency components in LLM inference,
including TTFT analysis and optimization impact.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Output directory
OUTPUT_DIR = "/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/llmops"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_latency_waterfall():
    """Create a waterfall chart showing latency breakdown."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Components and their durations (in ms)
    components = [
        ('Network\nIngress', 20),
        ('Queue\nWait', 50),
        ('Auth &\nRate Limit', 5),
        ('Cache\nLookup', 15),
        ('Tokenization', 10),
        ('Prefill\n(TTFT)', 200),
        ('Decode\n(Generation)', 1500),
        ('Detokenization', 5),
        ('Network\nEgress', 15),
    ]

    labels = [c[0] for c in components]
    durations = [c[1] for c in components]

    # Calculate cumulative for waterfall
    cumulative = np.cumsum([0] + durations[:-1])

    # Colors by category
    colors = ['#90CAF9', '#90CAF9', '#FFCC80', '#A5D6A7', '#CE93D8',
              '#EF9A9A', '#EF9A9A', '#CE93D8', '#90CAF9']

    # Draw bars
    bars = ax.barh(labels, durations, left=cumulative, color=colors,
                   edgecolor='white', linewidth=2, height=0.6)

    # Add duration labels
    for i, (bar, duration, start) in enumerate(zip(bars, durations, cumulative)):
        # Label inside bar if big enough
        if duration > 50:
            ax.text(start + duration/2, i, f'{duration}ms',
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        else:
            ax.text(start + duration + 10, i, f'{duration}ms',
                   ha='left', va='center', fontsize=8)

    # Mark TTFT point
    ttft = sum(durations[:6])
    ax.axvline(x=ttft, color='red', linestyle='--', linewidth=2)
    ax.text(ttft + 20, len(labels) - 1, f'TTFT: {ttft}ms',
            fontsize=11, fontweight='bold', color='red', va='center')

    # Total latency
    total = sum(durations)
    ax.axvline(x=total, color='#1565C0', linestyle='-', linewidth=2)
    ax.text(total + 20, 0, f'Total: {total}ms',
            fontsize=11, fontweight='bold', color='#1565C0', va='center')

    ax.set_xlabel('Time (milliseconds)', fontsize=12)
    ax.set_title('LLM Request Latency Waterfall', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2000)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#90CAF9', label='Network'),
        mpatches.Patch(color='#FFCC80', label='Gateway'),
        mpatches.Patch(color='#A5D6A7', label='Caching'),
        mpatches.Patch(color='#CE93D8', label='Processing'),
        mpatches.Patch(color='#EF9A9A', label='Inference'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_waterfall.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved latency waterfall chart to {OUTPUT_DIR}")


def create_ttft_vs_tps_comparison():
    """Create a chart comparing TTFT vs TPS across scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: TTFT by scenario
    ax1 = axes[0]
    scenarios = ['Small Model\n(7B)', 'Medium Model\n(70B)', 'Large Model\n(70B, 4GPU)',
                 'Quantized\n(70B INT4)', 'Cached\nPrefix']
    ttft_values = [100, 350, 200, 150, 50]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(scenarios)))

    bars1 = ax1.bar(scenarios, ttft_values, color=colors, edgecolor='white', linewidth=2)
    ax1.axhline(y=500, color='red', linestyle='--', linewidth=1.5, label='SLO: 500ms')
    ax1.set_ylabel('TTFT (milliseconds)', fontsize=12)
    ax1.set_title('Time to First Token (TTFT)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 600)
    ax1.legend(fontsize=9)

    for bar, val in zip(bars1, ttft_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 15, f'{val}ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right: TPS by scenario
    ax2 = axes[1]
    tps_values = [120, 30, 80, 100, 80]
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(scenarios)))

    bars2 = ax2.bar(scenarios, tps_values, color=colors, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Tokens Per Second (TPS)', fontsize=12)
    ax2.set_title('Token Generation Rate (TPS)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 150)

    for bar, val in zip(bars2, tps_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 3, f'{val}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('TTFT vs TPS Trade-offs', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ttft_vs_tps.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved TTFT vs TPS comparison to {OUTPUT_DIR}")


def create_optimization_impact():
    """Create a chart showing latency impact of various optimizations."""
    fig, ax = plt.subplots(figsize=(12, 7))

    optimizations = [
        'Baseline',
        '+ Quantization\n(INT8)',
        '+ Flash\nAttention',
        '+ Continuous\nBatching',
        '+ Tensor\nParallelism',
        '+ Speculative\nDecoding',
        'Full Stack\nOptimized'
    ]

    # Latency values (ms)
    ttft = [500, 400, 350, 350, 200, 200, 150]
    total = [3000, 2400, 2200, 2000, 1500, 1000, 800]

    x = np.arange(len(optimizations))
    width = 0.35

    bars1 = ax.bar(x - width/2, ttft, width, label='TTFT (ms)',
                   color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x + width/2, total, width, label='Total Latency (ms)',
                   color='#FF9800', edgecolor='white')

    ax.set_xlabel('Optimization Stack', fontsize=12)
    ax.set_ylabel('Latency (milliseconds)', fontsize=12)
    ax.set_title('Cumulative Impact of Latency Optimizations\n(70B Model, 500 output tokens)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(optimizations, fontsize=9)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 3500)
    ax.grid(axis='y', alpha=0.3)

    # Add improvement percentages
    for i in range(1, len(optimizations)):
        ttft_imp = (ttft[0] - ttft[i]) / ttft[0] * 100
        total_imp = (total[0] - total[i]) / total[0] * 100
        ax.text(i, max(ttft[i], total[i]) + 100,
                f'-{total_imp:.0f}%', ha='center', fontsize=8, color='#E65100')

    # Arrow showing total improvement
    ax.annotate('', xy=(6, 800), xytext=(0, 3000),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=3))
    ax.text(3, 2500, '73% Latency\nReduction',
            fontsize=12, fontweight='bold', color='#388E3C',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_optimization_impact.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved latency optimization impact chart to {OUTPUT_DIR}")


def create_quantization_comparison():
    """Create a comparison of different quantization methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods = ['FP16\n(Baseline)', 'INT8', 'INT4\n(GPTQ)', 'INT4\n(AWQ)', 'FP8']

    # Left: Latency comparison
    ax1 = axes[0]
    latency = [200, 160, 120, 115, 140]
    colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#64B5F6']

    bars1 = ax1.bar(methods, latency, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Inference Latency by Quantization', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 250)

    for bar, val in zip(bars1, latency):
        improvement = (200 - val) / 200 * 100
        ax1.text(bar.get_x() + bar.get_width()/2, val + 5,
                f'{val}ms\n(-{improvement:.0f}%)',
                ha='center', va='bottom', fontsize=8)

    # Right: Quality vs Memory trade-off
    ax2 = axes[1]
    memory = [140, 70, 35, 35, 70]  # GB
    quality = [100, 99, 96, 97, 99]  # % of FP16 quality

    scatter = ax2.scatter(memory, quality, s=300, c=colors, edgecolor='black', linewidth=2)

    for i, method in enumerate(methods):
        ax2.annotate(method.replace('\n', ' '), (memory[i], quality[i]),
                    xytext=(10, 10), textcoords='offset points', fontsize=9)

    ax2.set_xlabel('Model Memory (GB)', fontsize=12)
    ax2.set_ylabel('Quality (% of FP16)', fontsize=12)
    ax2.set_title('Quality vs Memory Trade-off', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 160)
    ax2.set_ylim(94, 101)
    ax2.grid(alpha=0.3)

    # Add Pareto frontier
    ax2.plot([35, 70, 140], [97, 99, 100], 'r--', linewidth=1.5, alpha=0.7)
    ax2.text(80, 98, 'Pareto\nFrontier', fontsize=9, color='red', style='italic')

    plt.suptitle('Quantization Method Comparison (70B Model)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'quantization_comparison.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved quantization comparison chart to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_latency_waterfall()
    create_ttft_vs_tps_comparison()
    create_optimization_impact()
    create_quantization_comparison()
    print("All latency visualization charts created successfully!")
