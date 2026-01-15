#!/usr/bin/env python3
"""
KV Cache Memory Growth Visualization

This script visualizes how KV cache memory grows with sequence length
and compares different optimization strategies (GQA, quantization).

Output: ../images/llm-foundations/kv_cache_*.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Ensure output directory exists
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "llm-foundations"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_kv_cache_size(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    bytes_per_element: int = 2  # fp16
) -> float:
    """Calculate KV cache size in bytes."""
    # K and V each: [batch, num_heads, seq_len, head_dim]
    return 2 * batch_size * num_layers * num_heads * seq_len * head_dim * bytes_per_element


def create_memory_growth_chart():
    """Visualize KV cache memory growth with sequence length."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Model configurations
    models = {
        'LLaMA 7B': {'layers': 32, 'heads': 32, 'head_dim': 128},
        'LLaMA 13B': {'layers': 40, 'heads': 40, 'head_dim': 128},
        'LLaMA 70B': {'layers': 80, 'heads': 64, 'head_dim': 128},
    }

    seq_lengths = np.arange(1000, 130000, 1000)

    # Left: Linear scale
    ax = axes[0]
    for name, config in models.items():
        cache_sizes = [
            calculate_kv_cache_size(
                batch_size=1,
                seq_len=seq_len,
                num_layers=config['layers'],
                num_heads=config['heads'],
                head_dim=config['head_dim']
            ) / 1e9  # Convert to GB
            for seq_len in seq_lengths
        ]
        ax.plot(seq_lengths / 1000, cache_sizes, linewidth=2, label=name)

    # Add GPU memory reference lines
    ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='A100 40GB')
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='A100 80GB')

    ax.set_xlabel('Sequence Length (K tokens)', fontsize=11)
    ax.set_ylabel('KV Cache Size (GB)', fontsize=11)
    ax.set_title('KV Cache Memory Growth', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Show breakdown at specific sequence length
    ax = axes[1]
    seq_len = 32768

    model_names = list(models.keys())
    x = np.arange(len(model_names))
    width = 0.6

    cache_sizes = [
        calculate_kv_cache_size(
            batch_size=1,
            seq_len=seq_len,
            num_layers=models[name]['layers'],
            num_heads=models[name]['heads'],
            head_dim=models[name]['head_dim']
        ) / 1e9
        for name in model_names
    ]

    # Estimate model weights (rough: 2 bytes per param for fp16)
    model_weights = [7 * 2, 13 * 2, 70 * 2]  # GB

    bars1 = ax.bar(x - width / 4, model_weights, width / 2, label='Model Weights', color='steelblue')
    bars2 = ax.bar(x + width / 4, cache_sizes, width / 2, label='KV Cache (32K)', color='coral')

    ax.set_ylabel('Memory (GB)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_title('Memory Breakdown at 32K Context', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Add total annotations
    for i, (w, c) in enumerate(zip(model_weights, cache_sizes)):
        ax.annotate(f'Total: {w + c:.0f}GB', xy=(i, w + c + 2),
                    ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "kv_cache_growth.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved KV cache growth chart to: {output_path}")


def create_optimization_comparison():
    """Compare different KV cache optimization strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Configuration: LLaMA 70B
    base_config = {
        'layers': 80,
        'q_heads': 64,
        'kv_heads': 64,  # MHA baseline
        'head_dim': 128,
        'precision': 'fp16'
    }

    seq_len = 32768
    batch_size = 1

    # Optimization strategies
    strategies = [
        {'name': 'MHA (fp16)', 'kv_heads': 64, 'bytes': 2},
        {'name': 'GQA 8 (fp16)', 'kv_heads': 8, 'bytes': 2},  # LLaMA 2 style
        {'name': 'MHA (int8)', 'kv_heads': 64, 'bytes': 1},
        {'name': 'GQA 8 (int8)', 'kv_heads': 8, 'bytes': 1},
        {'name': 'GQA 8 (int4)', 'kv_heads': 8, 'bytes': 0.5},
    ]

    # Left: Bar chart of cache sizes
    ax = axes[0]

    cache_sizes = []
    colors = []
    color_map = {'fp16': 'steelblue', 'int8': 'coral', 'int4': 'forestgreen'}

    for strat in strategies:
        size = calculate_kv_cache_size(
            batch_size=batch_size,
            seq_len=seq_len,
            num_layers=base_config['layers'],
            num_heads=strat['kv_heads'],
            head_dim=base_config['head_dim'],
            bytes_per_element=strat['bytes']
        ) / 1e9
        cache_sizes.append(size)

        if 'int4' in strat['name']:
            colors.append(color_map['int4'])
        elif 'int8' in strat['name']:
            colors.append(color_map['int8'])
        else:
            colors.append(color_map['fp16'])

    x = np.arange(len(strategies))
    bars = ax.bar(x, cache_sizes, color=colors, alpha=0.8)

    ax.set_ylabel('KV Cache Size (GB)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([s['name'] for s in strategies], rotation=15, ha='right')
    ax.set_title('KV Cache Optimization Strategies\n(LLaMA 70B, 32K context)', fontsize=12, fontweight='bold')

    # Add value labels
    for bar, size in zip(bars, cache_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{size:.1f}GB', ha='center', fontsize=10, fontweight='bold')

    # Add reduction annotations
    baseline = cache_sizes[0]
    for i, size in enumerate(cache_sizes[1:], 1):
        reduction = (baseline - size) / baseline * 100
        ax.annotate(f'-{reduction:.0f}%', xy=(i, size / 2),
                    ha='center', fontsize=9, color='white', fontweight='bold')

    ax.axhline(y=40, color='red', linestyle='--', alpha=0.5)
    ax.text(len(strategies) - 1, 42, 'A100 40GB', fontsize=9, color='red')

    # Right: Batch size capacity
    ax = axes[1]

    gpu_memory = 80  # GB
    model_weights = 140  # GB for 70B in fp16 (but assume tensor parallel)
    available_for_kv = 30  # GB available after model weights

    max_batch_sizes = []
    for strat in strategies:
        per_seq_cache = calculate_kv_cache_size(
            batch_size=1,
            seq_len=seq_len,
            num_layers=base_config['layers'],
            num_heads=strat['kv_heads'],
            head_dim=base_config['head_dim'],
            bytes_per_element=strat['bytes']
        ) / 1e9
        max_batch = int(available_for_kv / per_seq_cache)
        max_batch_sizes.append(max_batch)

    bars = ax.bar(x, max_batch_sizes, color=colors, alpha=0.8)

    ax.set_ylabel('Max Batch Size (32K context)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([s['name'] for s in strategies], rotation=15, ha='right')
    ax.set_title('Batch Size Capacity\n(30GB available for KV cache)', fontsize=12, fontweight='bold')

    # Add value labels
    for bar, bs in zip(bars, max_batch_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{bs}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "kv_cache_optimization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved KV cache optimization comparison to: {output_path}")


def create_gqa_diagram():
    """Create a diagram explaining Grouped Query Attention."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def draw_attention_pattern(ax, q_heads, kv_heads, title):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Draw Q heads
        q_y = 8
        q_spacing = 8 / (q_heads + 1)
        q_positions = [(i * q_spacing + 1, q_y) for i in range(1, q_heads + 1)]

        for x, y in q_positions:
            circle = plt.Circle((x, y), 0.3, color='steelblue', alpha=0.8)
            ax.add_patch(circle)

        ax.text(0.5, q_y, 'Q:', fontsize=11, fontweight='bold', va='center')

        # Draw K/V heads
        kv_y = 4
        kv_spacing = 8 / (kv_heads + 1)
        kv_positions = [(i * kv_spacing + 1, kv_y) for i in range(1, kv_heads + 1)]

        for x, y in kv_positions:
            rect = Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, color='coral', alpha=0.8)
            ax.add_patch(rect)

        ax.text(0.5, kv_y, 'K/V:', fontsize=11, fontweight='bold', va='center')

        # Draw connections
        group_size = q_heads // kv_heads
        for i, (kv_x, kv_y_pos) in enumerate(kv_positions):
            for j in range(group_size):
                q_idx = i * group_size + j
                if q_idx < len(q_positions):
                    q_x, q_y_pos = q_positions[q_idx]
                    ax.plot([q_x, kv_x], [q_y_pos - 0.3, kv_y_pos + 0.3],
                            'k-', alpha=0.3, linewidth=1)

        # Add annotations
        ax.text(5, 1.5, f'{q_heads} Q heads', ha='center', fontsize=10)
        ax.text(5, 0.5, f'{kv_heads} K/V heads', ha='center', fontsize=10)

        ratio = q_heads / kv_heads
        ax.text(5, 2.5, f'KV cache: 1/{ratio:.0f}x', ha='center', fontsize=11,
                fontweight='bold', color='green')

    # MHA: All heads have their own K,V
    draw_attention_pattern(axes[0], 8, 8, 'Multi-Head Attention (MHA)')

    # GQA: Groups of Q heads share K,V
    draw_attention_pattern(axes[1], 8, 2, 'Grouped Query Attention (GQA)')

    # MQA: All Q heads share single K,V
    draw_attention_pattern(axes[2], 8, 1, 'Multi-Query Attention (MQA)')

    # Add legend
    legend_elements = [
        mpatches.Patch(color='steelblue', alpha=0.8, label='Query Head'),
        mpatches.Patch(color='coral', alpha=0.8, label='Key/Value Head'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    output_path = os.path.join(OUTPUT_DIR, "kv_cache_gqa_diagram.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved GQA diagram to: {output_path}")


def create_memory_timeline():
    """Visualize memory usage during generation."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Simulate generation
    prompt_len = 1000
    max_gen = 2000
    total_len = prompt_len + max_gen

    positions = np.arange(total_len)

    # Memory components
    model_weights = np.full(total_len, 70)  # GB, constant

    # KV cache grows linearly
    kv_cache = positions * 0.02  # ~20MB per token for 70B

    # Activations spike during forward pass
    activations = np.zeros(total_len)
    activations[:prompt_len] = 15  # Higher during prompt processing

    # Total
    total = model_weights + kv_cache + activations

    # Plot stacked area
    ax.fill_between(positions, 0, model_weights, alpha=0.7, label='Model Weights', color='steelblue')
    ax.fill_between(positions, model_weights, model_weights + kv_cache, alpha=0.7, label='KV Cache', color='coral')
    ax.fill_between(positions, model_weights + kv_cache, model_weights + kv_cache + activations,
                    alpha=0.7, label='Activations', color='forestgreen')

    # Mark prompt vs generation
    ax.axvline(x=prompt_len, color='black', linestyle='--', linewidth=2)
    ax.annotate('Prompt\nProcessing', xy=(prompt_len / 2, 90), ha='center', fontsize=11)
    ax.annotate('Token\nGeneration', xy=(prompt_len + max_gen / 2, 90), ha='center', fontsize=11)

    # GPU memory limit
    ax.axhline(y=80, color='red', linestyle='-', linewidth=2, label='A100 80GB')

    ax.set_xlabel('Sequence Position', fontsize=11)
    ax.set_ylabel('GPU Memory (GB)', fontsize=11)
    ax.set_title('Memory Usage During LLM Generation (70B Model)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, total_len)
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)

    # Add annotation for OOM risk
    oom_point = (80 - 70) / 0.02  # Where KV cache fills remaining memory
    ax.axvline(x=oom_point, color='red', linestyle=':', alpha=0.5)
    ax.annotate('OOM Risk', xy=(oom_point, 85), fontsize=10, color='red',
                ha='center', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "kv_cache_memory_timeline.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved memory timeline to: {output_path}")


if __name__ == "__main__":
    create_memory_growth_chart()
    create_optimization_comparison()
    create_gqa_diagram()
    create_memory_timeline()
    print("All KV cache visualizations created successfully!")
