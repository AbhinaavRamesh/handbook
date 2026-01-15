"""
PEFT Methods Comparison Visualization

This script creates radar charts and comparison visualizations for
different parameter-efficient fine-tuning methods.

Output: peft_radar.png, peft_comparison_bar.png, peft_tradeoffs.png
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "fine-tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def create_radar_chart():
    """Create radar chart comparing PEFT methods"""
    # Categories
    categories = ['Performance', 'Memory\nEfficiency', 'Training\nSpeed',
                  'Ease of\nUse', 'Flexibility', 'Inference\nSpeed']
    N = len(categories)

    # Method scores (0-10 scale)
    methods = {
        'LoRA': [9, 8, 8, 9, 8, 9],
        'QLoRA': [8, 10, 7, 8, 7, 8],
        'AdaLoRA': [9, 7, 6, 6, 9, 9],
        'IA3': [7, 9, 9, 8, 6, 10],
        'Full FT': [10, 3, 5, 7, 10, 10],
    }

    colors = {
        'LoRA': '#1976d2',
        'QLoRA': '#4caf50',
        'AdaLoRA': '#ff9800',
        'IA3': '#9c27b0',
        'Full FT': '#f44336',
    }

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot each method
    for method, scores in methods.items():
        values = scores + scores[:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[method])
        ax.fill(angles, values, alpha=0.1, color=colors[method])

    # Customize chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.title('PEFT Methods Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'peft_radar.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'peft_radar.png'}")


def create_comparison_bars():
    """Create bar chart comparing PEFT methods on key metrics"""
    methods = ['Full FT', 'LoRA\n(r=16)', 'LoRA\n(r=64)', 'QLoRA', 'AdaLoRA', 'IA3']

    # Data (relative scales)
    trainable_params = [100, 0.3, 1.0, 0.3, 0.4, 0.05]  # Percentage
    memory_usage = [100, 30, 35, 15, 32, 25]  # Relative to full FT
    performance = [100, 95, 98, 93, 97, 90]  # Percentage of full FT performance

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, trainable_params, width, label='Trainable Params (%)',
                   color='#1976d2', alpha=0.8)
    bars2 = ax.bar(x, memory_usage, width, label='Memory Usage (% of Full FT)',
                   color='#f57c00', alpha=0.8)
    bars3 = ax.bar(x + width, performance, width, label='Performance (% of Full FT)',
                   color='#4caf50', alpha=0.8)

    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('PEFT Methods: Efficiency vs Performance Trade-offs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=10)

    # Add value labels on bars
    def autolabel(bars, fmt='{:.0f}'):
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only label if tall enough
                ax.annotate(fmt.format(height),
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    autolabel(bars1, '{:.1f}')
    autolabel(bars2)
    autolabel(bars3)

    ax.set_ylim(0, 120)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'peft_comparison_bar.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'peft_comparison_bar.png'}")


def create_tradeoff_scatter():
    """Create scatter plot showing parameter efficiency vs performance tradeoff"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Data points: (trainable_params_pct, performance_pct, method_name, memory_gb)
    data = [
        (100, 100, 'Full Fine-Tuning', 56, '#f44336'),
        (0.3, 95, 'LoRA (r=16)', 16, '#1976d2'),
        (0.6, 97, 'LoRA (r=32)', 18, '#1976d2'),
        (1.0, 98, 'LoRA (r=64)', 20, '#1976d2'),
        (0.3, 93, 'QLoRA (r=16)', 6, '#4caf50'),
        (0.6, 95, 'QLoRA (r=32)', 8, '#4caf50'),
        (0.4, 97, 'AdaLoRA', 18, '#ff9800'),
        (0.05, 90, 'IA3', 14, '#9c27b0'),
        (0.5, 94, 'Adapters', 20, '#795548'),
    ]

    for params, perf, name, memory, color in data:
        size = memory * 10  # Scale for visibility
        ax.scatter(params, perf, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=1)
        # Offset labels to avoid overlap
        offset_x = 0.5 if params < 50 else -3
        offset_y = 0.3 if perf > 95 else -0.5
        ax.annotate(name, (params, perf), xytext=(offset_x, offset_y),
                   textcoords='offset points', fontsize=9, ha='left')

    ax.set_xlabel('Trainable Parameters (%)', fontsize=12)
    ax.set_ylabel('Performance (% of Full FT)', fontsize=12)
    ax.set_title('PEFT Methods: Efficiency vs Performance', fontsize=14, fontweight='bold')

    # Add reference lines
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.3)

    # Annotation for sweet spot
    ax.annotate('Sweet Spot\n(High performance,\nlow parameters)',
               xy=(0.6, 97), xytext=(5, 93),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray'),
               bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='#4caf50'))

    # Legend for bubble size
    ax.scatter([], [], s=60, c='gray', alpha=0.5, label='6 GB VRAM')
    ax.scatter([], [], s=200, c='gray', alpha=0.5, label='20 GB VRAM')
    ax.scatter([], [], s=560, c='gray', alpha=0.5, label='56 GB VRAM')
    ax.legend(title='Memory (bubble size)', loc='lower right', fontsize=9)

    ax.set_xlim(-5, 110)
    ax.set_ylim(85, 102)

    # Use log scale for x-axis to better show low parameter methods
    # ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'peft_tradeoffs.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'peft_tradeoffs.png'}")


def create_memory_comparison():
    """Create memory comparison visualization for different model sizes"""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_sizes = ['7B', '13B', '70B']
    x = np.arange(len(model_sizes))
    width = 0.2

    # Memory in GB
    full_ft = [56, 104, 560]
    lora = [16, 28, 160]
    qlora = [6, 10, 48]
    ia3 = [14, 24, 140]

    bars1 = ax.bar(x - 1.5*width, full_ft, width, label='Full FT', color='#f44336')
    bars2 = ax.bar(x - 0.5*width, lora, width, label='LoRA', color='#1976d2')
    bars3 = ax.bar(x + 0.5*width, qlora, width, label='QLoRA', color='#4caf50')
    bars4 = ax.bar(x + 1.5*width, ia3, width, label='IA3', color='#9c27b0')

    ax.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_title('GPU Memory Requirements by Method and Model Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_sizes, fontsize=11)
    ax.legend(fontsize=10)

    # Add reference lines for common GPU sizes
    ax.axhline(y=24, color='#ff9800', linestyle='--', alpha=0.5, label='RTX 3090 (24GB)')
    ax.axhline(y=48, color='#2196f3', linestyle='--', alpha=0.5, label='A6000 (48GB)')
    ax.axhline(y=80, color='#4caf50', linestyle='--', alpha=0.5, label='A100 (80GB)')

    ax.text(2.5, 25, 'RTX 3090', fontsize=9, color='#ff9800')
    ax.text(2.5, 49, 'A6000', fontsize=9, color='#2196f3')
    ax.text(2.5, 81, 'A100', fontsize=9, color='#4caf50')

    ax.set_ylim(0, 600)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'peft_memory_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'peft_memory_comparison.png'}")


if __name__ == "__main__":
    create_radar_chart()
    create_comparison_bars()
    create_tradeoff_scatter()
    create_memory_comparison()
    print("\nAll PEFT comparison visualizations created successfully!")
