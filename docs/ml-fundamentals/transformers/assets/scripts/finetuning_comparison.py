#!/usr/bin/env python3
"""
Fine-Tuning Methods Comparison Visualization

Compares Full Fine-tuning, LoRA, Adapter, and Prefix-tuning across
key metrics: Parameters trained, Memory usage, and Performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Data for comparison
methods = ['Full\nFine-Tuning', 'LoRA', 'Adapter', 'Prefix\nTuning']

# Metrics (normalized for visualization, with actual values shown)
params_trained = [100, 0.5, 3, 0.1]  # Percentage of parameters
memory_usage = [100, 10, 15, 8]  # Relative memory (Full FT = 100)
performance = [100, 97, 95, 92]  # Relative performance (Full FT = 100)
inference_speed = [100, 100, 85, 88]  # Relative speed (Full FT = 100)

# Colors
colors = {
    'params': '#2196F3',      # Blue
    'memory': '#4CAF50',      # Green
    'performance': '#FF9800', # Orange
    'speed': '#9C27B0'        # Purple
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fine-Tuning Methods Comparison\nTrade-offs Across Key Metrics',
             fontsize=16, fontweight='bold', y=0.98)

# 1. Parameters Trained (log scale representation)
ax1 = axes[0, 0]
bars1 = ax1.bar(methods, params_trained, color=colors['params'], edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Parameters Trained (%)', fontweight='bold')
ax1.set_title('Trainable Parameters', fontweight='bold')
ax1.set_yscale('log')
ax1.set_ylim(0.05, 200)
for bar, val in zip(bars1, params_trained):
    height = bar.get_height()
    ax1.annotate(f'{val}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=11)

# 2. Memory Usage
ax2 = axes[0, 1]
bars2 = ax2.bar(methods, memory_usage, color=colors['memory'], edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Memory Usage (Relative %)', fontweight='bold')
ax2.set_title('GPU Memory Requirements', fontweight='bold')
ax2.set_ylim(0, 120)
for bar, val in zip(bars2, memory_usage):
    height = bar.get_height()
    ax2.annotate(f'{val}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add memory in GB annotations (for BERT-base example)
memory_gb = [7.0, 0.7, 1.0, 0.5]
for bar, gb in zip(bars2, memory_gb):
    ax2.annotate(f'~{gb}GB',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()/2),
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# 3. Performance Retention
ax3 = axes[1, 0]
bars3 = ax3.bar(methods, performance, color=colors['performance'], edgecolor='black', linewidth=1.2)
ax3.set_ylabel('Performance Retention (%)', fontweight='bold')
ax3.set_title('Task Performance vs Full Fine-Tuning', fontweight='bold')
ax3.set_ylim(85, 105)
ax3.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Full FT Baseline')
for bar, val in zip(bars3, performance):
    height = bar.get_height()
    ax3.annotate(f'{val}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=11)
ax3.legend(loc='lower right')

# 4. Inference Speed
ax4 = axes[1, 1]
bars4 = ax4.bar(methods, inference_speed, color=colors['speed'], edgecolor='black', linewidth=1.2)
ax4.set_ylabel('Inference Speed (Relative %)', fontweight='bold')
ax4.set_title('Inference Latency Impact', fontweight='bold')
ax4.set_ylim(75, 110)
ax4.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline Speed')
for bar, val in zip(bars4, inference_speed):
    height = bar.get_height()
    ax4.annotate(f'{val}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=11)
ax4.legend(loc='lower right')

# Add text box with recommendations
textstr = ('Recommendation:\n'
           '- Rapid prototyping: LoRA (best efficiency)\n'
           '- Modular systems: Adapter (composable)\n'
           '- Few-shot learning: Prefix Tuning\n'
           '- Maximum accuracy: Full Fine-Tuning')

# Adjust layout
plt.tight_layout(rect=[0, 0.02, 1, 0.95])

# Save figure
output_path = os.path.join(output_dir, 'finetuning_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved to: {output_path}")
