#!/usr/bin/env python3
"""
Quantization Impact Visualization

Shows the effects of Float32 vs INT8 vs INT4 quantization on:
- Memory usage
- Inference speed
- Accuracy retention
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Data
precisions = ['Float32\n(Baseline)', 'Float16', 'INT8', 'INT4']

# Memory usage (relative, Float32 = 100)
memory = [100, 50, 25, 12.5]
# Inference speed (relative throughput, Float32 = 100)
speed = [100, 150, 250, 300]
# Accuracy retention (percentage)
accuracy = [100, 99.8, 99.2, 97.5]

# Actual memory for BERT-base (110M params)
memory_gb = [0.44, 0.22, 0.11, 0.055]  # GB for weights only

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Impact of Quantization on Transformer Models\n(BERT-base, 110M parameters)',
             fontsize=14, fontweight='bold', y=1.02)

# Color scheme
colors = ['#1976D2', '#42A5F5', '#4CAF50', '#8BC34A']

# 1. Memory Usage
ax1 = axes[0]
bars1 = ax1.bar(precisions, memory, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Memory Usage (%)', fontweight='bold')
ax1.set_title('Memory Reduction', fontweight='bold', fontsize=12)
ax1.set_ylim(0, 120)

# Add reduction labels
for i, (bar, mem, gb) in enumerate(zip(bars1, memory, memory_gb)):
    height = bar.get_height()
    ax1.annotate(f'{mem}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.annotate(f'({gb:.2f} GB)',
                xy=(bar.get_x() + bar.get_width() / 2, height/2),
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Add reduction annotations
ax1.annotate('2x\nreduction', xy=(1.5, 75), fontsize=10, ha='center',
            color='green', fontweight='bold')
ax1.annotate('4x\nreduction', xy=(2.5, 37), fontsize=10, ha='center',
            color='green', fontweight='bold')
ax1.annotate('8x\nreduction', xy=(3.3, 20), fontsize=10, ha='center',
            color='green', fontweight='bold')

# 2. Inference Speed
ax2 = axes[1]
bars2 = ax2.bar(precisions, speed, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Relative Throughput (%)', fontweight='bold')
ax2.set_title('Inference Speed Improvement', fontweight='bold', fontsize=12)
ax2.set_ylim(0, 350)

for bar, spd in zip(bars2, speed):
    height = bar.get_height()
    ax2.annotate(f'{spd}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    speedup = spd / 100
    ax2.annotate(f'{speedup:.1f}x',
                xy=(bar.get_x() + bar.get_width() / 2, height/2),
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# 3. Accuracy Retention
ax3 = axes[2]
bars3 = ax3.bar(precisions, accuracy, color=colors, edgecolor='black', linewidth=1.2)
ax3.set_ylabel('Accuracy Retention (%)', fontweight='bold')
ax3.set_title('Model Accuracy', fontweight='bold', fontsize=12)
ax3.set_ylim(95, 102)
ax3.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

for bar, acc in zip(bars3, accuracy):
    height = bar.get_height()
    loss = 100 - acc
    ax3.annotate(f'{acc:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    if loss > 0:
        ax3.annotate(f'-{loss:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, 96),
                    ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

plt.tight_layout()

# Add summary table at the bottom
summary_text = """
Summary: Quantization Trade-offs
| Precision | Memory | Speed   | Accuracy Loss | Best Use Case                    |
|-----------|--------|---------|---------------|----------------------------------|
| Float32   | 100%   | 1.0x    | 0%            | Training, highest precision      |
| Float16   | 50%    | 1.5x    | ~0.2%         | GPU training, mixed precision    |
| INT8      | 25%    | 2.5x    | ~0.8%         | Production inference             |
| INT4      | 12.5%  | 3.0x    | ~2.5%         | Edge/mobile deployment           |
"""

# Save figure
output_path = os.path.join(output_dir, 'quantization_impact.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved to: {output_path}")

# Create a second figure with detailed breakdown
fig2, ax = plt.subplots(figsize=(12, 8))

# Grouped bar chart
x = np.arange(len(precisions))
width = 0.25

# Normalize all metrics to 0-100 scale for comparison
memory_norm = memory
speed_norm = [s/3 for s in speed]  # Normalize to ~100 max
accuracy_norm = accuracy

bars_mem = ax.bar(x - width, memory_norm, width, label='Memory Usage', color='#2196F3', edgecolor='black')
bars_spd = ax.bar(x, speed_norm, width, label='Speed (normalized)', color='#4CAF50', edgecolor='black')
bars_acc = ax.bar(x + width, accuracy_norm, width, label='Accuracy', color='#FF9800', edgecolor='black')

ax.set_ylabel('Relative Value (%)', fontweight='bold')
ax.set_xlabel('Precision Level', fontweight='bold')
ax.set_title('Quantization: Memory vs Speed vs Accuracy Trade-offs', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(precisions)
ax.legend(loc='upper right')
ax.set_ylim(0, 120)

# Add reference line
ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Add annotations for key insights
ax.annotate('Best for\nProduction', xy=(2, 105), fontsize=10, ha='center',
           fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax.annotate('Best for\nEdge/Mobile', xy=(3, 105), fontsize=10, ha='center',
           fontweight='bold', color='orange',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

output_path2 = os.path.join(output_dir, 'quantization_tradeoffs.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved to: {output_path2}")
