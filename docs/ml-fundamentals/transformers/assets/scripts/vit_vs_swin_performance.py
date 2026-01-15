#!/usr/bin/env python3
"""
ViT vs Swin Transformer Performance Comparison

Dual-axis plot showing:
- X: Image resolution
- Y1: Latency
- Y2: Memory usage

Demonstrates Swin's O(n) advantage over ViT's O(n^2) complexity.
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

# Image resolutions
resolutions = [224, 384, 512, 768, 1024]
res_labels = ['224', '384', '512', '768', '1024']

# Number of patches for each resolution (patch size = 16)
patches = [(r // 16) ** 2 for r in resolutions]

# ViT: O(n^2) complexity - quadratic scaling
# Latency in ms (simulated realistic values)
vit_latency = [15, 45, 80, 180, 320]
# Memory in GB
vit_memory = [1.5, 4.5, 8.0, 18.0, 32.0]

# Swin: O(n) complexity - linear scaling (with window attention)
swin_latency = [12, 20, 28, 42, 56]
swin_memory = [1.2, 2.0, 2.8, 4.2, 5.6]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ViT vs Swin Transformer: Scaling with Image Resolution\nViT: O(n²) vs Swin: O(n) Attention Complexity',
             fontsize=14, fontweight='bold', y=1.02)

# Plot 1: Latency comparison (dual y-axis not needed, same scale)
ax1 = axes[0]

# Plot both on same axis
line1 = ax1.plot(res_labels, vit_latency, 'o-', color='#e53935', linewidth=2.5,
                 markersize=10, label='ViT (O(n²))')
line2 = ax1.plot(res_labels, swin_latency, 's-', color='#1e88e5', linewidth=2.5,
                 markersize=10, label='Swin (O(n))')

ax1.set_xlabel('Image Resolution (pixels)', fontweight='bold')
ax1.set_ylabel('Inference Latency (ms)', fontweight='bold')
ax1.set_title('Latency Scaling', fontweight='bold', fontsize=12)
ax1.legend(loc='upper left')
ax1.set_ylim(0, 350)

# Add annotations for latency
for i, (vl, sl) in enumerate(zip(vit_latency, swin_latency)):
    if i > 0:  # Skip first point for clarity
        speedup = vl / sl
        ax1.annotate(f'{speedup:.1f}x faster',
                    xy=(i, (vl + sl) / 2),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=9, color='green', fontweight='bold')

# Shade the gap
ax1.fill_between(res_labels, vit_latency, swin_latency, alpha=0.2, color='green')

# Plot 2: Memory comparison
ax2 = axes[1]

line3 = ax2.plot(res_labels, vit_memory, 'o-', color='#e53935', linewidth=2.5,
                 markersize=10, label='ViT (O(n²))')
line4 = ax2.plot(res_labels, swin_memory, 's-', color='#1e88e5', linewidth=2.5,
                 markersize=10, label='Swin (O(n))')

ax2.set_xlabel('Image Resolution (pixels)', fontweight='bold')
ax2.set_ylabel('GPU Memory (GB)', fontweight='bold')
ax2.set_title('Memory Scaling', fontweight='bold', fontsize=12)
ax2.legend(loc='upper left')
ax2.set_ylim(0, 35)

# Add annotations for memory
for i, (vm, sm) in enumerate(zip(vit_memory, swin_memory)):
    if i > 0:
        reduction = vm / sm
        ax2.annotate(f'{reduction:.1f}x less',
                    xy=(i, (vm + sm) / 2),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=9, color='green', fontweight='bold')

# Shade the gap
ax2.fill_between(res_labels, vit_memory, swin_memory, alpha=0.2, color='green')

plt.tight_layout()

# Save first figure
output_path = os.path.join(output_dir, 'vit_vs_swin_performance.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved to: {output_path}")

# Create second figure: Complexity explanation
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Why Swin Transformer is More Efficient\nShifted Window Attention Mechanism',
              fontsize=14, fontweight='bold', y=1.02)

# Left: Show attention complexity
ax_left = axes2[0]
n_values = np.array([196, 576, 1024, 2304, 4096])  # Number of patches
vit_ops = n_values ** 2  # O(n^2)
swin_ops = n_values * 49  # O(n * M^2) where M=7 is window size

ax_left.semilogy(n_values, vit_ops, 'o-', color='#e53935', linewidth=2.5,
                  markersize=10, label='ViT: O(n²)')
ax_left.semilogy(n_values, swin_ops, 's-', color='#1e88e5', linewidth=2.5,
                  markersize=10, label='Swin: O(n × M²)')

ax_left.set_xlabel('Number of Patches (n)', fontweight='bold')
ax_left.set_ylabel('Attention Operations (log scale)', fontweight='bold')
ax_left.set_title('Attention Complexity Scaling', fontweight='bold', fontsize=12)
ax_left.legend()
ax_left.grid(True, alpha=0.3)

# Add text annotation
ax_left.text(2000, 1e6, 'ViT: Global attention\nO(n²) operations',
            fontsize=10, color='#e53935', fontweight='bold')
ax_left.text(2000, 5e4, 'Swin: Window attention\nO(n × M²) operations\nM=7 (window size)',
            fontsize=10, color='#1e88e5', fontweight='bold')

# Right: Accuracy vs Efficiency trade-off
ax_right = axes2[1]

# ImageNet accuracy data (approximate)
models = ['ViT-B/16', 'ViT-L/16', 'Swin-T', 'Swin-S', 'Swin-B']
accuracy = [84.5, 87.1, 81.3, 83.0, 85.2]
throughput = [300, 85, 755, 440, 275]  # images/sec on V100

colors = ['#ef5350', '#c62828', '#42a5f5', '#1e88e5', '#1565c0']

scatter = ax_right.scatter(throughput, accuracy, c=colors, s=200, edgecolors='black', linewidths=1.5)

for i, model in enumerate(models):
    ax_right.annotate(model, (throughput[i], accuracy[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=10, fontweight='bold')

ax_right.set_xlabel('Throughput (images/sec)', fontweight='bold')
ax_right.set_ylabel('ImageNet Top-1 Accuracy (%)', fontweight='bold')
ax_right.set_title('Accuracy vs Throughput Trade-off', fontweight='bold', fontsize=12)

# Add regions
ax_right.axvline(x=400, color='gray', linestyle='--', alpha=0.5)
ax_right.text(200, 81, 'Higher\nAccuracy', fontsize=10, ha='center', alpha=0.7)
ax_right.text(600, 81, 'Higher\nThroughput', fontsize=10, ha='center', alpha=0.7)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e53935', edgecolor='black', label='ViT Family'),
                   Patch(facecolor='#1e88e5', edgecolor='black', label='Swin Family')]
ax_right.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()

output_path2 = os.path.join(output_dir, 'vit_vs_swin_complexity.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved to: {output_path2}")
