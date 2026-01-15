#!/usr/bin/env python3
"""
KV-Cache Performance Visualization

Shows the dramatic difference in inference time between:
- Without KV-cache: O(n^2) total computation
- With KV-cache: O(n) total computation

Marks practical sequence lengths used in modern LLMs.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set up the output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Sequence lengths to plot
seq_lengths = np.linspace(1, 8192, 500)

# Normalized time (relative to some baseline)
# Without cache: Each step takes O(n), total is O(n^2)
time_without_cache = seq_lengths ** 2 / 1000  # Normalized

# With cache: Each step takes O(1) for K,V, O(n) for attention
# But K,V are cached, so effectively O(n) total
time_with_cache = seq_lengths / 10  # Normalized (much smaller scale)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(seq_lengths, time_without_cache, color='#c62828', linewidth=2.5,
        label='Without KV-Cache: O(n\u00b2)', linestyle='--')
ax.plot(seq_lengths, time_with_cache, color='#2e7d32', linewidth=2.5,
        label='With KV-Cache: O(n)')

# Fill the area between to show savings
ax.fill_between(seq_lengths, time_with_cache, time_without_cache,
                alpha=0.2, color='#ff6b35', label='Time Saved')

# Mark practical sequence lengths
practical_lengths = [512, 2048, 4096, 8192]
practical_labels = ['512\n(GPT-2)', '2048\n(GPT-3)', '4096\n(GPT-4)', '8192\n(Claude)']

for length, label in zip(practical_lengths, practical_labels):
    # Vertical line
    ax.axvline(x=length, color='#ff6b35', linestyle=':', alpha=0.7)

    # Calculate y positions for both curves at this x
    idx = np.argmin(np.abs(seq_lengths - length))
    y_without = time_without_cache[idx]
    y_with = time_with_cache[idx]

    # Add annotation
    speedup = y_without / y_with if y_with > 0 else 0
    ax.annotate(f'{label}\n({speedup:.0f}x speedup)',
                xy=(length, y_with),
                xytext=(length + 200, y_with + y_without * 0.15),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='#666'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e6',
                         edgecolor='#ff6b35', alpha=0.9))

# Styling
ax.set_xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Inference Time (normalized)', fontsize=12, fontweight='bold')
ax.set_title('KV-Cache Performance Improvement\nDecoder Inference Time vs Sequence Length',
             fontsize=14, fontweight='bold', color='#d94a1e', pad=15)

# Set axis limits
ax.set_xlim(0, 8500)
ax.set_ylim(0, max(time_without_cache) * 1.1)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

# Grid
ax.grid(True, alpha=0.3, linestyle='-')
ax.set_axisbelow(True)

# Add complexity explanation box
complexity_text = (
    "Without Cache:\n"
    "  Step t computes K,V for all t tokens\n"
    "  Total: 1+2+...+n = O(n\u00b2)\n\n"
    "With Cache:\n"
    "  Step t reuses cached K,V\n"
    "  Only compute for new token\n"
    "  Total: O(n)"
)
ax.text(0.98, 0.55, complexity_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor='#ff6b35', alpha=0.95),
        family='monospace')

plt.tight_layout()

# Save figure
output_path = os.path.join(output_dir, 'kv_cache_performance.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Saved KV-cache performance plot to: {output_path}")
