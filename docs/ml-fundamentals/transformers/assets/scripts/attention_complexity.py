"""
Attention Complexity Visualization
Shows O(n^2) quadratic complexity of self-attention
Demonstrates why long context is expensive
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Sequence lengths to plot
seq_lengths = np.linspace(128, 65536, 500)

# Compute complexity (normalized)
# Standard attention: O(n^2 * d) where d is model dimension
# We'll normalize to show relative growth
quadratic = (seq_lengths ** 2) / (512 ** 2)  # Normalized to seq_len=512
linear = seq_lengths / 512  # For comparison

# Key sequence length markers
markers = {
    512: ('GPT-2\n(2019)', '#1976D2'),
    2048: ('GPT-3\n(2020)', '#388E3C'),
    4096: ('GPT-3.5\n(2022)', '#F57C00'),
    8192: ('GPT-4\n(2023)', '#7B1FA2'),
    32768: ('Claude/GPT-4\nTurbo (2023)', '#C62828'),
    131072: ('Claude 3\n(2024)', '#00838F'),
}

# Left plot: Linear scale showing quadratic growth
ax1.plot(seq_lengths, quadratic, 'b-', linewidth=2.5, label='O(n$^2$) - Standard Attention')
ax1.plot(seq_lengths, linear, 'g--', linewidth=2, label='O(n) - Linear Attention', alpha=0.7)

# Add markers for key sequence lengths
for seq_len, (label, color) in markers.items():
    if seq_len <= 8192:  # Only show smaller ones on linear plot
        y_val = (seq_len ** 2) / (512 ** 2)
        ax1.scatter([seq_len], [y_val], c=color, s=100, zorder=5, edgecolors='white', linewidths=2)
        ax1.annotate(label, (seq_len, y_val),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, fontweight='bold', color=color,
                    ha='left')

ax1.set_xlabel('Sequence Length (tokens)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Relative Compute/Memory (normalized to n=512)', fontsize=11, fontweight='bold')
ax1.set_title('Quadratic Attention Complexity\n(Linear Scale)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 10000)
ax1.set_ylim(0, 400)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', fontsize=9)

# Add shaded region for "practical limit"
ax1.axvspan(0, 4096, alpha=0.1, color='green', label='Practical range')
ax1.axvspan(4096, 10000, alpha=0.1, color='red')
ax1.text(2000, 350, 'Practical\nRange', fontsize=9, ha='center', color='green', fontweight='bold')
ax1.text(7000, 350, 'Expensive\nRange', fontsize=9, ha='center', color='red', fontweight='bold')

# Right plot: Log scale to show full range
ax2.plot(seq_lengths, quadratic, 'b-', linewidth=2.5, label='O(n$^2$) - Standard Attention')
ax2.plot(seq_lengths, linear, 'g--', linewidth=2, label='O(n) - Linear Attention', alpha=0.7)

# Add all markers on log scale
for seq_len, (label, color) in markers.items():
    y_val = (seq_len ** 2) / (512 ** 2)
    ax2.scatter([seq_len], [y_val], c=color, s=120, zorder=5, edgecolors='white', linewidths=2)

    # Adjust label positions
    if seq_len == 131072:
        xytext = (-60, -20)
        ha = 'right'
    elif seq_len >= 32768:
        xytext = (10, -20)
        ha = 'left'
    else:
        xytext = (10, 10)
        ha = 'left'

    ax2.annotate(label, (seq_len, y_val),
                xytext=xytext, textcoords='offset points',
                fontsize=8, fontweight='bold', color=color,
                ha=ha)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Sequence Length (tokens)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Relative Compute/Memory (log scale)', fontsize=11, fontweight='bold')
ax2.set_title('Quadratic vs Linear Complexity\n(Log Scale)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--', which='both')
ax2.legend(loc='upper left', fontsize=9)

# Add annotation explaining the cost
ax2.annotate('128K context costs 256x more\nthan 8K context!',
            xy=(131072, (131072**2)/(512**2)),
            xytext=(5000, 100000),
            fontsize=9, style='italic', color='#666',
            arrowprops=dict(arrowstyle='->', color='#999'),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Add cost comparison box
textstr = ('Cost Comparison:\n'
           'n=512:    1x (baseline)\n'
           'n=2048:   16x\n'
           'n=8192:   256x\n'
           'n=32768:  4096x\n'
           'n=131072: 65536x')
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
ax2.text(0.98, 0.02, textstr, transform=ax2.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=props, family='monospace')

plt.tight_layout()

# Save the figure
output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/attention_complexity.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
