"""
Efficiency Approaches Heatmap
Shows trade-offs between different attention efficiency methods
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 8))

# Efficiency approaches (methods)
methods = [
    'Standard Attention',
    'Flash Attention',
    'Sparse Attention',
    'Local/Sliding Window',
    'Linear Attention',
    'Multi-Query (MQA)',
    'Grouped-Query (GQA)',
    'Compression/Pooling',
    'Hierarchical'
]

# Evaluation metrics
metrics = [
    'Speed',
    'Memory\nEfficiency',
    'Quality\n(Accuracy)',
    'Long-Range\nModeling',
    'Implementation\nSimplicity',
    'Compatibility'
]

# Scores (1-5 scale, 5 is best)
# [Speed, Memory, Quality, Long-Range, Simplicity, Compatibility]
scores = np.array([
    [2, 1, 5, 5, 5, 5],  # Standard Attention
    [4, 4, 5, 5, 4, 5],  # Flash Attention
    [4, 4, 3, 3, 3, 3],  # Sparse Attention
    [5, 5, 3, 2, 4, 4],  # Local/Sliding Window
    [5, 5, 2, 2, 3, 2],  # Linear Attention
    [5, 5, 3, 4, 4, 4],  # Multi-Query (MQA)
    [4, 4, 4, 4, 4, 5],  # Grouped-Query (GQA)
    [4, 4, 3, 3, 3, 3],  # Compression/Pooling
    [3, 3, 4, 4, 2, 2],  # Hierarchical
])

# Create custom colormap (red-yellow-green)
from matplotlib.colors import LinearSegmentedColormap
colors = ['#E53935', '#FFEB3B', '#43A047']  # Red, Yellow, Green
n_bins = 100
cmap = LinearSegmentedColormap.from_list('ryg', colors, N=n_bins)

# Create heatmap
sns.heatmap(scores,
            annot=True,
            fmt='d',
            cmap=cmap,
            vmin=1, vmax=5,
            xticklabels=metrics,
            yticklabels=methods,
            ax=ax,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': 'Score (1=Poor, 5=Excellent)', 'shrink': 0.8})

# Customize
ax.set_title('Attention Efficiency Methods: Trade-off Analysis\n'
             '(Higher scores are better)',
             fontsize=14, fontweight='bold', pad=20)

# Rotate x-axis labels
plt.xticks(rotation=0, ha='center', fontsize=10, fontweight='bold')
plt.yticks(rotation=0, fontsize=10, fontweight='bold')

# Add annotations for key insights
# Flash Attention highlight
ax.add_patch(plt.Rectangle((0, 1), 6, 1, fill=False, edgecolor='#1976D2', lw=3))
ax.text(6.1, 1.5, 'Best overall\n(use by default)', fontsize=8,
        color='#1976D2', fontweight='bold', va='center')

# GQA highlight
ax.add_patch(plt.Rectangle((0, 6), 6, 1, fill=False, edgecolor='#7B1FA2', lw=3))
ax.text(6.1, 6.5, 'Best for\nproduction', fontsize=8,
        color='#7B1FA2', fontweight='bold', va='center')

# Add legend/guide box
textstr = ('Recommendation Guide:\n'
           '1. Start with Flash Attention (free speedup)\n'
           '2. Use GQA for inference optimization\n'
           '3. Local attention for very long sequences\n'
           '4. Avoid linear attention (quality loss)')
props = dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.95)
fig.text(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
         bbox=props, family='sans-serif')

# Add complexity annotations on the side
complexity_notes = [
    'O(n$^2$)',      # Standard
    'O(n$^2$)*',     # Flash (same complexity, faster constant)
    'O(n$\\sqrt{n}$)',  # Sparse
    'O(nk)',         # Local (k = window size)
    'O(n)',          # Linear
    'O(n)',          # MQA
    'O(n)',          # GQA
    'O(n/r)',        # Compression (r = compression ratio)
    'O(n log n)',    # Hierarchical
]

for i, note in enumerate(complexity_notes):
    ax.text(-0.3, i + 0.5, note, fontsize=8, ha='right', va='center',
            color='#666', family='monospace')

ax.text(-0.3, -0.5, 'Complexity', fontsize=9, ha='right', va='center',
        fontweight='bold', color='#333')

plt.tight_layout()

# Save the figure
output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/efficiency_tradeoffs_heatmap.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
