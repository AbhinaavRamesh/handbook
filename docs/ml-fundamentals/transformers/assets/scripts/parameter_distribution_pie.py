#!/usr/bin/env python3
"""
Parameter Distribution in Transformer Encoder (BERT-base)

Pie chart showing where parameters live in a transformer encoder:
- Multi-head Attention: ~25%
- Feed-Forward Network: ~50%
- Embeddings: ~20%
- Other (LayerNorm, output): ~5%

Based on BERT-base (110M parameters).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Green color palette
GREEN_DARK = '#1B5E20'
GREEN_MEDIUM = '#43A047'
GREEN_LIGHT = '#81C784'
GREEN_PALE = '#C8E6C9'
GREEN_VERY_PALE = '#E8F5E9'

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# ===== BERT-base Parameter Calculation =====
d_model = 768
d_ff = 3072
num_heads = 12
num_layers = 12
vocab_size = 30522
max_seq_len = 512

# Per-layer parameters
attention_per_layer = 4 * d_model * d_model  # Q, K, V, O projections
ffn_per_layer = d_model * d_ff + d_ff * d_model  # W1 and W2
layernorm_per_layer = 4 * d_model  # 2 LayerNorms with gamma and beta

# Total parameters
attention_total = attention_per_layer * num_layers
ffn_total = ffn_per_layer * num_layers
layernorm_total = layernorm_per_layer * num_layers

# Embedding parameters
token_embeddings = vocab_size * d_model
position_embeddings = max_seq_len * d_model
embeddings_total = token_embeddings + position_embeddings

# Output layer (often tied with embeddings in BERT)
output_layer = d_model * vocab_size  # Not counted if tied

# For simplicity, use typical distribution percentages
total_params = 110_000_000  # BERT-base total

# ===== Subplot 1: Pie Chart =====
categories = ['Feed-Forward\nNetwork (FFN)', 'Multi-Head\nAttention',
              'Token & Position\nEmbeddings', 'LayerNorm &\nOther']
sizes = [50, 25, 22, 3]
actual_params = [55.0, 27.5, 24.2, 3.3]  # In millions

colors = [GREEN_DARK, GREEN_MEDIUM, GREEN_LIGHT, GREEN_PALE]
explode = (0.05, 0, 0, 0)  # Emphasize FFN

# Create pie chart
wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=categories,
                                    colors=colors, autopct='%1.0f%%',
                                    startangle=90, pctdistance=0.75,
                                    wedgeprops=dict(linewidth=2, edgecolor='white'))

# Style the text
for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')
    autotext.set_color('white')

# Add center circle for donut effect
centre_circle = plt.Circle((0, 0), 0.45, fc='white', ec=GREEN_DARK, linewidth=2)
ax1.add_patch(centre_circle)

# Add center text
ax1.text(0, 0.05, 'BERT-base', ha='center', va='center', fontsize=14, fontweight='bold', color=GREEN_DARK)
ax1.text(0, -0.15, '110M params', ha='center', va='center', fontsize=12, color=GREEN_DARK)

ax1.set_title('Parameter Distribution in Transformer Encoder', fontsize=16,
              fontweight='bold', color=GREEN_DARK, pad=20)

# ===== Subplot 2: Detailed Breakdown Bar Chart =====
components = [
    'FFN (W₁: 768→3072)',
    'FFN (W₂: 3072→768)',
    'Attention Q projection',
    'Attention K projection',
    'Attention V projection',
    'Attention O projection',
    'Token Embeddings',
    'Position Embeddings',
    'LayerNorm (×24)',
    'Biases & Other'
]

# Calculate actual parameter counts
params_millions = [
    (d_model * d_ff * num_layers) / 1e6,           # FFN W1: 28.3M
    (d_ff * d_model * num_layers) / 1e6,           # FFN W2: 28.3M
    (d_model * d_model * num_layers) / 1e6,        # Q: 7.1M
    (d_model * d_model * num_layers) / 1e6,        # K: 7.1M
    (d_model * d_model * num_layers) / 1e6,        # V: 7.1M
    (d_model * d_model * num_layers) / 1e6,        # O: 7.1M
    (vocab_size * d_model) / 1e6,                   # Token: 23.4M
    (max_seq_len * d_model) / 1e6,                  # Position: 0.4M
    (4 * d_model * num_layers) / 1e6,              # LayerNorm: 0.07M
    0.3                                              # Biases, etc.
]

# Color mapping
bar_colors = [
    GREEN_DARK, GREEN_DARK,  # FFN
    GREEN_MEDIUM, GREEN_MEDIUM, GREEN_MEDIUM, GREEN_MEDIUM,  # Attention
    GREEN_LIGHT, GREEN_LIGHT,  # Embeddings
    GREEN_PALE, GREEN_PALE  # Other
]

y_pos = np.arange(len(components))
bars = ax2.barh(y_pos, params_millions, color=bar_colors, edgecolor=GREEN_DARK, linewidth=1)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(components, fontsize=11)
ax2.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
ax2.set_title('Detailed Parameter Breakdown\n(BERT-base: 12 layers)',
              fontsize=14, fontweight='bold', color=GREEN_DARK)

# Add value labels
for bar, val in zip(bars, params_millions):
    if val > 1:
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}M', va='center', fontsize=10, fontweight='bold', color=GREEN_DARK)

# Add legend for component groups
legend_elements = [
    plt.Rectangle((0,0), 1, 1, facecolor=GREEN_DARK, edgecolor='white', label='FFN (~51%)'),
    plt.Rectangle((0,0), 1, 1, facecolor=GREEN_MEDIUM, edgecolor='white', label='Attention (~26%)'),
    plt.Rectangle((0,0), 1, 1, facecolor=GREEN_LIGHT, edgecolor='white', label='Embeddings (~22%)'),
    plt.Rectangle((0,0), 1, 1, facecolor=GREEN_PALE, edgecolor='white', label='Other (~1%)'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

ax2.set_xlim(0, 35)
ax2.grid(True, axis='x', alpha=0.3)

# Add key insight box
insight_text = (
    "Key Insight: FFN contains ~50% of parameters!\n"
    "Attention gets more attention, but FFN dominates\n"
    "in parameter count (4× expansion: 768 → 3072 → 768)"
)
fig.text(0.5, 0.02, insight_text, ha='center', va='bottom', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK),
         fontweight='bold', color=GREEN_DARK)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
output_dir = Path(__file__).parent.parent / 'images'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'parameter_distribution_pie.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {output_path}")

plt.close()
