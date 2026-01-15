"""
Chinchilla Scaling Law Visualization
Shows optimal tokens = 20 x parameters relationship
Identifies under-trained and over-trained models
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 9))

# Model data: (name, params_billions, tokens_billions, year)
models = [
    # Under-trained models (below Chinchilla optimal line)
    ('GPT-3', 175, 300, 2020),
    ('Gopher', 280, 300, 2021),
    ('PaLM', 540, 780, 2022),
    ('Megatron-LM', 530, 270, 2021),
    ('OPT-175B', 175, 180, 2022),

    # Near Chinchilla optimal
    ('Chinchilla', 70, 1400, 2022),
    ('LLaMA-65B', 65, 1400, 2023),
    ('LLaMA-7B', 7, 1000, 2023),
    ('LLaMA-13B', 13, 1000, 2023),
    ('LLaMA 2-70B', 70, 2000, 2023),
    ('Mistral-7B', 7, 1000, 2023),

    # Over-trained models (above Chinchilla optimal line)
    ('LLaMA 3-8B', 8, 15000, 2024),
    ('LLaMA 3-70B', 70, 15000, 2024),
    ('Gemma-7B', 7, 6000, 2024),
    ('Phi-2', 2.7, 1400, 2023),

    # Smaller models
    ('GPT-2', 1.5, 40, 2019),
    ('BERT', 0.34, 3.3, 2018),
]

# Calculate Chinchilla optimal line
params_range = np.logspace(-1, 3.5, 100)  # 0.1B to 3000B parameters
optimal_tokens = 20 * params_range  # Chinchilla rule: tokens = 20 * params

# Plot Chinchilla optimal line
ax.plot(params_range, optimal_tokens, 'k-', linewidth=2.5,
        label='Chinchilla Optimal (tokens = 20 x params)', zorder=3)

# Fill regions
ax.fill_between(params_range, optimal_tokens * 0.5, optimal_tokens,
                alpha=0.15, color='#F44336', label='Under-trained zone')
ax.fill_between(params_range, optimal_tokens, optimal_tokens * 3,
                alpha=0.15, color='#4CAF50', label='Over-trained zone')

# Color mapping based on training efficiency
def get_color_and_marker(params, tokens):
    ratio = tokens / (20 * params)
    if ratio < 0.7:
        return '#E53935', 'v'  # Red, down triangle (under-trained)
    elif ratio > 1.5:
        return '#43A047', '^'  # Green, up triangle (over-trained)
    else:
        return '#1E88E5', 'o'  # Blue, circle (optimal)

# Plot each model
for name, params, tokens, year in models:
    color, marker = get_color_and_marker(params, tokens)

    # Size based on recency
    size = 80 + (year - 2018) * 25

    ax.scatter(params, tokens, c=color, marker=marker, s=size,
               alpha=0.85, edgecolors='white', linewidths=1.5, zorder=5)

    # Label positioning
    ratio = tokens / (20 * params)
    if ratio < 0.5:
        offset = (1.1, 0.7)
        va = 'top'
    elif ratio > 2:
        offset = (1.1, 1.2)
        va = 'bottom'
    else:
        offset = (1.15, 1.0)
        va = 'center'

    # Special positioning for crowded areas
    if name in ['LLaMA-65B', 'Chinchilla']:
        offset = (0.85, 1.3)
    elif name in ['LLaMA 2-70B']:
        offset = (1.2, 1.0)
    elif name in ['GPT-3', 'OPT-175B']:
        offset = (1.15, 0.85)
    elif name in ['LLaMA 3-8B', 'LLaMA 3-70B']:
        offset = (1.2, 1.15)

    ax.annotate(name, (params * offset[0], tokens * offset[1]),
                fontsize=8, fontweight='bold', color=color,
                ha='left', va=va)

# Set log scale for both axes
ax.set_xscale('log')
ax.set_yscale('log')

# Customize axes
ax.set_xlabel('Parameters (Billions)', fontsize=12, fontweight='bold')
ax.set_ylabel('Training Tokens (Billions)', fontsize=12, fontweight='bold')
ax.set_title('Chinchilla Scaling: Optimal Training Tokens vs Parameters\n'
             'Optimal ratio: tokens = 20 x parameters',
             fontsize=14, fontweight='bold', pad=15)

# Set axis limits
ax.set_xlim(0.1, 3000)
ax.set_ylim(1, 30000)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', which='both')
ax.set_axisbelow(True)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='k', linewidth=2.5, label='Chinchilla Optimal Line'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#E53935',
           markersize=10, label='Under-trained (< 0.7x optimal)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E88E5',
           markersize=10, label='Near-optimal (0.7x - 1.5x)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#43A047',
           markersize=10, label='Over-trained (> 1.5x optimal)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

# Add annotation box explaining the law
textstr = 'Chinchilla Law (2022):\nFor compute budget C,\noptimal allocation is\ntokens = 20 x params'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

# Add annotation for key insight
ax.annotate('GPT-3: 175B params on only 300B tokens\n(Should have used ~3.5T tokens!)',
            xy=(175, 300), xytext=(30, 50),
            fontsize=8, style='italic', color='#666',
            arrowprops=dict(arrowstyle='->', color='#999'),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.annotate('LLaMA 3: Pushing beyond Chinchilla\nwith high-quality data',
            xy=(70, 15000), xytext=(200, 8000),
            fontsize=8, style='italic', color='#666',
            arrowprops=dict(arrowstyle='->', color='#999'),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the figure
output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/chinchilla_scaling.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
