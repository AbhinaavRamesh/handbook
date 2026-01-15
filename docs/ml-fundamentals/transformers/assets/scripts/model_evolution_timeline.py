"""
Model Evolution Timeline: Transformer models from 2018-2024
Shows parameter count growth over time, colored by model type.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 8))

# Model data: (name, year, params_billions, model_type)
# Model types: 'encoder' (BERT-like), 'decoder' (GPT-like), 'encoder-decoder' (T5-like)
models = [
    # Encoder-only models
    ('BERT', 2018.9, 0.34, 'encoder'),
    ('RoBERTa', 2019.6, 0.355, 'encoder'),
    ('ALBERT', 2019.9, 0.223, 'encoder'),
    ('DeBERTa', 2020.6, 0.4, 'encoder'),

    # Decoder-only models
    ('GPT-1', 2018.5, 0.117, 'decoder'),
    ('GPT-2', 2019.2, 1.5, 'decoder'),
    ('GPT-3', 2020.5, 175, 'decoder'),
    ('Gopher', 2021.9, 280, 'decoder'),
    ('PaLM', 2022.4, 540, 'decoder'),
    ('GPT-4', 2023.3, 1760, 'decoder'),  # Estimated MoE total
    ('LLaMA', 2023.2, 65, 'decoder'),
    ('LLaMA 2', 2023.7, 70, 'decoder'),
    ('Mistral', 2023.9, 7, 'decoder'),
    ('Mixtral', 2023.12, 46.7, 'decoder'),  # MoE
    ('LLaMA 3', 2024.4, 405, 'decoder'),

    # Encoder-decoder models
    ('T5', 2019.10, 11, 'encoder-decoder'),
    ('BART', 2019.10, 0.4, 'encoder-decoder'),
    ('T5-XXL', 2020.1, 11, 'encoder-decoder'),
    ('mT5', 2020.10, 13, 'encoder-decoder'),
    ('FLAN-T5', 2022.10, 11, 'encoder-decoder'),
]

# Color scheme for model types
colors = {
    'encoder': '#1976D2',          # Blue
    'decoder': '#F57C00',          # Orange
    'encoder-decoder': '#388E3C',  # Green
}

# Markers for model types
markers = {
    'encoder': 'o',
    'decoder': 's',
    'encoder-decoder': '^',
}

# Plot each model
for name, year, params, model_type in models:
    ax.scatter(year, params,
               c=colors[model_type],
               marker=markers[model_type],
               s=150,
               alpha=0.8,
               edgecolors='white',
               linewidths=1.5,
               zorder=5)

    # Add labels with offset based on position
    offset_y = 1.15 if params > 10 else 1.3
    offset_x = 0.05

    # Special positioning for crowded areas
    if name in ['GPT-4', 'PaLM']:
        offset_x = -0.15
        ha = 'right'
    elif name in ['BERT', 'GPT-1']:
        offset_x = 0.1
        ha = 'left'
    else:
        ha = 'left'

    ax.annotate(name, (year + offset_x, params * offset_y),
                fontsize=8, fontweight='bold',
                ha=ha, va='bottom',
                color=colors[model_type])

# Set log scale for y-axis
ax.set_yscale('log')

# Customize axes
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Parameters (Billions)', fontsize=12, fontweight='bold')
ax.set_title('Transformer Model Evolution: Parameter Count Over Time (2018-2024)',
             fontsize=14, fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(2018, 2025)
ax.set_ylim(0.1, 5000)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Create legend
legend_elements = [
    plt.scatter([], [], c=colors['encoder'], marker=markers['encoder'],
                s=100, label='Encoder-Only (BERT-like)'),
    plt.scatter([], [], c=colors['decoder'], marker=markers['decoder'],
                s=100, label='Decoder-Only (GPT-like)'),
    plt.scatter([], [], c=colors['encoder-decoder'], marker=markers['encoder-decoder'],
                s=100, label='Encoder-Decoder (T5-like)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

# Add annotation for scale milestone
ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, zorder=1)
ax.text(2024.5, 100, '100B params', fontsize=8, va='center', color='gray')

ax.axhline(y=1000, color='gray', linestyle=':', alpha=0.5, zorder=1)
ax.text(2024.5, 1000, '1T params', fontsize=8, va='center', color='gray')

# Add trend annotation
ax.annotate('Exponential growth\nin decoder models',
            xy=(2022, 300), xytext=(2021, 30),
            fontsize=9, style='italic', color='#666',
            arrowprops=dict(arrowstyle='->', color='#999'))

plt.tight_layout()

# Save the figure
output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/model_evolution_timeline.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
