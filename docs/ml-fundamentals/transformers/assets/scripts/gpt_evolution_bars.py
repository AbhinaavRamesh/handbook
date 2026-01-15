"""
GPT Series Evolution: Grouped Bar Chart
Compares GPT-1, GPT-2, GPT-3, and GPT-4 across key metrics
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 8))

# GPT series data
models = ['GPT-1\n(2018)', 'GPT-2\n(2019)', 'GPT-3\n(2020)', 'GPT-4\n(2023)']

# Metrics (using log scale later, so actual values)
# Parameters in billions
parameters = [0.117, 1.5, 175, 1760]  # GPT-4 estimated as MoE total

# Context window in tokens
context_window = [512, 1024, 2048, 8192]

# Training tokens in billions
training_tokens = [1, 40, 300, 13000]  # GPT-4 estimated

# Bar positions
x = np.arange(len(models))
width = 0.25

# Create bars
bars1 = ax.bar(x - width, parameters, width, label='Parameters (B)',
               color='#1976D2', alpha=0.85, edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x, context_window, width, label='Context Window (tokens)',
               color='#388E3C', alpha=0.85, edgecolor='white', linewidth=1.5)
bars3 = ax.bar(x + width, training_tokens, width, label='Training Tokens (B)',
               color='#F57C00', alpha=0.85, edgecolor='white', linewidth=1.5)

# Use log scale
ax.set_yscale('log')

# Add value labels on bars
def add_labels(bars, values, color):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if val >= 1000:
            label = f'{val/1000:.1f}T'
        elif val >= 1:
            label = f'{val:.0f}B' if val == int(val) else f'{val:.1f}B'
        else:
            label = f'{val*1000:.0f}M'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=color)

add_labels(bars1, parameters, '#1565C0')
add_labels(bars2, context_window, '#2E7D32')
add_labels(bars3, training_tokens, '#E65100')

# Customize axes
ax.set_xlabel('Model Version', fontsize=12, fontweight='bold')
ax.set_ylabel('Value (Log Scale)', fontsize=12, fontweight='bold')
ax.set_title('GPT Series Evolution: Parameters, Context, and Training Data',
             fontsize=14, fontweight='bold', pad=20)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')

# Set y-axis limits
ax.set_ylim(0.1, 50000)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_axisbelow(True)

# Add legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

# Add growth annotations
# Parameter growth
ax.annotate('', xy=(3.1, 1760), xytext=(0, 0.117),
            arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5, ls='--'))
ax.text(1.5, 5, '15,000x growth\nin parameters', fontsize=9,
        color='#1976D2', ha='center', style='italic')

# Training data growth
ax.annotate('', xy=(3.3, 13000), xytext=(0.2, 1),
            arrowprops=dict(arrowstyle='->', color='#F57C00', lw=1.5, ls='--'))
ax.text(2, 100, '13,000x growth\nin training data', fontsize=9,
        color='#F57C00', ha='center', style='italic')

# Add milestone annotations
milestones = [
    (0, 1, 'First GPT\nTransfer learning'),
    (1, 50, 'Emergent few-shot\nlearning begins'),
    (2, 1000, 'In-context learning\ndemonstrated'),
    (3, 20000, 'Multimodal + RLHF\nInstruction following'),
]

for x_pos, y_pos, text in milestones:
    ax.annotate(text, xy=(x_pos, y_pos), fontsize=7, ha='center',
                color='#666', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add info box
textstr = ('Key Scaling Insights:\n'
           '- Parameters: 117M to 1.76T (MoE)\n'
           '- Context: 512 to 8K+ tokens\n'
           '- Training: 1B to 13T tokens\n'
           '- Emergent abilities at scale')
props = dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9)
ax.text(0.98, 0.45, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()

# Save the figure
output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/gpt_evolution_bars.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
