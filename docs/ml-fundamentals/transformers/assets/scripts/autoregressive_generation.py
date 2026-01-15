#!/usr/bin/env python3
"""
Autoregressive Generation Visualization

Multi-panel figure showing step-by-step token generation in a decoder.
Each panel shows:
- Current sequence
- Logit bars for top candidate tokens
- Selected next token
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# Set up the output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Generation sequence
steps = [
    {
        'input': ['<START>'],
        'candidates': [('The', 0.45), ('A', 0.30), ('One', 0.12), ('This', 0.08), ('My', 0.05)],
        'selected': 'The'
    },
    {
        'input': ['<START>', 'The'],
        'candidates': [('cat', 0.35), ('dog', 0.25), ('quick', 0.18), ('big', 0.12), ('small', 0.10)],
        'selected': 'cat'
    },
    {
        'input': ['<START>', 'The', 'cat'],
        'candidates': [('sat', 0.42), ('jumped', 0.22), ('ran', 0.15), ('slept', 0.12), ('ate', 0.09)],
        'selected': 'sat'
    },
    {
        'input': ['<START>', 'The', 'cat', 'sat'],
        'candidates': [('on', 0.55), ('down', 0.20), ('quietly', 0.12), ('there', 0.08), ('still', 0.05)],
        'selected': 'on'
    },
    {
        'input': ['<START>', 'The', 'cat', 'sat', 'on'],
        'candidates': [('the', 0.48), ('a', 0.28), ('my', 0.12), ('her', 0.07), ('his', 0.05)],
        'selected': 'the'
    }
]

# Create figure
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

# Colors
colors = {
    'selected': '#ff6b35',
    'candidate': '#ffa07a',
    'input_token': '#4a90d9',
    'new_token': '#ff6b35',
    'background': '#fff3e6'
}

for idx, step in enumerate(steps):
    row = idx // 2
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])

    # === Left side: Current sequence ===
    input_tokens = step['input']

    # Create token boxes
    token_y = 0.85
    total_width = len(input_tokens) * 0.15
    start_x = 0.02

    for i, token in enumerate(input_tokens):
        x_pos = start_x + i * 0.16
        rect = mpatches.FancyBboxPatch(
            (x_pos, token_y - 0.08), 0.14, 0.16,
            boxstyle="round,pad=0.02",
            facecolor=colors['input_token'],
            edgecolor='#2e5a8a',
            linewidth=2,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        ax.text(x_pos + 0.07, token_y, token, transform=ax.transAxes,
               ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Arrow to prediction
    arrow_x = start_x + len(input_tokens) * 0.16 + 0.02
    ax.annotate('', xy=(arrow_x + 0.08, token_y), xytext=(arrow_x, token_y),
               transform=ax.transAxes,
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # Question mark box (prediction target)
    rect = mpatches.FancyBboxPatch(
        (arrow_x + 0.1, token_y - 0.08), 0.14, 0.16,
        boxstyle="round,pad=0.02",
        facecolor=colors['new_token'],
        edgecolor='#d94a1e',
        linewidth=2,
        linestyle='--',
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(arrow_x + 0.17, token_y, '?', transform=ax.transAxes,
           ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    # === Bottom: Logit bars ===
    candidates = step['candidates']
    tokens_list = [c[0] for c in candidates]
    probs = [c[1] for c in candidates]
    selected = step['selected']

    y_positions = np.arange(len(candidates))
    bar_colors = [colors['selected'] if t == selected else colors['candidate'] for t in tokens_list]

    # Create horizontal bars
    bars = ax.barh(y_positions, probs, color=bar_colors, edgecolor='#333', linewidth=1, height=0.6)

    # Add probability labels
    for i, (prob, token) in enumerate(zip(probs, tokens_list)):
        ax.text(prob + 0.02, i, f'{prob:.0%}', va='center', fontsize=9, fontweight='bold')
        # Highlight selected with star
        if token == selected:
            ax.text(-0.02, i, '\u2605', va='center', ha='right', fontsize=12, color='#ff6b35')

    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tokens_list, fontsize=10)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(-0.5, len(candidates) - 0.5)
    ax.invert_yaxis()

    ax.set_xlabel('Probability', fontsize=10)
    ax.set_title(f'Step {idx + 1}: Predicting next token', fontsize=11, fontweight='bold',
                color='#d94a1e', pad=40)

    # Add result annotation
    result_text = f'Selected: "{selected}"'
    ax.text(0.5, -0.15, result_text, transform=ax.transAxes, ha='center',
           fontsize=10, fontweight='bold', color='#ff6b35',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e6', edgecolor='#ff6b35'))

    ax.set_axisbelow(True)
    ax.grid(True, axis='x', alpha=0.3)

# Final panel: Complete generated sequence
ax_final = fig.add_subplot(gs[2, :])
ax_final.set_xlim(0, 1)
ax_final.set_ylim(0, 1)
ax_final.axis('off')

# Draw final sequence
final_tokens = ['<START>', 'The', 'cat', 'sat', 'on', 'the', '...']
start_x = 0.1
for i, token in enumerate(final_tokens):
    x_pos = start_x + i * 0.11
    color = colors['input_token'] if i == 0 else colors['new_token']
    rect = mpatches.FancyBboxPatch(
        (x_pos, 0.4), 0.09, 0.2,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='#333',
        linewidth=2,
        transform=ax_final.transAxes
    )
    ax_final.add_patch(rect)
    ax_final.text(x_pos + 0.045, 0.5, token, transform=ax_final.transAxes,
                 ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Add arrows between tokens
    if i < len(final_tokens) - 1:
        ax_final.annotate('', xy=(x_pos + 0.11, 0.5), xytext=(x_pos + 0.095, 0.5),
                         transform=ax_final.transAxes,
                         arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

ax_final.set_title('Complete Autoregressive Generation', fontsize=14, fontweight='bold',
                  color='#d94a1e', pad=10)

# Add explanation
explanation = ("Autoregressive generation: Each token is predicted based on all previous tokens.\n"
               "The decoder attends to the full input sequence (via cross-attention) and "
               "previously generated tokens (via masked self-attention).")
ax_final.text(0.5, 0.1, explanation, transform=ax_final.transAxes, ha='center', va='center',
             fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3e6', edgecolor='#ff6b35'))

# Main title
fig.suptitle('Autoregressive Token Generation in Decoder', fontsize=16, fontweight='bold',
            color='#d94a1e', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = os.path.join(output_dir, 'autoregressive_generation.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Saved autoregressive generation figure to: {output_path}")
