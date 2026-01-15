#!/usr/bin/env python3
"""
Causal Mask Animation

Creates an animated GIF showing how the causal mask grows from 1x1 to 6x6
as tokens are generated autoregressively. Demonstrates that each new token
can only see previous tokens.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
import os

# Set up the output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Parameters
max_seq_len = 6
fig_size = (8, 7)

# Custom colormap
colors = ['#1a1a1a', '#ffffff']  # Black for masked, White for allowed
cmap = LinearSegmentedColormap.from_list('causal', colors)

# Token labels for demonstration
tokens = ['<START>', 'The', 'cat', 'sat', 'on', 'the']

frames = []

for step in range(1, max_seq_len + 1):
    fig, ax = plt.subplots(figsize=fig_size)

    # Create full 6x6 grid (greyed out for future positions)
    full_mask = np.ones((max_seq_len, max_seq_len)) * 0.3  # Grey background

    # Fill in the active portion with causal mask
    causal_mask = np.tril(np.ones((step, step)))
    full_mask[:step, :step] = causal_mask

    # Plot the mask
    im = ax.imshow(full_mask, cmap='gray', aspect='equal', vmin=0, vmax=1)

    # Add grid lines
    for i in range(max_seq_len + 1):
        ax.axhline(i - 0.5, color='#ff6b35', linewidth=1.5)
        ax.axvline(i - 0.5, color='#ff6b35', linewidth=1.5)

    # Highlight the active region with orange border
    from matplotlib.patches import Rectangle
    rect = Rectangle((-0.5, -0.5), step, step,
                     fill=False, edgecolor='#ff6b35', linewidth=4)
    ax.add_patch(rect)

    # Add checkmarks and X marks for active region
    for i in range(step):
        for j in range(step):
            if causal_mask[i, j] == 1:
                ax.text(j, i, '\u2713', ha='center', va='center',
                       fontsize=14, color='#2e7d32', fontweight='bold')
            else:
                ax.text(j, i, '\u2717', ha='center', va='center',
                       fontsize=14, color='#c62828', fontweight='bold')

    # Grey X marks for future positions
    for i in range(max_seq_len):
        for j in range(max_seq_len):
            if i >= step or j >= step:
                ax.text(j, i, '?', ha='center', va='center',
                       fontsize=12, color='#888888', fontweight='bold')

    # Labels with tokens
    position_labels = [f'{tokens[i]}' for i in range(max_seq_len)]
    ax.set_xticks(range(max_seq_len))
    ax.set_yticks(range(max_seq_len))
    ax.set_xticklabels(position_labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(position_labels, fontsize=10)

    # Highlight current token being generated
    for i in range(step):
        color = '#ff6b35' if i < step else '#888888'
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight('bold')
        ax.get_xticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_fontweight('bold')

    # Axis labels
    ax.set_xlabel('Key Position (attending to)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Query Position (generating)', fontsize=12, fontweight='bold', labelpad=10)

    # Title showing generation step
    ax.set_title(f'Autoregressive Generation: Step {step}/{max_seq_len}\n'
                f'Generating token: "{tokens[step-1]}"',
                fontsize=14, fontweight='bold', color='#d94a1e', pad=15)

    # Add explanation
    explanation = f"Token {step} can attend to tokens 1-{step}\n(itself and all previous tokens)"
    ax.annotate(explanation, xy=(0.5, -0.15), xycoords='axes fraction',
                ha='center', va='top', fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e6', edgecolor='#ff6b35'))

    plt.tight_layout()

    # Save frame to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    buf.close()
    plt.close()

# Save as GIF
output_path = os.path.join(output_dir, 'causal_mask_animation.gif')
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=1000,  # 1 second per frame
    loop=0  # Loop forever
)

print(f"Saved causal mask animation to: {output_path}")
