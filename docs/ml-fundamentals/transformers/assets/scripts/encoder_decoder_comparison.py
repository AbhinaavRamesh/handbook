#!/usr/bin/env python3
"""
Encoder vs Decoder vs Encoder-Decoder Architecture Comparison

Three-panel figure showing:
1. Encoder-only (BERT) - Bidirectional attention
2. Decoder-only (GPT) - Causal/unidirectional attention
3. Encoder-Decoder (T5) - Cross-attention between encoder and decoder
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.gridspec import GridSpec
import os

# Set up the output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Colors
ENCODER_COLOR = '#4a90d9'
DECODER_COLOR = '#ff6b35'
CROSS_ATTN_COLOR = '#9c27b0'
BG_COLOR = '#f5f5f5'

def draw_attention_matrix(ax, matrix_type, x_offset=0, y_offset=0, size=0.3):
    """Draw a small attention matrix pattern."""
    n = 4

    if matrix_type == 'bidirectional':
        # Full attention - all ones
        mask = np.ones((n, n))
        title = 'Bidirectional'
        color = ENCODER_COLOR
    elif matrix_type == 'causal':
        # Lower triangular
        mask = np.tril(np.ones((n, n)))
        title = 'Causal'
        color = DECODER_COLOR
    elif matrix_type == 'cross':
        # Full attention (encoder to decoder)
        mask = np.ones((n, n))
        title = 'Cross'
        color = CROSS_ATTN_COLOR

    # Draw the matrix
    for i in range(n):
        for j in range(n):
            x = x_offset + j * size/n
            y = y_offset + (n-1-i) * size/n
            if mask[i, j] == 1:
                rect = plt.Rectangle((x, y), size/n * 0.9, size/n * 0.9,
                                     facecolor=color, alpha=0.7)
            else:
                rect = plt.Rectangle((x, y), size/n * 0.9, size/n * 0.9,
                                     facecolor='#e0e0e0', alpha=0.5)
            ax.add_patch(rect)

    return title

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1], hspace=0.3, wspace=0.25)

# ============== Panel 1: Encoder-Only (BERT) ==============
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_title('Encoder-Only\n(BERT, RoBERTa)', fontsize=14, fontweight='bold',
              color=ENCODER_COLOR, pad=10)

# Draw encoder block
encoder_box = FancyBboxPatch((0.15, 0.3), 0.7, 0.5, boxstyle="round,pad=0.02",
                              facecolor='#e3f2fd', edgecolor=ENCODER_COLOR, linewidth=3)
ax1.add_patch(encoder_box)
ax1.text(0.5, 0.55, 'ENCODER', ha='center', va='center', fontsize=14,
         fontweight='bold', color=ENCODER_COLOR)

# Input tokens
input_tokens = ['[CLS]', 'The', 'cat', 'sat']
for i, token in enumerate(input_tokens):
    x = 0.2 + i * 0.18
    rect = FancyBboxPatch((x, 0.1), 0.12, 0.12, boxstyle="round,pad=0.01",
                          facecolor=ENCODER_COLOR, edgecolor='#1565c0', linewidth=1)
    ax1.add_patch(rect)
    ax1.text(x + 0.06, 0.16, token, ha='center', va='center', fontsize=8,
             color='white', fontweight='bold')
    # Arrow up
    ax1.annotate('', xy=(x + 0.06, 0.3), xytext=(x + 0.06, 0.22),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# Bidirectional arrows inside encoder
for i in range(len(input_tokens)):
    for j in range(len(input_tokens)):
        if i != j:
            x1 = 0.26 + i * 0.18
            x2 = 0.26 + j * 0.18
            ax1.annotate('', xy=(x2, 0.52), xytext=(x1, 0.52),
                        arrowprops=dict(arrowstyle='<->', color=ENCODER_COLOR,
                                       alpha=0.3, lw=0.5, connectionstyle='arc3,rad=0.1'))

# Output
ax1.text(0.5, 0.92, 'Contextual Embeddings', ha='center', va='center',
         fontsize=10, fontweight='bold', style='italic')
ax1.annotate('', xy=(0.5, 0.88), xytext=(0.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Draw attention pattern
draw_attention_matrix(ax1, 'bidirectional', x_offset=0.35, y_offset=0.35, size=0.3)
ax1.text(0.5, 0.38, 'Bidirectional\nAttention', ha='center', va='top', fontsize=8, alpha=0.7)

# ============== Panel 2: Decoder-Only (GPT) ==============
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('Decoder-Only\n(GPT, LLaMA, Claude)', fontsize=14, fontweight='bold',
              color=DECODER_COLOR, pad=10)

# Draw decoder block
decoder_box = FancyBboxPatch((0.15, 0.3), 0.7, 0.5, boxstyle="round,pad=0.02",
                              facecolor='#fff3e6', edgecolor=DECODER_COLOR, linewidth=3)
ax2.add_patch(decoder_box)
ax2.text(0.5, 0.55, 'DECODER', ha='center', va='center', fontsize=14,
         fontweight='bold', color=DECODER_COLOR)

# Input tokens
for i, token in enumerate(input_tokens):
    x = 0.2 + i * 0.18
    rect = FancyBboxPatch((x, 0.1), 0.12, 0.12, boxstyle="round,pad=0.01",
                          facecolor=DECODER_COLOR, edgecolor='#d94a1e', linewidth=1)
    ax2.add_patch(rect)
    ax2.text(x + 0.06, 0.16, token, ha='center', va='center', fontsize=8,
             color='white', fontweight='bold')
    ax2.annotate('', xy=(x + 0.06, 0.3), xytext=(x + 0.06, 0.22),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# Causal arrows (only left to right)
for i in range(len(input_tokens)):
    for j in range(i):
        x1 = 0.26 + j * 0.18
        x2 = 0.26 + i * 0.18
        ax2.annotate('', xy=(x2, 0.52), xytext=(x1, 0.52),
                    arrowprops=dict(arrowstyle='->', color=DECODER_COLOR,
                                   alpha=0.3, lw=0.5, connectionstyle='arc3,rad=0.1'))

# Output with next token prediction
ax2.text(0.5, 0.92, 'Next Token Prediction', ha='center', va='center',
         fontsize=10, fontweight='bold', style='italic')
ax2.annotate('', xy=(0.5, 0.88), xytext=(0.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Draw attention pattern
draw_attention_matrix(ax2, 'causal', x_offset=0.35, y_offset=0.35, size=0.3)
ax2.text(0.5, 0.38, 'Causal\nAttention', ha='center', va='top', fontsize=8, alpha=0.7)

# ============== Panel 3: Encoder-Decoder (T5) ==============
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('Encoder-Decoder\n(T5, BART, mT5)', fontsize=14, fontweight='bold',
              color=CROSS_ATTN_COLOR, pad=10)

# Draw encoder block (smaller, on left)
enc_box = FancyBboxPatch((0.05, 0.5), 0.35, 0.35, boxstyle="round,pad=0.02",
                          facecolor='#e3f2fd', edgecolor=ENCODER_COLOR, linewidth=2)
ax3.add_patch(enc_box)
ax3.text(0.225, 0.68, 'ENCODER', ha='center', va='center', fontsize=10,
         fontweight='bold', color=ENCODER_COLOR)

# Draw decoder block (smaller, on right)
dec_box = FancyBboxPatch((0.6, 0.5), 0.35, 0.35, boxstyle="round,pad=0.02",
                          facecolor='#fff3e6', edgecolor=DECODER_COLOR, linewidth=2)
ax3.add_patch(dec_box)
ax3.text(0.775, 0.68, 'DECODER', ha='center', va='center', fontsize=10,
         fontweight='bold', color=DECODER_COLOR)

# Cross-attention arrow
ax3.annotate('', xy=(0.6, 0.68), xytext=(0.4, 0.68),
            arrowprops=dict(arrowstyle='->', color=CROSS_ATTN_COLOR, lw=3,
                           connectionstyle='arc3,rad=0'))
ax3.text(0.5, 0.72, 'Cross\nAttention', ha='center', va='bottom', fontsize=8,
         color=CROSS_ATTN_COLOR, fontweight='bold')

# Encoder input tokens
enc_tokens = ['Hello', 'world']
for i, token in enumerate(enc_tokens):
    x = 0.1 + i * 0.15
    rect = FancyBboxPatch((x, 0.35), 0.1, 0.1, boxstyle="round,pad=0.01",
                          facecolor=ENCODER_COLOR, edgecolor='#1565c0', linewidth=1)
    ax3.add_patch(rect)
    ax3.text(x + 0.05, 0.4, token, ha='center', va='center', fontsize=7,
             color='white', fontweight='bold')
    ax3.annotate('', xy=(x + 0.05, 0.5), xytext=(x + 0.05, 0.45),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1))

# Decoder input/output tokens
dec_tokens = ['<s>', 'Bonjour']
for i, token in enumerate(dec_tokens):
    x = 0.65 + i * 0.15
    rect = FancyBboxPatch((x, 0.35), 0.1, 0.1, boxstyle="round,pad=0.01",
                          facecolor=DECODER_COLOR, edgecolor='#d94a1e', linewidth=1)
    ax3.add_patch(rect)
    ax3.text(x + 0.05, 0.4, token, ha='center', va='center', fontsize=7,
             color='white', fontweight='bold')
    ax3.annotate('', xy=(x + 0.05, 0.5), xytext=(x + 0.05, 0.45),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1))

# Output
ax3.text(0.775, 0.92, 'monde', ha='center', va='center', fontsize=10,
         fontweight='bold', style='italic', color=DECODER_COLOR)
ax3.annotate('', xy=(0.775, 0.88), xytext=(0.775, 0.85),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Labels for source and target
ax3.text(0.225, 0.28, 'Source\n(English)', ha='center', va='top', fontsize=8, alpha=0.7)
ax3.text(0.775, 0.28, 'Target\n(French)', ha='center', va='top', fontsize=8, alpha=0.7)

# ============== Bottom row: Use cases and characteristics ==============
characteristics = [
    {
        'title': 'Encoder-Only',
        'attention': 'Bidirectional (full)',
        'examples': 'BERT, RoBERTa, ALBERT',
        'use_cases': 'Classification, NER,\nSentiment Analysis,\nQuestion Answering',
        'color': ENCODER_COLOR
    },
    {
        'title': 'Decoder-Only',
        'attention': 'Causal (left-to-right)',
        'examples': 'GPT, LLaMA, Claude, PaLM',
        'use_cases': 'Text Generation,\nCode Completion,\nChatbots',
        'color': DECODER_COLOR
    },
    {
        'title': 'Encoder-Decoder',
        'attention': 'Bidirectional + Causal + Cross',
        'examples': 'T5, BART, mBART',
        'use_cases': 'Translation,\nSummarization,\nParaphrasing',
        'color': CROSS_ATTN_COLOR
    }
]

for i, char in enumerate(characteristics):
    ax = fig.add_subplot(gs[1, i])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Create info box
    info_text = (f"Attention: {char['attention']}\n\n"
                f"Models: {char['examples']}\n\n"
                f"Use Cases:\n{char['use_cases']}")

    box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.02",
                          facecolor='white', edgecolor=char['color'], linewidth=2)
    ax.add_patch(box)
    ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=9,
           transform=ax.transAxes, linespacing=1.5)

# Main title
fig.suptitle('Transformer Architecture Comparison', fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
output_path = os.path.join(output_dir, 'encoder_decoder_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Saved encoder-decoder comparison to: {output_path}")
