#!/usr/bin/env python3
"""
Whisper Architecture Visualization

Generates visualizations showing:
1. The complete Whisper encoder-decoder architecture
2. Audio preprocessing pipeline (waveform to mel spectrogram)
3. The multi-task training format with special tokens

Output: ../images/multimodal/whisper_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'multimodal')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_whisper_architecture():
    """Create a comprehensive Whisper architecture visualization."""
    fig = plt.figure(figsize=(18, 12))

    # Color scheme
    colors = {
        'audio': '#E3F2FD',
        'encoder': '#BBDEFB',
        'decoder': '#F3E5F5',
        'attention': '#FFECB3',
        'output': '#E8F5E9',
        'token': '#FFF9C4',
        'border': '#37474F'
    }

    # === Panel 1: Full Architecture ===
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.set_xlim(0, 18)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Whisper: Encoder-Decoder Architecture for Speech Recognition',
                  fontsize=14, fontweight='bold', pad=15)

    # --- Audio Input Section ---
    audio_box = FancyBboxPatch((0.5, 3), 3, 4, boxstyle="round,pad=0.1",
                                facecolor=colors['audio'], edgecolor=colors['border'], linewidth=2)
    ax1.add_patch(audio_box)
    ax1.text(2, 6.5, 'Audio Input', ha='center', va='center', fontsize=11, fontweight='bold')

    # Waveform visualization
    t = np.linspace(0, 2*np.pi, 50)
    wave = np.sin(3*t) * 0.3
    ax1.plot(np.linspace(0.8, 2.2, 50), wave + 5.8, 'b-', linewidth=1.5)
    ax1.text(1.5, 5.3, 'Waveform', ha='center', va='center', fontsize=8)

    # Mel spectrogram visualization
    mel_rect = Rectangle((0.8, 3.5), 1.4, 1.2, facecolor='white',
                          edgecolor=colors['border'], linewidth=1)
    ax1.add_patch(mel_rect)

    # Draw mel spectrogram heatmap
    for i in range(6):
        for j in range(8):
            intensity = np.random.rand() * 0.5 + 0.2
            small_rect = Rectangle((0.85 + j*0.16, 3.55 + i*0.18), 0.14, 0.16,
                                    facecolor=plt.cm.viridis(intensity), edgecolor='none')
            ax1.add_patch(small_rect)
    ax1.text(1.5, 3.2, 'Mel Spectrogram\n80 x 3000', ha='center', va='center', fontsize=8)

    # Arrow from waveform to mel
    ax1.annotate('', xy=(1.5, 4.8), xytext=(1.5, 5.2),
                 arrowprops=dict(arrowstyle='->', color=colors['border'], lw=1.5))

    # --- Encoder Section ---
    encoder_box = FancyBboxPatch((4, 2.5), 4, 5.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['encoder'], edgecolor=colors['border'], linewidth=2)
    ax1.add_patch(encoder_box)
    ax1.text(6, 7.5, 'Audio Encoder', ha='center', va='center', fontsize=11, fontweight='bold')

    # Conv layers
    conv_box = FancyBboxPatch((4.3, 6.5), 3.4, 0.7, boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor=colors['border'], linewidth=1)
    ax1.add_patch(conv_box)
    ax1.text(6, 6.85, 'Conv1D + Conv1D (stride 2)', ha='center', va='center', fontsize=8)

    # Position encoding
    pos_box = FancyBboxPatch((4.3, 5.6), 3.4, 0.6, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor=colors['border'], linewidth=1)
    ax1.add_patch(pos_box)
    ax1.text(6, 5.9, 'Sinusoidal Position Encoding', ha='center', va='center', fontsize=8)

    # Transformer blocks
    for i in range(4):
        y = 5.0 - i * 0.6
        block = FancyBboxPatch((4.3, y), 3.4, 0.5, boxstyle="round,pad=0.03",
                                facecolor='white', edgecolor=colors['border'], linewidth=1)
        ax1.add_patch(block)
        if i < 3:
            ax1.text(6, y + 0.25, 'Self-Attention + FFN', ha='center', va='center', fontsize=7)
        else:
            ax1.text(6, y + 0.25, '... x N layers', ha='center', va='center', fontsize=7)

    # Encoder output
    ax1.text(6, 2.7, 'Encoder Output\n[1500, dim]', ha='center', va='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#C8E6C9', edgecolor='none'))

    # Arrow from audio to encoder
    ax1.annotate('', xy=(4, 5), xytext=(3.5, 5),
                 arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))

    # --- Decoder Section ---
    decoder_box = FancyBboxPatch((9, 2.5), 4, 5.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['decoder'], edgecolor=colors['border'], linewidth=2)
    ax1.add_patch(decoder_box)
    ax1.text(11, 7.5, 'Text Decoder', ha='center', va='center', fontsize=11, fontweight='bold')

    # Token embedding
    tok_box = FancyBboxPatch((9.3, 6.5), 3.4, 0.7, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor=colors['border'], linewidth=1)
    ax1.add_patch(tok_box)
    ax1.text(11, 6.85, 'Token + Position Embedding', ha='center', va='center', fontsize=8)

    # Decoder blocks with cross-attention
    for i in range(4):
        y = 5.8 - i * 0.8
        # Self attention
        block = FancyBboxPatch((9.3, y), 3.4, 0.35, boxstyle="round,pad=0.02",
                                facecolor='white', edgecolor=colors['border'], linewidth=1)
        ax1.add_patch(block)
        ax1.text(11, y + 0.18, 'Masked Self-Attention', ha='center', va='center', fontsize=7)

        if i < 3:
            # Cross attention
            cross_block = FancyBboxPatch((9.3, y - 0.4), 3.4, 0.35, boxstyle="round,pad=0.02",
                                          facecolor=colors['attention'], edgecolor=colors['border'], linewidth=1)
            ax1.add_patch(cross_block)
            ax1.text(11, y - 0.22, 'Cross-Attention', ha='center', va='center', fontsize=7)
        else:
            ax1.text(11, y - 0.3, '... x N layers', ha='center', va='center', fontsize=7)

    # Cross-attention arrow from encoder
    ax1.annotate('', xy=(9.3, 5.2), xytext=(8, 4),
                 arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2,
                                connectionstyle='arc3,rad=0.2'))
    ax1.text(8.5, 5.0, 'K, V from\nEncoder', ha='center', va='center', fontsize=7, color='#FF9800')

    # LM Head
    lm_box = FancyBboxPatch((9.3, 2.8), 3.4, 0.6, boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor=colors['border'], linewidth=1)
    ax1.add_patch(lm_box)
    ax1.text(11, 3.1, 'LM Head (Linear)', ha='center', va='center', fontsize=8)

    # --- Output Section ---
    output_box = FancyBboxPatch((14, 4), 3.5, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['output'], edgecolor=colors['border'], linewidth=2)
    ax1.add_patch(output_box)
    ax1.text(15.75, 5.5, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(15.75, 4.8, '"Hello, how are', ha='center', va='center', fontsize=9)
    ax1.text(15.75, 4.4, 'you doing today?"', ha='center', va='center', fontsize=9)

    # Arrow to output
    ax1.annotate('', xy=(14, 5), xytext=(13, 5),
                 arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))

    # --- Token Input to Decoder ---
    token_input = FancyBboxPatch((9.5, 7.8), 3, 0.6, boxstyle="round,pad=0.05",
                                  facecolor=colors['token'], edgecolor=colors['border'], linewidth=1)
    ax1.add_patch(token_input)
    ax1.text(11, 8.1, 'Previous Tokens (shifted)', ha='center', va='center', fontsize=8)

    ax1.annotate('', xy=(11, 7.2), xytext=(11, 7.8),
                 arrowprops=dict(arrowstyle='->', color=colors['border'], lw=1.5))

    # === Panel 2: Audio Preprocessing ===
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('Audio Preprocessing Pipeline', fontsize=12, fontweight='bold', pad=10)

    # Pipeline steps
    steps = [
        ('Raw Audio\n16kHz', 0.5),
        ('Frame &\nWindow\n25ms', 2.5),
        ('FFT\nFrequency', 4.5),
        ('Mel Filter\nBank', 6.5),
        ('Log Mel\nSpectrogram', 8.5)
    ]

    for i, (label, x) in enumerate(steps):
        box = FancyBboxPatch((x, 2.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor=colors['audio'] if i == 0 else colors['encoder'] if i < 4 else colors['output'],
                              edgecolor=colors['border'], linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(x + 0.75, 3.25, label, ha='center', va='center', fontsize=8)

        if i < len(steps) - 1:
            ax2.annotate('', xy=(steps[i+1][1], 3.25), xytext=(x + 1.5, 3.25),
                         arrowprops=dict(arrowstyle='->', color=colors['border'], lw=1.5))

    # Dimensions
    dims = ['[samples]', '[frames, 400]', '[frames, 201]', '[frames, 80]', '[80, 3000]']
    for i, (dim, (_, x)) in enumerate(zip(dims, steps)):
        ax2.text(x + 0.75, 2.2, dim, ha='center', va='center', fontsize=7, style='italic')

    # Key parameters
    params = [
        'Sample Rate: 16,000 Hz',
        'Window: 25ms (400 samples)',
        'Hop: 10ms (160 samples)',
        'Mel Bins: 80',
        'Max Duration: 30 seconds'
    ]
    for i, param in enumerate(params):
        ax2.text(5, 1.5 - i*0.25, param, ha='center', va='center', fontsize=8)

    # === Panel 3: Multi-task Token Format ===
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.set_xlim(0, 14)
    ax3.set_ylim(0, 6)
    ax3.axis('off')
    ax3.set_title('Whisper Multi-task Format: Special Tokens', fontsize=12, fontweight='bold', pad=10)

    # Token sequence
    tokens = [
        ('<|startoftranscript|>', colors['token'], 0.5),
        ('<|en|>', '#BBDEFB', 3),
        ('<|transcribe|>', '#E1BEE7', 5),
        ('<|notimestamps|>', '#B2DFDB', 7.5),
        ('text tokens...', 'white', 10),
        ('<|endoftext|>', colors['token'], 12.5)
    ]

    y = 4
    for token, color, x in tokens:
        width = 2 if len(token) > 15 else 1.5
        box = FancyBboxPatch((x, y), width, 0.8, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor=colors['border'], linewidth=1)
        ax3.add_patch(box)
        ax3.text(x + width/2, y + 0.4, token, ha='center', va='center', fontsize=7)

    # Labels
    labels = [
        ('Start', 1.25),
        ('Language', 3.75),
        ('Task', 5.75),
        ('Timestamps', 8.25),
        ('Content', 11),
        ('End', 13.25)
    ]
    for label, x in labels:
        ax3.text(x, y - 0.4, label, ha='center', va='center', fontsize=8, style='italic')

    # Task options
    ax3.text(3, 2.5, 'Language Options:', ha='left', va='center', fontsize=9, fontweight='bold')
    ax3.text(3, 2.1, '<|en|>, <|zh|>, <|es|>, <|fr|>, ... (99 languages)', ha='left', va='center', fontsize=8)

    ax3.text(3, 1.5, 'Task Options:', ha='left', va='center', fontsize=9, fontweight='bold')
    ax3.text(3, 1.1, '<|transcribe|> - Speech to same-language text', ha='left', va='center', fontsize=8)
    ax3.text(3, 0.7, '<|translate|> - Speech to English text', ha='left', va='center', fontsize=8)

    # Model sizes
    ax3.text(10, 2.5, 'Model Sizes:', ha='left', va='center', fontsize=9, fontweight='bold')
    sizes = ['tiny (39M)', 'base (74M)', 'small (244M)', 'medium (769M)', 'large (1.5B)']
    for i, size in enumerate(sizes):
        ax3.text(10, 2.1 - i*0.35, f'  {size}', ha='left', va='center', fontsize=8)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'whisper_architecture.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved Whisper architecture diagram to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_whisper_architecture()
