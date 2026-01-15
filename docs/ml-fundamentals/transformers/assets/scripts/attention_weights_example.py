"""
Machine Translation Attention Weights Visualization

This script demonstrates cross-attention in a sequence-to-sequence model:
- Source: "Je suis etudiant" (French)
- Target: "I am a student" (English)

Shows how the decoder attends to different source tokens when generating
each target token, creating an alignment matrix that reveals the
translation correspondence.

This is the classic attention visualization that made attention mechanisms
famous in the Bahdanau et al. (2015) paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_translation_attention_viz():
    """Create a heatmap showing French-to-English translation attention."""

    # Source and target sentences
    source_tokens = ["Je", "suis", "etudiant", "<EOS>"]
    target_tokens = ["<BOS>", "I", "am", "a", "student", "<EOS>"]

    n_source = len(source_tokens)
    n_target = len(target_tokens)

    # Create realistic attention weights
    # Each row = target token attending to source tokens
    # Weights should reflect natural translation alignment
    attention_weights = np.array([
        # <BOS> - attends mostly to beginning
        [0.6, 0.2, 0.1, 0.1],
        # I - attends to "Je"
        [0.85, 0.08, 0.05, 0.02],
        # am - attends to "suis"
        [0.10, 0.80, 0.08, 0.02],
        # a - no direct French equivalent, distributes attention
        [0.15, 0.25, 0.50, 0.10],
        # student - attends to "etudiant"
        [0.05, 0.10, 0.82, 0.03],
        # <EOS> - attends to <EOS>
        [0.05, 0.05, 0.10, 0.80],
    ])

    # Create the main figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=11)

    # Set ticks and labels
    ax.set_xticks(range(n_source))
    ax.set_yticks(range(n_target))
    ax.set_xticklabels(source_tokens, fontsize=12, fontweight='bold')
    ax.set_yticklabels(target_tokens, fontsize=12, fontweight='bold')

    # Labels
    ax.set_xlabel('Source (French)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target (English)', fontsize=12, fontweight='bold')

    # Title
    ax.set_title('Cross-Attention in Machine Translation\n"Je suis etudiant" -> "I am a student"',
                 fontsize=14, fontweight='bold', pad=20)

    # Add attention weight values to cells
    for i in range(n_target):
        for j in range(n_source):
            weight = attention_weights[i, j]
            text_color = 'white' if weight > 0.5 else 'black'
            ax.text(j, i, f'{weight:.2f}', ha='center', va='center',
                   fontsize=11, color=text_color, fontweight='bold')

    # Add annotation explaining the alignment
    # Draw lines connecting aligned words
    ax.annotate('', xy=(0, 1), xytext=(0, 1.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                annotation_clip=False)

    # Add text annotations for key alignments
    alignment_notes = [
        (0, 1, '"Je" -> "I"', 'right'),
        (1, 2, '"suis" -> "am"', 'right'),
        (2, 4, '"etudiant" -> "student"', 'right'),
    ]

    plt.tight_layout()

    # Add a text box with insights
    textstr = ('Key Observations:\n'
               '1. "I" attends strongly to "Je" (0.85)\n'
               '2. "am" attends strongly to "suis" (0.80)\n'
               '3. "student" attends to "etudiant" (0.82)\n'
               '4. "a" (article) has no direct French\n'
               '   equivalent, so attention is spread\n'
               '5. Each row sums to 1.0')

    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    fig.text(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
             bbox=props, family='monospace')

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/attention_weights_example.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")

    # Also create a second figure showing the alignment as a diagram
    create_alignment_diagram(source_tokens, target_tokens, attention_weights)

    return output_path

def create_alignment_diagram(source_tokens, target_tokens, attention_weights):
    """Create a diagram showing word alignments with connecting lines."""

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 3)
    ax.axis('off')

    # Position source tokens at top
    source_y = 2.5
    target_y = 0.5
    source_positions = np.linspace(1, 8, len(source_tokens))
    target_positions = np.linspace(0, 9, len(target_tokens))

    # Draw source tokens
    for i, (pos, token) in enumerate(zip(source_positions, source_tokens)):
        ax.text(pos, source_y, token, ha='center', va='center',
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='blue'))

    # Draw target tokens
    for i, (pos, token) in enumerate(zip(target_positions, target_tokens)):
        ax.text(pos, target_y, token, ha='center', va='center',
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='green'))

    # Draw attention lines (only for weights > 0.3)
    for i, target_pos in enumerate(target_positions):
        for j, source_pos in enumerate(source_positions):
            weight = attention_weights[i, j]
            if weight > 0.3:
                alpha = min(weight * 1.2, 1.0)
                linewidth = weight * 3
                ax.plot([source_pos, target_pos], [source_y - 0.3, target_y + 0.3],
                       'b-', alpha=alpha, linewidth=linewidth)

    # Labels
    ax.text(4.5, 3, 'Source: French', ha='center', fontsize=12, fontweight='bold', color='blue')
    ax.text(4.5, -0.2, 'Target: English', ha='center', fontsize=12, fontweight='bold', color='green')
    ax.set_title('Word Alignment via Attention\nLine thickness proportional to attention weight',
                fontsize=14, fontweight='bold', y=1.05)

    # Save
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/attention_alignment_diagram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    create_translation_attention_viz()
