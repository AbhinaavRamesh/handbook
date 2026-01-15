#!/usr/bin/env python3
"""
Head Specialization Grid Visualization

Shows different attention patterns learned by different heads in multi-head attention.
Each head naturally specializes in different types of relationships:
- Local attention (nearby tokens)
- Syntactic patterns (subject-verb)
- Long-range dependencies
- Positional patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set random seed for reproducibility
np.random.seed(42)

# Define sequence (tokens)
tokens = ["The", "president", "met", "with", "the", "ambassador", ",", "who", "discussed", "trade"]
seq_len = len(tokens)

# Custom purple colormap
purple_cmap = LinearSegmentedColormap.from_list(
    'purple_attention',
    ['#ffffff', '#e8daef', '#d5a6e6', '#9b59b6', '#7d3c98', '#4a235a']
)

def create_local_attention(seq_len, width=2):
    """Head 1: Local attention - focuses on nearby tokens"""
    attention = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            dist = abs(i - j)
            if dist <= width:
                attention[i, j] = np.exp(-dist * 0.8) + np.random.uniform(0, 0.1)
    # Normalize rows
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention

def create_syntactic_attention(seq_len, tokens):
    """Head 2: Syntactic patterns - subject-verb relationships"""
    attention = np.random.uniform(0.02, 0.08, (seq_len, seq_len))
    # "president" (1) attends to "met" (2) - subject to verb
    attention[2, 1] = 0.7  # verb attends to subject
    attention[1, 2] = 0.5  # subject attends to verb
    # "ambassador" (5) attends to "discussed" (8)
    attention[8, 5] = 0.65
    attention[5, 8] = 0.4
    # "who" (7) attends to "ambassador" (5) - relative pronoun
    attention[7, 5] = 0.8
    # Normalize
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention

def create_longrange_attention(seq_len):
    """Head 3: Long-range dependencies"""
    attention = np.random.uniform(0.02, 0.05, (seq_len, seq_len))
    # Long-range connections
    attention[7, 1] = 0.4   # "who" connects to "president" context
    attention[8, 2] = 0.5   # "discussed" connects to "met"
    attention[9, 1] = 0.45  # "trade" connects to "president"
    attention[9, 5] = 0.5   # "trade" connects to "ambassador"
    attention[6, 2] = 0.3   # comma connects back to verb
    # Some diagonal for context
    for i in range(seq_len):
        attention[i, i] = 0.2
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention

def create_positional_attention(seq_len, pattern='first'):
    """Head 4: Positional patterns - attends to specific positions"""
    attention = np.random.uniform(0.02, 0.05, (seq_len, seq_len))
    if pattern == 'first':
        # Strong attention to first token (CLS-like)
        attention[:, 0] = np.linspace(0.6, 0.3, seq_len)
    elif pattern == 'prev':
        # Strong attention to previous token
        for i in range(1, seq_len):
            attention[i, i-1] = 0.7
        attention[0, 0] = 0.7
    elif pattern == 'next':
        # Attention to next token
        for i in range(seq_len-1):
            attention[i, i+1] = 0.6
        attention[seq_len-1, seq_len-1] = 0.6
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention

def create_content_attention(seq_len, tokens):
    """Head 5: Content-based - attends to nouns/important words"""
    attention = np.random.uniform(0.03, 0.07, (seq_len, seq_len))
    # Nouns: president(1), ambassador(5), trade(9)
    noun_indices = [1, 5, 9]
    for i in range(seq_len):
        for j in noun_indices:
            attention[i, j] += 0.25
    # Self attention for nouns
    for idx in noun_indices:
        attention[idx, idx] = 0.3
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention

def create_verb_attention(seq_len, tokens):
    """Head 6: Verb-focused attention"""
    attention = np.random.uniform(0.03, 0.06, (seq_len, seq_len))
    # Verbs: met(2), discussed(8)
    verb_indices = [2, 8]
    for i in range(seq_len):
        for j in verb_indices:
            attention[i, j] += 0.3
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention

def create_separator_attention(seq_len, tokens):
    """Head 7: Separator/punctuation attention"""
    attention = np.random.uniform(0.05, 0.1, (seq_len, seq_len))
    # Comma at position 6 gets attention
    # Everything after comma attends to it
    for i in range(7, seq_len):
        attention[i, 6] = 0.5
    # Things before comma attend to beginning
    for i in range(6):
        attention[i, 0] = 0.3
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention

def create_uniform_attention(seq_len):
    """Head 8: Broad/uniform attention - aggregates global context"""
    attention = np.ones((seq_len, seq_len)) + np.random.uniform(-0.1, 0.1, (seq_len, seq_len))
    attention = np.maximum(attention, 0.1)
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention

# Create all attention patterns
attention_patterns = [
    (create_local_attention(seq_len), "Head 1: Local\n(nearby tokens)"),
    (create_syntactic_attention(seq_len, tokens), "Head 2: Syntactic\n(subject-verb)"),
    (create_longrange_attention(seq_len), "Head 3: Long-Range\n(dependencies)"),
    (create_positional_attention(seq_len, 'first'), "Head 4: Positional\n(attend to start)"),
    (create_content_attention(seq_len, tokens), "Head 5: Content\n(nouns)"),
    (create_verb_attention(seq_len, tokens), "Head 6: Content\n(verbs)"),
    (create_separator_attention(seq_len, tokens), "Head 7: Structural\n(separators)"),
    (create_uniform_attention(seq_len), "Head 8: Global\n(broad context)"),
]

# Create figure with 2x4 grid
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('Multi-Head Attention: Each Head Learns Different Patterns',
             fontsize=16, fontweight='bold', y=0.98)

for idx, (ax, (attention, title)) in enumerate(zip(axes.flatten(), attention_patterns)):
    im = ax.imshow(attention, cmap=purple_cmap, aspect='auto', vmin=0, vmax=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#4a235a')

    # Set tick labels
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    # Labels
    ax.set_xlabel('Key (attending to)', fontsize=9)
    ax.set_ylabel('Query (from)', fontsize=9)

    # Add grid
    ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.grid(which='minor', color='#d5a6e6', linestyle='-', linewidth=0.5, alpha=0.5)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Attention Weight', fontsize=11)

# Add explanatory text
fig.text(0.5, 0.02,
         'Each head in multi-head attention naturally specializes in different relationship types.\n'
         'This diversity emerges during training without explicit supervision.',
         ha='center', fontsize=10, style='italic', color='#666666')

plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/head_specialization_grid.png',
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Generated: head_specialization_grid.png")
