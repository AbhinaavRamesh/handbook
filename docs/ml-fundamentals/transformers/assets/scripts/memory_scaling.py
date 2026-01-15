#!/usr/bin/env python3
"""
Memory Scaling Analysis: Multi-Head Attention

Shows how memory scales with sequence length for different head configurations.
Key insight: Parallelization across heads doesn't increase total memory -
the attention matrices are smaller per head (d_k instead of d_model).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configuration
d_model = 512
batch_size = 32
float_size = 4  # bytes for float32

def calculate_attention_memory(seq_len, num_heads, d_model, batch_size):
    """
    Calculate memory for attention matrices.
    Per head: batch x seq x seq attention matrix (but at d_k scale)
    Total: batch x num_heads x seq x seq
    """
    # Attention matrix memory
    attention_memory = batch_size * num_heads * seq_len * seq_len * float_size

    # Q, K, V matrices (per head dimension is d_k = d_model/num_heads)
    d_k = d_model // num_heads
    qkv_memory = 3 * batch_size * seq_len * d_model * float_size

    # Output memory
    output_memory = batch_size * seq_len * d_model * float_size

    return attention_memory + qkv_memory + output_memory

def calculate_single_head_memory(seq_len, d_model, batch_size):
    """Calculate memory for single-head attention"""
    # Attention matrix
    attention_memory = batch_size * seq_len * seq_len * float_size

    # Q, K, V matrices (full d_model)
    qkv_memory = 3 * batch_size * seq_len * d_model * float_size

    # Output
    output_memory = batch_size * seq_len * d_model * float_size

    return attention_memory + qkv_memory + output_memory

# Sequence lengths to analyze
seq_lengths = np.array([64, 128, 256, 512, 768, 1024, 1536, 2048])

# Calculate memory for different configurations
memory_1h = [calculate_single_head_memory(s, d_model, batch_size) / (1024**2)
             for s in seq_lengths]
memory_4h = [calculate_attention_memory(s, 4, d_model, batch_size) / (1024**2)
             for s in seq_lengths]
memory_8h = [calculate_attention_memory(s, 8, d_model, batch_size) / (1024**2)
             for s in seq_lengths]
memory_16h = [calculate_attention_memory(s, 16, d_model, batch_size) / (1024**2)
              for s in seq_lengths]

# Create figure with 2 subplots
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

# ============ Left: Line Plot ============
ax1 = fig.add_subplot(gs[0])

ax1.plot(seq_lengths, memory_1h, 'o-', color='#888888', linewidth=2.5,
         markersize=8, label='1 head (d_k=512)', markeredgecolor='white')
ax1.plot(seq_lengths, memory_4h, 's-', color='#d5a6e6', linewidth=2.5,
         markersize=8, label='4 heads (d_k=128)', markeredgecolor='white')
ax1.plot(seq_lengths, memory_8h, '^-', color='#9b59b6', linewidth=2.5,
         markersize=8, label='8 heads (d_k=64)', markeredgecolor='white')
ax1.plot(seq_lengths, memory_16h, 'D-', color='#4a235a', linewidth=2.5,
         markersize=8, label='16 heads (d_k=32)', markeredgecolor='white')

# Fill between to show O(n^2) growth
ax1.fill_between(seq_lengths, 0, memory_8h, alpha=0.1, color='#9b59b6')

ax1.set_xlabel('Sequence Length', fontsize=12)
ax1.set_ylabel('Memory Usage (MB)', fontsize=12)
ax1.set_title('Memory Scaling with Sequence Length\n(batch=32, d_model=512)',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Add O(n^2) annotation
ax1.annotate('O(n^2) growth\n(attention matrices)',
             xy=(1536, memory_8h[-2]), xytext=(1200, memory_8h[-1]*0.7),
             fontsize=10, style='italic',
             arrowprops=dict(arrowstyle='->', color='#666666'))

# ============ Right: Bar Chart Breakdown ============
ax2 = fig.add_subplot(gs[1])

seq_len = 512  # Fixed sequence length for breakdown
configs = ['1 head', '4 heads', '8 heads', '16 heads']
heads = [1, 4, 8, 16]

# Calculate component breakdowns
attention_mem = []
qkv_mem = []
output_mem = []

for h in heads:
    d_k = d_model // h if h > 1 else d_model
    # Attention matrices: batch x num_heads x seq x seq
    attn = batch_size * max(h, 1) * seq_len * seq_len * float_size / (1024**2)
    # QKV: 3 x batch x seq x d_model
    qkv = 3 * batch_size * seq_len * d_model * float_size / (1024**2)
    # Output: batch x seq x d_model
    out = batch_size * seq_len * d_model * float_size / (1024**2)

    attention_mem.append(attn)
    qkv_mem.append(qkv)
    output_mem.append(out)

x = np.arange(len(configs))
width = 0.5

# Stacked bars
bars1 = ax2.bar(x, attention_mem, width, label='Attention Matrices', color='#9b59b6')
bars2 = ax2.bar(x, qkv_mem, width, bottom=attention_mem, label='Q, K, V', color='#d5a6e6')
bars3 = ax2.bar(x, output_mem, width, bottom=np.array(attention_mem)+np.array(qkv_mem),
                label='Output', color='#e8daef')

# Add total labels
for i, (a, q, o) in enumerate(zip(attention_mem, qkv_mem, output_mem)):
    total = a + q + o
    ax2.text(i, total + 5, f'{total:.1f} MB', ha='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Configuration', fontsize=12)
ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
ax2.set_title(f'Memory Breakdown (seq_len={seq_len})\nWhy More Heads != More Memory',
              fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=10)
ax2.legend(loc='upper right', fontsize=10)

# Add insight box
insight_text = '\n'.join([
    'Key Insight:',
    '-' * 30,
    'More heads = same total memory!',
    '',
    'Why?',
    '- Attention: h matrices of (n x n)',
    '  but each head uses d_k = d_model/h',
    '- Q, K, V: Always batch x n x d_model',
    '- Total: Same compute, parallel exec',
])
props = dict(boxstyle='round,pad=0.5', facecolor='#f5eef8', edgecolor='#9b59b6', alpha=0.95)
ax2.text(0.02, 0.98, insight_text, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=props, family='monospace')

# Main title
fig.suptitle('Multi-Head Attention Memory Analysis', fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/memory_scaling.png',
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Generated: memory_scaling.png")
