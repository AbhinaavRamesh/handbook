#!/usr/bin/env python3
"""
Learning Rate Schedules Comparison
Compares 4 different LR schedules over 100K training steps:
- Warmup + Cosine decay
- Warmup + Linear decay
- Warmup + Step decay
- No warmup baseline
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuration
TOTAL_STEPS = 100_000
WARMUP_STEPS = 10_000
BASE_LR = 1e-4
STEP_DECAY_GAMMA = 0.5
STEP_DECAY_INTERVAL = 25_000

# Create steps array
steps = np.arange(TOTAL_STEPS)

def warmup_factor(step, warmup_steps):
    """Linear warmup from 0 to 1"""
    return np.minimum(step / warmup_steps, 1.0)

def cosine_decay(step, warmup_steps, total_steps):
    """Cosine decay after warmup"""
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = np.clip(progress, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))

def linear_decay(step, warmup_steps, total_steps):
    """Linear decay after warmup"""
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = np.clip(progress, 0, 1)
    return 1 - progress

def step_decay(step, gamma, interval):
    """Step decay: multiply by gamma every interval steps"""
    return gamma ** (step // interval)

# Calculate learning rates for each schedule
lr_cosine = np.where(
    steps < WARMUP_STEPS,
    BASE_LR * warmup_factor(steps, WARMUP_STEPS),
    BASE_LR * cosine_decay(steps, WARMUP_STEPS, TOTAL_STEPS)
)

lr_linear = np.where(
    steps < WARMUP_STEPS,
    BASE_LR * warmup_factor(steps, WARMUP_STEPS),
    BASE_LR * linear_decay(steps, WARMUP_STEPS, TOTAL_STEPS)
)

lr_step = np.where(
    steps < WARMUP_STEPS,
    BASE_LR * warmup_factor(steps, WARMUP_STEPS),
    BASE_LR * step_decay(steps - WARMUP_STEPS, STEP_DECAY_GAMMA, STEP_DECAY_INTERVAL)
)

lr_no_warmup = BASE_LR * cosine_decay(steps, 0, TOTAL_STEPS)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Color scheme
colors = {
    'cosine': '#1f77b4',      # Blue
    'linear': '#2ca02c',      # Green
    'step': '#ff7f0e',        # Orange
    'no_warmup': '#d62728',   # Red
    'warmup_region': '#e6f2ff'
}

# Shade warmup region
ax.axvspan(0, WARMUP_STEPS, alpha=0.3, color=colors['warmup_region'],
           label='Warmup Region', zorder=0)
ax.axvline(x=WARMUP_STEPS, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Plot learning rate schedules
ax.plot(steps, lr_cosine * 1e4, label=r'Warmup + Cosine: $\frac{1}{2}(1 + \cos(\pi \cdot progress))$',
        color=colors['cosine'], linewidth=2.5)
ax.plot(steps, lr_linear * 1e4, label=r'Warmup + Linear: $1 - progress$',
        color=colors['linear'], linewidth=2.5)
ax.plot(steps, lr_step * 1e4, label=r'Warmup + Step: $\gamma^{\lfloor step / interval \rfloor}$',
        color=colors['step'], linewidth=2.5)
ax.plot(steps, lr_no_warmup * 1e4, label='No Warmup (Cosine only)',
        color=colors['no_warmup'], linewidth=2.5, linestyle='--')

# Add annotation for warmup period
ax.annotate('Warmup Period\n(10K steps)',
            xy=(WARMUP_STEPS/2, 0.5), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Step decay annotation
ax.annotate('Step decay\n(50% reduction)',
            xy=(35000, lr_step[35000] * 1e4 + 0.1),
            xytext=(45000, 0.7),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Formatting
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Learning Rate (x10$^{-4}$)', fontsize=12)
ax.set_title('Learning Rate Schedules Comparison\n(Base LR = 1e-4, Warmup = 10K steps)', fontsize=14, fontweight='bold')

ax.set_xlim(0, TOTAL_STEPS)
ax.set_ylim(0, 1.15)

# Format x-axis with K notation
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))

# Legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

# Grid
ax.grid(True, alpha=0.3, linestyle='-')
ax.set_axisbelow(True)

plt.tight_layout()

# Save
output_path = Path(__file__).parent.parent / 'images' / 'lr_schedules_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
