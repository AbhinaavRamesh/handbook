#!/usr/bin/env python3
"""
Curriculum Learning - Sequence Length Curriculum
Stacked area chart showing training phases:
- Phase 1 (10%): len=128
- Phase 2 (20%): len=256
- Phase 3 (70%): len=512
Overlay throughput line (tokens/sec)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
TOTAL_STEPS = 100_000

# Phase boundaries (as fractions)
PHASE1_END = 0.10  # 10%
PHASE2_END = 0.30  # 10% + 20% = 30%
PHASE3_END = 1.00  # 30% + 70% = 100%

# Sequence lengths per phase
SEQ_LENS = {1: 128, 2: 256, 3: 512}

# Base throughput (tokens/sec) - decreases with longer sequences
BASE_THROUGHPUT = {128: 150000, 256: 80000, 512: 35000}

steps = np.arange(TOTAL_STEPS)
step_fractions = steps / TOTAL_STEPS

# Determine phase for each step
phases = np.zeros_like(steps)
phases[step_fractions < PHASE1_END] = 1
phases[(step_fractions >= PHASE1_END) & (step_fractions < PHASE2_END)] = 2
phases[step_fractions >= PHASE2_END] = 3

# Create sequence length array
seq_lengths = np.array([SEQ_LENS[int(p)] for p in phases])

# Calculate throughput (with some variation)
np.random.seed(42)
throughput = np.zeros_like(steps, dtype=float)
for phase in [1, 2, 3]:
    mask = phases == phase
    base = BASE_THROUGHPUT[SEQ_LENS[phase]]
    # Add gradual warmup within each phase and some noise
    phase_steps = np.cumsum(mask) * mask
    max_phase_steps = phase_steps.max()
    if max_phase_steps > 0:
        warmup_factor = np.minimum(phase_steps / (max_phase_steps * 0.1), 1.0)
        throughput[mask] = base * warmup_factor[mask] * (1 + np.random.normal(0, 0.05, mask.sum()))

# Create figure
fig, ax1 = plt.subplots(figsize=(14, 7))

# Colors for phases
phase_colors = {
    1: '#2ca02c',   # Green for short sequences
    2: '#1f77b4',   # Blue for medium
    3: '#ff7f0e',   # Orange for long
}

# Create stacked area data
area_128 = np.where(phases == 1, 128, 0)
area_256 = np.where(phases == 2, 256, 0)
area_512 = np.where(phases == 3, 512, 0)

# Stacked area chart
ax1.fill_between(steps, 0, area_128, color=phase_colors[1], alpha=0.7,
                  label='Phase 1: len=128 (10%)')
ax1.fill_between(steps, 0, area_256, color=phase_colors[2], alpha=0.7,
                  label='Phase 2: len=256 (20%)')
ax1.fill_between(steps, 0, area_512, color=phase_colors[3], alpha=0.7,
                  label='Phase 3: len=512 (70%)')

# Phase boundaries
phase1_step = int(PHASE1_END * TOTAL_STEPS)
phase2_step = int(PHASE2_END * TOTAL_STEPS)

ax1.axvline(x=phase1_step, color='gray', linestyle='--', linewidth=2, alpha=0.8)
ax1.axvline(x=phase2_step, color='gray', linestyle='--', linewidth=2, alpha=0.8)

# Phase labels
ax1.text(phase1_step/2, 580, 'Phase 1\n(10%)', fontsize=11, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
ax1.text((phase1_step + phase2_step)/2, 580, 'Phase 2\n(20%)', fontsize=11, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
ax1.text((phase2_step + TOTAL_STEPS)/2, 580, 'Phase 3\n(70%)', fontsize=11, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Sequence Length', fontsize=12, color='black')
ax1.set_xlim(0, TOTAL_STEPS)
ax1.set_ylim(0, 650)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax1.grid(True, alpha=0.3, axis='x')

# Secondary y-axis for throughput
ax2 = ax1.twinx()

# Smooth the throughput for plotting
from scipy.ndimage import uniform_filter1d
throughput_smooth = uniform_filter1d(throughput, size=500)

ax2.plot(steps, throughput_smooth / 1000, color='#d62728', linewidth=2.5,
         label='Throughput (tokens/sec)', zorder=10)
ax2.set_ylabel('Throughput (K tokens/sec)', fontsize=12, color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')
ax2.set_ylim(0, 200)

# Throughput annotations
ax2.annotate(f'{BASE_THROUGHPUT[128]/1000:.0f}K tok/s',
            xy=(phase1_step * 0.5, BASE_THROUGHPUT[128]/1000),
            xytext=(phase1_step * 0.5, BASE_THROUGHPUT[128]/1000 + 20),
            fontsize=10, ha='center', color='#d62728',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.annotate(f'{BASE_THROUGHPUT[256]/1000:.0f}K tok/s',
            xy=((phase1_step + phase2_step)/2, BASE_THROUGHPUT[256]/1000),
            xytext=((phase1_step + phase2_step)/2, BASE_THROUGHPUT[256]/1000 + 15),
            fontsize=10, ha='center', color='#d62728',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.annotate(f'{BASE_THROUGHPUT[512]/1000:.0f}K tok/s',
            xy=((phase2_step + TOTAL_STEPS)/2, BASE_THROUGHPUT[512]/1000),
            xytext=((phase2_step + TOTAL_STEPS)/2, BASE_THROUGHPUT[512]/1000 + 15),
            fontsize=10, ha='center', color='#d62728',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)

# Title
ax1.set_title('Sequence Length Curriculum Learning\n(Gradually increase complexity)',
              fontsize=14, fontweight='bold')

# Benefits annotation at bottom
benefits = """
Benefits of Curriculum Learning:
1. Faster initial training (shorter sequences = more throughput)
2. More stable early gradients (simpler examples first)
3. Better final performance (+0.5-1% perplexity improvement)
"""
fig.text(0.5, -0.02, benefits.strip(), ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.95, edgecolor='orange'),
         fontfamily='monospace')

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# Save
output_path = Path(__file__).parent.parent / 'images' / 'curriculum_learning.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
