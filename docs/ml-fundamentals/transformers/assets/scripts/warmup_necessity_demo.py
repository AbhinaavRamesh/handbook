#!/usr/bin/env python3
"""
Warmup Necessity Demo
Demonstrates why warmup is critical for transformer training:
- Left panel: WITH warmup (smooth convergence)
- Right panel: WITHOUT warmup (divergence/instability)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

# Configuration
TOTAL_STEPS = 50_000
WARMUP_STEPS = 5_000

# Generate synthetic training loss curves
steps = np.arange(TOTAL_STEPS)

# WITH WARMUP: Smooth exponential decay with minor noise
def generate_warmup_loss(steps, warmup_steps):
    loss = np.zeros_like(steps, dtype=float)

    # During warmup: gradual stabilization
    warmup_mask = steps < warmup_steps
    loss[warmup_mask] = 4.5 - 1.5 * (steps[warmup_mask] / warmup_steps) + \
                        np.random.normal(0, 0.05, warmup_mask.sum())

    # After warmup: smooth exponential decay
    post_warmup = steps >= warmup_steps
    post_steps = steps[post_warmup] - warmup_steps
    decay = np.exp(-post_steps / 15000)
    loss[post_warmup] = 3.0 * decay + 0.8 + np.random.normal(0, 0.03, post_warmup.sum())

    return loss

# WITHOUT WARMUP: Initial instability, slow recovery, higher final loss
def generate_no_warmup_loss(steps):
    loss = np.zeros_like(steps, dtype=float)

    # Initial phase: high instability with spikes
    early_phase = steps < 5000
    loss[early_phase] = 5.0 + 2.0 * np.random.random(early_phase.sum()) + \
                        np.abs(np.random.normal(0, 1.5, early_phase.sum()))
    # Add dramatic spikes
    spike_indices = np.random.choice(np.where(early_phase)[0], size=15, replace=False)
    loss[spike_indices] += np.random.uniform(3, 8, len(spike_indices))

    # Middle phase: gradual but noisy recovery
    mid_phase = (steps >= 5000) & (steps < 20000)
    mid_steps = steps[mid_phase] - 5000
    loss[mid_phase] = 5.5 - 1.5 * (mid_steps / 15000) + \
                      np.random.normal(0, 0.3, mid_phase.sum())
    # Still some instability
    spike_indices_mid = np.random.choice(np.where(mid_phase)[0], size=10, replace=False)
    loss[spike_indices_mid] += np.random.uniform(0.5, 2, len(spike_indices_mid))

    # Late phase: converges but to higher final loss
    late_phase = steps >= 20000
    late_steps = steps[late_phase] - 20000
    decay = np.exp(-late_steps / 20000)
    loss[late_phase] = 3.2 * decay + 1.0 + np.random.normal(0, 0.05, late_phase.sum())

    return loss

loss_warmup = generate_warmup_loss(steps, WARMUP_STEPS)
loss_no_warmup = generate_no_warmup_loss(steps)

# Create dual panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Color scheme
colors = {
    'loss': '#d62728',        # Red for loss
    'warmup_region': '#e6f2ff',
    'instability': '#ffcccc'
}

# LEFT PANEL: WITH WARMUP
ax1.axvspan(0, WARMUP_STEPS, alpha=0.3, color=colors['warmup_region'],
            label='Warmup Period', zorder=0)
ax1.axvline(x=WARMUP_STEPS, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax1.plot(steps, loss_warmup, color=colors['loss'], linewidth=1.5, alpha=0.9)

# Smoothed trend line
from scipy.ndimage import uniform_filter1d
smooth_warmup = uniform_filter1d(loss_warmup, size=500)
ax1.plot(steps, smooth_warmup, color='#1f77b4', linewidth=2.5,
         label='Smoothed trend', zorder=5)

# Annotations
ax1.annotate('Warmup\nPeriod',
            xy=(WARMUP_STEPS/2, 3.8), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax1.annotate(f'Final Loss: {loss_warmup[-1]:.2f}',
            xy=(45000, loss_warmup[-1] + 0.15),
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('WITH Warmup\n(Smooth Convergence)', fontsize=14, fontweight='bold', color='green')
ax1.set_xlim(0, TOTAL_STEPS)
ax1.set_ylim(0, 10)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# RIGHT PANEL: WITHOUT WARMUP
ax2.axvspan(0, 5000, alpha=0.3, color=colors['instability'],
            label='Instability Zone', zorder=0)

ax2.plot(steps, loss_no_warmup, color=colors['loss'], linewidth=1.5, alpha=0.9)

# Smoothed trend line
smooth_no_warmup = uniform_filter1d(loss_no_warmup, size=500)
ax2.plot(steps, smooth_no_warmup, color='#1f77b4', linewidth=2.5,
         label='Smoothed trend', zorder=5)

# Annotations
ax2.annotate('Gradient\nExplosion!',
            xy=(2500, 9),
            fontsize=11, ha='center', fontweight='bold',
            color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.annotate('Slow\nRecovery',
            xy=(12500, 4.2),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.annotate(f'Final Loss: {loss_no_warmup[-1]:.2f}',
            xy=(45000, loss_no_warmup[-1] + 0.15),
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Performance gap annotation
gap = loss_no_warmup[-1] - loss_warmup[-1]
gap_pct = (gap / loss_warmup[-1]) * 100

ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Training Loss', fontsize=12)
ax2.set_title('WITHOUT Warmup\n(Divergence & Instability)', fontsize=14, fontweight='bold', color='red')
ax2.set_xlim(0, TOTAL_STEPS)
ax2.set_ylim(0, 10)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Add performance gap annotation between panels
fig.text(0.5, 0.02, f'Performance Gap: {gap_pct:.1f}% higher final loss without warmup',
         ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.9, edgecolor='orange'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# Save
output_path = Path(__file__).parent.parent / 'images' / 'warmup_necessity_demo.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
