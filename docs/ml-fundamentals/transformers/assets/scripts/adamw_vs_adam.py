#!/usr/bin/env python3
"""
AdamW vs Adam Comparison
2-panel comparison showing:
- Left: Effective regularization strength over training (Adam+L2 varies with LR)
- Right: AdamW (constant regularization)
- Bar chart showing final performance difference
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

# Configuration
TOTAL_STEPS = 100_000
WARMUP_STEPS = 10_000
BASE_LR = 1e-4
WEIGHT_DECAY = 0.01
LAMBDA_L2 = 0.01

# Generate learning rate schedule (warmup + cosine decay)
steps = np.arange(TOTAL_STEPS)

def get_lr_schedule(steps, warmup_steps, total_steps, base_lr):
    lr = np.zeros_like(steps, dtype=float)

    # Warmup
    warmup_mask = steps < warmup_steps
    lr[warmup_mask] = base_lr * (steps[warmup_mask] / warmup_steps)

    # Cosine decay
    decay_mask = steps >= warmup_steps
    progress = (steps[decay_mask] - warmup_steps) / (total_steps - warmup_steps)
    lr[decay_mask] = base_lr * 0.5 * (1 + np.cos(np.pi * progress))

    return lr

lr_schedule = get_lr_schedule(steps, WARMUP_STEPS, TOTAL_STEPS, BASE_LR)

# For Adam+L2: effective regularization varies with adaptive learning rate
# Simplified model: effective_reg = lambda * lr / sqrt(v_t)
# As v_t stabilizes, effective_reg ~ lambda * lr
effective_reg_adam = LAMBDA_L2 * lr_schedule  # Proportional to LR

# Add some noise to simulate per-parameter variation
noise = np.random.normal(0, 0.00001, len(steps))
effective_reg_adam = effective_reg_adam + noise
effective_reg_adam = np.clip(effective_reg_adam, 0, None)

# For AdamW: constant regularization
effective_reg_adamw = np.full_like(steps, WEIGHT_DECAY * BASE_LR, dtype=float)

# Create figure with 3 panels
fig = plt.figure(figsize=(15, 6))

# Left panel: Adam+L2 effective regularization
ax1 = fig.add_subplot(131)

# Shade warmup region
ax1.axvspan(0, WARMUP_STEPS, alpha=0.2, color='#e6f2ff', zorder=0)
ax1.axvline(x=WARMUP_STEPS, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Plot effective regularization and LR
ax1.plot(steps, effective_reg_adam * 1e6, color='#d62728', linewidth=2, label='Effective reg strength')
ax1.plot(steps, lr_schedule * 1e4, color='#1f77b4', linewidth=2, linestyle='--', alpha=0.7, label='Learning rate (scaled)')

ax1.set_xlabel('Training Steps', fontsize=11)
ax1.set_ylabel('Regularization Strength (x10$^{-6}$)', fontsize=11)
ax1.set_title('Adam + L2 Regularization\n(Regularization couples to LR)', fontsize=12, fontweight='bold', color='#d62728')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, TOTAL_STEPS)

# Annotation
ax1.annotate('Regularization\nvaries with LR!',
            xy=(50000, effective_reg_adam[50000] * 1e6),
            xytext=(60000, 0.6),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.9),
            color='#d62728')

# Middle panel: AdamW constant regularization
ax2 = fig.add_subplot(132)

# Shade warmup region
ax2.axvspan(0, WARMUP_STEPS, alpha=0.2, color='#e6f2ff', zorder=0)
ax2.axvline(x=WARMUP_STEPS, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Plot constant regularization
ax2.plot(steps, effective_reg_adamw * 1e6, color='#2ca02c', linewidth=2.5, label='Effective reg strength')
ax2.plot(steps, lr_schedule * 1e4, color='#1f77b4', linewidth=2, linestyle='--', alpha=0.7, label='Learning rate (scaled)')

ax2.set_xlabel('Training Steps', fontsize=11)
ax2.set_ylabel('Regularization Strength (x10$^{-6}$)', fontsize=11)
ax2.set_title('AdamW (Decoupled Weight Decay)\n(Constant regularization)', fontsize=12, fontweight='bold', color='#2ca02c')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, TOTAL_STEPS)
ax2.set_ylim(ax1.get_ylim())  # Match y-axis

# Annotation
ax2.annotate('Constant regularization\nregardless of LR!',
            xy=(70000, effective_reg_adamw[70000] * 1e6),
            xytext=(55000, 0.6),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='#e6ffe6', alpha=0.9),
            color='#2ca02c')

# Right panel: Performance comparison bar chart
ax3 = fig.add_subplot(133)

# Simulated performance metrics
metrics = ['Val Accuracy', 'Test Accuracy', 'Perplexity\n(lower=better)']
adam_scores = [85.2, 84.8, 15.3]
adamw_scores = [86.1, 85.9, 14.1]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x - width/2, adam_scores, width, label='Adam + L2', color='#d62728', alpha=0.8)
bars2 = ax3.bar(x + width/2, adamw_scores, width, label='AdamW', color='#2ca02c', alpha=0.8)

# Add value labels on bars
for bar, score in zip(bars1, adam_scores):
    ax3.annotate(f'{score}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
for bar, score in zip(bars2, adamw_scores):
    ax3.annotate(f'{score}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)

ax3.set_ylabel('Score (%)', fontsize=11)
ax3.set_title('Final Performance\n(Typical Transformer)', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=10)
ax3.legend(loc='lower right', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 100)

# Improvement annotation
improvement = (adamw_scores[1] - adam_scores[1])
ax3.annotate(f'+{improvement:.1f}%\nbetter',
            xy=(1 + width/2, adamw_scores[1]),
            xytext=(1.5, 75),
            fontsize=11, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2),
            bbox=dict(boxstyle='round', facecolor='#e6ffe6', alpha=0.9),
            color='#2ca02c')

# Add formula annotations at bottom
formulas = """
Adam + L2:  $\\theta_t \\leftarrow \\theta_t - lr \\cdot (\\hat{m}_t + \\lambda\\theta_t)$  (regularization divided by adaptive LR)
AdamW:      $\\theta_t \\leftarrow \\theta_t - lr \\cdot \\hat{m}_t - wd \\cdot \\theta_t$  (decoupled weight decay)
"""
fig.text(0.5, -0.02, formulas.strip(), ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.95), fontfamily='monospace')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save
output_path = Path(__file__).parent.parent / 'images' / 'adamw_vs_adam.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
