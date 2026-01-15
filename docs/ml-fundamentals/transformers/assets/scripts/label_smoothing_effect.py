#!/usr/bin/env python3
"""
Label Smoothing Effect
Two histograms side-by-side showing confidence distributions:
- epsilon=0: Sharp peaks at 0 and 1 (overconfident)
- epsilon=0.1: More distributed confidence (better calibrated)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

# Configuration
N_SAMPLES = 10_000
EPSILON = 0.1
VOCAB_SIZE = 30_000  # For calculating smoothed probability

# Generate synthetic model confidence data
def generate_hard_label_confidence(n_samples):
    """Model trained with hard labels (epsilon=0): overconfident"""
    # Most predictions are very confident (close to 0 or 1)
    # Correct predictions: very high confidence
    correct = np.random.beta(20, 1, int(n_samples * 0.85))  # Peaked near 1
    # Incorrect predictions: also often confident
    incorrect_confident = np.random.beta(1, 5, int(n_samples * 0.10))  # Some near 0
    incorrect_overconfident = np.random.beta(8, 2, int(n_samples * 0.05))  # Wrong but confident!

    return np.concatenate([correct, incorrect_confident, incorrect_overconfident])

def generate_smoothed_label_confidence(n_samples, epsilon):
    """Model trained with label smoothing: better calibrated"""
    # More distributed confidence, less overconfident
    # Correct predictions: confident but not extreme
    correct = np.random.beta(8, 1.5, int(n_samples * 0.85))  # Peaked but less extreme
    # Uncertain predictions more common
    uncertain = np.random.beta(3, 3, int(n_samples * 0.10))  # Middle range
    # Low confidence
    low_conf = np.random.beta(1.5, 5, int(n_samples * 0.05))

    return np.concatenate([correct, uncertain, low_conf])

conf_hard = generate_hard_label_confidence(N_SAMPLES)
conf_smooth = generate_smoothed_label_confidence(N_SAMPLES, EPSILON)

# Create dual panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Common histogram settings
bins = np.linspace(0, 1, 51)
hist_kwargs = {'bins': bins, 'edgecolor': 'white', 'linewidth': 0.5}

# Color scheme
colors = {
    'hard': '#d62728',     # Red for overconfident
    'smooth': '#2ca02c',   # Green for calibrated
}

# LEFT PANEL: Hard labels (epsilon = 0)
ax1.hist(conf_hard, color=colors['hard'], alpha=0.8, **hist_kwargs)

# Add vertical lines for problematic regions
ax1.axvline(x=0.9, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(x=0.1, color='darkred', linestyle='--', linewidth=2, alpha=0.7)

# Annotations
ax1.annotate('Overconfident\n(wrong predictions\nwith high confidence)',
            xy=(0.92, 1200),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.9),
            color='darkred')

ax1.annotate('Sharp peak at\nconfidence = 1',
            xy=(0.98, 3000),
            xytext=(0.75, 3500),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Statistics
mean_hard = np.mean(conf_hard)
std_hard = np.std(conf_hard)
over_90 = (conf_hard > 0.9).mean() * 100

stats_text = f'Mean confidence: {mean_hard:.3f}\nStd deviation: {std_hard:.3f}\n>90% confident: {over_90:.1f}%'
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax1.set_xlabel('Predicted Probability (Confidence)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title(r'Hard Labels ($\epsilon = 0$)' + '\nOverconfident Predictions',
              fontsize=14, fontweight='bold', color='darkred')
ax1.set_xlim(0, 1)
ax1.grid(True, alpha=0.3, axis='y')

# RIGHT PANEL: Label smoothing (epsilon = 0.1)
ax2.hist(conf_smooth, color=colors['smooth'], alpha=0.8, **hist_kwargs)

# Add target smoothing annotation
smooth_target = 1 - EPSILON + EPSILON / VOCAB_SIZE
other_target = EPSILON / (VOCAB_SIZE - 1)

ax2.axvline(x=0.9, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)

# Annotations
ax2.annotate('More distributed\nconfidence',
            xy=(0.6, 600),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#e6ffe6', alpha=0.9),
            color='darkgreen')

ax2.annotate('Peak shifted from 1.0\n(less extreme)',
            xy=(0.88, 2200),
            xytext=(0.65, 3000),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Statistics
mean_smooth = np.mean(conf_smooth)
std_smooth = np.std(conf_smooth)
over_90_smooth = (conf_smooth > 0.9).mean() * 100

stats_text = f'Mean confidence: {mean_smooth:.3f}\nStd deviation: {std_smooth:.3f}\n>90% confident: {over_90_smooth:.1f}%'
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.set_xlabel('Predicted Probability (Confidence)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title(r'Label Smoothing ($\epsilon = 0.1$)' + '\nBetter Calibrated',
              fontsize=14, fontweight='bold', color='darkgreen')
ax2.set_xlim(0, 1)
ax2.grid(True, alpha=0.3, axis='y')

# Add formula annotation at bottom
formula_text = r'Smoothed target: $p_i = (1-\epsilon)$ if correct, $\frac{\epsilon}{K-1}$ otherwise'
fig.text(0.5, 0.02, formula_text, ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

# Generalization gap annotation
gap_text = 'Generalization: Label smoothing typically improves test accuracy by 0.5-1.5%\nand produces better calibrated probabilities for downstream tasks.'
fig.text(0.5, -0.05, gap_text, ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.9, edgecolor='orange'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# Save
output_path = Path(__file__).parent.parent / 'images' / 'label_smoothing_effect.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
