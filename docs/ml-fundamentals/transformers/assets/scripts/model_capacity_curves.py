#!/usr/bin/env python3
"""
Model Capacity vs Overfitting Learning Curves
Shows three scenarios:
1. Small model (underfitting): train ~ val, both high
2. Optimal model: train < val, both decreasing
3. Large model (overfitting): train << val, val increases
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

np.random.seed(42)

# Create figure
fig = plt.figure(figsize=(15, 5))

# Common x-axis (epochs)
epochs = np.arange(1, 51)

# ==============================================================================
# Panel 1: Underfitting (Model too small)
# ==============================================================================
ax1 = fig.add_subplot(1, 3, 1)

# Both curves stay high, close together
train_loss_underfit = 1.5 * np.exp(-epochs / 30) + 0.8 + np.random.randn(50) * 0.02
val_loss_underfit = 1.5 * np.exp(-epochs / 30) + 0.85 + np.random.randn(50) * 0.03

ax1.plot(epochs, train_loss_underfit, 'b-', linewidth=2.5, label='Training Loss')
ax1.plot(epochs, val_loss_underfit, 'r-', linewidth=2.5, label='Validation Loss')

ax1.fill_between(epochs, train_loss_underfit, val_loss_underfit, alpha=0.2, color='gray')

ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('UNDERFITTING\n(Model too small)', fontsize=14, fontweight='bold', color='#1565C0')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim(1, 50)
ax1.set_ylim(0, 3.5)
ax1.grid(True, alpha=0.3)

# Add annotations
ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
ax1.text(40, 0.6, 'Optimal\nloss', fontsize=9, color='gray', ha='center')

ax1.annotate('Both losses\nstay high',
             xy=(40, 1.0), xytext=(30, 1.8),
             fontsize=10, ha='center',
             arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

# Symptom box
ax1.text(0.5, -0.15,
         'Symptoms:\n- High train AND val loss\n- Small gap between curves\n- Model lacks capacity',
         transform=ax1.transAxes, fontsize=9, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.9))

# ==============================================================================
# Panel 2: Optimal (Just right capacity)
# ==============================================================================
ax2 = fig.add_subplot(1, 3, 2)

# Both curves decrease, small gap that stabilizes
train_loss_optimal = 2.0 * np.exp(-epochs / 15) + 0.2 + np.random.randn(50) * 0.015
val_loss_optimal = 2.0 * np.exp(-epochs / 15) + 0.35 + np.random.randn(50) * 0.02

ax2.plot(epochs, train_loss_optimal, 'b-', linewidth=2.5, label='Training Loss')
ax2.plot(epochs, val_loss_optimal, 'r-', linewidth=2.5, label='Validation Loss')

ax2.fill_between(epochs, train_loss_optimal, val_loss_optimal, alpha=0.2, color='green')

ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('OPTIMAL\n(Right capacity)', fontsize=14, fontweight='bold', color='#2E7D32')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlim(1, 50)
ax2.set_ylim(0, 3.5)
ax2.grid(True, alpha=0.3)

# Add annotations
ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
ax2.text(40, 0.05, 'Optimal\nloss', fontsize=9, color='gray', ha='center')

ax2.annotate('Both decrease\nand converge',
             xy=(40, 0.4), xytext=(30, 1.5),
             fontsize=10, ha='center',
             arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

ax2.annotate('Small stable\ngap', xy=(45, 0.3), xytext=(35, 0.8),
             fontsize=9, ha='center', color='green',
             arrowprops=dict(arrowstyle='->', color='green', lw=1))

# Symptom box
ax2.text(0.5, -0.15,
         'Symptoms:\n- Both losses decrease\n- Small, stable gap\n- Model generalizes well',
         transform=ax2.transAxes, fontsize=9, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.9))

# ==============================================================================
# Panel 3: Overfitting (Model too large)
# ==============================================================================
ax3 = fig.add_subplot(1, 3, 3)

# Training goes down, validation goes up after some point
train_loss_overfit = 2.5 * np.exp(-epochs / 8) + 0.05 + np.random.randn(50) * 0.01
val_loss_overfit = 2.0 * np.exp(-epochs / 12) + 0.3 + 0.015 * (epochs - 20) * (epochs > 20) + np.random.randn(50) * 0.025

ax3.plot(epochs, train_loss_overfit, 'b-', linewidth=2.5, label='Training Loss')
ax3.plot(epochs, val_loss_overfit, 'r-', linewidth=2.5, label='Validation Loss')

ax3.fill_between(epochs, train_loss_overfit, val_loss_overfit, alpha=0.2, color='red')

ax3.set_xlabel('Epochs', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('OVERFITTING\n(Model too large)', fontsize=14, fontweight='bold', color='#C62828')
ax3.legend(loc='upper right', fontsize=10)
ax3.set_xlim(1, 50)
ax3.set_ylim(0, 3.5)
ax3.grid(True, alpha=0.3)

# Mark the overfitting point
overfit_epoch = 20
ax3.axvline(x=overfit_epoch, color='red', linestyle=':', alpha=0.7)
ax3.text(overfit_epoch + 1, 2.8, 'Overfitting\nbegins', fontsize=9, color='#C62828')

# Add annotations
ax3.annotate('Train loss\nkeeps dropping',
             xy=(45, 0.1), xytext=(35, 0.8),
             fontsize=9, ha='center', color='blue',
             arrowprops=dict(arrowstyle='->', color='blue', lw=1))

ax3.annotate('Val loss\nincreases!',
             xy=(45, 0.75), xytext=(35, 1.8),
             fontsize=10, ha='center', color='#C62828',
             arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

ax3.annotate('Growing gap',
             xy=(45, 0.4), xytext=(38, 1.3),
             fontsize=9, ha='center', color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1))

# Symptom box
ax3.text(0.5, -0.15,
         'Symptoms:\n- Train loss << Val loss\n- Val loss increases\n- Gap keeps growing',
         transform=ax3.transAxes, fontsize=9, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#C62828', alpha=0.9))

# Main title
fig.suptitle('Diagnosing Model Capacity from Learning Curves',
             fontsize=16, fontweight='bold', y=1.02)

# Add solution text at bottom
fig.text(0.17, -0.08, 'Solution: Increase model size\nor get more data',
         ha='center', fontsize=9, style='italic', color='#1565C0')
fig.text(0.5, -0.08, 'Solution: This is the goal!\nContinue training or deploy',
         ha='center', fontsize=9, style='italic', color='#2E7D32')
fig.text(0.83, -0.08, 'Solution: Early stopping,\ndropout, or reduce model',
         ha='center', fontsize=9, style='italic', color='#C62828')

plt.tight_layout()

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, '..', 'images', 'model_capacity_curves.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
plt.close()
