#!/usr/bin/env python3
"""
Knowledge Distillation Flow Visualization

Shows Teacher (large) -> Student (small) training process:
- Soft labels transfer
- Size reduction statistics
- Performance retention
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Create output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

fig = plt.figure(figsize=(16, 12))

# Main distillation diagram
ax_main = fig.add_axes([0.05, 0.45, 0.9, 0.5])
ax_main.set_xlim(0, 10)
ax_main.set_ylim(0, 6)
ax_main.axis('off')
ax_main.set_title('Knowledge Distillation: Training Small Models from Large Teachers\n'
                  '$\\mathcal{L}_{total} = \\alpha \\mathcal{L}_{task} + (1-\\alpha) \\mathcal{L}_{KL}(student, teacher)$',
                  fontsize=14, fontweight='bold', pad=20)

# Draw Teacher Model (large box)
teacher_box = FancyBboxPatch((0.5, 3), 2.5, 2.5, boxstyle="round,pad=0.1",
                              facecolor='#ffcdd2', edgecolor='#c62828', linewidth=3)
ax_main.add_patch(teacher_box)
ax_main.text(1.75, 5.2, 'Teacher Model', fontsize=12, fontweight='bold', ha='center')
ax_main.text(1.75, 4.5, 'BERT-base', fontsize=11, ha='center', style='italic')
ax_main.text(1.75, 4.0, '110M params', fontsize=10, ha='center')
ax_main.text(1.75, 3.5, '12 layers', fontsize=10, ha='center')
ax_main.text(1.75, 3.2, 'Frozen', fontsize=9, ha='center', color='#c62828', fontweight='bold')

# Draw Student Model (smaller box)
student_box = FancyBboxPatch((7, 3.5), 2, 2, boxstyle="round,pad=0.1",
                              facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=3)
ax_main.add_patch(student_box)
ax_main.text(8, 5.2, 'Student Model', fontsize=12, fontweight='bold', ha='center')
ax_main.text(8, 4.6, 'DistilBERT', fontsize=11, ha='center', style='italic')
ax_main.text(8, 4.1, '66M params', fontsize=10, ha='center')
ax_main.text(8, 3.7, '6 layers', fontsize=10, ha='center')

# Draw Input
input_box = FancyBboxPatch((4, 0.3), 2, 1, boxstyle="round,pad=0.05",
                            facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
ax_main.add_patch(input_box)
ax_main.text(5, 0.8, 'Input Text', fontsize=11, fontweight='bold', ha='center')

# Soft Labels output from teacher
soft_box = FancyBboxPatch((0.8, 0.3), 1.9, 1, boxstyle="round,pad=0.05",
                           facecolor='#fff3e0', edgecolor='#f57c00', linewidth=2)
ax_main.add_patch(soft_box)
ax_main.text(1.75, 1.0, 'Soft Labels', fontsize=10, fontweight='bold', ha='center')
ax_main.text(1.75, 0.55, '(T=4)', fontsize=9, ha='center', style='italic')

# Hard Labels (ground truth)
hard_box = FancyBboxPatch((4, 5), 2, 0.7, boxstyle="round,pad=0.05",
                           facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=2)
ax_main.add_patch(hard_box)
ax_main.text(5, 5.35, 'Hard Labels (y)', fontsize=10, fontweight='bold', ha='center')

# Student output
student_out = FancyBboxPatch((7.3, 0.3), 1.4, 1, boxstyle="round,pad=0.05",
                              facecolor='#e8f5e9', edgecolor='#43a047', linewidth=2)
ax_main.add_patch(student_out)
ax_main.text(8, 1.0, 'Student', fontsize=10, fontweight='bold', ha='center')
ax_main.text(8, 0.55, 'Output', fontsize=10, ha='center')

# Loss computation box
loss_box = FancyBboxPatch((4, 1.8), 2, 1.2, boxstyle="round,pad=0.05",
                           facecolor='#fce4ec', edgecolor='#c2185b', linewidth=2)
ax_main.add_patch(loss_box)
ax_main.text(5, 2.7, 'Loss Computation', fontsize=10, fontweight='bold', ha='center')
ax_main.text(5, 2.2, '$\\mathcal{L}_{KL} + \\mathcal{L}_{CE}$', fontsize=11, ha='center')

# Arrows
arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=4"

# Input to Teacher
ax_main.annotate('', xy=(1.75, 3), xytext=(4.5, 1.3),
                arrowprops=dict(arrowstyle='->', color='#1976d2', lw=2))

# Input to Student
ax_main.annotate('', xy=(8, 3.5), xytext=(5.5, 1.3),
                arrowprops=dict(arrowstyle='->', color='#1976d2', lw=2))

# Teacher to Soft Labels
ax_main.annotate('', xy=(1.75, 1.3), xytext=(1.75, 3),
                arrowprops=dict(arrowstyle='->', color='#f57c00', lw=2))

# Soft Labels to Loss
ax_main.annotate('', xy=(4, 2.4), xytext=(2.7, 1.0),
                arrowprops=dict(arrowstyle='->', color='#f57c00', lw=2))

# Hard Labels to Loss
ax_main.annotate('', xy=(5, 3), xytext=(5, 5),
                arrowprops=dict(arrowstyle='->', color='#7b1fa2', lw=2))

# Student output to Loss
ax_main.annotate('', xy=(6, 2.4), xytext=(7.3, 1.0),
                arrowprops=dict(arrowstyle='->', color='#43a047', lw=2))

# Loss back to Student (gradient)
ax_main.annotate('', xy=(7, 4.2), xytext=(6, 2.8),
                arrowprops=dict(arrowstyle='->', color='#c2185b', lw=2, ls='--'))
ax_main.text(6.2, 3.7, 'Gradient', fontsize=9, color='#c2185b', fontweight='bold', rotation=45)

# Bottom: Statistics and comparison
ax_stats = fig.add_axes([0.05, 0.05, 0.55, 0.35])

# Data for comparison
models = ['BERT-base\n(Teacher)', 'DistilBERT\n(Student)', 'TinyBERT\n(Student)']
params = [110, 66, 14.5]  # Millions
layers = [12, 6, 4]
accuracy = [100, 97, 95]  # Relative to teacher
speedup = [1.0, 1.6, 9.4]

x = np.arange(len(models))
width = 0.2

bars1 = ax_stats.bar(x - 1.5*width, [p/110*100 for p in params], width,
                      label='Parameters (%)', color='#2196F3', edgecolor='black')
bars2 = ax_stats.bar(x - 0.5*width, [l/12*100 for l in layers], width,
                      label='Layers (%)', color='#4CAF50', edgecolor='black')
bars3 = ax_stats.bar(x + 0.5*width, accuracy, width,
                      label='Accuracy (%)', color='#FF9800', edgecolor='black')
bars4 = ax_stats.bar(x + 1.5*width, [s/9.4*100 for s in speedup], width,
                      label='Relative Speed', color='#9C27B0', edgecolor='black')

ax_stats.set_ylabel('Relative Value (%)', fontweight='bold')
ax_stats.set_title('Knowledge Distillation Results', fontweight='bold', fontsize=12)
ax_stats.set_xticks(x)
ax_stats.set_xticklabels(models)
ax_stats.legend(loc='upper right')
ax_stats.set_ylim(0, 120)

# Add specific annotations
for i, (p, l, a, s) in enumerate(zip(params, layers, accuracy, speedup)):
    if i > 0:  # Skip teacher
        ax_stats.annotate(f'{p}M\n{100-a}% loss\n{s}x faster',
                         xy=(i, 105), ha='center', fontsize=8, fontweight='bold')

# Right panel: Soft vs Hard labels visualization
ax_labels = fig.add_axes([0.65, 0.05, 0.32, 0.35])

# Simulate probability distributions
classes = ['Cat', 'Dog', 'Bird', 'Fish', 'Other']
hard_labels = [1, 0, 0, 0, 0]  # One-hot
soft_labels_t1 = [0.95, 0.03, 0.01, 0.005, 0.005]  # T=1 (sharp)
soft_labels_t4 = [0.65, 0.20, 0.08, 0.04, 0.03]   # T=4 (softer)

x = np.arange(len(classes))
width = 0.25

bars_hard = ax_labels.bar(x - width, hard_labels, width, label='Hard Labels',
                           color='#7b1fa2', edgecolor='black')
bars_t1 = ax_labels.bar(x, soft_labels_t1, width, label='Soft (T=1)',
                         color='#f57c00', edgecolor='black')
bars_t4 = ax_labels.bar(x + width, soft_labels_t4, width, label='Soft (T=4)',
                         color='#ffb74d', edgecolor='black')

ax_labels.set_ylabel('Probability', fontweight='bold')
ax_labels.set_title('Hard vs Soft Labels\n(Temperature T controls softness)', fontweight='bold', fontsize=11)
ax_labels.set_xticks(x)
ax_labels.set_xticklabels(classes)
ax_labels.legend(loc='upper right')
ax_labels.set_ylim(0, 1.1)

# Add annotation explaining soft labels
ax_labels.text(2.5, 0.85, 'Soft labels preserve\n"dark knowledge":\nCat is similar to Dog',
              fontsize=9, ha='center', style='italic',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(os.path.join(output_dir, 'distillation_flow.png'),
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved to: {os.path.join(output_dir, 'distillation_flow.png')}")
