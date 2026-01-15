#!/usr/bin/env python3
"""
Scaling Laws 3D Visualization
Shows Chinchilla scaling: Loss vs Parameters vs Training Tokens
Demonstrates optimal compute allocation for transformer training
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# Set up the figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Chinchilla scaling law: L(N, D) = E + A/N^alpha + B/D^beta
# Where N = parameters, D = tokens
# Typical values: alpha ~ 0.34, beta ~ 0.28, E ~ 1.69, A ~ 406.4, B ~ 410.7

E = 1.69  # Irreducible loss
A = 406.4
B = 410.7
alpha = 0.34
beta = 0.28

# Create mesh for parameters (N) and tokens (D) in log scale
log_N = np.linspace(6, 12, 50)  # 10^6 to 10^12 parameters
log_D = np.linspace(9, 15, 50)  # 10^9 to 10^15 tokens

N_mesh, D_mesh = np.meshgrid(10**log_N, 10**log_D)
log_N_mesh, log_D_mesh = np.meshgrid(log_N, log_D)

# Calculate loss
Loss = E + A / (N_mesh ** alpha) + B / (D_mesh ** beta)

# Plot surface
surf = ax.plot_surface(log_N_mesh, log_D_mesh, Loss,
                        cmap=cm.viridis, alpha=0.8,
                        linewidth=0, antialiased=True)

# Find optimal ridge (where N ~ D for compute-optimal training)
# Chinchilla shows optimal is roughly N tokens per parameter
optimal_log_N = np.linspace(7, 11, 20)
optimal_log_D = optimal_log_N + 0.5  # Approximate Chinchilla ratio
optimal_N = 10 ** optimal_log_N
optimal_D = 10 ** optimal_log_D
optimal_loss = E + A / (optimal_N ** alpha) + B / (optimal_D ** beta)

# Plot optimal ridge
ax.plot(optimal_log_N, optimal_log_D, optimal_loss,
        'r-', linewidth=3, label='Chinchilla Optimal Ridge', zorder=10)

# Add iso-loss contours on the bottom
contour_levels = [2.5, 3.0, 3.5, 4.0, 5.0]
contour = ax.contour(log_N_mesh, log_D_mesh, Loss,
                      levels=contour_levels, zdir='z', offset=Loss.min(),
                      cmap=cm.plasma, linewidths=1.5)

# Mark some famous models (approximate)
models = {
    'GPT-2 (1.5B)': (9.18, 10.6),  # log10(1.5B), log10(40B tokens)
    'GPT-3 (175B)': (11.24, 11.48),  # log10(175B), log10(300B tokens)
    'Chinchilla (70B)': (10.85, 12.15),  # log10(70B), log10(1.4T tokens)
    'LLaMA-7B': (9.85, 12.0),  # log10(7B), log10(1T tokens)
}

for name, (log_n, log_d) in models.items():
    n, d = 10**log_n, 10**log_d
    loss = E + A / (n ** alpha) + B / (d ** beta)
    ax.scatter([log_n], [log_d], [loss], s=100, c='red', marker='*', zorder=15)
    ax.text(log_n, log_d, loss + 0.15, name, fontsize=8, ha='center')

# Labels and title
ax.set_xlabel('log10(Parameters)', fontsize=12, labelpad=10)
ax.set_ylabel('log10(Training Tokens)', fontsize=12, labelpad=10)
ax.set_zlabel('Loss', fontsize=12, labelpad=10)
ax.set_title('Chinchilla Scaling Laws: Loss vs Model Size vs Data\n'
             'Optimal training follows the ridge (N ~ D)', fontsize=14, fontweight='bold')

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('Loss', fontsize=10)

# Add legend
ax.legend(loc='upper left', fontsize=10)

# Set viewing angle
ax.view_init(elev=25, azim=45)

# Add text annotation
fig.text(0.02, 0.02,
         'Key Insight: Chinchilla showed that models should be trained on\n'
         'roughly as many tokens as they have parameters for optimal compute use.\n'
         'GPT-3 was undertrained; Chinchilla matched its performance with 4x fewer params.',
         fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, '..', 'images', 'scaling_laws_3d.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
plt.close()
