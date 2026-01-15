"""
Training Loss Curves Visualization

This script creates visualizations of training dynamics including:
- Normal training curves
- Overfitting patterns
- Learning rate schedules

Output: training_curves.png, overfitting_patterns.png, lr_schedules.png
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "fine-tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def generate_loss_curve(steps, initial_loss, final_loss, noise_scale=0.05, curve_type='normal'):
    """Generate synthetic loss curves for visualization"""
    x = np.linspace(0, 1, steps)

    if curve_type == 'normal':
        # Smooth decay
        loss = initial_loss * np.exp(-3 * x) + final_loss
    elif curve_type == 'overfit_val':
        # Validation loss that increases
        loss = initial_loss * np.exp(-3 * x) + final_loss
        loss = loss + 0.3 * x**2  # Add upward curve
    elif curve_type == 'plateau':
        # Loss that plateaus early
        loss = initial_loss * np.exp(-6 * x) + final_loss
    elif curve_type == 'unstable':
        # Unstable training
        loss = initial_loss * np.exp(-2 * x) + final_loss
        noise_scale = 0.15

    # Add noise
    noise = np.random.normal(0, noise_scale, steps) * (loss * 0.5)
    loss = loss + noise
    loss = np.maximum(loss, final_loss * 0.8)

    return loss


def create_normal_training_curves():
    """Create visualization of normal training dynamics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = 500
    x = np.arange(steps)

    # Left plot: Good training
    ax = axes[0]
    train_loss = generate_loss_curve(steps, 3.0, 0.5, noise_scale=0.03)
    val_loss = generate_loss_curve(steps, 3.2, 0.6, noise_scale=0.05)

    ax.plot(x, train_loss, label='Training Loss', color='#1976d2', linewidth=2)
    ax.plot(x, val_loss, label='Validation Loss', color='#f57c00', linewidth=2)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Healthy Training Dynamics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # Add annotations
    ax.annotate('Both losses\ndecrease together',
                xy=(200, 1.0), xytext=(280, 1.8),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.annotate('Small train-val gap',
                xy=(400, 0.55), xytext=(350, 1.2),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # Right plot: Learning rate impact
    ax = axes[1]

    # Different learning rates
    for lr, color, label in [(1e-5, '#4caf50', 'LR=1e-5 (good)'),
                              (1e-4, '#ff9800', 'LR=1e-4 (fast)'),
                              (1e-6, '#9c27b0', 'LR=1e-6 (slow)')]:
        if lr == 1e-5:
            loss = generate_loss_curve(steps, 3.0, 0.5, noise_scale=0.03)
        elif lr == 1e-4:
            loss = generate_loss_curve(steps, 3.0, 0.4, noise_scale=0.08, curve_type='unstable')
        else:
            loss = generate_loss_curve(steps, 3.0, 1.5, noise_scale=0.02, curve_type='plateau')

        ax.plot(x, loss, label=label, linewidth=2, color=color)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Learning Rate Impact', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'training_curves.png'}")


def create_overfitting_patterns():
    """Create visualization of overfitting patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = 500
    x = np.arange(steps)

    # Pattern 1: Classic overfitting
    ax = axes[0, 0]
    train_loss = generate_loss_curve(steps, 3.0, 0.2, noise_scale=0.02)
    val_loss = generate_loss_curve(steps, 3.2, 0.8, noise_scale=0.04, curve_type='overfit_val')

    ax.plot(x, train_loss, label='Training Loss', color='#1976d2', linewidth=2)
    ax.plot(x, val_loss, label='Validation Loss', color='#f57c00', linewidth=2)
    ax.axvline(x=280, color='red', linestyle='--', alpha=0.5, label='Optimal stopping point')

    ax.fill_between(x[280:], train_loss[280:], val_loss[280:], alpha=0.3, color='red')
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Classic Overfitting', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.annotate('Overfitting zone', xy=(400, 0.9), fontsize=10, ha='center', color='red')

    # Pattern 2: Large train-val gap
    ax = axes[0, 1]
    train_loss = generate_loss_curve(steps, 3.0, 0.3, noise_scale=0.02)
    val_loss = generate_loss_curve(steps, 3.2, 1.2, noise_scale=0.05)

    ax.plot(x, train_loss, label='Training Loss', color='#1976d2', linewidth=2)
    ax.plot(x, val_loss, label='Validation Loss', color='#f57c00', linewidth=2)

    ax.fill_between(x, train_loss, val_loss, alpha=0.2, color='orange')
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Large Train-Val Gap (Memorization)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    ax.annotate('Gap indicates\nmemorization',
                xy=(350, 0.7), xytext=(250, 1.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='orange'))

    # Pattern 3: Early plateau
    ax = axes[1, 0]
    train_loss = generate_loss_curve(steps, 3.0, 1.5, noise_scale=0.03, curve_type='plateau')
    val_loss = generate_loss_curve(steps, 3.2, 1.6, noise_scale=0.04, curve_type='plateau')

    ax.plot(x, train_loss, label='Training Loss', color='#1976d2', linewidth=2)
    ax.plot(x, val_loss, label='Validation Loss', color='#f57c00', linewidth=2)

    ax.axhline(y=1.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Early Plateau (Learning Rate Too Low)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    ax.annotate('Not learning!\nTry higher LR',
                xy=(400, 1.55), xytext=(350, 2.2),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # Pattern 4: Unstable training
    ax = axes[1, 1]
    np.random.seed(42)
    train_loss = generate_loss_curve(steps, 3.0, 0.6, noise_scale=0.15, curve_type='unstable')

    # Add some spikes
    spike_indices = [100, 250, 380]
    for idx in spike_indices:
        train_loss[idx:idx+10] += np.random.uniform(0.5, 1.5)

    val_loss = train_loss * 1.1 + np.random.normal(0, 0.1, steps)

    ax.plot(x, train_loss, label='Training Loss', color='#1976d2', linewidth=2)
    ax.plot(x, val_loss, label='Validation Loss', color='#f57c00', linewidth=2)

    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Unstable Training (Learning Rate Too High)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    for idx in spike_indices:
        ax.annotate('spike', xy=(idx, train_loss[idx]), fontsize=8,
                   xytext=(idx, train_loss[idx] + 0.5),
                   arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'overfitting_patterns.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'overfitting_patterns.png'}")


def create_lr_schedules():
    """Create visualization of learning rate schedules"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = 1000
    x = np.arange(steps)
    warmup_steps = 100
    max_lr = 2e-5

    # Constant
    ax = axes[0, 0]
    lr = np.ones(steps) * max_lr
    ax.plot(x, lr * 1e5, color='#1976d2', linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Learning Rate (×1e-5)', fontsize=11)
    ax.set_title('Constant Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 2.5)

    # Linear with warmup
    ax = axes[0, 1]
    lr = np.zeros(steps)
    lr[:warmup_steps] = np.linspace(0, max_lr, warmup_steps)
    lr[warmup_steps:] = np.linspace(max_lr, 0, steps - warmup_steps)
    ax.plot(x, lr * 1e5, color='#4caf50', linewidth=2)
    ax.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Learning Rate (×1e-5)', fontsize=11)
    ax.set_title('Linear Decay with Warmup', fontsize=12, fontweight='bold')
    ax.annotate('Warmup', xy=(50, 1.0), fontsize=10, ha='center')
    ax.set_ylim(0, 2.5)

    # Cosine with warmup
    ax = axes[1, 0]
    lr = np.zeros(steps)
    lr[:warmup_steps] = np.linspace(0, max_lr, warmup_steps)
    lr[warmup_steps:] = max_lr * 0.5 * (1 + np.cos(np.pi * np.arange(steps - warmup_steps) / (steps - warmup_steps)))
    ax.plot(x, lr * 1e5, color='#f57c00', linewidth=2)
    ax.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Learning Rate (×1e-5)', fontsize=11)
    ax.set_title('Cosine Decay with Warmup (Recommended)', fontsize=12, fontweight='bold')
    ax.annotate('Smooth decay', xy=(500, 1.0), fontsize=10, ha='center')
    ax.set_ylim(0, 2.5)

    # Add star to indicate recommended
    ax.plot(950, 2.3, marker='*', markersize=15, color='gold')
    ax.text(900, 2.35, 'Recommended', fontsize=9, color='gold', fontweight='bold')

    # Cosine with restarts
    ax = axes[1, 1]
    lr = np.zeros(steps)
    lr[:warmup_steps] = np.linspace(0, max_lr, warmup_steps)

    # Multiple cycles
    cycle_length = (steps - warmup_steps) // 3
    for i in range(3):
        start = warmup_steps + i * cycle_length
        end = start + cycle_length
        cycle_x = np.arange(cycle_length)
        lr[start:end] = max_lr * 0.5 * (1 + np.cos(np.pi * cycle_x / cycle_length))

    ax.plot(x, lr * 1e5, color='#9c27b0', linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Learning Rate (×1e-5)', fontsize=11)
    ax.set_title('Cosine with Warm Restarts', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 2.5)

    for i in range(1, 3):
        ax.axvline(x=warmup_steps + i * cycle_length, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('Restarts', xy=(400, 2.0), fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lr_schedules.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'lr_schedules.png'}")


if __name__ == "__main__":
    create_normal_training_curves()
    create_overfitting_patterns()
    create_lr_schedules()
    print("\nAll training curve visualizations created successfully!")
