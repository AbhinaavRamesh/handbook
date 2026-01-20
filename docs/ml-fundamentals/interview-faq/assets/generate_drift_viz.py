#!/usr/bin/env python3
"""
Generate visualizations for Model Drift FAQ documentation.

Creates:
1. Feature distribution drift (overlapping histograms)
2. Concept drift animation (decision boundary shifting) - GIF
3. PSI/KL divergence monitoring time series
4. Prediction distribution over time
5. Monitoring dashboard layout (2x2 grid)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import os

# Set consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_feature_distribution_drift():
    """
    Create overlapping histograms showing feature distribution drift
    between training and production data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    np.random.seed(42)

    # Example 1: Age distribution shift (data drift)
    train_age = np.random.normal(35, 8, 5000)
    prod_age = np.random.normal(28, 10, 5000)

    axes[0].hist(train_age, bins=40, alpha=0.6, color='#2E86AB', label='Training', density=True, edgecolor='white', linewidth=0.5)
    axes[0].hist(prod_age, bins=40, alpha=0.6, color='#E94F37', label='Production', density=True, edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Data Drift: Age Distribution Shift\nPSI = 0.23 (Moderate Drift)', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].axvline(np.mean(train_age), color='#2E86AB', linestyle='--', linewidth=2, alpha=0.8)
    axes[0].axvline(np.mean(prod_age), color='#E94F37', linestyle='--', linewidth=2, alpha=0.8)

    # Example 2: Transaction amount shift
    train_amount = np.random.exponential(100, 5000)
    prod_amount = np.random.exponential(150, 5000) + 50

    axes[1].hist(train_amount, bins=50, alpha=0.6, color='#2E86AB', label='Training', density=True, edgecolor='white', linewidth=0.5, range=(0, 500))
    axes[1].hist(prod_amount, bins=50, alpha=0.6, color='#E94F37', label='Production', density=True, edgecolor='white', linewidth=0.5, range=(0, 500))
    axes[1].set_xlabel('Transaction Amount ($)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Data Drift: Transaction Amount Shift\nPSI = 0.31 (Significant Drift)', fontweight='bold')
    axes[1].legend(loc='upper right')

    # Example 3: Categorical feature shift
    categories = ['Low', 'Medium', 'High', 'Premium']
    train_dist = [0.4, 0.35, 0.2, 0.05]
    prod_dist = [0.2, 0.3, 0.35, 0.15]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = axes[2].bar(x - width/2, train_dist, width, label='Training', color='#2E86AB', edgecolor='white', linewidth=0.5)
    bars2 = axes[2].bar(x + width/2, prod_dist, width, label='Production', color='#E94F37', edgecolor='white', linewidth=0.5)
    axes[2].set_xlabel('Customer Segment')
    axes[2].set_ylabel('Proportion')
    axes[2].set_title('Label Drift: Customer Segment Shift\nChi-square p < 0.001', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories)
    axes[2].legend(loc='upper right')
    axes[2].set_ylim(0, 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distribution_drift.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: feature_distribution_drift.png")


def create_concept_drift_animation():
    """
    Create GIF showing decision boundary shifting over time (concept drift).
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    np.random.seed(42)

    def generate_data(phase):
        """Generate data where the optimal boundary shifts based on phase."""
        n_samples = 100

        # Class 0 (blue) - stays relatively fixed
        class0_x = np.random.normal(-1, 0.8, n_samples)
        class0_y = np.random.normal(0, 0.8, n_samples)

        # Class 1 (red) - shifts over time
        shift = phase * 0.15  # Gradual shift
        class1_x = np.random.normal(1 + shift, 0.8, n_samples)
        class1_y = np.random.normal(shift * 0.5, 0.8, n_samples)

        return class0_x, class0_y, class1_x, class1_y

    def get_decision_boundary(phase, boundary_type='optimal'):
        """Get decision boundary parameters."""
        if boundary_type == 'original':
            # Original trained boundary (doesn't move)
            return 0, -3, 3
        else:
            # Optimal boundary that should shift with data
            shift = phase * 0.15
            return 0.5 + shift/2, -3, 3

    frames = 40

    def animate(frame):
        ax.clear()
        phase = frame

        c0_x, c0_y, c1_x, c1_y = generate_data(phase)

        # Plot data points
        ax.scatter(c0_x, c0_y, c='#2E86AB', alpha=0.6, s=40, label='Class 0 (Legitimate)', edgecolor='white', linewidth=0.5)
        ax.scatter(c1_x, c1_y, c='#E94F37', alpha=0.6, s=40, label='Class 1 (Fraud)', edgecolor='white', linewidth=0.5)

        # Plot original boundary (trained)
        orig_x, orig_y1, orig_y2 = get_decision_boundary(0, 'original')
        ax.axvline(x=orig_x, color='#333333', linestyle='--', linewidth=2,
                   label='Original Boundary (Trained)', alpha=0.8)

        # Plot optimal boundary
        opt_x, opt_y1, opt_y2 = get_decision_boundary(phase, 'optimal')
        ax.axvline(x=opt_x, color='#28A745', linestyle='-', linewidth=2,
                   label='Optimal Boundary (Current)', alpha=0.8)

        # Shade misclassified region
        if phase > 5:
            ax.axvspan(orig_x, opt_x, alpha=0.15, color='#FFC107', label='Misclassification Zone')

        ax.set_xlim(-4, 5)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.set_title(f'Concept Drift: Decision Boundary Should Shift\nTime Period: {frame + 1}/{frames}',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add annotation for drift
        if phase > 10:
            misclass_rate = min(0.3, phase * 0.008)
            ax.annotate(f'Misclassification\nRate: {misclass_rate:.1%}',
                       xy=(2.5, -2.3), fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='#FFC107', alpha=0.3))

        return ax,

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)
    anim.save(os.path.join(OUTPUT_DIR, 'concept_drift_animation.gif'),
              writer='pillow', fps=5)
    plt.close()
    print("Created: concept_drift_animation.gif")


def create_psi_monitoring_timeseries():
    """
    Create time series showing PSI/KL divergence monitoring over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150)

    np.random.seed(42)

    # Generate time series data
    days = 90
    dates = np.arange(days)

    # PSI values with drift events
    base_psi = np.random.uniform(0.02, 0.06, days)
    # Add gradual drift starting day 30
    drift_component = np.zeros(days)
    drift_component[30:60] = np.linspace(0, 0.15, 30)
    drift_component[60:] = 0.15 + np.random.uniform(-0.02, 0.02, 30)
    # Add sudden spike at day 75
    drift_component[75:80] = 0.35
    drift_component[80:] = np.linspace(0.35, 0.08, 10)

    psi_values = base_psi + drift_component

    # Plot PSI
    axes[0].plot(dates, psi_values, color='#2E86AB', linewidth=2, label='PSI Score')
    axes[0].fill_between(dates, 0, psi_values, alpha=0.3, color='#2E86AB')

    # Add threshold lines
    axes[0].axhline(y=0.1, color='#FFC107', linestyle='--', linewidth=2, label='Warning (0.1)')
    axes[0].axhline(y=0.25, color='#E94F37', linestyle='--', linewidth=2, label='Critical (0.25)')

    # Shade regions
    axes[0].axhspan(0, 0.1, alpha=0.1, color='#28A745', label='Stable')
    axes[0].axhspan(0.1, 0.25, alpha=0.1, color='#FFC107')
    axes[0].axhspan(0.25, 0.5, alpha=0.1, color='#E94F37')

    # Add event annotations
    axes[0].annotate('Gradual drift\nbegins', xy=(35, 0.12), xytext=(20, 0.25),
                     arrowprops=dict(arrowstyle='->', color='#333333'),
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].annotate('Data pipeline\nissue', xy=(76, 0.38), xytext=(65, 0.42),
                     arrowprops=dict(arrowstyle='->', color='#333333'),
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('PSI Score')
    axes[0].set_title('Population Stability Index (PSI) Monitoring Over Time', fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0, 0.5)
    axes[0].set_xlim(0, days)

    # Create feature-level heatmap
    features = ['age', 'income', 'tenure', 'balance', 'num_products', 'credit_score']

    # Generate PSI values for each feature over time (weekly)
    weeks = 13
    psi_matrix = np.random.uniform(0.01, 0.08, (len(features), weeks))
    # Add drift to some features
    psi_matrix[0, 5:] += np.linspace(0, 0.2, 8)  # age drifts
    psi_matrix[2, 8:] += 0.15  # tenure sudden shift
    psi_matrix[4, :] = np.random.uniform(0.02, 0.05, weeks)  # stable

    im = axes[1].imshow(psi_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.3)
    axes[1].set_xticks(np.arange(weeks))
    axes[1].set_xticklabels([f'W{i+1}' for i in range(weeks)])
    axes[1].set_yticks(np.arange(len(features)))
    axes[1].set_yticklabels(features)
    axes[1].set_xlabel('Week')
    axes[1].set_ylabel('Feature')
    axes[1].set_title('Feature-Level PSI Heatmap (Green=Stable, Red=Drifting)', fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1], orientation='vertical', pad=0.02)
    cbar.set_label('PSI Score')

    # Add text annotations for high PSI values
    for i in range(len(features)):
        for j in range(weeks):
            if psi_matrix[i, j] > 0.15:
                axes[1].text(j, i, f'{psi_matrix[i, j]:.2f}', ha='center', va='center',
                            color='white', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'psi_monitoring_timeseries.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: psi_monitoring_timeseries.png")


def create_prediction_distribution_over_time():
    """
    Create visualization showing how prediction distribution changes over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)

    np.random.seed(42)

    # Top Left: Prediction confidence distribution shift
    ax = axes[0, 0]
    train_conf = np.random.beta(5, 2, 5000)  # Training: high confidence
    week1_conf = np.random.beta(4.5, 2.2, 5000)
    week4_conf = np.random.beta(3, 2.5, 5000)  # Later: lower confidence

    ax.hist(train_conf, bins=50, alpha=0.4, color='#2E86AB', label='Training', density=True, edgecolor='none')
    ax.hist(week1_conf, bins=50, alpha=0.4, color='#28A745', label='Week 1', density=True, edgecolor='none')
    ax.hist(week4_conf, bins=50, alpha=0.4, color='#E94F37', label='Week 4', density=True, edgecolor='none')
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Confidence Distribution Shift', fontweight='bold')
    ax.legend()
    ax.axvline(0.5, color='#333333', linestyle=':', linewidth=1, alpha=0.5)

    # Top Right: Class prediction ratio over time
    ax = axes[0, 1]
    weeks = np.arange(1, 13)
    class0_ratio = [0.72, 0.71, 0.70, 0.68, 0.65, 0.62, 0.58, 0.55, 0.52, 0.50, 0.48, 0.47]
    class1_ratio = [1 - r for r in class0_ratio]

    ax.fill_between(weeks, 0, class0_ratio, alpha=0.6, color='#2E86AB', label='Class 0 (Negative)')
    ax.fill_between(weeks, class0_ratio, 1, alpha=0.6, color='#E94F37', label='Class 1 (Positive)')
    ax.axhline(y=0.7, color='#333333', linestyle='--', linewidth=1, alpha=0.7)
    ax.annotate('Training\nratio', xy=(1.5, 0.73), fontsize=9)
    ax.set_xlabel('Week')
    ax.set_ylabel('Prediction Ratio')
    ax.set_title('Predicted Class Ratio Over Time (Label Drift)', fontweight='bold')
    ax.legend(loc='center right')
    ax.set_xlim(1, 12)
    ax.set_ylim(0, 1)

    # Bottom Left: Score calibration drift
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

    # Training calibration (good)
    train_actual = bin_centers + np.random.normal(0, 0.02, len(bin_centers))
    ax.plot(bin_centers, train_actual, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='Training')

    # Production calibration (degraded)
    prod_actual = bin_centers * 0.8 + 0.1 + np.random.normal(0, 0.03, len(bin_centers))
    ax.plot(bin_centers, prod_actual, 's-', color='#E94F37', linewidth=2, markersize=8, label='Production (Week 4)')

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Positive Rate')
    ax.set_title('Calibration Drift (Reliability Diagram)', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.fill_between([0, 1], [0, 1], [0.1, 0.9], alpha=0.1, color='gray')

    # Bottom Right: Error rate by prediction confidence
    ax = axes[1, 1]
    conf_bins = ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    train_errors = [0.42, 0.28, 0.18, 0.08, 0.03]
    prod_errors = [0.48, 0.38, 0.30, 0.18, 0.10]

    x = np.arange(len(conf_bins))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_errors, width, label='Training', color='#2E86AB', edgecolor='white')
    bars2 = ax.bar(x + width/2, prod_errors, width, label='Production', color='#E94F37', edgecolor='white')

    ax.set_xlabel('Confidence Bucket')
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rate by Confidence Bucket', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conf_bins)
    ax.legend()
    ax.set_ylim(0, 0.6)

    # Add percentage increase annotations
    for i, (t, p) in enumerate(zip(train_errors, prod_errors)):
        increase = ((p - t) / t) * 100
        if increase > 10:
            ax.annotate(f'+{increase:.0f}%', xy=(i + width/2, p + 0.02),
                       ha='center', fontsize=9, color='#E94F37', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_distribution_over_time.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: prediction_distribution_over_time.png")


def create_monitoring_dashboard():
    """
    Create a 2x2 monitoring dashboard layout showing key metrics.
    """
    fig = plt.figure(figsize=(14, 10), dpi=150)

    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    np.random.seed(42)

    # Generate common time axis
    hours = 168  # 7 days
    time_axis = np.arange(hours)

    # Panel 1: Accuracy over time
    ax1 = fig.add_subplot(gs[0, 0])
    base_accuracy = 0.92
    accuracy = base_accuracy - np.cumsum(np.random.normal(0, 0.001, hours))
    accuracy = np.clip(accuracy, 0.80, 0.95)
    # Add sudden drop
    accuracy[100:120] -= 0.05
    accuracy[120:] -= 0.03

    ax1.plot(time_axis, accuracy, color='#2E86AB', linewidth=2)
    ax1.fill_between(time_axis, accuracy, alpha=0.3, color='#2E86AB')
    ax1.axhline(y=0.90, color='#FFC107', linestyle='--', linewidth=2, label='Warning (90%)')
    ax1.axhline(y=0.85, color='#E94F37', linestyle='--', linewidth=2, label='Critical (85%)')
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy', fontweight='bold', fontsize=12)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_ylim(0.80, 0.96)
    ax1.set_xlim(0, hours)

    # Add alert annotation
    ax1.annotate('Alert\nTriggered', xy=(105, 0.86), xytext=(70, 0.82),
                arrowprops=dict(arrowstyle='->', color='#E94F37'),
                fontsize=9, color='#E94F37', fontweight='bold')

    # Current value indicator
    current_acc = accuracy[-1]
    color = '#28A745' if current_acc > 0.90 else '#FFC107' if current_acc > 0.85 else '#E94F37'
    ax1.text(0.98, 0.95, f'Current: {current_acc:.1%}', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', color=color, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: Latency percentiles
    ax2 = fig.add_subplot(gs[0, 1])
    p50 = 15 + np.random.normal(0, 2, hours)
    p95 = 45 + np.random.normal(0, 5, hours)
    p99 = 80 + np.random.normal(0, 10, hours)
    # Add spike
    p95[80:90] += 30
    p99[80:90] += 60

    ax2.plot(time_axis, p50, color='#28A745', linewidth=2, label='p50')
    ax2.plot(time_axis, p95, color='#FFC107', linewidth=2, label='p95')
    ax2.plot(time_axis, p99, color='#E94F37', linewidth=2, label='p99')
    ax2.axhline(y=100, color='#E94F37', linestyle=':', linewidth=1, alpha=0.7, label='SLA (100ms)')
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Inference Latency', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 180)
    ax2.set_xlim(0, hours)

    # Spike annotation
    ax2.annotate('Spike', xy=(85, 150), fontsize=9, color='#E94F37', fontweight='bold')

    # Panel 3: Request volume
    ax3 = fig.add_subplot(gs[1, 0])
    # Create daily pattern
    daily_pattern = 50 + 30 * np.sin(np.linspace(0, 7 * 2 * np.pi, hours))
    volume = daily_pattern + np.random.normal(0, 5, hours)
    volume = np.clip(volume, 10, None)

    ax3.fill_between(time_axis, 0, volume, alpha=0.6, color='#2E86AB')
    ax3.plot(time_axis, volume, color='#2E86AB', linewidth=1)
    ax3.set_xlabel('Hours')
    ax3.set_ylabel('Requests/min')
    ax3.set_title('Request Volume', fontweight='bold', fontsize=12)
    ax3.set_xlim(0, hours)
    ax3.set_ylim(0, 100)

    # Add day markers
    for day in range(1, 8):
        ax3.axvline(x=day*24, color='gray', linestyle=':', alpha=0.5)
        ax3.text(day*24 - 12, 95, f'Day {day}', ha='center', fontsize=8, alpha=0.7)

    # Total volume indicator
    total_vol = np.sum(volume) * 60  # per hour to total
    ax3.text(0.98, 0.95, f'Total: {total_vol/1000:.0f}K requests', transform=ax3.transAxes,
             fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 4: Drift score
    ax4 = fig.add_subplot(gs[1, 1])
    drift_score = 0.05 + np.cumsum(np.random.normal(0.0005, 0.002, hours))
    drift_score = np.clip(drift_score, 0.02, 0.4)
    # Add drift event
    drift_score[120:] += 0.08

    colors = ['#28A745' if d < 0.1 else '#FFC107' if d < 0.25 else '#E94F37' for d in drift_score]

    for i in range(len(time_axis)-1):
        ax4.fill_between(time_axis[i:i+2], 0, drift_score[i:i+2], color=colors[i], alpha=0.6)
    ax4.plot(time_axis, drift_score, color='#333333', linewidth=1.5)

    ax4.axhline(y=0.1, color='#FFC107', linestyle='--', linewidth=2, label='Moderate (0.1)')
    ax4.axhline(y=0.25, color='#E94F37', linestyle='--', linewidth=2, label='Significant (0.25)')
    ax4.set_xlabel('Hours')
    ax4.set_ylabel('PSI Score')
    ax4.set_title('Data Drift (PSI)', fontweight='bold', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.set_ylim(0, 0.4)
    ax4.set_xlim(0, hours)

    # Current status
    current_drift = drift_score[-1]
    status = 'STABLE' if current_drift < 0.1 else 'WARNING' if current_drift < 0.25 else 'CRITICAL'
    status_color = '#28A745' if current_drift < 0.1 else '#FFC107' if current_drift < 0.25 else '#E94F37'
    ax4.text(0.98, 0.95, f'Status: {status}', transform=ax4.transAxes,
             fontsize=11, fontweight='bold', color=status_color, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add overall title
    fig.suptitle('ML Model Monitoring Dashboard', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(OUTPUT_DIR, 'monitoring_dashboard.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: monitoring_dashboard.png")


def create_drift_types_comparison():
    """
    Create a visual comparison of the three types of drift.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    np.random.seed(42)

    # Data Drift: P(X) changes
    ax = axes[0]
    x = np.linspace(-4, 6, 200)
    train_dist = 0.6 * np.exp(-0.5 * ((x - 0) / 1) ** 2) / (1 * np.sqrt(2 * np.pi))
    prod_dist = 0.6 * np.exp(-0.5 * ((x - 1.5) / 1.3) ** 2) / (1.3 * np.sqrt(2 * np.pi))

    ax.fill_between(x, train_dist, alpha=0.5, color='#2E86AB', label='Training P(X)')
    ax.fill_between(x, prod_dist, alpha=0.5, color='#E94F37', label='Production P(X)')
    ax.plot(x, train_dist, color='#2E86AB', linewidth=2)
    ax.plot(x, prod_dist, color='#E94F37', linewidth=2)

    # Add arrows showing shift
    ax.annotate('', xy=(1.5, 0.25), xytext=(0, 0.25),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
    ax.text(0.75, 0.27, 'Shift', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Feature Value (X)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Data Drift (Covariate Shift)\nP(X) changes, P(Y|X) same', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.45)

    # Concept Drift: P(Y|X) changes
    ax = axes[1]

    # Generate 2D data showing boundary shift
    n = 200
    x1 = np.random.randn(n) - 1
    y1 = np.random.randn(n)
    x2 = np.random.randn(n) + 1
    y2 = np.random.randn(n)

    ax.scatter(x1, y1, c='#2E86AB', alpha=0.5, s=30, label='Class 0')
    ax.scatter(x2, y2, c='#E94F37', alpha=0.5, s=30, label='Class 1')

    # Original boundary
    ax.axvline(x=0, color='#333333', linestyle='--', linewidth=2, label='Original Boundary')
    # New optimal boundary
    ax.axvline(x=0.7, color='#28A745', linestyle='-', linewidth=2, label='New Optimal')

    # Shade the problem region
    ax.axvspan(0, 0.7, alpha=0.2, color='#FFC107')
    ax.text(0.35, 2.3, 'Now\nmisclassified', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#FFC107', alpha=0.3))

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Concept Drift\nP(Y|X) changes', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)

    # Label Drift: P(Y) changes
    ax = axes[2]
    categories = ['Class 0', 'Class 1']
    train_priors = [0.9, 0.1]
    prod_priors = [0.7, 0.3]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_priors, width, label='Training P(Y)',
                   color='#2E86AB', edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, prod_priors, width, label='Production P(Y)',
                   color='#E94F37', edgecolor='white', linewidth=2)

    # Add value labels
    for bar, val in zip(bars1, train_priors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=11, fontweight='bold', color='#2E86AB')
    for bar, val in zip(bars2, prod_priors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=11, fontweight='bold', color='#E94F37')

    # Add arrows showing change
    ax.annotate('', xy=(1.175, 0.3), xytext=(1.175, 0.1),
                arrowprops=dict(arrowstyle='->', color='#28A745', lw=2))
    ax.text(1.35, 0.2, '+200%', fontsize=10, color='#28A745', fontweight='bold')

    ax.set_xlabel('Class')
    ax.set_ylabel('Prior Probability P(Y)')
    ax.set_title('Label Drift (Prior Probability Shift)\nP(Y) changes', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'drift_types_comparison.png'),
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: drift_types_comparison.png")


def main():
    """Generate all visualizations."""
    print(f"Generating visualizations in: {OUTPUT_DIR}")
    print("-" * 50)

    create_feature_distribution_drift()
    create_concept_drift_animation()
    create_psi_monitoring_timeseries()
    create_prediction_distribution_over_time()
    create_monitoring_dashboard()
    create_drift_types_comparison()

    print("-" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
