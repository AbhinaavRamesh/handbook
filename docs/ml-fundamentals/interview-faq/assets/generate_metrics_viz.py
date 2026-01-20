"""
Generate visualizations for Evaluation Metrics FAQ
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Common styling
FIGSIZE = (10, 6)
DPI = 150
COLORS = {
    'primary': '#2563eb',
    'secondary': '#dc2626',
    'tertiary': '#16a34a',
    'quaternary': '#9333ea',
    'fill': '#93c5fd',
    'grid': '#e5e7eb'
}


def plot_confusion_matrix():
    """Generate confusion matrix heatmap with annotations."""
    # Sample confusion matrix values
    cm = np.array([[85, 15],
                   [10, 90]])

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=12)

    # Configure axes
    classes = ['Negative (0)', 'Positive (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='Actual Label',
           xlabel='Predicted Label')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Add text annotations with metrics
    thresh = cm.max() / 2.
    annotations = [['TN\n85', 'FP\n15'],
                   ['FN\n10', 'TP\n90']]

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, annotations[i][j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')

    # Add metrics box
    precision = 90 / (90 + 15)
    recall = 90 / (90 + 10)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (85 + 90) / 200

    metrics_text = f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1: {f1:.2f}\nAccuracy: {accuracy:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props)

    ax.set_title('Confusion Matrix with Classification Metrics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: confusion_matrix.png")


def plot_roc_curve():
    """Generate ROC curve with AUC shaded area and diagonal reference."""
    # Generate sample data for a good classifier
    np.random.seed(42)
    n_samples = 1000

    # True labels
    y_true = np.concatenate([np.zeros(500), np.ones(500)])

    # Scores from a good model (positive class scores higher)
    y_scores = np.concatenate([
        np.random.beta(2, 5, 500),  # Negatives: lower scores
        np.random.beta(5, 2, 500)   # Positives: higher scores
    ])

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.3, color=COLORS['fill'], label=f'AUC = {roc_auc:.3f}')

    # Plot ROC curve
    ax.plot(fpr, tpr, color=COLORS['primary'], lw=2.5, label='ROC Curve')

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color=COLORS['secondary'], lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')

    # Mark some operating points
    # Find threshold closest to 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    ax.scatter([fpr[idx_05]], [tpr[idx_05]], color=COLORS['tertiary'], s=100, zorder=5,
               label=f'Threshold=0.5 (TPR={tpr[idx_05]:.2f}, FPR={fpr[idx_05]:.2f})')

    # Configure plot
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR) / Recall', fontsize=12)
    ax.set_title('ROC Curve with AUC', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    # Add interpretation guide
    ax.annotate('Better', xy=(0.1, 0.9), fontsize=12, fontweight='bold', color=COLORS['tertiary'])
    ax.annotate('Worse', xy=(0.7, 0.2), fontsize=12, fontweight='bold', color=COLORS['secondary'])

    # Add arrow pointing to ideal corner
    ax.annotate('', xy=(0.02, 0.98), xytext=(0.15, 0.85),
                arrowprops=dict(arrowstyle='->', color=COLORS['tertiary'], lw=2))

    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: roc_curve.png")


def plot_precision_recall_curve():
    """Generate Precision-Recall curve with iso-F1 curves."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # Imbalanced: 20% positive
    y_true = np.concatenate([np.zeros(800), np.ones(200)])

    # Scores from model
    y_scores = np.concatenate([
        np.random.beta(2, 5, 800),  # Negatives
        np.random.beta(4, 2, 200)   # Positives
    ])

    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Plot iso-F1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        y[y < 0] = np.nan
        y[y > 1] = np.nan
        ax.plot(x, y, color='gray', alpha=0.4, linestyle='--', lw=1)
        # Label the iso-F1 curves
        label_x = f_score / 2
        label_y = f_score
        if label_y <= 1:
            ax.annotate(f'F1={f_score:.1f}', xy=(label_x + 0.02, label_y - 0.02),
                       fontsize=9, color='gray', alpha=0.7)

    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.3, color=COLORS['fill'])

    # Plot PR curve
    ax.plot(recall, precision, color=COLORS['primary'], lw=2.5,
            label=f'PR Curve (AUC = {pr_auc:.3f})')

    # Plot baseline (random classifier = positive class ratio)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color=COLORS['secondary'], linestyle='--', lw=2,
               label=f'Random Baseline ({baseline:.2f})')

    # Mark operating point (e.g., threshold = 0.5)
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    ax.scatter([recall[idx_05]], [precision[idx_05]], color=COLORS['tertiary'], s=100, zorder=5,
               label=f'Threshold=0.5 (P={precision[idx_05]:.2f}, R={recall[idx_05]:.2f})')

    # Configure plot
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve with Iso-F1 Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)

    # Add interpretation
    ax.annotate('Better\n(top-right)', xy=(0.85, 0.95), fontsize=11, fontweight='bold',
                color=COLORS['tertiary'], ha='center')

    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: precision_recall_curve.png")


def plot_threshold_metrics():
    """Generate threshold vs metrics (precision, recall, F1) multi-line plot."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    y_true = np.concatenate([np.zeros(700), np.ones(300)])
    y_scores = np.concatenate([
        np.random.beta(2, 5, 700),
        np.random.beta(5, 2, 300)
    ])

    # Calculate metrics at different thresholds
    thresholds = np.linspace(0.05, 0.95, 50)
    precisions = []
    recalls = []
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Plot metrics
    ax.plot(thresholds, precisions, color=COLORS['primary'], lw=2.5, label='Precision', marker='o', markersize=3)
    ax.plot(thresholds, recalls, color=COLORS['secondary'], lw=2.5, label='Recall', marker='s', markersize=3)
    ax.plot(thresholds, f1_scores, color=COLORS['tertiary'], lw=2.5, label='F1 Score', marker='^', markersize=3)

    # Find optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_idx]
    ax.axvline(x=optimal_thresh, color=COLORS['quaternary'], linestyle='--', lw=2, alpha=0.7,
               label=f'Optimal F1 Threshold ({optimal_thresh:.2f})')
    ax.scatter([optimal_thresh], [f1_scores[optimal_idx]], color=COLORS['quaternary'], s=150, zorder=5, marker='*')

    # Add shaded regions for decision zones
    ax.axvspan(0, 0.3, alpha=0.1, color=COLORS['secondary'], label='High Recall Zone')
    ax.axvspan(0.7, 1.0, alpha=0.1, color=COLORS['primary'], label='High Precision Zone')

    # Configure plot
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Precision, Recall, and F1 vs Classification Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Add annotations
    ax.annotate('Low threshold:\nHigh Recall,\nLow Precision', xy=(0.1, 0.3), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate('High threshold:\nLow Recall,\nHigh Precision', xy=(0.75, 0.3), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('threshold_metrics.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: threshold_metrics.png")


def plot_calibration():
    """Generate calibration plot (reliability diagram)."""
    np.random.seed(42)
    n_samples = 2000

    # Generate true labels
    y_true = np.random.binomial(1, 0.4, n_samples)

    # Well-calibrated model: predictions close to true probabilities
    well_calibrated = np.clip(y_true + np.random.normal(0, 0.2, n_samples), 0.01, 0.99)

    # Overconfident model: pushes predictions toward 0 and 1
    overconfident = np.where(y_true == 1,
                              np.clip(0.7 + np.random.normal(0, 0.15, n_samples), 0.5, 0.99),
                              np.clip(0.3 + np.random.normal(0, 0.15, n_samples), 0.01, 0.5))

    # Under-confident model: predictions close to 0.5
    underconfident = 0.5 + (y_true - 0.5) * 0.3 + np.random.normal(0, 0.1, n_samples)
    underconfident = np.clip(underconfident, 0.01, 0.99)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Calculate calibration curves
    n_bins = 10

    # Well calibrated
    prob_true_wc, prob_pred_wc = calibration_curve(y_true, well_calibrated, n_bins=n_bins)
    ax.plot(prob_pred_wc, prob_true_wc, marker='o', lw=2.5, color=COLORS['tertiary'],
            label='Well-Calibrated Model', markersize=8)

    # Overconfident
    prob_true_oc, prob_pred_oc = calibration_curve(y_true, overconfident, n_bins=n_bins)
    ax.plot(prob_pred_oc, prob_true_oc, marker='s', lw=2.5, color=COLORS['secondary'],
            label='Overconfident Model', markersize=8)

    # Under-confident
    prob_true_uc, prob_pred_uc = calibration_curve(y_true, underconfident, n_bins=n_bins)
    ax.plot(prob_pred_uc, prob_true_uc, marker='^', lw=2.5, color=COLORS['quaternary'],
            label='Under-confident Model', markersize=8)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color=COLORS['primary'],
            label='Perfectly Calibrated')

    # Add shaded regions for interpretation
    ax.fill_between([0, 1], [0, 1], [0.15, 1.15], alpha=0.1, color=COLORS['secondary'],
                    label='Overconfident Region')
    ax.fill_between([0, 1], [-0.15, 0.85], [0, 1], alpha=0.1, color=COLORS['quaternary'],
                    label='Under-confident Region')

    # Configure plot
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives (True Probability)', fontsize=12)
    ax.set_title('Calibration Plot (Reliability Diagram)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    # Add interpretation text
    ax.annotate('Above diagonal:\nModel under-confident\n(predicts lower than actual)',
                xy=(0.25, 0.65), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate('Below diagonal:\nModel overconfident\n(predicts higher than actual)',
                xy=(0.55, 0.25), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('calibration_plot.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: calibration_plot.png")


if __name__ == '__main__':
    print("Generating evaluation metrics visualizations...\n")

    plot_confusion_matrix()
    plot_roc_curve()
    plot_precision_recall_curve()
    plot_threshold_metrics()
    plot_calibration()

    print("\nAll visualizations generated successfully!")
