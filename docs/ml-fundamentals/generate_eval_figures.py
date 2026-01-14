"""
Generate evaluation visualizations for ML Fundamentals documentation.
Creates: ROC curve, Precision-Recall curve, Confusion matrix, Bias-variance tradeoff
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import os

# Set style for clean, professional visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Output directory
OUTPUT_DIR = "/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images"

# Generate synthetic data for demonstrations
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, n_classes=2, weights=[0.7, 0.3],
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model for demonstration
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Color palette
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#dc2626',    # Red
    'accent': '#16a34a',       # Green
    'neutral': '#6b7280',      # Gray
    'light': '#e5e7eb',        # Light gray
    'dark': '#1f2937'          # Dark gray
}


def plot_roc_curve():
    """Generate ROC curve with AUC visualization."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    ax.plot(fpr, tpr, color=COLORS['primary'], lw=2.5,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')

    # Fill under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['primary'])

    # Random classifier line
    ax.plot([0, 1], [0, 1], color=COLORS['neutral'], lw=1.5, linestyle='--',
            label='Random Classifier (AUC = 0.5)')

    # Perfect classifier point
    ax.scatter([0], [1], s=100, color=COLORS['accent'], zorder=5,
               label='Perfect Classifier', marker='*')

    # Annotations
    ax.annotate('Better', xy=(0.15, 0.85), fontsize=10, color=COLORS['dark'],
                fontweight='bold')
    ax.annotate('Worse', xy=(0.7, 0.25), fontsize=10, color=COLORS['neutral'],
                fontweight='bold')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR / Recall)', fontweight='bold')
    ax.set_title('ROC Curve', fontweight='bold', pad=15)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eval-roc-curve.svg'), format='svg',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: eval-roc-curve.svg")


def plot_precision_recall_curve():
    """Generate Precision-Recall curve visualization."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)

    # Plot PR curve
    ax.plot(recall, precision, color=COLORS['secondary'], lw=2.5,
            label=f'PR Curve (AP = {avg_precision:.3f})')

    # Fill under curve
    ax.fill_between(recall, precision, alpha=0.15, color=COLORS['secondary'])

    # Baseline (random classifier for imbalanced data)
    baseline = y_test.sum() / len(y_test)
    ax.axhline(y=baseline, color=COLORS['neutral'], lw=1.5, linestyle='--',
               label=f'Random Baseline ({baseline:.2f})')

    # Perfect classifier point
    ax.scatter([1], [1], s=100, color=COLORS['accent'], zorder=5,
               label='Perfect Classifier', marker='*')

    # Annotations for trade-off
    ax.annotate('High Precision\nLow Recall', xy=(0.15, 0.85), fontsize=9,
                color=COLORS['dark'], ha='center')
    ax.annotate('High Recall\nLow Precision', xy=(0.85, 0.45), fontsize=9,
                color=COLORS['dark'], ha='center')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('Recall (TPR)', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontweight='bold', pad=15)
    ax.legend(loc='lower left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eval-pr-curve.svg'), format='svg',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: eval-pr-curve.svg")


def plot_confusion_matrix():
    """Generate confusion matrix heatmap visualization."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Normalize for display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion', fontweight='bold')

    # Labels
    classes = ['Negative (0)', 'Positive (1)']
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Add text annotations
    thresh = 0.5
    for i in range(2):
        for j in range(2):
            # Show both count and percentage
            text = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'
            color = 'white' if cm_normalized[i, j] > thresh else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color,
                   fontsize=12, fontweight='bold')

    # Add quadrant labels
    ax.text(-0.35, 0, 'TN', ha='center', va='center', fontsize=10,
            color=COLORS['accent'], fontweight='bold')
    ax.text(1.35, 0, 'FP', ha='center', va='center', fontsize=10,
            color=COLORS['secondary'], fontweight='bold')
    ax.text(-0.35, 1, 'FN', ha='center', va='center', fontsize=10,
            color=COLORS['secondary'], fontweight='bold')
    ax.text(1.35, 1, 'TP', ha='center', va='center', fontsize=10,
            color=COLORS['accent'], fontweight='bold')

    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('Actual Label', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eval-confusion-matrix.svg'), format='svg',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: eval-confusion-matrix.svg")


def plot_bias_variance_tradeoff():
    """Generate bias-variance tradeoff illustration."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create x-axis (model complexity)
    complexity = np.linspace(0, 10, 100)

    # Simulated curves
    bias_squared = 3 * np.exp(-0.5 * complexity) + 0.1
    variance = 0.1 * np.exp(0.4 * complexity)
    irreducible = np.ones_like(complexity) * 0.3
    total_error = bias_squared + variance + irreducible

    # Plot curves
    ax.plot(complexity, bias_squared, color=COLORS['primary'], lw=2.5,
            label=r'Bias$^2$', linestyle='-')
    ax.plot(complexity, variance, color=COLORS['secondary'], lw=2.5,
            label='Variance', linestyle='-')
    ax.plot(complexity, irreducible, color=COLORS['neutral'], lw=1.5,
            label='Irreducible Error', linestyle=':')
    ax.plot(complexity, total_error, color=COLORS['dark'], lw=3,
            label='Total Error', linestyle='-')

    # Find optimal complexity
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    optimal_error = total_error[optimal_idx]

    # Mark optimal point
    ax.axvline(x=optimal_complexity, color=COLORS['accent'], lw=1.5, linestyle='--', alpha=0.7)
    ax.scatter([optimal_complexity], [optimal_error], s=120, color=COLORS['accent'],
               zorder=5, label='Optimal Complexity', marker='o', edgecolors='white', linewidth=2)

    # Add region labels
    ax.annotate('Underfitting\n(High Bias)', xy=(1.5, 3.5), fontsize=11,
                color=COLORS['primary'], ha='center', fontweight='bold')
    ax.annotate('Overfitting\n(High Variance)', xy=(8.5, 3.5), fontsize=11,
                color=COLORS['secondary'], ha='center', fontweight='bold')

    # Add arrows
    ax.annotate('', xy=(0.5, 2.8), xytext=(2.5, 2.8),
                arrowprops=dict(arrowstyle='<->', color=COLORS['neutral'], lw=1.5))
    ax.annotate('', xy=(7.5, 2.8), xytext=(9.5, 2.8),
                arrowprops=dict(arrowstyle='<->', color=COLORS['neutral'], lw=1.5))

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 5])
    ax.set_xlabel('Model Complexity', fontweight='bold')
    ax.set_ylabel('Error', fontweight='bold')
    ax.set_title('Bias-Variance Tradeoff', fontweight='bold', pad=15)
    ax.legend(loc='upper center', framealpha=0.95, ncol=3, bbox_to_anchor=(0.5, -0.12))

    # Remove x-axis ticks (conceptual diagram)
    ax.set_xticks([])
    ax.text(0, -0.3, 'Simple', fontsize=10, ha='left', color=COLORS['neutral'])
    ax.text(10, -0.3, 'Complex', fontsize=10, ha='right', color=COLORS['neutral'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eval-bias-variance.svg'), format='svg',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: eval-bias-variance.svg")


if __name__ == "__main__":
    print(f"Generating visualizations to: {OUTPUT_DIR}\n")

    plot_roc_curve()
    plot_precision_recall_curve()
    plot_confusion_matrix()
    plot_bias_variance_tradeoff()

    print(f"\nAll visualizations generated successfully!")
