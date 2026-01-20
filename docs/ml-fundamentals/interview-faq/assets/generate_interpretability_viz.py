"""
Generate visualizations for Model Interpretability FAQ.
Creates matplotlib-based diagrams for SHAP, LIME, and related concepts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')

def create_shap_summary_plot():
    """Create a SHAP summary/beeswarm plot visualization."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Feature names and data
    features = ['Monthly Charges', 'Contract Type', 'Tenure', 'Internet Service',
                'Payment Method', 'Online Security', 'Tech Support']

    np.random.seed(42)
    n_points = 100

    # Create beeswarm-like plot for each feature
    colors = plt.cm.RdBu_r

    for i, feature in enumerate(features):
        # Generate SHAP values (normally distributed with different means)
        shap_vals = np.random.randn(n_points) * (0.3 - i * 0.03) + (0.1 - i * 0.02)

        # Generate feature values (for coloring)
        feat_vals = np.random.rand(n_points)

        # Add jitter to y position
        y_positions = np.ones(n_points) * i + np.random.randn(n_points) * 0.1

        # Plot scatter with color based on feature value
        scatter = ax.scatter(shap_vals, y_positions, c=feat_vals, cmap=colors,
                            s=20, alpha=0.7, edgecolors='none')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Feature Value', shrink=0.8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    # Styling
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=12)
    ax.set_title('SHAP Summary Plot (Beeswarm)\nFeatures sorted by importance', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlim(-0.6, 0.6)

    # Add annotation
    ax.annotate('Increases\nprediction', xy=(0.4, -0.8), fontsize=10, ha='center', color='#d73027')
    ax.annotate('Decreases\nprediction', xy=(-0.4, -0.8), fontsize=10, ha='center', color='#4575b4')

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/interview-faq/assets/shap_summary_plot.png',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: shap_summary_plot.png")


def create_shap_waterfall_plot():
    """Create a SHAP waterfall plot for single prediction explanation."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Feature contributions
    features = ['income = $85,000', 'age = 42', 'credit_score = 720',
                'employment = stable', 'debt_ratio = 0.25', 'years_employed = 8']
    contributions = [0.18, 0.09, 0.06, 0.03, -0.02, -0.01]

    base_value = 0.35
    final_pred = base_value + sum(contributions)

    # Colors for positive/negative
    colors = ['#d73027' if c > 0 else '#4575b4' for c in contributions]

    # Calculate cumulative positions
    cumulative = [base_value]
    for c in contributions:
        cumulative.append(cumulative[-1] + c)

    # Draw waterfall bars
    bar_height = 0.6
    y_positions = list(range(len(features), 0, -1))

    for i, (feat, contrib, y) in enumerate(zip(features, contributions, y_positions)):
        start = cumulative[i]
        end = cumulative[i + 1]
        left = min(start, end)
        width = abs(contrib)

        # Draw bar
        bar = ax.barh(y, width, left=left, height=bar_height,
                     color=colors[i], alpha=0.85, edgecolor='white', linewidth=1)

        # Add contribution label
        label_x = end + 0.02 if contrib > 0 else end - 0.02
        ha = 'left' if contrib > 0 else 'right'
        sign = '+' if contrib > 0 else ''
        ax.text(label_x, y, f'{sign}{contrib:.2f}', va='center', ha=ha, fontsize=10, fontweight='bold')

        # Add connector line
        if i < len(features) - 1:
            ax.plot([end, end], [y - 0.5, y - 1 + 0.3], color='gray', linewidth=1, linestyle='-', alpha=0.5)

    # Add base value and final prediction markers
    ax.axvline(x=base_value, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(base_value, len(features) + 0.7, f'Base Value\nE[f(x)] = {base_value:.2f}',
            ha='center', fontsize=10, color='gray')

    ax.axvline(x=final_pred, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.8)
    ax.text(final_pred, 0.3, f'Final Prediction\nf(x) = {final_pred:.2f}',
            ha='center', fontsize=10, color='#2ca02c', fontweight='bold')

    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Model Output', fontsize=12)
    ax.set_title('SHAP Waterfall Plot\nSingle Prediction Explanation (Loan Approval)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.2, 0.85)
    ax.set_ylim(0, len(features) + 1.5)

    # Add legend
    legend_elements = [mpatches.Patch(facecolor='#d73027', label='Positive contribution'),
                      mpatches.Patch(facecolor='#4575b4', label='Negative contribution')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/interview-faq/assets/shap_waterfall_plot.png',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: shap_waterfall_plot.png")


def create_shap_dependence_plot():
    """Create a SHAP dependence plot with interaction coloring."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    np.random.seed(42)
    n_points = 200

    # Generate age values
    age = np.random.uniform(20, 70, n_points)

    # Generate SHAP values with non-linear relationship
    shap_values = 0.01 * (age - 45) + 0.0005 * (age - 45) ** 2 + np.random.randn(n_points) * 0.05

    # Generate income for coloring (interaction feature)
    income = np.random.uniform(30000, 150000, n_points)
    # Add some correlation with age
    income = income + (age - 45) * 1000

    # Modify SHAP values based on income interaction
    shap_values = shap_values + (income - 90000) / 500000

    # Create scatter plot
    scatter = ax.scatter(age, shap_values, c=income, cmap='viridis',
                        s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Add trend line
    z = np.polyfit(age, shap_values, 2)
    p = np.poly1d(z)
    age_sorted = np.sort(age)
    ax.plot(age_sorted, p(age_sorted), color='#d62728', linewidth=2,
            linestyle='--', label='Trend', alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Income ($)', shrink=0.8)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

    # Styling
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('SHAP Value for Age', fontsize=12)
    ax.set_title('SHAP Dependence Plot\nAge effect with Income interaction', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Add annotation
    ax.annotate('Higher income\nincreases effect', xy=(60, 0.25), fontsize=10,
                ha='center', color='#440154', fontweight='bold')
    ax.annotate('Lower income\ndecreases effect', xy=(30, -0.15), fontsize=10,
                ha='center', color='#fde725', fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/interview-faq/assets/shap_dependence_plot.png',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: shap_dependence_plot.png")


def create_lime_explanation_plot():
    """Create a LIME local explanation bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Feature contributions from LIME
    features = ['monthly_charges > $70', 'tenure <= 12 months', 'contract = Month-to-month',
                'payment = Electronic check', 'internet = Fiber optic',
                'online_security = No', 'tech_support = No', 'streaming_tv = Yes']
    contributions = [0.24, 0.19, 0.15, 0.08, 0.06, 0.04, 0.03, -0.02]

    # Colors
    colors = ['#d73027' if c > 0 else '#4575b4' for c in contributions]

    # Sort by absolute contribution
    sorted_pairs = sorted(zip(contributions, features, colors), key=lambda x: abs(x[0]), reverse=True)
    contributions, features, colors = zip(*sorted_pairs)

    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, contributions, color=colors, alpha=0.85, edgecolor='white', linewidth=1)

    # Add value labels
    for i, (v, bar) in enumerate(zip(contributions, bars)):
        sign = '+' if v > 0 else ''
        ha = 'left' if v > 0 else 'right'
        offset = 0.01 if v > 0 else -0.01
        ax.text(v + offset, i, f'{sign}{v:.2f}', va='center', ha=ha, fontsize=10, fontweight='bold')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Feature Contribution to Prediction', fontsize=12)
    ax.set_title('LIME Local Explanation\nPrediction: Churn (78% probability)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlim(-0.1, 0.35)

    # Add prediction box
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', edgecolor='#ffc107', linewidth=2)
    ax.text(0.25, 7.5, 'Local Linear\nApproximation', fontsize=10, ha='center',
            bbox=bbox_props, fontweight='bold')

    # Add legend
    legend_elements = [mpatches.Patch(facecolor='#d73027', label='Pushes toward Churn'),
                      mpatches.Patch(facecolor='#4575b4', label='Pushes toward Not Churn')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/interview-faq/assets/lime_explanation_plot.png',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: lime_explanation_plot.png")


def create_pdp_ice_plot():
    """Create a Partial Dependence Plot with ICE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    np.random.seed(42)

    # Feature values
    age = np.linspace(20, 80, 100)

    # Left plot: ICE curves
    ax1 = axes[0]
    n_instances = 50

    for i in range(n_instances):
        # Different baseline for each instance
        baseline = np.random.uniform(-0.3, 0.3)
        # Different slope/curve for each instance
        slope = np.random.uniform(0.005, 0.015)
        curve = np.random.uniform(-0.0002, 0.0002)

        ice_curve = baseline + slope * (age - 50) + curve * (age - 50) ** 2
        ax1.plot(age, ice_curve, color='#1f77b4', alpha=0.15, linewidth=1)

    # Average PDP line
    pdp = 0.01 * (age - 50) + 0.00005 * (age - 50) ** 2
    ax1.plot(age, pdp, color='#d62728', linewidth=3, label='PDP (Average)')

    ax1.set_xlabel('Age', fontsize=12)
    ax1.set_ylabel('Predicted Probability Change', fontsize=12)
    ax1.set_title('ICE Plot with PDP Overlay', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add annotation
    ax1.annotate('Individual\nICE curves', xy=(70, 0.15), fontsize=10,
                color='#1f77b4', ha='center')

    # Right plot: Centered ICE (c-ICE)
    ax2 = axes[1]

    for i in range(n_instances):
        # Centered at age=20
        baseline = np.random.uniform(-0.1, 0.1)
        slope = np.random.uniform(0.005, 0.015)
        curve = np.random.uniform(-0.0002, 0.0002)

        ice_curve = baseline + slope * (age - 50) + curve * (age - 50) ** 2
        # Center at minimum value
        ice_centered = ice_curve - ice_curve[0]
        ax2.plot(age, ice_centered, color='#2ca02c', alpha=0.15, linewidth=1)

    # Centered PDP
    pdp_centered = pdp - pdp[0]
    ax2.plot(age, pdp_centered, color='#d62728', linewidth=3, label='Centered PDP')

    ax2.set_xlabel('Age', fontsize=12)
    ax2.set_ylabel('Centered Predicted Probability Change', fontsize=12)
    ax2.set_title('Centered ICE Plot (c-ICE)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add annotation about heterogeneity
    ax2.annotate('Heterogeneous\neffects visible', xy=(65, 0.5), fontsize=10,
                ha='center', color='#2ca02c')

    # Add crossing lines indicator
    ax2.annotate('Crossing lines =\ninteraction effect', xy=(35, 0.3), fontsize=9,
                ha='center', color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/interview-faq/assets/pdp_ice_plot.png',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: pdp_ice_plot.png")


def create_permutation_importance_plot():
    """Create a permutation importance plot with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Feature names and importance values with standard deviations
    features = ['contract_type', 'monthly_charges', 'tenure', 'internet_service',
                'payment_method', 'total_charges', 'online_security', 'tech_support',
                'streaming_tv', 'phone_service']
    importances = [0.152, 0.124, 0.098, 0.076, 0.054, 0.043, 0.032, 0.028, 0.015, 0.008]
    stds = [0.018, 0.015, 0.012, 0.009, 0.008, 0.007, 0.005, 0.004, 0.003, 0.002]

    # Create horizontal bar chart with error bars
    y_pos = np.arange(len(features))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))[::-1]

    bars = ax.barh(y_pos, importances, xerr=stds, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1, capsize=4, error_kw={'linewidth': 1.5})

    # Add value labels
    for i, (imp, std) in enumerate(zip(importances, stds)):
        ax.text(imp + std + 0.008, i, f'{imp:.3f}', va='center', ha='left', fontsize=10)

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Decrease in Accuracy (mean +/- std)', fontsize=12)
    ax.set_title('Permutation Importance\n(10 repeats, test set)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.22)

    # Add annotation
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='#e8f4ea', edgecolor='#28a745', linewidth=1.5)
    ax.text(0.16, 1, 'Error bars show\nvariability across\npermutations', fontsize=9,
            ha='center', bbox=bbox_props)

    # Add vertical line at 0
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/interview-faq/assets/permutation_importance_plot.png',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: permutation_importance_plot.png")


def create_interpretability_comparison():
    """Create a comparison diagram of SHAP vs LIME vs other methods."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'Interpretability Methods Comparison', fontsize=16,
            fontweight='bold', ha='center')

    # Method boxes
    methods = [
        {'name': 'SHAP', 'x': 1.5, 'y': 5.5, 'color': '#3498db',
         'pros': ['Theoretically grounded', 'Consistent & additive', 'Local + Global'],
         'cons': ['Can be slow', 'Complex for non-experts']},
        {'name': 'LIME', 'x': 5, 'y': 5.5, 'color': '#e74c3c',
         'pros': ['Intuitive output', 'Model-agnostic', 'Fast'],
         'cons': ['Unstable results', 'Local only', 'Sampling dependent']},
        {'name': 'Permutation\nImportance', 'x': 8.5, 'y': 5.5, 'color': '#2ecc71',
         'pros': ['Model-agnostic', 'Easy to compute', 'Clear interpretation'],
         'cons': ['Global only', 'Correlated features issue', 'Slow']},
    ]

    for method in methods:
        # Main box
        box = FancyBboxPatch((method['x'] - 1.3, method['y'] - 0.4), 2.6, 1.2,
                             boxstyle='round,pad=0.05', facecolor=method['color'],
                             edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(method['x'], method['y'] + 0.15, method['name'], fontsize=12,
                fontweight='bold', ha='center', va='center', color='white')

        # Pros box
        pros_y = method['y'] - 1.8
        pros_text = '\n'.join([f'+ {p}' for p in method['pros']])
        ax.text(method['x'], pros_y, pros_text, fontsize=9, ha='center', va='top',
                color='#27ae60', family='monospace')

        # Cons box
        cons_y = method['y'] - 3.5
        cons_text = '\n'.join([f'- {c}' for c in method['cons']])
        ax.text(method['x'], cons_y, cons_text, fontsize=9, ha='center', va='top',
                color='#c0392b', family='monospace')

    # Usage guide at bottom
    ax.text(6, 0.8, 'When to Use:', fontsize=12, fontweight='bold', ha='center')
    usage_text = ('SHAP: Need theoretical guarantees, tree models, feature interactions\n'
                  'LIME: Quick intuitive explanations, text/image data, prototyping\n'
                  'Permutation: Model comparison, feature selection, global overview')
    ax.text(6, 0.3, usage_text, fontsize=10, ha='center', va='top', family='monospace')

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/interview-faq/assets/interpretability_comparison.png',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: interpretability_comparison.png")


if __name__ == '__main__':
    print("Generating Interpretability visualizations...")
    print("=" * 50)

    create_shap_summary_plot()
    create_shap_waterfall_plot()
    create_shap_dependence_plot()
    create_lime_explanation_plot()
    create_pdp_ice_plot()
    create_permutation_importance_plot()
    create_interpretability_comparison()

    print("=" * 50)
    print("All visualizations generated successfully!")
