"""
Few-Shot Learning Performance Visualization

This script creates visualizations showing how performance varies with the number
of examples, example ordering effects, and the importance of example quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def create_few_shot_performance_viz():
    """Create visualizations for few-shot learning performance."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ============== Panel 1: Performance vs Number of Examples ==============
    ax1 = fig.add_subplot(gs[0, 0])

    n_examples = np.arange(0, 17)

    # Different task types with different learning curves
    # Simple task: quick saturation
    simple_task = 60 + 35 * (1 - np.exp(-n_examples / 2))
    simple_task = np.clip(simple_task + np.random.randn(17) * 2, 0, 100)

    # Complex task: slower learning
    complex_task = 30 + 50 * (1 - np.exp(-n_examples / 5))
    complex_task = np.clip(complex_task + np.random.randn(17) * 3, 0, 100)

    # Very complex: needs many examples
    very_complex = 20 + 40 * (1 - np.exp(-n_examples / 8))
    very_complex = np.clip(very_complex + np.random.randn(17) * 3, 0, 100)

    ax1.plot(n_examples, simple_task, 'o-', color='#48bb78', linewidth=2,
            markersize=6, label='Simple (format conversion)')
    ax1.plot(n_examples, complex_task, 's-', color='#4299e1', linewidth=2,
            markersize=6, label='Medium (classification)')
    ax1.plot(n_examples, very_complex, '^-', color='#9f7aea', linewidth=2,
            markersize=6, label='Complex (reasoning)')

    # Mark diminishing returns region
    ax1.axvspan(8, 16, alpha=0.1, color='#fc8181', label='Diminishing returns')
    ax1.axvline(x=8, color='#c53030', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Number of Examples', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Performance vs Number of Examples', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(-0.5, 16.5)

    # Annotation
    ax1.annotate('Sweet spot:\n3-8 examples', xy=(5, 85), fontsize=10,
                ha='center', color='#276749', fontweight='bold')

    # ============== Panel 2: Ordering Effects ==============
    ax2 = fig.add_subplot(gs[0, 1])

    orderings = ['Random', 'Easy→Hard', 'Similar\nLast', 'Label\nBalanced', 'Diverse\nFirst']
    math_scores = [65, 72, 78, 68, 70]
    classification_scores = [75, 73, 82, 85, 78]

    x = np.arange(len(orderings))
    width = 0.35

    bars1 = ax2.bar(x - width/2, math_scores, width, label='Math Reasoning',
                   color='#4299e1', edgecolor='#2b6cb0')
    bars2 = ax2.bar(x + width/2, classification_scores, width, label='Classification',
                   color='#48bb78', edgecolor='#276749')

    # Highlight best for each
    bars1[2].set_color('#2b6cb0')
    bars2[3].set_color('#276749')

    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Example Ordering Effects', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(orderings, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 100)

    # Add improvement annotations
    ax2.annotate('+13%', xy=(2, 80), fontsize=10, color='#2b6cb0',
                fontweight='bold', ha='center')
    ax2.annotate('+10%', xy=(3.18, 88), fontsize=10, color='#276749',
                fontweight='bold', ha='center')

    # ============== Panel 3: Example Quality Impact ==============
    ax3 = fig.add_subplot(gs[1, 0])

    quality_levels = ['Random\nExamples', 'Representative\nExamples', 'Diverse\nExamples',
                      'Dynamic\nSelection', 'Curated +\nDynamic']
    performance = [58, 72, 78, 85, 91]
    colors = ['#fc8181', '#fbd38d', '#9ae6b4', '#4299e1', '#48bb78']

    bars = ax3.bar(quality_levels, performance, color=colors, edgecolor='#1a202c', linewidth=1.5)

    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Impact of Example Quality', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 100)

    # Add value labels
    for bar, val in zip(bars, performance):
        ax3.annotate(f'{val}%', xy=(bar.get_x() + bar.get_width()/2, val + 2),
                    ha='center', fontsize=10, fontweight='bold')

    # Add trend line
    x_pos = np.arange(len(quality_levels))
    z = np.polyfit(x_pos, performance, 2)
    p = np.poly1d(z)
    ax3.plot(x_pos, p(x_pos), '--', color='#718096', linewidth=2, alpha=0.7)

    ax3.text(2, 50, 'Quality > Quantity', fontsize=12, ha='center',
            style='italic', color='#2d3748')

    # ============== Panel 4: Label Distribution Bias ==============
    ax4 = fig.add_subplot(gs[1, 1])

    scenarios = ['Balanced\n(50/50)', 'Slight Bias\n(60/40)', 'Moderate\n(70/30)',
                'Heavy Bias\n(80/20)', 'Extreme\n(90/10)']
    true_positive_rate = [95, 92, 85, 72, 55]
    true_negative_rate = [93, 88, 75, 58, 40]

    x = np.arange(len(scenarios))
    width = 0.35

    ax4.bar(x - width/2, true_positive_rate, width, label='Majority Class Recall',
           color='#4299e1', edgecolor='#2b6cb0')
    ax4.bar(x + width/2, true_negative_rate, width, label='Minority Class Recall',
           color='#fc8181', edgecolor='#c53030')

    ax4.set_ylabel('Recall (%)', fontsize=11)
    ax4.set_title('Label Distribution Bias Effect', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios, fontsize=9)
    ax4.legend(fontsize=9, loc='lower left')
    ax4.set_ylim(0, 105)

    # Add warning zone
    ax4.axhspan(0, 60, xmin=0.6, xmax=1, alpha=0.2, color='#fc8181')
    ax4.text(4, 48, 'Danger zone:\nMinority class\nneglected', fontsize=9,
            ha='center', color='#c53030')

    plt.suptitle('Few-Shot Learning: Performance Factors',
                fontsize=16, fontweight='bold', y=0.98)

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/prompt-engineering/few_shot_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_few_shot_performance_viz()
