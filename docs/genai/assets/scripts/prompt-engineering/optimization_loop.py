"""
Prompt Optimization Loop Visualization

This script visualizes the APE (Automatic Prompt Engineering) optimization
process, DSPy compilation, and A/B testing workflows.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def create_optimization_visualization():
    """Create visualizations for prompt optimization processes."""

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # ============== Panel 1: APE Pipeline ==============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Automatic Prompt Engineering (APE) Pipeline', fontsize=14, fontweight='bold', pad=20)

    # Step 1: Task + Examples
    box1 = FancyBboxPatch((0.5, 7), 2.5, 2.5, boxstyle="round,pad=0.05",
                          facecolor='#4299e1', edgecolor='#2b6cb0', linewidth=2)
    ax1.add_patch(box1)
    ax1.text(1.75, 8.8, '1. Input', fontsize=10, color='white', ha='center', fontweight='bold')
    ax1.text(1.75, 8.1, 'Task description\n+\nExamples', fontsize=9, color='white', ha='center')

    # Arrow
    ax1.annotate('', xy=(3.8, 8.25), xytext=(3.2, 8.25),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))

    # Step 2: Generate Candidates
    box2 = FancyBboxPatch((3.8, 7), 2.5, 2.5, boxstyle="round,pad=0.05",
                          facecolor='#48bb78', edgecolor='#276749', linewidth=2)
    ax1.add_patch(box2)
    ax1.text(5.05, 8.8, '2. Generate', fontsize=10, color='white', ha='center', fontweight='bold')
    ax1.text(5.05, 8.1, 'LLM creates\n10-100 prompt\ncandidates', fontsize=9, color='white', ha='center')

    # Arrow
    ax1.annotate('', xy=(7.1, 8.25), xytext=(6.5, 8.25),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))

    # Step 3: Evaluate
    box3 = FancyBboxPatch((7.1, 7), 2.5, 2.5, boxstyle="round,pad=0.05",
                          facecolor='#ed8936', edgecolor='#c05621', linewidth=2)
    ax1.add_patch(box3)
    ax1.text(8.35, 8.8, '3. Evaluate', fontsize=10, color='white', ha='center', fontweight='bold')
    ax1.text(8.35, 8.1, 'Score each\non test set', fontsize=9, color='white', ha='center')

    # Step 4: Select
    box4 = FancyBboxPatch((5.3, 3), 2.5, 2.5, boxstyle="round,pad=0.05",
                          facecolor='#9f7aea', edgecolor='#6b46c1', linewidth=2)
    ax1.add_patch(box4)
    ax1.text(6.55, 4.8, '4. Select', fontsize=10, color='white', ha='center', fontweight='bold')
    ax1.text(6.55, 4.1, 'Keep top-K\ncandidates', fontsize=9, color='white', ha='center')

    # Arrow from evaluate to select
    ax1.annotate('', xy=(7.8, 5.7), xytext=(8.35, 6.8),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2,
                              connectionstyle="arc3,rad=-0.3"))

    # Iteration loop
    ax1.annotate('', xy=(4, 4.25), xytext=(5.1, 4.25),
                arrowprops=dict(arrowstyle='->', color='#6b46c1', lw=2))
    ax1.text(4.5, 4.7, 'Refine &\nMutate', fontsize=8, color='#6b46c1', ha='center')

    # Back to evaluate
    ax1.annotate('', xy=(8.35, 6.8), xytext=(4.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='#6b46c1', lw=2,
                              connectionstyle="arc3,rad=0.4"))

    # Output
    box5 = FancyBboxPatch((0.5, 3), 2.5, 2.5, boxstyle="round,pad=0.05",
                          facecolor='#38a169', edgecolor='#276749', linewidth=2)
    ax1.add_patch(box5)
    ax1.text(1.75, 4.8, '5. Output', fontsize=10, color='white', ha='center', fontweight='bold')
    ax1.text(1.75, 4.1, 'Best prompt\nfound!', fontsize=9, color='white', ha='center')

    ax1.annotate('', xy=(3.2, 4.25), xytext=(5.1, 4.25),
                arrowprops=dict(arrowstyle='->', color='#38a169', lw=2))
    ax1.text(4.15, 3.7, 'Converged', fontsize=8, color='#38a169', ha='center')

    # ============== Panel 2: Optimization Progress ==============
    ax2 = fig.add_subplot(gs[0, 1])

    iterations = np.arange(0, 21)
    np.random.seed(42)

    # Simulated optimization curves
    best_score = 0.45 + 0.45 * (1 - np.exp(-iterations / 5))
    best_score += np.random.randn(21) * 0.02
    best_score = np.maximum.accumulate(best_score)

    mean_score = 0.35 + 0.35 * (1 - np.exp(-iterations / 4))
    mean_score += np.random.randn(21) * 0.03

    worst_score = 0.25 + 0.25 * (1 - np.exp(-iterations / 6))
    worst_score += np.random.randn(21) * 0.04

    ax2.fill_between(iterations, worst_score, best_score, alpha=0.2, color='#4299e1')
    ax2.plot(iterations, best_score, 'o-', color='#48bb78', linewidth=2, label='Best candidate', markersize=5)
    ax2.plot(iterations, mean_score, 's--', color='#4299e1', linewidth=1.5, label='Mean score', markersize=4)
    ax2.plot(iterations, worst_score, '^--', color='#fc8181', linewidth=1.5, label='Worst candidate', markersize=4)

    # Mark convergence
    ax2.axvline(x=12, color='#718096', linestyle=':', linewidth=2)
    ax2.text(12.5, 0.3, 'Convergence\nzone', fontsize=9, color='#718096')

    ax2.set_xlabel('Optimization Iteration', fontsize=11)
    ax2.set_ylabel('Accuracy on Test Set', fontsize=11)
    ax2.set_title('APE Optimization Progress', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.set_ylim(0.1, 1.0)
    ax2.set_xlim(-0.5, 20.5)

    # Annotations
    ax2.annotate('Exploration\nphase', xy=(3, 0.55), fontsize=9, ha='center')
    ax2.annotate('Refinement\nphase', xy=(16, 0.8), fontsize=9, ha='center')

    # ============== Panel 3: DSPy Compilation Flow ==============
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('DSPy: From Signature to Optimized Prompt', fontsize=14, fontweight='bold', pad=20)

    # Signature
    sig_box = FancyBboxPatch((0.3, 7), 3, 2.5, boxstyle="round,pad=0.05",
                             facecolor='#4299e1', edgecolor='#2b6cb0', linewidth=2)
    ax3.add_patch(sig_box)
    ax3.text(1.8, 8.8, 'Signature', fontsize=11, color='white', ha='center', fontweight='bold')
    ax3.text(1.8, 7.8, 'question -> answer\n(declarative)', fontsize=9, color='white', ha='center',
            family='monospace')

    # Module
    mod_box = FancyBboxPatch((0.3, 3.5), 3, 2.5, boxstyle="round,pad=0.05",
                             facecolor='#48bb78', edgecolor='#276749', linewidth=2)
    ax3.add_patch(mod_box)
    ax3.text(1.8, 5.3, 'Module', fontsize=11, color='white', ha='center', fontweight='bold')
    ax3.text(1.8, 4.3, 'ChainOfThought(\n  Signature\n)', fontsize=9, color='white', ha='center',
            family='monospace')

    ax3.annotate('', xy=(1.8, 6.2), xytext=(1.8, 6.8),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))

    # Teleprompter
    tel_box = FancyBboxPatch((4, 5), 3, 2.5, boxstyle="round,pad=0.05",
                             facecolor='#9f7aea', edgecolor='#6b46c1', linewidth=2)
    ax3.add_patch(tel_box)
    ax3.text(5.5, 6.8, 'Teleprompter', fontsize=11, color='white', ha='center', fontweight='bold')
    ax3.text(5.5, 5.8, 'BootstrapFewShot\nor MIPRO', fontsize=9, color='white', ha='center')

    ax3.annotate('', xy=(3.8, 6.25), xytext=(3.5, 4.75),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2,
                              connectionstyle="arc3,rad=-0.3"))

    # Training data
    data_box = FancyBboxPatch((4, 1.5), 3, 2.5, boxstyle="round,pad=0.05",
                              facecolor='#ed8936', edgecolor='#c05621', linewidth=2)
    ax3.add_patch(data_box)
    ax3.text(5.5, 3.3, 'Training Data', fontsize=11, color='white', ha='center', fontweight='bold')
    ax3.text(5.5, 2.3, 'Examples for\noptimization', fontsize=9, color='white', ha='center')

    ax3.annotate('', xy=(5.5, 4.2), xytext=(5.5, 4.8),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))

    # Compiled Module
    comp_box = FancyBboxPatch((7.5, 3.5), 2.2, 5, boxstyle="round,pad=0.05",
                              facecolor='#38a169', edgecolor='#276749', linewidth=2)
    ax3.add_patch(comp_box)
    ax3.text(8.6, 8, 'Compiled', fontsize=11, color='white', ha='center', fontweight='bold')
    ax3.text(8.6, 7.3, 'Optimized\nprompt\n+\nFew-shot\nexamples\n+\nChain-of-\nthought', fontsize=9, color='white', ha='center')

    ax3.annotate('', xy=(7.3, 6.25), xytext=(7.2, 6.25),
                arrowprops=dict(arrowstyle='->', color='#38a169', lw=2))

    # Key insight
    ax3.text(5, 0.5, 'DSPy separates WHAT (signature) from HOW (prompt)\nEnabling systematic optimization', fontsize=10, ha='center',
            style='italic', color='#4a5568')

    # ============== Panel 4: A/B Testing Framework ==============
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('A/B Testing Prompt Variants', fontsize=14, fontweight='bold', pad=20)

    # Traffic source
    traffic_box = FancyBboxPatch((0.3, 4), 1.8, 2, boxstyle="round,pad=0.05",
                                 facecolor='#718096', edgecolor='#4a5568', linewidth=2)
    ax4.add_patch(traffic_box)
    ax4.text(1.2, 5, 'User\nTraffic', fontsize=10, color='white', ha='center')

    # Splitter
    ax4.annotate('', xy=(2.8, 5.5), xytext=(2.3, 5),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))
    ax4.annotate('', xy=(2.8, 4.5), xytext=(2.3, 5),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))

    # Variant A
    var_a_box = FancyBboxPatch((3, 5.5), 2.5, 2, boxstyle="round,pad=0.05",
                               facecolor='#4299e1', edgecolor='#2b6cb0', linewidth=2)
    ax4.add_patch(var_a_box)
    ax4.text(4.25, 6.8, 'Variant A (Control)', fontsize=9, color='white', ha='center', fontweight='bold')
    ax4.text(4.25, 6.1, '50% traffic', fontsize=9, color='white', ha='center')

    # Variant B
    var_b_box = FancyBboxPatch((3, 2.5), 2.5, 2, boxstyle="round,pad=0.05",
                               facecolor='#48bb78', edgecolor='#276749', linewidth=2)
    ax4.add_patch(var_b_box)
    ax4.text(4.25, 3.8, 'Variant B (Treatment)', fontsize=9, color='white', ha='center', fontweight='bold')
    ax4.text(4.25, 3.1, '50% traffic', fontsize=9, color='white', ha='center')

    # Results collection
    ax4.annotate('', xy=(6.3, 6.5), xytext=(5.7, 6.5),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))
    ax4.annotate('', xy=(6.3, 3.5), xytext=(5.7, 3.5),
                arrowprops=dict(arrowstyle='->', color='#718096', lw=2))

    # Analysis
    analysis_box = FancyBboxPatch((6.3, 3.5), 3.2, 3.5, boxstyle="round,pad=0.05",
                                  facecolor='#9f7aea', edgecolor='#6b46c1', linewidth=2)
    ax4.add_patch(analysis_box)
    ax4.text(7.9, 6.5, 'Statistical Analysis', fontsize=10, color='white', ha='center', fontweight='bold')
    ax4.text(7.9, 5.5, 'A: 82.3% accuracy\nB: 87.1% accuracy\n\np-value: 0.003\nSignificant!', fontsize=9, color='white', ha='center')

    # Decision
    ax4.annotate('', xy=(7.9, 2.8), xytext=(7.9, 3.3),
                arrowprops=dict(arrowstyle='->', color='#38a169', lw=2))

    decision_box = FancyBboxPatch((6.3, 1.5), 3.2, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='#38a169', edgecolor='#276749', linewidth=2)
    ax4.add_patch(decision_box)
    ax4.text(7.9, 2.1, 'Deploy Variant B', fontsize=11, color='white', ha='center', fontweight='bold')

    # Metrics list
    ax4.text(1.2, 1.5, 'Track:\n- Accuracy\n- Latency\n- Cost\n- User satisfaction', fontsize=8, ha='center', color='#4a5568')

    plt.suptitle('Prompt Optimization: From Manual to Automated',
                fontsize=16, fontweight='bold', y=0.98)

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/prompt-engineering/optimization_loop.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_optimization_visualization()
