#!/usr/bin/env python3
"""
LLM Judge Calibration Visualization

Creates visualizations for LLM-as-judge calibration analysis,
including calibration curves, agreement plots, and bias analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_calibration_curve():
    """
    Create calibration curves comparing LLM judges to human evaluators.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

    # Simulated calibration data for different judges
    judge_bins = np.linspace(0.1, 0.9, 9)

    # GPT-4 Judge (well-calibrated)
    gpt4_human = judge_bins + np.array([0.02, 0.01, -0.03, 0.02, 0.00, -0.02, 0.01, 0.03, -0.01])
    gpt4_human = np.clip(gpt4_human, 0, 1)

    # Claude Judge (slightly overconfident)
    claude_human = judge_bins * 0.85 + 0.05
    claude_human = np.clip(claude_human, 0, 1)

    # Smaller open-source model (underconfident)
    oss_human = judge_bins * 1.1 - 0.05
    oss_human = np.clip(oss_human, 0, 1)

    # Plot calibration curves
    ax.plot(judge_bins, gpt4_human, 'o-', label='GPT-4 Judge', color='#3498db',
            linewidth=2, markersize=8)
    ax.plot(judge_bins, claude_human, 's-', label='Claude Judge', color='#e74c3c',
            linewidth=2, markersize=8)
    ax.plot(judge_bins, oss_human, '^-', label='Open-Source Judge', color='#2ecc71',
            linewidth=2, markersize=8)

    # Add confidence interval shading for GPT-4
    ax.fill_between(judge_bins, gpt4_human - 0.05, gpt4_human + 0.05,
                    alpha=0.2, color='#3498db')

    ax.set_xlabel('LLM Judge Score', fontsize=12)
    ax.set_ylabel('Human Agreement Rate', fontsize=12)
    ax.set_title('LLM Judge Calibration Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add ECE annotations
    ax.text(0.05, 0.9, 'ECE (GPT-4): 0.023', fontsize=9, color='#3498db')
    ax.text(0.05, 0.85, 'ECE (Claude): 0.078', fontsize=9, color='#e74c3c')
    ax.text(0.05, 0.80, 'ECE (OSS): 0.052', fontsize=9, color='#2ecc71')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'judge_calibration_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'judge_calibration_curves.png'}")


def create_agreement_heatmap():
    """
    Create a heatmap showing agreement between different evaluation methods.
    """
    methods = ['Human', 'GPT-4', 'Claude', 'Llama', 'ROUGE', 'BERTScore']

    # Agreement matrix (simulated)
    agreement = np.array([
        [1.00, 0.85, 0.82, 0.68, 0.55, 0.72],  # Human
        [0.85, 1.00, 0.88, 0.75, 0.58, 0.78],  # GPT-4
        [0.82, 0.88, 1.00, 0.72, 0.55, 0.75],  # Claude
        [0.68, 0.75, 0.72, 1.00, 0.60, 0.70],  # Llama
        [0.55, 0.58, 0.55, 0.60, 1.00, 0.62],  # ROUGE
        [0.72, 0.78, 0.75, 0.70, 0.62, 1.00],  # BERTScore
    ])

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(agreement, cmap='RdYlGn', vmin=0.4, vmax=1.0)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Agreement (Cohen\'s Kappa)', rotation=-90, va="bottom", fontsize=11)

    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_yticklabels(methods, fontsize=11)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add values
    for i in range(len(methods)):
        for j in range(len(methods)):
            text_color = "white" if agreement[i, j] < 0.7 else "black"
            ax.text(j, i, f'{agreement[i, j]:.2f}',
                   ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

    ax.set_title('Inter-Evaluator Agreement Matrix', fontsize=14, fontweight='bold')

    # Add annotation
    ax.text(1.15, 0.02, 'LLM judges show\nstrong agreement\nwith humans',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'judge_agreement_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'judge_agreement_heatmap.png'}")


def create_bias_analysis():
    """
    Create a visualization showing common LLM judge biases.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Position Bias
    ax1 = axes[0]
    positions = ['First', 'Second']
    models = ['GPT-4', 'Claude', 'Llama']
    bias_data = {
        'GPT-4': [0.52, 0.48],
        'Claude': [0.58, 0.42],
        'Llama': [0.65, 0.35],
    }

    x = np.arange(len(positions))
    width = 0.25
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for i, (model, prefs) in enumerate(bias_data.items()):
        ax1.bar(x + i * width, prefs, width, label=model, color=colors[i], alpha=0.8)

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='No bias')
    ax1.set_ylabel('Preference Rate', fontsize=11)
    ax1.set_title('Position Bias', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(positions, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 0.8)

    # 2. Verbosity Bias
    ax2 = axes[1]
    length_ratios = ['1x', '1.5x', '2x', '3x']
    verbosity_preference = [0.50, 0.58, 0.65, 0.72]  # Preference for longer

    ax2.bar(length_ratios, verbosity_preference, color='#9b59b6', alpha=0.8)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='No bias')
    ax2.set_ylabel('Preference for Longer Response', fontsize=11)
    ax2.set_xlabel('Length Ratio (Longer/Shorter)', fontsize=11)
    ax2.set_title('Verbosity Bias', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 0.9)

    # Add trend arrow
    ax2.annotate('', xy=(3.2, 0.72), xytext=(0, 0.50),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.text(1.5, 0.78, 'Bias increases\nwith length diff', fontsize=9, ha='center')

    # 3. Self-Preference Bias
    ax3 = axes[2]
    judged_by = ['GPT-4\nJudge', 'Claude\nJudge', 'Human\nJudge']
    gpt4_win_rate = [0.62, 0.45, 0.48]  # GPT-4 output win rate

    bars = ax3.bar(judged_by, gpt4_win_rate, color=['#3498db', '#e74c3c', '#95a5a6'], alpha=0.8)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='No bias')
    ax3.set_ylabel('GPT-4 Output Win Rate', fontsize=11)
    ax3.set_title('Self-Preference Bias', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 0.8)

    # Highlight the bias
    bars[0].set_edgecolor('red')
    bars[0].set_linewidth(3)
    ax3.text(0, 0.67, 'Self-preference!', fontsize=9, ha='center', color='red', fontweight='bold')

    plt.suptitle('Common LLM Judge Biases', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'judge_bias_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'judge_bias_analysis.png'}")


def create_score_distribution():
    """
    Create a visualization comparing score distributions between LLM and human judges.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scores = np.arange(1, 6)

    # Human judge distribution (more uniform)
    human_dist = np.array([0.08, 0.18, 0.35, 0.28, 0.11])

    # LLM judge distribution (center-biased)
    llm_dist = np.array([0.03, 0.12, 0.45, 0.32, 0.08])

    # Left plot: Side by side
    ax1 = axes[0]
    width = 0.35
    ax1.bar(scores - width/2, human_dist, width, label='Human Judge', color='#3498db', alpha=0.8)
    ax1.bar(scores + width/2, llm_dist, width, label='LLM Judge', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Score (1-5)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Score Distribution Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xticks(scores)

    # Add annotation about central tendency
    ax1.annotate('LLM judges show\nstronger central tendency',
                xy=(3, 0.45), xytext=(4.2, 0.35),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Right plot: Cumulative distribution
    ax2 = axes[1]
    human_cumsum = np.cumsum(human_dist)
    llm_cumsum = np.cumsum(llm_dist)

    ax2.plot(scores, human_cumsum, 'o-', label='Human Judge', color='#3498db', linewidth=2, markersize=8)
    ax2.plot(scores, llm_cumsum, 's-', label='LLM Judge', color='#e74c3c', linewidth=2, markersize=8)
    ax2.set_xlabel('Score (1-5)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xticks(scores)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'judge_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'judge_score_distribution.png'}")


def create_calibration_improvement():
    """
    Show how calibration improves with different techniques.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    techniques = [
        'Baseline\nLLM Judge',
        'With\nPosition Swap',
        'With\nMulti-Judge',
        'With Human\nCalibration',
        'Full\nPipeline'
    ]

    # Metrics for each technique
    metrics = {
        'Agreement with Human': [0.72, 0.78, 0.82, 0.86, 0.89],
        'Consistency': [0.65, 0.80, 0.85, 0.88, 0.91],
        'Bias Reduction': [0.40, 0.65, 0.75, 0.82, 0.88],
    }

    x = np.arange(len(techniques))
    width = 0.25
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, (metric, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)

    ax.set_xlabel('Calibration Technique', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Impact of Calibration Techniques on LLM Judge Quality', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(techniques, fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3)

    # Add improvement arrow
    ax.annotate('', xy=(4.25, 0.89), xytext=(0, 0.72),
                arrowprops=dict(arrowstyle='->', color='green', lw=3, ls='--'))
    ax.text(2, 0.95, '+24% improvement', fontsize=11, ha='center',
            color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'calibration_improvement.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'calibration_improvement.png'}")


def main():
    """Generate all judge calibration visualizations."""
    print("Generating judge calibration visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    create_calibration_curve()
    create_agreement_heatmap()
    create_bias_analysis()
    create_score_distribution()
    create_calibration_improvement()

    print()
    print("All judge calibration visualizations generated successfully!")


if __name__ == "__main__":
    main()
