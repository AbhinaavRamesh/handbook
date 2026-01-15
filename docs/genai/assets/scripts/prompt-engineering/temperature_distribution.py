"""
Temperature and Token Probability Distribution Visualization

This script visualizes how temperature affects token probability distributions
and the resulting output diversity in LLM generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def create_temperature_visualization():
    """Create visualizations for temperature effects on token distributions."""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ============== Panel 1: Softmax at Different Temperatures ==============
    ax1 = fig.add_subplot(gs[0, 0])

    # Simulated logits for 10 tokens
    logits = np.array([2.5, 2.3, 1.8, 1.5, 1.2, 0.8, 0.5, 0.2, -0.1, -0.5])
    tokens = ['token_A', 'token_B', 'token_C', 'token_D', 'token_E',
              'token_F', 'token_G', 'token_H', 'token_I', 'token_J']

    temperatures = [0.3, 0.7, 1.0, 1.5, 2.0]
    colors = ['#c53030', '#ed8936', '#48bb78', '#4299e1', '#9f7aea']

    x = np.arange(len(tokens))
    width = 0.15

    for i, (temp, color) in enumerate(zip(temperatures, colors)):
        probs = softmax(logits / temp)
        offset = (i - 2) * width
        bars = ax1.bar(x + offset, probs, width, label=f'T={temp}', color=color, alpha=0.85)

    ax1.set_xlabel('Token', fontsize=11)
    ax1.set_ylabel('Probability', fontsize=11)
    ax1.set_title('Token Probabilities at Different Temperatures', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{i}' for i in range(10)], fontsize=9)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylim(0, 0.8)

    # Annotations
    ax1.annotate('Low T:\nPeaky, deterministic', xy=(0, 0.65), fontsize=9,
                color='#c53030', ha='center')
    ax1.annotate('High T:\nFlat, random', xy=(8, 0.25), fontsize=9,
                color='#9f7aea', ha='center')

    # ============== Panel 2: Temperature Effect on Top-1 Probability ==============
    ax2 = fig.add_subplot(gs[0, 1])

    temp_range = np.linspace(0.1, 3.0, 100)
    top1_probs = []
    entropy_vals = []

    for t in temp_range:
        probs = softmax(logits / t)
        top1_probs.append(probs[0])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropy_vals.append(entropy)

    ax2_twin = ax2.twinx()

    line1, = ax2.plot(temp_range, top1_probs, color='#4299e1', linewidth=2.5, label='Top-1 Probability')
    line2, = ax2_twin.plot(temp_range, entropy_vals, color='#ed8936', linewidth=2.5, label='Entropy')

    ax2.set_xlabel('Temperature', fontsize=11)
    ax2.set_ylabel('Top-1 Probability', fontsize=11, color='#4299e1')
    ax2_twin.set_ylabel('Entropy', fontsize=11, color='#ed8936')
    ax2.set_title('Temperature vs Determinism and Diversity', fontsize=14, fontweight='bold')

    ax2.tick_params(axis='y', labelcolor='#4299e1')
    ax2_twin.tick_params(axis='y', labelcolor='#ed8936')

    # Mark typical usage zones
    ax2.axvspan(0.1, 0.5, alpha=0.15, color='#48bb78', label='Factual tasks')
    ax2.axvspan(0.6, 1.0, alpha=0.15, color='#4299e1', label='Balanced')
    ax2.axvspan(1.0, 2.0, alpha=0.15, color='#9f7aea', label='Creative tasks')

    ax2.legend([line1, line2], ['Top-1 Probability', 'Entropy'], loc='center right', fontsize=9)

    # ============== Panel 3: Sampling Distribution Histogram ==============
    ax3 = fig.add_subplot(gs[1, 0])

    np.random.seed(42)

    # Simulate token selection over many samples
    temps_to_show = [0.3, 0.7, 1.0, 1.5]
    samples_per_temp = 1000

    for temp, color in zip(temps_to_show, ['#c53030', '#ed8936', '#48bb78', '#9f7aea']):
        probs = softmax(logits / temp)
        samples = np.random.choice(len(tokens), size=samples_per_temp, p=probs)
        counts = np.bincount(samples, minlength=len(tokens))
        ax3.bar(np.arange(len(tokens)) + temps_to_show.index(temp) * 0.2 - 0.3,
               counts / samples_per_temp, width=0.2, label=f'T={temp}',
               color=color, alpha=0.8)

    ax3.set_xlabel('Token Index', fontsize=11)
    ax3.set_ylabel('Selection Frequency', fontsize=11)
    ax3.set_title('Empirical Token Selection Distribution (1000 samples)', fontsize=14, fontweight='bold')
    ax3.set_xticks(np.arange(len(tokens)))
    ax3.set_xticklabels([f'T{i}' for i in range(10)], fontsize=9)
    ax3.legend(fontsize=9)

    # ============== Panel 4: Temperature Guidance Chart ==============
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_title('Temperature Selection Guide', fontsize=14, fontweight='bold', pad=20)

    # Create a table-like visualization
    guidance = [
        ('0.0 - 0.3', 'Very Low', 'Factual Q&A, code generation,\nmath problems', '#c53030'),
        ('0.3 - 0.7', 'Low', 'Summarization, translation,\ndata extraction', '#ed8936'),
        ('0.7 - 1.0', 'Moderate', 'General conversation,\nbalanced tasks', '#48bb78'),
        ('1.0 - 1.5', 'High', 'Creative writing,\nbrainstorming', '#4299e1'),
        ('1.5 - 2.0', 'Very High', 'Poetry, experimental content,\nmaximum diversity', '#9f7aea'),
    ]

    # Column headers
    ax4.text(0.1, 0.95, 'Temperature', fontsize=11, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.35, 0.95, 'Level', fontsize=11, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.6, 0.95, 'Best For', fontsize=11, fontweight='bold', transform=ax4.transAxes)

    ax4.plot([0.05, 0.95], [0.92, 0.92], color='#1a202c', linewidth=1.5, transform=ax4.transAxes)

    for i, (temp_range, level, use_case, color) in enumerate(guidance):
        y = 0.85 - i * 0.17

        # Color bar
        ax4.barh(y, 0.08, height=0.12, left=0.02, color=color, transform=ax4.transAxes)

        # Text
        ax4.text(0.12, y, temp_range, fontsize=10, va='center', transform=ax4.transAxes)
        ax4.text(0.35, y, level, fontsize=10, va='center', transform=ax4.transAxes)
        ax4.text(0.55, y, use_case, fontsize=9, va='center', transform=ax4.transAxes)

    # Add a note
    ax4.text(0.5, 0.02, 'Note: Optimal temperature depends on the specific model and task.\nAlways experiment with your use case.',
            fontsize=9, ha='center', va='bottom', style='italic', color='#718096', transform=ax4.transAxes)

    plt.suptitle('Temperature: Controlling LLM Output Diversity',
                fontsize=16, fontweight='bold', y=0.98)

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/prompt-engineering/temperature_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_temperature_visualization()
