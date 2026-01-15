"""
GenAI System Design Framework Visualization
Generates diagrams for the 6-step design framework
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'images', 'system-design')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_framework_overview():
    """Create the 6-step framework overview diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('GenAI System Design: 6-Step Framework', fontsize=16, fontweight='bold', pad=20)

    # Steps with colors
    steps = [
        ('1. Clarify', '5 min', 'Requirements', '#e3f2fd', '#1976d2'),
        ('2. Architecture', '10 min', 'High-level', '#e8f5e9', '#388e3c'),
        ('3. Components', '15 min', 'Deep dive', '#fff3e0', '#f57c00'),
        ('4. Optimize', '5 min', 'Performance', '#fce4ec', '#c2185b'),
        ('5. Safety', '5 min', 'Guardrails', '#f3e5f5', '#7b1fa2'),
        ('6. Operations', '5 min', 'Production', '#e0f2f1', '#00796b')
    ]

    # Draw steps
    for i, (name, time, focus, bg_color, edge_color) in enumerate(steps):
        x = 1 + i * 2.1
        y = 3

        # Main box
        rect = FancyBboxPatch((x-0.9, y-1), 1.8, 2.5, boxstyle="round,pad=0.1",
                               facecolor=bg_color, edgecolor=edge_color, linewidth=2)
        ax.add_patch(rect)

        # Step name
        ax.text(x, y+0.8, name, ha='center', va='center', fontsize=10, fontweight='bold')

        # Time
        ax.text(x, y+0.3, time, ha='center', va='center', fontsize=9, color='gray')

        # Focus
        ax.text(x, y-0.3, focus, ha='center', va='center', fontsize=8)

        # Arrow to next
        if i < len(steps) - 1:
            ax.annotate("", xy=(x+1.1, y), xytext=(x+0.9, y),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Phase labels
    ax.text(2.05, 5.2, 'Understanding', ha='center', va='center', fontsize=11,
            fontweight='bold', color='#1976d2')
    ax.text(6.25, 5.2, 'Design', ha='center', va='center', fontsize=11,
            fontweight='bold', color='#f57c00')
    ax.text(10.45, 5.2, 'Production', ha='center', va='center', fontsize=11,
            fontweight='bold', color='#7b1fa2')

    # Phase underlines
    ax.plot([0.5, 3.6], [5, 5], color='#1976d2', linewidth=2)
    ax.plot([4, 8.5], [5, 5], color='#f57c00', linewidth=2)
    ax.plot([8.9, 12.8], [5, 5], color='#7b1fa2', linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'framework_overview.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_optimization_triangle():
    """Create the Quality-Latency-Cost optimization triangle."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 1.5)
    ax.axis('off')
    ax.set_title('The Optimization Triangle: Pick Two', fontsize=14, fontweight='bold', pad=20)

    # Triangle vertices
    vertices = np.array([
        [0, 1.2],      # Quality (top)
        [-1, -0.5],    # Cost (bottom left)
        [1, -0.5]      # Latency (bottom right)
    ])

    # Draw triangle
    triangle = plt.Polygon(vertices, fill=False, edgecolor='gray', linewidth=3)
    ax.add_patch(triangle)

    # Labels
    labels = [
        ('Quality', 0, 1.35, '#4caf50'),
        ('Cost', -1.15, -0.5, '#f44336'),
        ('Latency', 1.15, -0.5, '#2196f3')
    ]

    for label, x, y, color in labels:
        ax.text(x, y, label, ha='center', va='center', fontsize=14,
                fontweight='bold', color=color)

    # Center text
    ax.text(0, 0.2, 'Pick\nTwo', ha='center', va='center', fontsize=16,
            fontweight='bold', color='gray')

    # Trade-off examples on edges
    edge_labels = [
        ('Better model\n= Higher cost', -0.7, 0.5, '#9c27b0'),
        ('Faster response\n= Lower quality', 0.7, 0.5, '#ff9800'),
        ('Cheaper model\n= Slower', 0, -0.7, '#607d8b')
    ]

    for label, x, y, color in edge_labels:
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                style='italic', color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimization_triangle.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_safety_layers():
    """Create safety guardrails layer diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Safety Guardrails Architecture', fontsize=14, fontweight='bold', pad=20)

    # Input flow
    ax.text(1, 5, 'User\nInput', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#e0e0e0', edgecolor='gray', linewidth=2))

    # Input guardrails
    input_guards = ['PII Detection', 'Toxicity Filter', 'Injection Detection']
    colors = ['#ffcdd2', '#fff9c4', '#c8e6c9']

    for i, (guard, color) in enumerate(zip(input_guards, colors)):
        rect = FancyBboxPatch((2.5, 5.5 - i * 1.5), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='gray', linewidth=1)
        ax.add_patch(rect)
        ax.text(3.5, 6 - i * 1.5, guard, ha='center', va='center', fontsize=9)

    ax.text(3.5, 6.7, 'Input Guardrails', ha='center', va='center', fontsize=11, fontweight='bold')

    # LLM Processing
    llm_box = FancyBboxPatch((5.5, 3.5), 2, 2, boxstyle="round,pad=0.1",
                              facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(llm_box)
    ax.text(6.5, 4.5, 'LLM\nProcessing', ha='center', va='center', fontsize=11, fontweight='bold')

    # Output guardrails
    output_guards = ['Output Moderation', 'Fact Checking', 'Response Formatting']
    colors = ['#ffcdd2', '#c8e6c9', '#bbdefb']

    for i, (guard, color) in enumerate(zip(output_guards, colors)):
        rect = FancyBboxPatch((8.5, 5.5 - i * 1.5), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='gray', linewidth=1)
        ax.add_patch(rect)
        ax.text(9.5, 6 - i * 1.5, guard, ha='center', va='center', fontsize=9)

    ax.text(9.5, 6.7, 'Output Guardrails', ha='center', va='center', fontsize=11, fontweight='bold')

    # Output
    ax.text(11, 5, 'Safe\nResponse', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#c8e6c9', edgecolor='#388e3c', linewidth=2))

    # Fallback
    ax.text(6.5, 1.5, 'Fallback\nResponse', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#ffcdd2', edgecolor='#c62828', linewidth=2))

    # Arrows
    ax.annotate("", xy=(2.5, 5), xytext=(1.5, 5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate("", xy=(5.5, 4.5), xytext=(4.5, 4.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate("", xy=(8.5, 4.5), xytext=(7.5, 4.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate("", xy=(10.5, 5), xytext=(10.5, 5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Fallback arrows
    ax.annotate("", xy=(6.5, 2), xytext=(3.5, 2.5),
               arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.5, ls='--'))
    ax.annotate("", xy=(6.5, 2), xytext=(9.5, 2.5),
               arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.5, ls='--'))

    ax.text(5, 2.5, 'Blocked', ha='center', va='center', fontsize=8, color='#c62828')
    ax.text(8, 2.5, 'Blocked', ha='center', va='center', fontsize=8, color='#c62828')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'safety_layers.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_evaluation_pyramid():
    """Create evaluation strategy pyramid."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('GenAI Evaluation Strategy Pyramid', fontsize=14, fontweight='bold', pad=20)

    # Pyramid layers (bottom to top)
    layers = [
        ('Automated Metrics', 'BLEU, ROUGE, custom classifiers', 0, '#e8f5e9', 5),
        ('LLM-as-Judge', 'GPT-4 evaluation on criteria', 1.5, '#fff3e0', 3.5),
        ('Human Evaluation', 'Expert annotation sampling', 3, '#fce4ec', 2.5),
        ('A/B Testing', 'Business metrics', 4.5, '#e3f2fd', 1.5)
    ]

    for label, desc, y, color, width in layers:
        # Trapezoid
        left = 5 - width
        right = 5 + width
        bottom = y
        top = y + 1.3

        trap_x = [left, right, right - 0.75, left + 0.75]
        trap_y = [bottom, bottom, top, top]

        ax.fill(trap_x, trap_y, color=color, edgecolor='gray', linewidth=2)
        ax.text(5, y + 0.65, label, ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(5, y + 0.2, desc, ha='center', va='center', fontsize=8, style='italic')

    # Side labels
    ax.text(1, 2.5, 'High\nVolume', ha='center', va='center', fontsize=10, color='gray')
    ax.text(1, 5.5, 'Low\nVolume', ha='center', va='center', fontsize=10, color='gray')
    ax.annotate("", xy=(1, 6), xytext=(1, 2),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.text(9, 2.5, 'Low\nCost', ha='center', va='center', fontsize=10, color='gray')
    ax.text(9, 5.5, 'High\nCost', ha='center', va='center', fontsize=10, color='gray')
    ax.annotate("", xy=(9, 6), xytext=(9, 2),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'evaluation_pyramid.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_time_allocation():
    """Create interview time allocation chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    steps = ['Clarify', 'Architecture', 'Components', 'Optimize', 'Safety', 'Operations']
    times = [5, 10, 15, 5, 5, 5]
    colors = ['#1976d2', '#388e3c', '#f57c00', '#c2185b', '#7b1fa2', '#00796b']

    bars = ax.barh(steps, times, color=colors, edgecolor='white', linewidth=2)

    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_title('Interview Time Allocation (45 min total)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 20)

    # Add value labels
    for bar, time in zip(bars, times):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{time} min', va='center', fontsize=10, fontweight='bold')

    # Total line
    ax.axvline(x=sum(times), color='red', linestyle='--', linewidth=2)
    ax.text(sum(times) + 0.5, 2.5, f'Total: {sum(times)} min', fontsize=10, color='red')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_allocation.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_question_categories():
    """Create interview question categories pie chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    categories = ['Chatbots &\nAssistants', 'RAG &\nKnowledge', 'Code\nGeneration',
                  'Content\nCreation', 'Real-time\nSystems', 'Moderation\n& Safety']
    sizes = [30, 25, 15, 15, 10, 5]
    colors = ['#42a5f5', '#66bb6a', '#ff7043', '#ab47bc', '#26c6da', '#ef5350']
    explode = (0.05, 0.05, 0, 0, 0, 0)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=categories,
                                       autopct='%1.0f%%', colors=colors,
                                       startangle=90, pctdistance=0.75,
                                       wedgeprops=dict(edgecolor='white', linewidth=2))

    ax.set_title('GenAI System Design Question Categories', fontsize=14, fontweight='bold')

    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'question_categories.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    print("Generating design framework visualizations...")
    create_framework_overview()
    create_optimization_triangle()
    create_safety_layers()
    create_evaluation_pyramid()
    create_time_allocation()
    create_question_categories()
    print(f"Visualizations saved to {OUTPUT_DIR}")
