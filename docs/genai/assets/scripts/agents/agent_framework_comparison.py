"""
Agent Framework Comparison Visualization

Creates radar charts and comparison diagrams for different
agent frameworks (LangChain, LlamaIndex, AutoGen, CrewAI).

Output: /docs/genai/assets/images/agents/agent_framework_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "agents"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Framework data
FRAMEWORKS = {
    'LangChain': {
        'color': '#4CAF50',
        'scores': {
            'Flexibility': 9,
            'RAG Support': 7,
            'Multi-Agent': 6,
            'Ease of Use': 6,
            'Ecosystem': 9,
            'Documentation': 8,
            'Production Ready': 8,
            'Tool Support': 9,
        }
    },
    'LlamaIndex': {
        'color': '#2196F3',
        'scores': {
            'Flexibility': 7,
            'RAG Support': 10,
            'Multi-Agent': 4,
            'Ease of Use': 7,
            'Ecosystem': 6,
            'Documentation': 8,
            'Production Ready': 8,
            'Tool Support': 6,
        }
    },
    'AutoGen': {
        'color': '#FF9800',
        'scores': {
            'Flexibility': 6,
            'RAG Support': 4,
            'Multi-Agent': 10,
            'Ease of Use': 7,
            'Ecosystem': 5,
            'Documentation': 6,
            'Production Ready': 6,
            'Tool Support': 7,
        }
    },
    'CrewAI': {
        'color': '#E91E63',
        'scores': {
            'Flexibility': 5,
            'RAG Support': 5,
            'Multi-Agent': 9,
            'Ease of Use': 9,
            'Ecosystem': 4,
            'Documentation': 7,
            'Production Ready': 5,
            'Tool Support': 6,
        }
    }
}


def create_radar_chart():
    """Create radar chart comparing all frameworks."""
    categories = list(FRAMEWORKS['LangChain']['scores'].keys())
    N = len(categories)

    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Draw one framework at a time
    for name, data in FRAMEWORKS.items():
        values = list(data['scores'].values())
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=name,
                color=data['color'], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=data['color'])

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    # Set y-axis
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)

    # Add title and legend
    plt.title('Agent Framework Comparison\n', size=16, fontweight='bold',
              color='#1a237e', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'agent_framework_radar.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_individual_radars():
    """Create individual radar charts for each framework."""
    categories = list(FRAMEWORKS['LangChain']['scores'].keys())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                             subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(FRAMEWORKS.items()):
        ax = axes[idx]
        values = list(data['scores'].values())
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=data['color'],
                markersize=8)
        ax.fill(angles, values, alpha=0.25, color=data['color'])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['', '', '', '', ''], size=8)
        ax.set_title(name, size=14, fontweight='bold', color=data['color'],
                     pad=15)

    plt.suptitle('Framework Comparison: Individual Profiles', size=16,
                 fontweight='bold', color='#1a237e', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'agent_framework_individual.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_feature_heatmap():
    """Create a heatmap comparing framework features."""
    fig, ax = plt.subplots(figsize=(12, 8))

    frameworks = list(FRAMEWORKS.keys())
    features = list(FRAMEWORKS['LangChain']['scores'].keys())

    # Create data matrix
    data = np.array([[FRAMEWORKS[f]['scores'][feat] for feat in features]
                     for f in frameworks])

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)

    # Set ticks
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(frameworks)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(frameworks, fontsize=11)

    # Add color bar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Score (0-10)', rotation=-90, va="bottom", fontsize=10)

    # Add text annotations
    for i in range(len(frameworks)):
        for j in range(len(features)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="black",
                           fontsize=10, fontweight='bold')

    # Add framework colors on left
    for i, framework in enumerate(frameworks):
        color = FRAMEWORKS[framework]['color']
        rect = patches.Rectangle((-0.6, i - 0.4), 0.15, 0.8,
                                  facecolor=color, edgecolor='none')
        ax.add_patch(rect)

    ax.set_title('Framework Feature Comparison Heatmap', fontsize=14,
                 fontweight='bold', color='#1a237e', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'agent_framework_heatmap.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_use_case_matrix():
    """Create a use case recommendation matrix."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, "Framework Selection Guide", fontsize=16,
            ha='center', fontweight='bold', color='#1a237e')

    # Use cases and recommendations
    use_cases = [
        {
            'case': 'RAG / Document QA',
            'best': 'LlamaIndex',
            'reason': 'Best indexing & retrieval',
            'y': 6.2
        },
        {
            'case': 'General Agent / Chatbot',
            'best': 'LangChain',
            'reason': 'Most flexible, large ecosystem',
            'y': 5.2
        },
        {
            'case': 'Multi-Agent Collaboration',
            'best': 'AutoGen',
            'reason': 'Built-in multi-agent support',
            'y': 4.2
        },
        {
            'case': 'Role-Based Team Tasks',
            'best': 'CrewAI',
            'reason': 'Intuitive role abstraction',
            'y': 3.2
        },
        {
            'case': 'Complex Workflows',
            'best': 'LangGraph',
            'reason': 'State machines for agents',
            'y': 2.2
        },
        {
            'case': 'Quick Prototype',
            'best': 'CrewAI',
            'reason': 'Fastest to get started',
            'y': 1.2
        },
    ]

    # Draw header
    headers = [('Use Case', 2.5), ('Recommended', 6.5), ('Why', 11)]
    for text, x in headers:
        ax.text(x, 6.8, text, fontsize=11, fontweight='bold',
                ha='center', color='#333')

    # Draw separator line
    ax.plot([0.5, 13.5], [6.6, 6.6], 'k-', linewidth=1)

    # Draw each row
    for uc in use_cases:
        y = uc['y']

        # Use case text
        ax.text(2.5, y, uc['case'], fontsize=10, ha='center', va='center')

        # Recommended framework (with color)
        framework_color = FRAMEWORKS.get(uc['best'], {}).get('color', '#666')
        rect = FancyBboxPatch(
            (5, y - 0.3), 3, 0.6,
            boxstyle="round,pad=0.02",
            facecolor=framework_color,
            edgecolor='none',
            alpha=0.2
        )
        ax.add_patch(rect)
        ax.text(6.5, y, uc['best'], fontsize=10, ha='center', va='center',
                fontweight='bold', color=framework_color)

        # Reason
        ax.text(11, y, uc['reason'], fontsize=9, ha='center', va='center',
                style='italic', color='#666')

        # Row separator
        ax.plot([0.5, 13.5], [y - 0.45, y - 0.45], color='#e0e0e0',
                linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'agent_framework_guide.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_bar_comparison():
    """Create a grouped bar chart comparison."""
    fig, ax = plt.subplots(figsize=(14, 8))

    frameworks = list(FRAMEWORKS.keys())
    features = list(FRAMEWORKS['LangChain']['scores'].keys())
    n_frameworks = len(frameworks)
    n_features = len(features)

    # Bar positions
    x = np.arange(n_features)
    width = 0.2

    # Draw bars for each framework
    for i, (name, data) in enumerate(FRAMEWORKS.items()):
        values = list(data['scores'].values())
        offset = (i - n_frameworks/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=data['color'], alpha=0.8)

    # Customize
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Score (0-10)', fontsize=12)
    ax.set_title('Agent Framework Feature Comparison', fontsize=14,
                 fontweight='bold', color='#1a237e', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 11)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'agent_framework_bars.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    print("Creating agent framework comparison visualizations...")

    create_radar_chart()
    print(f"Radar chart saved to: {OUTPUT_DIR}/agent_framework_radar.png")

    create_individual_radars()
    print(f"Individual radars saved to: {OUTPUT_DIR}/agent_framework_individual.png")

    create_feature_heatmap()
    print(f"Feature heatmap saved to: {OUTPUT_DIR}/agent_framework_heatmap.png")

    create_use_case_matrix()
    print(f"Use case guide saved to: {OUTPUT_DIR}/agent_framework_guide.png")

    create_bar_comparison()
    print(f"Bar comparison saved to: {OUTPUT_DIR}/agent_framework_bars.png")

    print("Done!")
