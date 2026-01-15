#!/usr/bin/env python3
"""
Evaluation Taxonomy Tree Visualization

Creates tree diagrams and hierarchical visualizations for the
LLM evaluation taxonomy, showing different evaluation dimensions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_taxonomy_tree():
    """
    Create a hierarchical tree visualization of evaluation taxonomy.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Root node
    root = {'x': 8, 'y': 11, 'label': 'LLM Evaluation', 'color': '#34495e'}

    # Level 1: What to evaluate
    level1 = [
        {'x': 3, 'y': 9, 'label': 'Capability', 'color': '#3498db'},
        {'x': 8, 'y': 9, 'label': 'Safety', 'color': '#e74c3c'},
        {'x': 13, 'y': 9, 'label': 'Alignment', 'color': '#2ecc71'},
    ]

    # Level 2: Specific aspects
    level2 = {
        'Capability': [
            {'x': 1, 'y': 6.5, 'label': 'Reasoning', 'color': '#5dade2'},
            {'x': 3, 'y': 6.5, 'label': 'Knowledge', 'color': '#5dade2'},
            {'x': 5, 'y': 6.5, 'label': 'Coding', 'color': '#5dade2'},
        ],
        'Safety': [
            {'x': 6.5, 'y': 6.5, 'label': 'Toxicity', 'color': '#ec7063'},
            {'x': 8, 'y': 6.5, 'label': 'Bias', 'color': '#ec7063'},
            {'x': 9.5, 'y': 6.5, 'label': 'Jailbreak', 'color': '#ec7063'},
        ],
        'Alignment': [
            {'x': 11.5, 'y': 6.5, 'label': 'Helpful', 'color': '#58d68d'},
            {'x': 13, 'y': 6.5, 'label': 'Honest', 'color': '#58d68d'},
            {'x': 14.5, 'y': 6.5, 'label': 'Harmless', 'color': '#58d68d'},
        ],
    }

    # Level 3: How to evaluate
    level3 = [
        {'x': 4, 'y': 4, 'label': 'Intrinsic\n(Model Properties)', 'color': '#f39c12'},
        {'x': 12, 'y': 4, 'label': 'Extrinsic\n(Task Performance)', 'color': '#9b59b6'},
    ]

    # Level 4: Who evaluates
    level4 = [
        {'x': 2, 'y': 1.5, 'label': 'Automated\nMetrics', 'color': '#1abc9c'},
        {'x': 6, 'y': 1.5, 'label': 'LLM\nJudge', 'color': '#e67e22'},
        {'x': 10, 'y': 1.5, 'label': 'Human\nEvaluation', 'color': '#8e44ad'},
        {'x': 14, 'y': 1.5, 'label': 'Benchmarks', 'color': '#16a085'},
    ]

    # Helper function to draw nodes
    def draw_node(node, width=1.8, height=0.8):
        rect = mpatches.FancyBboxPatch(
            (node['x'] - width/2, node['y'] - height/2),
            width, height,
            boxstyle="round,pad=0.05",
            facecolor=node['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(node['x'], node['y'], node['label'],
                fontsize=10, fontweight='bold', ha='center', va='center', color='white')

    # Helper function to draw connections
    def draw_connection(parent, child, style='-'):
        ax.plot([parent['x'], child['x']], [parent['y'] - 0.4, child['y'] + 0.4],
                style, color='gray', linewidth=1.5)

    # Draw root
    draw_node(root, width=2.5, height=0.9)

    # Draw level 1
    for node in level1:
        draw_node(node, width=2.0, height=0.8)
        draw_connection(root, node)

    # Draw level 2
    for parent_label, children in level2.items():
        parent = next(n for n in level1 if n['label'] == parent_label)
        for child in children:
            draw_node(child, width=1.5, height=0.7)
            draw_connection(parent, child)

    # Draw level 3 with bracket
    ax.plot([4, 4], [5.8, 4.5], 'k-', linewidth=1.5)
    ax.plot([12, 12], [5.8, 4.5], 'k-', linewidth=1.5)
    for node in level3:
        draw_node(node, width=2.5, height=0.9)

    # Draw level 4
    ax.text(8, 3, 'Evaluator Type', fontsize=11, ha='center', fontweight='bold')
    ax.plot([3, 13], [2.7, 2.7], 'k-', linewidth=1)
    for node in level4:
        draw_node(node, width=2.0, height=0.9)
        ax.plot([node['x'], node['x']], [2.7, 2.0], 'k-', linewidth=1)

    # Add legend boxes
    legend_data = [
        ('What to Evaluate', '#3498db', 10.5),
        ('Evaluation Method', '#f39c12', 9.8),
        ('Who Evaluates', '#1abc9c', 9.1),
    ]
    for label, color, y in legend_data:
        ax.add_patch(mpatches.Rectangle((0.3, y - 0.15), 0.4, 0.3, facecolor=color, edgecolor='black'))
        ax.text(0.9, y, label, fontsize=9, va='center')

    ax.set_title('LLM Evaluation Taxonomy', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'evaluation_taxonomy_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'evaluation_taxonomy_tree.png'}")


def create_evaluation_decision_flowchart():
    """
    Create a flowchart for choosing evaluation methods.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define flowchart elements
    def draw_box(x, y, text, color, width=2.2, height=0.8, shape='rect'):
        if shape == 'diamond':
            diamond = mpatches.RegularPolygon(
                (x, y), 4, radius=0.7,
                facecolor=color, edgecolor='black', linewidth=2, alpha=0.85
            )
            ax.add_patch(diamond)
        else:
            rect = mpatches.FancyBboxPatch(
                (x - width/2, y - height/2),
                width, height,
                boxstyle="round,pad=0.03",
                facecolor=color,
                edgecolor='black',
                linewidth=2,
                alpha=0.85
            )
            ax.add_patch(rect)
        ax.text(x, y, text, fontsize=9, fontweight='bold', ha='center', va='center',
                color='white' if shape != 'diamond' else 'black')

    def draw_arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=8, ha='left')

    # Start
    draw_box(7, 9.5, 'Start:\nEvaluation Need', '#34495e')

    # Decision 1: Task type
    ax.add_patch(mpatches.RegularPolygon((7, 8), numVertices=4, radius=0.8,
                facecolor='#f39c12', edgecolor='black', linewidth=2))
    ax.text(7, 8, 'Task\nType?', fontsize=9, ha='center', va='center')
    draw_arrow(7, 9.1, 7, 8.7)

    # Task branches
    draw_box(3, 6.5, 'Generation\n(Open-ended)', '#3498db')
    draw_box(7, 6.5, 'Classification\n(Closed)', '#2ecc71')
    draw_box(11, 6.5, 'Code/Math\n(Verifiable)', '#9b59b6')

    draw_arrow(6.3, 7.5, 3.5, 6.9, 'Open')
    draw_arrow(7, 7.3, 7, 6.9, 'Closed')
    draw_arrow(7.7, 7.5, 10.5, 6.9, 'Code')

    # Generation path
    ax.add_patch(mpatches.RegularPolygon((3, 5.2), numVertices=4, radius=0.7,
                facecolor='#f39c12', edgecolor='black', linewidth=2))
    ax.text(3, 5.2, 'Budget?', fontsize=8, ha='center', va='center')
    draw_arrow(3, 6.1, 3, 5.8)

    draw_box(1.2, 3.8, 'LLM Judge\n+ Sampling', '#e74c3c')
    draw_box(4.8, 3.8, 'Human Eval\n+ LLM Judge', '#e67e22')

    draw_arrow(2.4, 4.7, 1.5, 4.2, 'Low')
    draw_arrow(3.6, 4.7, 4.5, 4.2, 'High')

    # Classification path
    draw_box(7, 5, 'Automated Metrics\n(F1, Accuracy)', '#27ae60')
    draw_arrow(7, 6.1, 7, 5.4)

    # Code path
    draw_box(11, 5, 'Execution Tests\n(pass@k)', '#8e44ad')
    draw_arrow(11, 6.1, 11, 5.4)

    # Safety check
    ax.add_patch(mpatches.RegularPolygon((7, 3), numVertices=4, radius=0.8,
                facecolor='#e74c3c', edgecolor='black', linewidth=2))
    ax.text(7, 3, 'Safety\nCritical?', fontsize=8, ha='center', va='center')

    draw_arrow(1.2, 3.4, 5.5, 3.2)
    draw_arrow(4.8, 3.4, 6.3, 3.1)
    draw_arrow(7, 4.6, 7, 3.7)
    draw_arrow(11, 4.6, 7.7, 3.4)

    # Safety branches
    draw_box(4, 1.5, 'Add Safety\nBenchmarks', '#c0392b')
    draw_box(10, 1.5, 'Standard\nPipeline', '#27ae60')

    draw_arrow(6.3, 2.5, 4.5, 1.9, 'Yes')
    draw_arrow(7.7, 2.5, 9.5, 1.9, 'No')

    # Title
    ax.set_title('Evaluation Method Selection Flowchart', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'evaluation_decision_flow.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'evaluation_decision_flow.png'}")


def create_evaluation_coverage_map():
    """
    Create a visual map showing coverage of different evaluation approaches.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define evaluation dimensions and methods
    dimensions = [
        'Factual\nAccuracy',
        'Fluency',
        'Relevance',
        'Safety',
        'Creativity',
        'Reasoning',
        'Code\nCorrectness'
    ]

    methods = ['Automated\nMetrics', 'LLM Judge', 'Human Eval', 'Benchmarks']

    # Coverage matrix (0-1 scale of how well each method covers each dimension)
    coverage = np.array([
        [0.4, 0.7, 0.9, 0.6],   # Factual Accuracy
        [0.8, 0.8, 0.9, 0.5],   # Fluency
        [0.5, 0.8, 0.9, 0.4],   # Relevance
        [0.6, 0.7, 0.8, 0.8],   # Safety
        [0.2, 0.6, 0.9, 0.3],   # Creativity
        [0.3, 0.7, 0.8, 0.9],   # Reasoning
        [0.9, 0.5, 0.7, 0.95],  # Code Correctness
    ])

    # Create heatmap
    im = ax.imshow(coverage, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Coverage Quality', rotation=-90, va="bottom", fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(dimensions)))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_yticklabels(dimensions, fontsize=11)

    # Add values
    for i in range(len(dimensions)):
        for j in range(len(methods)):
            text_color = "white" if coverage[i, j] < 0.5 else "black"
            ax.text(j, i, f'{coverage[i, j]:.1f}',
                   ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')

    ax.set_title('Evaluation Coverage Map:\nWhich Method Covers Which Dimension?',
                 fontsize=14, fontweight='bold', pad=20)

    # Add recommendation text
    ax.text(1.02, 0.5, 'Recommendation:\nCombine multiple\nmethods for\ncomprehensive\ncoverage',
            transform=ax.transAxes, fontsize=10, va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'evaluation_coverage_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'evaluation_coverage_map.png'}")


def create_cost_quality_tradeoff():
    """
    Create a visualization showing cost vs quality tradeoffs for evaluation methods.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Evaluation methods with their characteristics
    methods = [
        ('Automated\nMetrics', 5, 55, 800, '#3498db'),
        ('LLM Judge\n(GPT-4)', 50, 78, 500, '#e74c3c'),
        ('LLM Judge\n(Open)', 20, 68, 400, '#9b59b6'),
        ('Crowdsource', 200, 75, 350, '#f39c12'),
        ('Expert\nHuman', 1000, 92, 200, '#2ecc71'),
        ('Multi-Judge\nEnsemble', 150, 85, 300, '#1abc9c'),
    ]

    for name, cost, quality, speed, color in methods:
        ax.scatter(cost, quality, s=speed, c=color, alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(name, (cost, quality), xytext=(10, 5), textcoords='offset points',
                   fontsize=9, ha='left')

    ax.set_xscale('log')
    ax.set_xlabel('Cost per 1000 Evaluations ($)', fontsize=12)
    ax.set_ylabel('Correlation with Gold Standard (%)', fontsize=12)
    ax.set_title('Evaluation Methods: Cost vs Quality Tradeoff', fontsize=14, fontweight='bold')

    ax.set_xlim(1, 5000)
    ax.set_ylim(45, 100)
    ax.grid(True, alpha=0.3)

    # Add Pareto frontier annotation
    ax.plot([5, 50, 150, 1000], [55, 78, 85, 92], 'k--', alpha=0.5, linewidth=2)
    ax.text(100, 88, 'Pareto\nFrontier', fontsize=9, style='italic')

    # Add size legend
    sizes = [200, 500, 800]
    labels = ['Slow', 'Medium', 'Fast']
    for i, (size, label) in enumerate(zip(sizes, labels)):
        ax.scatter([], [], s=size, c='gray', alpha=0.5, label=f'{label} ({size} eval/hr)')
    ax.legend(title='Throughput', loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cost_quality_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'cost_quality_tradeoff.png'}")


def main():
    """Generate all evaluation taxonomy visualizations."""
    print("Generating evaluation taxonomy visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    create_taxonomy_tree()
    create_evaluation_decision_flowchart()
    create_evaluation_coverage_map()
    create_cost_quality_tradeoff()

    print()
    print("All evaluation taxonomy visualizations generated successfully!")


if __name__ == "__main__":
    main()
