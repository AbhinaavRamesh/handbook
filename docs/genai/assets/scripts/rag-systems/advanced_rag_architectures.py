#!/usr/bin/env python3
"""
Advanced RAG Architectures Visualization
Diagrams for FLARE, Self-RAG, CRAG, GraphRAG, and multi-hop retrieval patterns.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'images', 'rag-systems')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_flare_diagram():
    """Create FLARE (Forward-Looking Active REtrieval) diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    colors = {
        'generation': '#E3F2FD',
        'confidence': '#FFF3E0',
        'retrieval': '#E8F5E9',
        'low_conf': '#FFCDD2'
    }

    # Title
    ax.text(7, 7.5, 'FLARE: Forward-Looking Active Retrieval', fontsize=14, fontweight='bold',
            ha='center')

    # Generation step 1
    gen1 = FancyBboxPatch((0.5, 5), 3, 1.5, boxstyle="round,pad=0.05",
                           facecolor=colors['generation'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(gen1)
    ax.text(2, 6, 'Generate', fontsize=10, fontweight='bold', ha='center')
    ax.text(2, 5.4, '"The capital of France..."', fontsize=8, ha='center', style='italic')

    # Confidence check
    conf1 = FancyBboxPatch((4, 5), 2.5, 1.5, boxstyle="round,pad=0.05",
                            facecolor=colors['confidence'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(conf1)
    ax.text(5.25, 6, 'Confidence?', fontsize=10, fontweight='bold', ha='center')
    ax.text(5.25, 5.4, 'Check logprobs', fontsize=8, ha='center')

    # High confidence path
    high_conf = FancyBboxPatch((7, 6), 2.5, 1, boxstyle="round,pad=0.05",
                                facecolor=colors['generation'], edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(high_conf)
    ax.text(8.25, 6.5, 'HIGH: Continue', fontsize=9, fontweight='bold', ha='center', color='#4CAF50')

    # Low confidence path
    low_conf = FancyBboxPatch((7, 4.5), 2.5, 1, boxstyle="round,pad=0.05",
                               facecolor=colors['low_conf'], edgecolor='#F44336', linewidth=2)
    ax.add_patch(low_conf)
    ax.text(8.25, 5, 'LOW: Retrieve', fontsize=9, fontweight='bold', ha='center', color='#F44336')

    # Retrieval step
    retrieve = FancyBboxPatch((10, 4), 3, 1.5, boxstyle="round,pad=0.05",
                               facecolor=colors['retrieval'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(retrieve)
    ax.text(11.5, 5, 'Retrieve Docs', fontsize=10, fontweight='bold', ha='center')
    ax.text(11.5, 4.4, 'Using low-conf tokens', fontsize=8, ha='center')

    # Resume generation
    resume = FancyBboxPatch((10, 2), 3, 1.5, boxstyle="round,pad=0.05",
                             facecolor=colors['generation'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(resume)
    ax.text(11.5, 3, 'Resume Generation', fontsize=10, fontweight='bold', ha='center')
    ax.text(11.5, 2.4, 'With new context', fontsize=8, ha='center')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#666', lw=1.5)
    ax.annotate('', xy=(4, 5.75), xytext=(3.5, 5.75), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 6.5), xytext=(6.5, 5.75), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 5), xytext=(6.5, 5.75), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 4.75), xytext=(9.5, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(11.5, 3.5), xytext=(11.5, 4), arrowprops=arrow_style)

    # Loop back
    ax.annotate('', xy=(2, 5), xytext=(10, 2.5),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2,
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(5, 2.8, 'Loop until complete', fontsize=9, color='#2196F3', ha='center')

    # Continue to output
    ax.annotate('', xy=(12, 6.5), xytext=(9.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))
    ax.text(12.5, 6.5, 'Final\nOutput', fontsize=10, fontweight='bold', color='#4CAF50')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'flare_architecture.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_self_rag_diagram():
    """Create Self-RAG diagram with reflection tokens."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    colors = {
        'query': '#E3F2FD',
        'decision': '#FFF3E0',
        'retrieval': '#E8F5E9',
        'critique': '#F3E5F5',
        'output': '#E0F2F1'
    }

    # Title
    ax.text(7, 9.5, 'Self-RAG: Self-Reflective Retrieval-Augmented Generation',
            fontsize=14, fontweight='bold', ha='center')

    # Query input
    query = FancyBboxPatch((5.5, 8), 3, 1, boxstyle="round,pad=0.05",
                            facecolor=colors['query'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(query)
    ax.text(7, 8.5, 'User Query', fontsize=10, fontweight='bold', ha='center')

    # Retrieve decision
    retrieve_dec = FancyBboxPatch((5.5, 6), 3, 1.5, boxstyle="round,pad=0.05",
                                   facecolor=colors['decision'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(retrieve_dec)
    ax.text(7, 7, '[Retrieve]', fontsize=11, fontweight='bold', ha='center', color='#E65100')
    ax.text(7, 6.4, 'Should I retrieve?', fontsize=9, ha='center')

    # Retrieval
    retrieval = FancyBboxPatch((1, 4), 3, 1.5, boxstyle="round,pad=0.05",
                                facecolor=colors['retrieval'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(retrieval)
    ax.text(2.5, 5, 'Retrieve Docs', fontsize=10, fontweight='bold', ha='center')

    # IsRel check
    is_rel = FancyBboxPatch((1, 2), 3, 1.5, boxstyle="round,pad=0.05",
                             facecolor=colors['critique'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(is_rel)
    ax.text(2.5, 3, '[IsRel]', fontsize=11, fontweight='bold', ha='center', color='#7B1FA2')
    ax.text(2.5, 2.4, 'Is doc relevant?', fontsize=9, ha='center')

    # Generate
    generate = FancyBboxPatch((5.5, 4), 3, 1.5, boxstyle="round,pad=0.05",
                               facecolor=colors['query'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(generate)
    ax.text(7, 5, 'Generate', fontsize=10, fontweight='bold', ha='center')
    ax.text(7, 4.4, 'Response', fontsize=9, ha='center')

    # IsSup check
    is_sup = FancyBboxPatch((5.5, 2), 3, 1.5, boxstyle="round,pad=0.05",
                             facecolor=colors['critique'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(is_sup)
    ax.text(7, 3, '[IsSup]', fontsize=11, fontweight='bold', ha='center', color='#7B1FA2')
    ax.text(7, 2.4, 'Is answer supported?', fontsize=9, ha='center')

    # IsUse check
    is_use = FancyBboxPatch((10, 4), 3, 1.5, boxstyle="round,pad=0.05",
                             facecolor=colors['critique'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(is_use)
    ax.text(11.5, 5, '[IsUse]', fontsize=11, fontweight='bold', ha='center', color='#7B1FA2')
    ax.text(11.5, 4.4, 'Is answer useful?', fontsize=9, ha='center')

    # Output
    output = FancyBboxPatch((10, 2), 3, 1.5, boxstyle="round,pad=0.05",
                             facecolor=colors['output'], edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(output)
    ax.text(11.5, 3, 'Final Output', fontsize=10, fontweight='bold', ha='center', color='#4CAF50')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#666', lw=1.5)

    ax.annotate('', xy=(7, 7.5), xytext=(7, 8), arrowprops=arrow_style)
    ax.annotate('', xy=(2.5, 5.5), xytext=(5.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5,
                               connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(2.5, 3.5), xytext=(2.5, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(5.5, 4.75), xytext=(4, 2.75),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5,
                               connectionstyle='arc3,rad=-0.2'))
    ax.annotate('', xy=(7, 3.5), xytext=(7, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 4.75), xytext=(8.5, 4.75), arrowprops=arrow_style)
    ax.annotate('', xy=(11.5, 3.5), xytext=(11.5, 4), arrowprops=arrow_style)

    # Skip retrieval path
    ax.annotate('', xy=(7, 5.5), xytext=(7, 6),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5))
    ax.text(8.2, 5.8, 'No retrieval\nneeded', fontsize=8, color='#2196F3')

    # Regenerate loop
    ax.annotate('', xy=(5.5, 2.75), xytext=(8.5, 2.75),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5,
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(7, 1.2, 'Regenerate if\nnot supported', fontsize=8, color='#F44336', ha='center')

    # Legend for tokens
    ax.text(0.5, 0.5, 'Reflection Tokens:', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.1, '[Retrieve] = Should retrieve?  [IsRel] = Doc relevant?  [IsSup] = Supported?  [IsUse] = Useful?',
            fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'self_rag_architecture.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_crag_diagram():
    """Create CRAG (Corrective RAG) diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    colors = {
        'query': '#E3F2FD',
        'retrieval': '#E8F5E9',
        'evaluator': '#FFF3E0',
        'correct': '#C8E6C9',
        'incorrect': '#FFCDD2',
        'web': '#E1BEE7'
    }

    # Title
    ax.text(7, 7.5, 'CRAG: Corrective Retrieval-Augmented Generation',
            fontsize=14, fontweight='bold', ha='center')

    # Query and initial retrieval
    query = FancyBboxPatch((0.5, 5.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                            facecolor=colors['query'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(query)
    ax.text(1.75, 6.3, 'Query', fontsize=10, fontweight='bold', ha='center')
    ax.text(1.75, 5.9, 'Input', fontsize=9, ha='center')

    retrieval = FancyBboxPatch((3.5, 5.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                                facecolor=colors['retrieval'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(retrieval)
    ax.text(4.75, 6.3, 'Retrieve', fontsize=10, fontweight='bold', ha='center')
    ax.text(4.75, 5.9, 'Documents', fontsize=9, ha='center')

    # Evaluator
    evaluator = FancyBboxPatch((6.5, 5.5), 3, 1.2, boxstyle="round,pad=0.05",
                                facecolor=colors['evaluator'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(evaluator)
    ax.text(8, 6.3, 'Retrieval Evaluator', fontsize=10, fontweight='bold', ha='center')
    ax.text(8, 5.9, 'Score relevance', fontsize=9, ha='center')

    # Three paths
    # Correct path
    correct = FancyBboxPatch((1, 3), 3, 1.2, boxstyle="round,pad=0.05",
                              facecolor=colors['correct'], edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(correct)
    ax.text(2.5, 3.8, 'CORRECT', fontsize=10, fontweight='bold', ha='center', color='#4CAF50')
    ax.text(2.5, 3.3, 'Use retrieved docs', fontsize=9, ha='center')

    # Ambiguous path
    ambiguous = FancyBboxPatch((5, 3), 4, 1.2, boxstyle="round,pad=0.05",
                                facecolor=colors['evaluator'], edgecolor='#FF9800', linewidth=2)
    ax.add_patch(ambiguous)
    ax.text(7, 3.8, 'AMBIGUOUS', fontsize=10, fontweight='bold', ha='center', color='#FF9800')
    ax.text(7, 3.3, 'Refine + Web supplement', fontsize=9, ha='center')

    # Incorrect path
    incorrect = FancyBboxPatch((10, 3), 3, 1.2, boxstyle="round,pad=0.05",
                                facecolor=colors['incorrect'], edgecolor='#F44336', linewidth=2)
    ax.add_patch(incorrect)
    ax.text(11.5, 3.8, 'INCORRECT', fontsize=10, fontweight='bold', ha='center', color='#F44336')
    ax.text(11.5, 3.3, 'Web search fallback', fontsize=9, ha='center')

    # Web search
    web = FancyBboxPatch((10, 1), 3, 1.2, boxstyle="round,pad=0.05",
                          facecolor=colors['web'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(web)
    ax.text(11.5, 1.8, 'Web Search', fontsize=10, fontweight='bold', ha='center')
    ax.text(11.5, 1.3, 'Google/Bing API', fontsize=9, ha='center')

    # Generation
    generate = FancyBboxPatch((5, 0.5), 3, 1.2, boxstyle="round,pad=0.05",
                               facecolor=colors['query'], edgecolor='#333', linewidth=1.5)
    ax.add_patch(generate)
    ax.text(6.5, 1.3, 'Generate', fontsize=10, fontweight='bold', ha='center')
    ax.text(6.5, 0.8, 'Final Response', fontsize=9, ha='center')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#666', lw=1.5)
    ax.annotate('', xy=(3.5, 6.1), xytext=(3, 6.1), arrowprops=arrow_style)
    ax.annotate('', xy=(6.5, 6.1), xytext=(6, 6.1), arrowprops=arrow_style)

    # To paths
    ax.annotate('', xy=(2.5, 4.2), xytext=(7.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5,
                               connectionstyle='arc3,rad=0.3'))
    ax.annotate('', xy=(7, 4.2), xytext=(8, 5.5),
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5))
    ax.annotate('', xy=(11.5, 4.2), xytext=(8.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5,
                               connectionstyle='arc3,rad=-0.3'))

    # To web search
    ax.annotate('', xy=(11.5, 2.2), xytext=(11.5, 3), arrowprops=arrow_style)

    # To generation
    ax.annotate('', xy=(5.5, 1.5), xytext=(2.5, 3),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5,
                               connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(6.5, 1.7), xytext=(7, 3),
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5))
    ax.annotate('', xy=(7.5, 1.5), xytext=(11.5, 1),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5,
                               connectionstyle='arc3,rad=-0.2'))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'crag_architecture.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_pattern_comparison():
    """Create comparison chart of advanced RAG patterns."""
    fig, ax = plt.subplots(figsize=(12, 8))

    patterns = ['Basic RAG', 'FLARE', 'Self-RAG', 'CRAG', 'GraphRAG', 'Multi-hop']

    # Metrics (normalized 0-1)
    metrics = {
        'Accuracy': [0.70, 0.82, 0.85, 0.83, 0.80, 0.78],
        'Latency (inv)': [0.90, 0.60, 0.55, 0.65, 0.50, 0.45],  # Inverted so higher is better
        'Complexity (inv)': [0.95, 0.60, 0.40, 0.55, 0.35, 0.50],  # Inverted
        'Hallucination Reduction': [0.50, 0.75, 0.88, 0.85, 0.70, 0.72],
        'Multi-hop Ability': [0.30, 0.65, 0.60, 0.55, 0.85, 0.95],
    }

    x = np.arange(len(patterns))
    width = 0.15

    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']

    for i, (metric, values) in enumerate(metrics.items()):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8)

    ax.set_xlabel('RAG Pattern')
    ax.set_ylabel('Score (normalized)')
    ax.set_title('Advanced RAG Patterns Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, rotation=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add best-for annotations
    annotations = {
        'Basic RAG': 'Simple QA',
        'FLARE': 'Long-form gen',
        'Self-RAG': 'High precision',
        'CRAG': 'Uncertain retrieval',
        'GraphRAG': 'Relationships',
        'Multi-hop': 'Complex queries'
    }

    for i, (pattern, best_for) in enumerate(annotations.items()):
        ax.text(i, -0.1, f'Best: {best_for}', ha='center', fontsize=8,
                style='italic', color='gray')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'advanced_rag_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    create_flare_diagram()
    create_self_rag_diagram()
    create_crag_diagram()
    create_pattern_comparison()
    print("Advanced RAG architecture visualizations complete!")
