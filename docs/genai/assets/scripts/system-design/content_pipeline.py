"""
Content Pipeline Architecture Visualization
Generates diagrams for content generation system design
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'images', 'system-design')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_content_pipeline():
    """Create content generation pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Content Generation Pipeline', fontsize=16, fontweight='bold', pad=20)

    # Color scheme
    colors = {
        'input': '#e3f2fd',
        'planning': '#e8f5e9',
        'generation': '#fff3e0',
        'quality': '#fce4ec',
        'review': '#f3e5f5',
        'publish': '#e0f2f1'
    }

    # Input Layer
    input_box = FancyBboxPatch((0.5, 6), 2.5, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['input'], edgecolor='#1976d2', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 7, 'Input', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.75, 6.4, 'Brief | Template | Style', ha='center', va='center', fontsize=8)

    # Planning Layer
    plan_box = FancyBboxPatch((3.5, 6), 2.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor=colors['planning'], edgecolor='#388e3c', linewidth=2)
    ax.add_patch(plan_box)
    ax.text(4.75, 7, 'Planning', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(4.75, 6.4, 'Outline | Research', ha='center', va='center', fontsize=8)

    # Generation Layer
    gen_box = FancyBboxPatch((6.5, 6), 2.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor=colors['generation'], edgecolor='#f57c00', linewidth=2)
    ax.add_patch(gen_box)
    ax.text(7.75, 7, 'Generation', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7.75, 6.4, 'Draft | Variations', ha='center', va='center', fontsize=8)

    # Quality Layer
    quality_box = FancyBboxPatch((0.5, 3.5), 4, 1.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['quality'], edgecolor='#c2185b', linewidth=2)
    ax.add_patch(quality_box)
    ax.text(2.5, 4.5, 'Automated Quality', ha='center', va='center', fontsize=11, fontweight='bold')
    quality_items = ['Grammar', 'Plagiarism', 'Facts', 'Brand', 'Safety']
    for i, item in enumerate(quality_items):
        ax.text(1 + i * 0.8, 3.9, item, ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#c2185b'))

    # Human Review Layer
    review_box = FancyBboxPatch((5, 3.5), 4, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['review'], edgecolor='#7b1fa2', linewidth=2)
    ax.add_patch(review_box)
    ax.text(7, 4.5, 'Human Review', ha='center', va='center', fontsize=11, fontweight='bold')
    review_items = ['Queue', 'Editor', 'Approval']
    for i, item in enumerate(review_items):
        ax.text(5.5 + i * 1.3, 3.9, item, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#7b1fa2'))

    # Publishing Layer
    publish_box = FancyBboxPatch((9.5, 3.5), 4, 1.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['publish'], edgecolor='#00796b', linewidth=2)
    ax.add_patch(publish_box)
    ax.text(11.5, 4.5, 'Publishing', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(11.5, 3.9, 'CMS | Distribution | Analytics', ha='center', va='center', fontsize=8)

    # Feedback Loop
    feedback_box = FancyBboxPatch((4, 1), 6, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='#fff8e1', edgecolor='#ffa000', linewidth=2)
    ax.add_patch(feedback_box)
    ax.text(7, 2, 'Feedback Loop', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7, 1.4, 'Learning | Prompt Improvement | Quality Tracking', ha='center', va='center', fontsize=8)

    # Arrows
    arrow_props = dict(arrowstyle='->', color='gray', lw=2)

    # Input to Planning
    ax.annotate("", xy=(3.5, 6.75), xytext=(3, 6.75), arrowprops=arrow_props)
    # Planning to Generation
    ax.annotate("", xy=(6.5, 6.75), xytext=(6, 6.75), arrowprops=arrow_props)
    # Generation to Quality
    ax.annotate("", xy=(2.5, 5), xytext=(7.75, 6), arrowprops=arrow_props)
    # Quality to Review
    ax.annotate("", xy=(5, 4.25), xytext=(4.5, 4.25), arrowprops=arrow_props)
    # Review to Publish
    ax.annotate("", xy=(9.5, 4.25), xytext=(9, 4.25), arrowprops=arrow_props)

    # Feedback connections
    ax.annotate("", xy=(7, 2.5), xytext=(7, 3.5),
               arrowprops=dict(arrowstyle='<->', color='#ffa000', lw=1.5, ls='--'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'content_pipeline.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_quality_scoring():
    """Create quality scoring breakdown visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    categories = ['Grammar', 'Readability', 'Originality', 'Accuracy', 'Brand', 'Safety']
    weights = [15, 10, 20, 25, 15, 15]
    colors = ['#64b5f6', '#81c784', '#ffb74d', '#e57373', '#ba68c8', '#4db6ac']

    # Create pie chart
    wedges, texts, autotexts = ax.pie(weights, labels=categories, autopct='%1.0f%%',
                                       colors=colors, startangle=90,
                                       wedgeprops=dict(edgecolor='white', linewidth=2))

    ax.set_title('Content Quality Score Weights', fontsize=14, fontweight='bold')

    # Make text bold
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'quality_scoring.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_human_review_flow():
    """Create human review workflow visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Human-in-the-Loop Review Workflow', fontsize=14, fontweight='bold', pad=20)

    # Stages
    stages = [
        ('Auto QC', 1.5, 4, '#e3f2fd'),
        ('Triage', 4, 4, '#e8f5e9'),
        ('Review', 6.5, 4, '#fff3e0'),
        ('Decision', 9, 4, '#fce4ec')
    ]

    for name, x, y, color in stages:
        rect = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1.2, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='gray', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows between stages
    for i in range(len(stages)-1):
        ax.annotate("", xy=(stages[i+1][1]-0.8, stages[i+1][2]),
                   xytext=(stages[i][1]+0.8, stages[i][2]),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Decision outcomes
    outcomes = [
        ('Auto-Publish\n70%', 10.5, 5, '#c8e6c9'),
        ('Publish', 10.5, 4, '#c8e6c9'),
        ('Reject', 10.5, 3, '#ffcdd2'),
        ('Regenerate', 10.5, 2, '#fff9c4')
    ]

    for name, x, y, color in outcomes:
        ax.text(x, y, name, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor=color, edgecolor='gray'))

    # Auto QC percentages
    ax.text(1.5, 2.5, '70% Pass\n→ Auto-Publish', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#c8e6c9', edgecolor='#388e3c'))
    ax.text(1.5, 1.5, '30% → Review', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#fff9c4', edgecolor='#ffa000'))

    ax.annotate("", xy=(1.5, 3), xytext=(1.5, 3.5),
               arrowprops=dict(arrowstyle='->', color='#388e3c', lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'human_review_flow.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_scaling_metrics():
    """Create content scaling metrics visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    content_types = ['Social Posts', 'Blog Articles', 'Product Desc', 'Email\nCampaigns', 'Technical\nDocs']
    volumes = [1000, 50, 500, 100, 10]  # per day
    latencies = [30, 300, 60, 120, 900]  # seconds
    auto_approve = [85, 70, 90, 80, 60]  # percentage

    x = np.arange(len(content_types))
    width = 0.25

    bars1 = ax.bar(x - width, [v/10 for v in volumes], width, label='Volume/day (/10)', color='#42a5f5')
    bars2 = ax.bar(x, [l/10 for l in latencies], width, label='Latency (s/10)', color='#ff7043')
    bars3 = ax.bar(x + width, auto_approve, width, label='Auto-Approve %', color='#66bb6a')

    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Content Generation SLAs by Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(content_types)
    ax.legend(loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scaling_metrics.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_multi_agent_system():
    """Create multi-agent content system visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Multi-Agent Content Generation System', fontsize=14, fontweight='bold', pad=20)

    # Orchestrator (center)
    orch_box = FancyBboxPatch((4.5, 3.5), 3, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#e0e0e0', edgecolor='#424242', linewidth=2)
    ax.add_patch(orch_box)
    ax.text(6, 4.25, 'Orchestrator', ha='center', va='center', fontsize=12, fontweight='bold')

    # Agents
    agents = [
        ('Research\nAgent', 1, 6, '#e3f2fd', 'Facts, Sources'),
        ('Writing\nAgent', 4, 6, '#e8f5e9', 'Draft Content'),
        ('Editor\nAgent', 8, 6, '#fff3e0', 'Style, Clarity'),
        ('SEO\nAgent', 11, 6, '#fce4ec', 'Keywords'),
        ('Critic\nAgent', 6, 1, '#f3e5f5', 'Quality Check')
    ]

    for name, x, y, color, desc in agents:
        rect = FancyBboxPatch((x-1, y-0.6), 2, 1.2, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='gray', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x, y-0.9, desc, ha='center', va='center', fontsize=7, style='italic')

    # Shared Memory
    memory_box = FancyBboxPatch((4.5, 0), 3, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#fff8e1', edgecolor='#ffa000', linewidth=2)
    ax.add_patch(memory_box)
    ax.text(6, 0.4, 'Shared Memory', ha='center', va='center', fontsize=10, fontweight='bold')

    # Connections to orchestrator
    for name, x, y, _, _ in agents[:-1]:  # Exclude critic
        ax.annotate("", xy=(6, 5), xytext=(x, y-0.6),
                   arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))

    # Critic connection
    ax.annotate("", xy=(6, 3.5), xytext=(6, 1.6),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))

    # Memory connections
    ax.annotate("", xy=(6, 0.8), xytext=(6, 3.5),
               arrowprops=dict(arrowstyle='<->', color='#ffa000', lw=1.5, ls='--'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_agent_system.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    print("Generating content pipeline visualizations...")
    create_content_pipeline()
    create_quality_scoring()
    create_human_review_flow()
    create_scaling_metrics()
    create_multi_agent_system()
    print(f"Visualizations saved to {OUTPUT_DIR}")
