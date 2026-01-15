"""
Chatbot System Architecture Visualization
Generates diagrams for chatbot system design
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'images', 'system-design')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_chatbot_architecture():
    """Create high-level chatbot architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Chatbot System Architecture', fontsize=16, fontweight='bold', pad=20)

    # Color scheme
    colors = {
        'client': '#e3f2fd',
        'gateway': '#fff3e0',
        'engine': '#e8f5e9',
        'ai': '#fce4ec',
        'data': '#f3e5f5',
        'handoff': '#fff8e1'
    }

    # Client Layer
    client_box = FancyBboxPatch((0.5, 8), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['client'], edgecolor='#1976d2', linewidth=2)
    ax.add_patch(client_box)
    ax.text(2, 8.75, 'Client Layer', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2, 8.35, 'Web | Mobile | SMS', ha='center', va='center', fontsize=8)

    # Gateway Layer
    gateway_box = FancyBboxPatch((0.5, 6), 3, 1.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['gateway'], edgecolor='#f57c00', linewidth=2)
    ax.add_patch(gateway_box)
    ax.text(2, 6.75, 'API Gateway', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2, 6.35, 'Auth | Rate Limit | Session', ha='center', va='center', fontsize=8)

    # Conversation Engine
    engine_box = FancyBboxPatch((4.5, 5.5, ), 5, 2.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['engine'], edgecolor='#388e3c', linewidth=2)
    ax.add_patch(engine_box)
    ax.text(7, 7.5, 'Conversation Engine', ha='center', va='center', fontsize=11, fontweight='bold')

    # Engine components
    components = ['Intent Router', 'Context Manager', 'Persona Engine', 'Dialog Manager']
    for i, comp in enumerate(components):
        x = 5 + (i % 2) * 2.2
        y = 6.5 - (i // 2) * 0.8
        ax.text(x, y, f'• {comp}', ha='left', va='center', fontsize=8)

    # AI Services
    ai_box = FancyBboxPatch((10, 5.5), 3.5, 2.5, boxstyle="round,pad=0.1",
                             facecolor=colors['ai'], edgecolor='#c2185b', linewidth=2)
    ax.add_patch(ai_box)
    ax.text(11.75, 7.5, 'AI Services', ha='center', va='center', fontsize=11, fontweight='bold')
    ai_components = ['NLU', 'RAG', 'LLM', 'Safety']
    for i, comp in enumerate(ai_components):
        ax.text(10.3, 6.7 - i * 0.4, f'• {comp}', ha='left', va='center', fontsize=8)

    # Data Layer
    data_box = FancyBboxPatch((4.5, 2.5), 5, 2, boxstyle="round,pad=0.1",
                               facecolor=colors['data'], edgecolor='#7b1fa2', linewidth=2)
    ax.add_patch(data_box)
    ax.text(7, 4, 'Data Layer', ha='center', va='center', fontsize=11, fontweight='bold')
    data_components = ['Conversation Store', 'User Profiles', 'Knowledge Base']
    for i, comp in enumerate(data_components):
        ax.text(5 + i * 1.6, 3.2, comp, ha='center', va='center', fontsize=7)

    # Human Handoff
    handoff_box = FancyBboxPatch((10, 2.5), 3.5, 2, boxstyle="round,pad=0.1",
                                  facecolor=colors['handoff'], edgecolor='#ffa000', linewidth=2)
    ax.add_patch(handoff_box)
    ax.text(11.75, 4, 'Human Handoff', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(11.75, 3.2, 'Queue | Routing | Agent UI', ha='center', va='center', fontsize=8)

    # Arrows
    arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=8"
    kw = dict(arrowstyle=arrow_style, color="gray", lw=1.5)

    # Client to Gateway
    ax.annotate("", xy=(2, 7.5), xytext=(2, 8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Gateway to Engine
    ax.annotate("", xy=(4.5, 6.75), xytext=(3.5, 6.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Engine to AI
    ax.annotate("", xy=(10, 6.75), xytext=(9.5, 6.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Engine to Data
    ax.annotate("", xy=(7, 5.5), xytext=(7, 4.5),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=2))

    # Engine to Handoff
    ax.annotate("", xy=(10, 4), xytext=(9.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chatbot_architecture.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_context_management():
    """Create context management strategy diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Conversation Context Management', fontsize=14, fontweight='bold', pad=20)

    # Token budget visualization
    categories = ['System Prompt', 'Summary', 'Recent History', 'Current Turn']
    percentages = [12.5, 12.5, 50, 25]
    colors = ['#e3f2fd', '#fff3e0', '#e8f5e9', '#fce4ec']

    # Draw stacked bar
    left = 1
    for cat, pct, color in zip(categories, percentages, colors):
        width = pct / 100 * 8
        rect = FancyBboxPatch((left, 3), width, 1.5, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='gray', linewidth=1)
        ax.add_patch(rect)
        ax.text(left + width/2, 3.75, f'{cat}\n{pct}%', ha='center', va='center', fontsize=8)
        left += width

    ax.text(5, 5, '4K Token Context Window Budget', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Add description
    descriptions = [
        ('~500 tokens', 'Persona, boundaries'),
        ('~500 tokens', 'Older turns summarized'),
        ('~2000 tokens', 'Last 10 turns verbatim'),
        ('~1000 tokens', 'Current + response space')
    ]

    for i, (tokens, desc) in enumerate(descriptions):
        ax.text(2 + i * 2.5, 2.2, tokens, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(2 + i * 2.5, 1.8, desc, ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'context_management.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_handoff_scoring():
    """Create handoff scoring visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    factors = ['Sentiment', 'Frustration', 'Bot Confidence', 'Turn Count', 'Repetitions']
    weights = [30, 25, 20, 15, 10]
    colors = ['#ef5350', '#ff7043', '#ffca28', '#66bb6a', '#42a5f5']

    bars = ax.barh(factors, weights, color=colors, edgecolor='white', linewidth=2)

    ax.set_xlabel('Weight (%)', fontsize=12)
    ax.set_title('Human Handoff Scoring Factors', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 35)

    # Add value labels
    for bar, weight in zip(bars, weights):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{weight}%', va='center', fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'handoff_scoring.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    print("Generating chatbot architecture visualizations...")
    create_chatbot_architecture()
    create_context_management()
    create_handoff_scoring()
    print(f"Visualizations saved to {OUTPUT_DIR}")
