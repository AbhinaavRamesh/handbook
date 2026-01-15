"""
Code Assistant Architecture Visualization
Generates diagrams for code assistant system design (Copilot-style)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'images', 'system-design')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_copilot_architecture():
    """Create GitHub Copilot-style architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Code Assistant Architecture', fontsize=16, fontweight='bold', pad=20)

    # Color scheme
    colors = {
        'ide': '#e3f2fd',
        'gateway': '#fff3e0',
        'context': '#e8f5e9',
        'model': '#fce4ec',
        'post': '#f3e5f5'
    }

    # IDE Layer
    ide_box = FancyBboxPatch((0.5, 7.5), 3, 2, boxstyle="round,pad=0.1",
                              facecolor=colors['ide'], edgecolor='#1976d2', linewidth=2)
    ax.add_patch(ide_box)
    ax.text(2, 9, 'IDE Integration', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2, 8.3, 'VS Code | JetBrains | Neovim', ha='center', va='center', fontsize=8)
    ax.text(2, 7.9, 'LSP Protocol', ha='center', va='center', fontsize=8)

    # Gateway Layer
    gateway_box = FancyBboxPatch((4.5, 7.5), 4, 2, boxstyle="round,pad=0.1",
                                  facecolor=colors['gateway'], edgecolor='#f57c00', linewidth=2)
    ax.add_patch(gateway_box)
    ax.text(6.5, 9, 'API Gateway', ha='center', va='center', fontsize=11, fontweight='bold')
    gateway_items = ['Auth', 'Rate Limit', 'Router']
    for i, item in enumerate(gateway_items):
        ax.text(5 + i * 1.3, 8.2, item, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#f57c00'))

    # Context Engine
    context_box = FancyBboxPatch((0.5, 4.5), 5, 2.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['context'], edgecolor='#388e3c', linewidth=2)
    ax.add_patch(context_box)
    ax.text(3, 6.5, 'Context Engine', ha='center', va='center', fontsize=11, fontweight='bold')

    context_items = [
        ('File Parser', 1.3, 5.5),
        ('Symbol Table', 3, 5.5),
        ('Repo Index', 4.7, 5.5),
        ('Semantic Index', 3, 4.9)
    ]
    for name, x, y in context_items:
        ax.text(x, y, name, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#388e3c'))

    # Model Layer
    model_box = FancyBboxPatch((6, 4.5), 4, 2.5, boxstyle="round,pad=0.1",
                                facecolor=colors['model'], edgecolor='#c2185b', linewidth=2)
    ax.add_patch(model_box)
    ax.text(8, 6.5, 'Model Layer', ha='center', va='center', fontsize=11, fontweight='bold')

    model_items = [
        ('Model Router', 8, 5.8),
        ('Small (Fast)', 6.8, 5.1),
        ('Large (Quality)', 8, 5.1),
        ('FIM Model', 9.2, 5.1)
    ]
    for name, x, y in model_items:
        ax.text(x, y, name, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#c2185b'))

    # Post-Processing
    post_box = FancyBboxPatch((10.5, 4.5), 3, 2.5, boxstyle="round,pad=0.1",
                               facecolor=colors['post'], edgecolor='#7b1fa2', linewidth=2)
    ax.add_patch(post_box)
    ax.text(12, 6.5, 'Post-Processing', ha='center', va='center', fontsize=11, fontweight='bold')

    post_items = ['Syntax Check', 'Security Scan', 'License Check', 'Formatter']
    for i, item in enumerate(post_items):
        ax.text(12, 6 - i * 0.4, f'• {item}', ha='center', va='center', fontsize=8)

    # Caching Layer
    cache_box = FancyBboxPatch((3, 1.5), 8, 2, boxstyle="round,pad=0.1",
                                facecolor='#fff8e1', edgecolor='#ffa000', linewidth=2)
    ax.add_patch(cache_box)
    ax.text(7, 3, 'Caching Layer', ha='center', va='center', fontsize=11, fontweight='bold')

    cache_items = ['Edge Cache', 'Semantic Cache', 'Result Cache']
    for i, item in enumerate(cache_items):
        ax.text(4.5 + i * 2.3, 2.2, item, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#ffa000'))

    # Arrows
    # IDE to Gateway
    ax.annotate("", xy=(4.5, 8.5), xytext=(3.5, 8.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Gateway to Context
    ax.annotate("", xy=(3, 7), xytext=(5, 7.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Context to Model
    ax.annotate("", xy=(6, 5.75), xytext=(5.5, 5.75),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Model to Post
    ax.annotate("", xy=(10.5, 5.75), xytext=(10, 5.75),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Cache connections
    ax.annotate("", xy=(7, 3.5), xytext=(7, 4.5),
               arrowprops=dict(arrowstyle='<->', color='#ffa000', lw=1.5, ls='--'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'code_assistant_architecture.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_context_priority():
    """Create context priority visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    priorities = ['Cursor Context\n(Immediate)', 'File Context\n(Imports, Class)',
                  'Related Files\n(Tests, Types)', 'Repo Context\n(Patterns)']
    percentages = [40, 25, 20, 15]
    colors = ['#1976d2', '#388e3c', '#f57c00', '#7b1fa2']

    bars = ax.barh(priorities, percentages, color=colors, edgecolor='white', linewidth=2)

    ax.set_xlabel('Token Budget (%)', fontsize=12)
    ax.set_title('Context Window Priority Allocation', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 50)

    # Add value labels
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct}%', va='center', fontsize=11, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'context_priority.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_latency_budget():
    """Create latency budget breakdown."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    components = ['IDE Processing', 'Network', 'Auth/Rate Limit', 'Context Collection',
                  'Model Inference', 'Post-Processing', 'Display']
    latencies = [10, 40, 5, 30, 100, 20, 10]
    colors = ['#e3f2fd', '#fff3e0', '#e8f5e9', '#fce4ec', '#ef5350', '#f3e5f5', '#e0e0e0']

    bars = ax.barh(components, latencies, color=colors, edgecolor='gray', linewidth=1)

    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_title('Code Completion Latency Budget (Target: <200ms)', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{lat}ms', va='center', fontsize=10)

    # Add total line
    ax.axvline(x=sum(latencies), color='red', linestyle='--', linewidth=2, label=f'Total: {sum(latencies)}ms')
    ax.legend()

    ax.set_xlim(0, 250)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_budget.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_model_routing():
    """Create model routing decision diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Task complexity vs model selection
    tasks = ['Simple\nAutocomplete', 'Method\nCompletion', 'Complex\nLogic', 'Code\nExplanation']
    complexity = [1, 2, 4, 5]
    latency_req = [100, 150, 300, 1000]  # ms
    model_size = ['1-3B', '3-7B', '7-13B', '70B+']

    colors = ['#c8e6c9', '#fff9c4', '#ffcc80', '#ef9a9a']

    bars = ax.bar(tasks, complexity, color=colors, edgecolor='gray', linewidth=2)

    ax2 = ax.twinx()
    ax2.plot(tasks, latency_req, 'ro-', markersize=10, linewidth=2, label='Latency Budget (ms)')
    ax2.set_ylabel('Latency Budget (ms)', fontsize=11, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_ylabel('Task Complexity', fontsize=11)
    ax.set_title('Model Routing by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 6)

    # Add model size labels
    for i, (bar, size) in enumerate(zip(bars, model_size)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                size, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_routing.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_completion_metrics():
    """Create completion quality metrics visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    metrics = ['Acceptance\nRate', 'Persistence\nRate', 'Character\nSavings', 'Latency\nP50', 'Latency\nP95']
    targets = [30, 70, 50, 200, 500]
    actuals = [35, 75, 55, 180, 450]
    units = ['%', '%', '%', 'ms', 'ms']

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, targets, width, label='Target', color='#bbdefb', edgecolor='#1976d2')
    bars2 = ax.bar(x + width/2, actuals, width, label='Actual', color='#c8e6c9', edgecolor='#388e3c')

    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Code Completion Quality Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels with units
    for bars, values in [(bars1, targets), (bars2, actuals)]:
        for bar, val, unit in zip(bars, values, units):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val}{unit}', ha='center', va='bottom', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'completion_metrics.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    print("Generating code assistant visualizations...")
    create_copilot_architecture()
    create_context_priority()
    create_latency_budget()
    create_model_routing()
    create_completion_metrics()
    print(f"Visualizations saved to {OUTPUT_DIR}")
