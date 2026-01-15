"""
LLM Serving Architecture Visualization

Creates a comprehensive diagram showing the architecture of a production
LLM serving infrastructure including load balancing, caching, and GPU layers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Output directory
OUTPUT_DIR = "/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/llmops"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_serving_architecture_diagram():
    """Create a comprehensive LLM serving architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # Color scheme
    colors = {
        'client': '#E3F2FD',
        'gateway': '#FFF3E0',
        'cache': '#E8F5E9',
        'serving': '#FCE4EC',
        'infra': '#F3E5F5',
        'arrow': '#546E7A',
        'text': '#263238'
    }

    # Title
    ax.text(8, 11.5, 'LLM Serving Architecture', fontsize=18, fontweight='bold',
            ha='center', va='center', color=colors['text'])

    # Layer 1: Client Layer
    client_box = FancyBboxPatch((0.5, 9.5), 15, 1.5, boxstyle="round,pad=0.05",
                                 facecolor=colors['client'], edgecolor='#1976D2', linewidth=2)
    ax.add_patch(client_box)
    ax.text(8, 10.25, 'Client Layer', fontsize=12, fontweight='bold', ha='center', va='center')

    # Client icons
    for i, (name, x) in enumerate([('Web App', 3), ('Mobile App', 8), ('API Clients', 13)]):
        rect = FancyBboxPatch((x-1, 9.7), 2, 0.9, boxstyle="round,pad=0.02",
                               facecolor='white', edgecolor='#1976D2', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 10.15, name, fontsize=9, ha='center', va='center')

    # Layer 2: API Gateway
    gateway_box = FancyBboxPatch((0.5, 7.2), 15, 1.8, boxstyle="round,pad=0.05",
                                  facecolor=colors['gateway'], edgecolor='#F57C00', linewidth=2)
    ax.add_patch(gateway_box)
    ax.text(8, 8.8, 'API Gateway Layer', fontsize=12, fontweight='bold', ha='center', va='center')

    # Gateway components
    gateway_components = [
        ('Load Balancer', 2.5),
        ('Authentication', 6),
        ('Rate Limiting', 9.5),
        ('Request Router', 13)
    ]
    for name, x in gateway_components:
        rect = FancyBboxPatch((x-1.2, 7.4), 2.4, 0.8, boxstyle="round,pad=0.02",
                               facecolor='white', edgecolor='#F57C00', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 7.8, name, fontsize=8, ha='center', va='center')

    # Layer 3: Caching Layer
    cache_box = FancyBboxPatch((0.5, 5), 15, 1.8, boxstyle="round,pad=0.05",
                                facecolor=colors['cache'], edgecolor='#388E3C', linewidth=2)
    ax.add_patch(cache_box)
    ax.text(8, 6.6, 'Caching Layer', fontsize=12, fontweight='bold', ha='center', va='center')

    # Cache components
    cache_components = [
        ('Semantic Cache\n(Vector DB)', 3),
        ('KV-Cache\n(Shared)', 8),
        ('Response Cache\n(Redis)', 13)
    ]
    for name, x in cache_components:
        rect = FancyBboxPatch((x-1.5, 5.15), 3, 1, boxstyle="round,pad=0.02",
                               facecolor='white', edgecolor='#388E3C', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 5.65, name, fontsize=8, ha='center', va='center')

    # Layer 4: Model Serving Layer
    serving_box = FancyBboxPatch((0.5, 2.5), 15, 2, boxstyle="round,pad=0.05",
                                  facecolor=colors['serving'], edgecolor='#C2185B', linewidth=2)
    ax.add_patch(serving_box)
    ax.text(8, 4.3, 'Model Serving Layer (vLLM / TGI)', fontsize=12, fontweight='bold',
            ha='center', va='center')

    # Serving instances
    for i, x in enumerate([2.5, 6, 9.5, 13]):
        rect = FancyBboxPatch((x-1.2, 2.7), 2.4, 1.2, boxstyle="round,pad=0.02",
                               facecolor='white', edgecolor='#C2185B', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 3.5, f'Instance {i+1}', fontsize=9, fontweight='bold', ha='center', va='center')
        ax.text(x, 3.1, f'4x A100 GPU', fontsize=7, ha='center', va='center', color='#666')

    # Layer 5: Infrastructure Layer
    infra_box = FancyBboxPatch((0.5, 0.3), 15, 1.8, boxstyle="round,pad=0.05",
                                facecolor=colors['infra'], edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(infra_box)
    ax.text(8, 1.9, 'Infrastructure Layer', fontsize=12, fontweight='bold', ha='center', va='center')

    # Infrastructure components
    infra_components = [
        ('GPU Cluster\n(Kubernetes)', 3),
        ('Model Storage\n(S3/GCS)', 8),
        ('Observability\n(Prometheus/Grafana)', 13)
    ]
    for name, x in infra_components:
        rect = FancyBboxPatch((x-1.5, 0.5), 3, 0.9, boxstyle="round,pad=0.02",
                               facecolor='white', edgecolor='#7B1FA2', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 0.95, name, fontsize=8, ha='center', va='center')

    # Draw arrows between layers
    arrow_style = dict(arrowstyle='->', color=colors['arrow'], lw=2, mutation_scale=15)

    # Clients to Gateway
    ax.annotate('', xy=(8, 9), xytext=(8, 9.5), arrowprops=arrow_style)

    # Gateway to Cache
    ax.annotate('', xy=(8, 6.8), xytext=(8, 7.2), arrowprops=arrow_style)

    # Cache to Serving (with cache hit bypass shown)
    ax.annotate('', xy=(8, 4.5), xytext=(8, 5), arrowprops=arrow_style)

    # Cache hit arrow (bypass to response)
    ax.annotate('', xy=(14, 8), xytext=(14, 5.5),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5,
                               connectionstyle='arc3,rad=-0.3', mutation_scale=12))
    ax.text(15.2, 6.8, 'Cache\nHit', fontsize=7, ha='center', va='center', color='#4CAF50')

    # Serving to Infrastructure
    ax.annotate('', xy=(8, 2.1), xytext=(8, 2.5), arrowprops=arrow_style)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['client'], edgecolor='#1976D2', label='Client Layer'),
        mpatches.Patch(facecolor=colors['gateway'], edgecolor='#F57C00', label='API Gateway'),
        mpatches.Patch(facecolor=colors['cache'], edgecolor='#388E3C', label='Caching'),
        mpatches.Patch(facecolor=colors['serving'], edgecolor='#C2185B', label='Model Serving'),
        mpatches.Patch(facecolor=colors['infra'], edgecolor='#7B1FA2', label='Infrastructure'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
              framealpha=0.9, edgecolor='#ccc')

    # Add metrics annotations
    metrics_text = """Key Metrics:
- TTFT: <500ms (P95)
- TPS: 50-200 tokens/s
- Cache Hit Rate: 30-50%
- GPU Utilization: 70-85%"""
    ax.text(0.8, 0.3, metrics_text, fontsize=7, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ccc'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'serving_architecture.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'serving_architecture.svg'),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved serving architecture diagram to {OUTPUT_DIR}")


def create_framework_comparison():
    """Create a comparison chart of LLM serving frameworks."""
    fig, ax = plt.subplots(figsize=(12, 8))

    frameworks = ['vLLM', 'TGI', 'Triton', 'TensorRT-LLM']
    metrics = ['Ease of Use', 'Performance', 'Flexibility', 'GPU Efficiency', 'Community']

    # Scores (1-10 scale)
    scores = {
        'vLLM': [9, 9, 7, 9, 9],
        'TGI': [8, 8, 7, 8, 8],
        'Triton': [5, 9, 9, 8, 7],
        'TensorRT-LLM': [4, 10, 6, 10, 6]
    }

    x = np.arange(len(metrics))
    width = 0.2
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    for i, (framework, color) in enumerate(zip(frameworks, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, scores[framework], width, label=framework,
                     color=color, edgecolor='white', linewidth=1)

    ax.set_xlabel('Evaluation Criteria', fontsize=12)
    ax.set_ylabel('Score (1-10)', fontsize=12)
    ax.set_title('LLM Serving Framework Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 11)
    ax.grid(axis='y', alpha=0.3)

    # Add score labels on bars
    for i, (framework, color) in enumerate(zip(frameworks, colors)):
        offset = (i - 1.5) * width
        for j, score in enumerate(scores[framework]):
            ax.text(j + offset, score + 0.2, str(score), ha='center', va='bottom',
                   fontsize=7, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'framework_comparison.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved framework comparison to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_serving_architecture_diagram()
    create_framework_comparison()
    print("All serving architecture visualizations created successfully!")
