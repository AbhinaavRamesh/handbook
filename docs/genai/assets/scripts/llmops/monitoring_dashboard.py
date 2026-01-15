"""
LLM Monitoring Dashboard Visualization

Creates mockup visualizations of monitoring dashboards, including
metrics panels, alerting status, and observability components.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import os

# Output directory
OUTPUT_DIR = "/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/llmops"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_monitoring_dashboard():
    """Create a comprehensive monitoring dashboard mockup."""
    fig = plt.figure(figsize=(16, 12))

    # Create grid for dashboard panels
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # Dashboard title
    fig.suptitle('LLM Service Monitoring Dashboard', fontsize=18, fontweight='bold', y=0.98)

    # Panel 1: Request Rate (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    hours = np.linspace(0, 24, 100)
    request_rate = 150 + 100 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 10, 100)
    ax1.fill_between(hours, request_rate, alpha=0.3, color='#2196F3')
    ax1.plot(hours, request_rate, color='#1565C0', linewidth=2)
    ax1.set_title('Request Rate', fontsize=11, fontweight='bold')
    ax1.set_ylabel('req/s', fontsize=9)
    ax1.set_xlim(0, 24)
    ax1.set_xlabel('Hour', fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: Latency Percentiles (top-left-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    p50 = 500 + np.random.normal(0, 30, 100)
    p95 = 1500 + np.random.normal(0, 100, 100)
    p99 = 3000 + np.random.normal(0, 200, 100)
    ax2.plot(hours, p50, label='P50', color='#4CAF50', linewidth=1.5)
    ax2.plot(hours, p95, label='P95', color='#FF9800', linewidth=1.5)
    ax2.plot(hours, p99, label='P99', color='#F44336', linewidth=1.5)
    ax2.axhline(y=5000, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title('Latency (ms)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.set_xlim(0, 24)
    ax2.set_xlabel('Hour', fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: TTFT (top-right-middle)
    ax3 = fig.add_subplot(gs[0, 2])
    ttft = 200 + 50 * np.sin(hours * np.pi / 12) + np.random.normal(0, 20, 100)
    ax3.fill_between(hours, ttft, alpha=0.3, color='#9C27B0')
    ax3.plot(hours, ttft, color='#7B1FA2', linewidth=2)
    ax3.axhline(y=500, color='red', linestyle='--', linewidth=1, label='SLO')
    ax3.set_title('Time to First Token (ms)', fontsize=11, fontweight='bold')
    ax3.set_xlim(0, 24)
    ax3.set_xlabel('Hour', fontsize=8)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # Panel 4: Error Rate (top-right)
    ax4 = fig.add_subplot(gs[0, 3])
    error_rate = 0.5 + np.abs(np.random.normal(0, 0.3, 100))
    error_rate[40:45] = 2.5  # Spike
    ax4.fill_between(hours, error_rate, alpha=0.3, color='#F44336')
    ax4.plot(hours, error_rate, color='#C62828', linewidth=2)
    ax4.axhline(y=1, color='orange', linestyle='--', linewidth=1, label='Warning')
    ax4.axhline(y=5, color='red', linestyle='--', linewidth=1, label='Critical')
    ax4.set_title('Error Rate (%)', fontsize=11, fontweight='bold')
    ax4.set_xlim(0, 24)
    ax4.set_xlabel('Hour', fontsize=8)
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    # Panel 5: GPU Utilization (row 2, left half)
    ax5 = fig.add_subplot(gs[1, :2])
    gpus = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3', 'GPU 4', 'GPU 5', 'GPU 6', 'GPU 7']
    utilization = [78, 82, 75, 80, 77, 85, 72, 79]
    colors = ['#4CAF50' if u < 85 else '#FF9800' for u in utilization]

    bars = ax5.barh(gpus, utilization, color=colors, edgecolor='white')
    ax5.axvline(x=85, color='orange', linestyle='--', linewidth=1.5, label='Warning')
    ax5.axvline(x=95, color='red', linestyle='--', linewidth=1.5, label='Critical')
    ax5.set_xlabel('Utilization (%)', fontsize=9)
    ax5.set_title('GPU Utilization by Device', fontsize=11, fontweight='bold')
    ax5.set_xlim(0, 100)
    ax5.legend(fontsize=7, loc='lower right')

    for bar, util in zip(bars, utilization):
        ax5.text(util + 1, bar.get_y() + bar.get_height()/2, f'{util}%',
                va='center', fontsize=8)

    # Panel 6: Cache Hit Rate (row 2, right half)
    ax6 = fig.add_subplot(gs[1, 2:])
    cache_types = ['Exact Match', 'Semantic', 'KV-Cache', 'Miss']
    hit_rates = [15, 30, 25, 30]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9E9E9E']

    wedges, texts, autotexts = ax6.pie(hit_rates, labels=cache_types, colors=colors,
                                        autopct='%1.0f%%', startangle=90,
                                        explode=(0, 0.05, 0, 0))
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    ax6.set_title('Cache Distribution', fontsize=11, fontweight='bold')

    # Panel 7: Token Throughput (row 3, left)
    ax7 = fig.add_subplot(gs[2, :2])
    input_tokens = 50000 + 20000 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 3000, 100)
    output_tokens = 30000 + 15000 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 2000, 100)

    ax7.fill_between(hours, input_tokens, alpha=0.3, color='#2196F3')
    ax7.fill_between(hours, output_tokens, alpha=0.3, color='#FF9800')
    ax7.plot(hours, input_tokens, color='#1565C0', linewidth=1.5, label='Input')
    ax7.plot(hours, output_tokens, color='#E65100', linewidth=1.5, label='Output')
    ax7.set_title('Token Throughput (tokens/min)', fontsize=11, fontweight='bold')
    ax7.set_xlabel('Hour', fontsize=8)
    ax7.set_xlim(0, 24)
    ax7.legend(fontsize=8)
    ax7.grid(alpha=0.3)

    # Panel 8: Cost Tracking (row 3, right)
    ax8 = fig.add_subplot(gs[2, 2:])
    cumulative_cost = np.cumsum(request_rate * 0.01)  # Simulated cost
    budget_line = np.linspace(0, 1000, 100)

    ax8.fill_between(hours, cumulative_cost, alpha=0.3, color='#4CAF50')
    ax8.plot(hours, cumulative_cost, color='#388E3C', linewidth=2, label='Actual')
    ax8.plot(hours, budget_line, '--', color='#666', linewidth=1.5, label='Budget')
    ax8.set_title('Daily Cost ($)', fontsize=11, fontweight='bold')
    ax8.set_xlabel('Hour', fontsize=8)
    ax8.set_xlim(0, 24)
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)

    # Panel 9: Status Indicators (bottom row)
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')

    # Create status boxes
    statuses = [
        ('API Gateway', 'Healthy', '#4CAF50'),
        ('Model Server', 'Healthy', '#4CAF50'),
        ('Cache Layer', 'Healthy', '#4CAF50'),
        ('GPU Cluster', 'Warning', '#FF9800'),
        ('Monitoring', 'Healthy', '#4CAF50'),
        ('Cost Budget', 'Healthy', '#4CAF50'),
    ]

    for i, (name, status, color) in enumerate(statuses):
        x = 0.08 + (i % 6) * 0.15
        y = 0.3

        # Status box
        box = FancyBboxPatch((x, y), 0.12, 0.5, transform=ax9.transAxes,
                              boxstyle="round,pad=0.02", facecolor='white',
                              edgecolor=color, linewidth=3)
        ax9.add_patch(box)

        # Status indicator circle
        circle = plt.Circle((x + 0.06, y + 0.4), 0.02, transform=ax9.transAxes,
                            facecolor=color, edgecolor='white')
        ax9.add_patch(circle)

        # Labels
        ax9.text(x + 0.06, y + 0.25, name, transform=ax9.transAxes,
                ha='center', fontsize=9, fontweight='bold')
        ax9.text(x + 0.06, y + 0.1, status, transform=ax9.transAxes,
                ha='center', fontsize=8, color=color)

    ax9.text(0.5, 0.9, 'System Status', transform=ax9.transAxes,
            ha='center', fontsize=12, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, 'monitoring_dashboard.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved monitoring dashboard to {OUTPUT_DIR}")


def create_alert_overview():
    """Create an alert status overview visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Alert severity distribution
    ax1 = axes[0]
    severities = ['Critical', 'Warning', 'Info']
    counts = [2, 5, 12]
    colors = ['#F44336', '#FF9800', '#2196F3']

    bars = ax1.barh(severities, counts, color=colors, edgecolor='white', height=0.6)
    ax1.set_xlabel('Active Alerts', fontsize=11)
    ax1.set_title('Alert Severity Distribution', fontsize=13, fontweight='bold')

    for bar, count in zip(bars, counts):
        ax1.text(count + 0.3, bar.get_y() + bar.get_height()/2, str(count),
                va='center', fontsize=12, fontweight='bold')

    ax1.set_xlim(0, 15)

    # Right: Alert timeline
    ax2 = axes[1]
    np.random.seed(42)
    hours = np.arange(24)

    # Simulate alert counts per hour
    critical = np.random.poisson(0.5, 24)
    warning = np.random.poisson(1, 24)
    info = np.random.poisson(2, 24)

    ax2.bar(hours, critical, label='Critical', color='#F44336', alpha=0.8)
    ax2.bar(hours, warning, bottom=critical, label='Warning', color='#FF9800', alpha=0.8)
    ax2.bar(hours, info, bottom=critical+warning, label='Info', color='#2196F3', alpha=0.8)

    ax2.set_xlabel('Hour', fontsize=11)
    ax2.set_ylabel('Alert Count', fontsize=11)
    ax2.set_title('Alerts Over 24 Hours', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(-0.5, 23.5)

    plt.suptitle('Alert Management Overview', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'alert_overview.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved alert overview to {OUTPUT_DIR}")


def create_metrics_hierarchy():
    """Create a visualization of the metrics hierarchy."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'LLM Observability Metrics Hierarchy', fontsize=16, fontweight='bold',
            ha='center', va='center')

    # Main categories
    categories = [
        ('Performance', 2, 7, '#E3F2FD', [
            'TTFT (ms)', 'TPS', 'E2E Latency', 'Queue Wait'
        ]),
        ('Resource', 6, 7, '#FFF3E0', [
            'GPU Util %', 'GPU Memory', 'CPU Usage', 'Network I/O'
        ]),
        ('Business', 10, 7, '#E8F5E9', [
            'Cost/Request', 'Token Usage', 'Cache Hit Rate', 'Error Rate'
        ]),
        ('Quality', 6, 3, '#FCE4EC', [
            'User Feedback', 'Regeneration Rate', 'Hallucination', 'Relevance'
        ]),
    ]

    for cat_name, x, y, color, metrics in categories:
        # Category box
        cat_box = FancyBboxPatch((x-1.5, y-1), 3, 1.8, boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='#666', linewidth=2)
        ax.add_patch(cat_box)
        ax.text(x, y+0.5, cat_name, fontsize=12, fontweight='bold', ha='center')

        # Metrics
        for i, metric in enumerate(metrics):
            metric_y = y - 0.1 - i * 0.3
            ax.text(x, metric_y, f'- {metric}', fontsize=8, ha='center', color='#444')

    # Arrows showing relationships
    arrow_style = dict(arrowstyle='->', color='#666', lw=1.5, mutation_scale=12,
                      connectionstyle='arc3,rad=0.1')

    # Performance affects Business
    ax.annotate('', xy=(10-1.5, 7), xytext=(2+1.5, 7), arrowprops=arrow_style)
    ax.text(6, 7.3, 'impacts', fontsize=8, ha='center', color='#666', style='italic')

    # Resource affects Performance
    ax.annotate('', xy=(2+1.5, 6.8), xytext=(6-1.5, 6.8),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5,
                               mutation_scale=12, connectionstyle='arc3,rad=-0.1'))

    # Business connects to Quality
    ax.annotate('', xy=(6+1.5, 4), xytext=(10-1.5, 6),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5,
                               mutation_scale=12, connectionstyle='arc3,rad=0.3'))

    # SLI/SLO box
    slo_box = FancyBboxPatch((1, 1), 12, 1.5, boxstyle="round,pad=0.05",
                              facecolor='#E1BEE7', edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(slo_box)
    ax.text(7, 2.2, 'Service Level Objectives (SLOs)', fontsize=12, fontweight='bold',
            ha='center')

    slos = [
        'TTFT P95 < 500ms',
        'E2E P99 < 10s',
        'Error Rate < 0.1%',
        'Availability > 99.9%',
    ]
    for i, slo in enumerate(slos):
        ax.text(2 + i * 3, 1.5, slo, fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_hierarchy.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved metrics hierarchy to {OUTPUT_DIR}")


def create_tracing_visualization():
    """Create a visualization of distributed tracing spans."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1800)
    ax.set_ylim(-0.5, 8.5)

    # Spans data: (name, start, duration, level, color)
    spans = [
        ('llm_request', 0, 1700, 0, '#E3F2FD'),
        ('auth_middleware', 10, 20, 1, '#FFECB3'),
        ('rate_limiter', 35, 10, 1, '#FFECB3'),
        ('cache_lookup', 50, 50, 1, '#C8E6C9'),
        ('semantic_search', 55, 40, 2, '#A5D6A7'),
        ('preprocessing', 110, 30, 1, '#E1BEE7'),
        ('tokenization', 115, 25, 2, '#CE93D8'),
        ('inference', 150, 1450, 1, '#FFCDD2'),
        ('prefill', 160, 200, 2, '#EF9A9A'),
        ('kv_cache_alloc', 165, 30, 3, '#FFAB91'),
        ('attention_compute', 200, 150, 3, '#FFAB91'),
        ('decode', 370, 1200, 2, '#EF9A9A'),
        ('token_gen_1', 380, 15, 3, '#FFAB91'),
        ('token_gen_2', 400, 15, 3, '#FFAB91'),
        ('token_gen_N', 1550, 15, 3, '#FFAB91'),
        ('postprocessing', 1600, 40, 1, '#E1BEE7'),
        ('detokenization', 1605, 30, 2, '#CE93D8'),
        ('response_streaming', 1650, 45, 1, '#B2DFDB'),
    ]

    for name, start, duration, level, color in spans:
        y = 7.5 - level
        height = 0.6

        # Draw span bar
        rect = FancyBboxPatch((start, y), duration, height, boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='#666', linewidth=1)
        ax.add_patch(rect)

        # Label (inside if fits, else above)
        if duration > 80:
            ax.text(start + duration/2, y + height/2, f'{name}\n{duration}ms',
                   ha='center', va='center', fontsize=7)
        else:
            ax.text(start + duration/2, y + height + 0.1, name,
                   ha='center', va='bottom', fontsize=6, rotation=45)

    # Time axis
    ax.set_xlabel('Time (milliseconds)', fontsize=12)
    ax.set_title('Distributed Trace: LLM Request Lifecycle', fontsize=14, fontweight='bold')

    # Y-axis labels
    ax.set_yticks([7.5, 6.5, 5.5, 4.5])
    ax.set_yticklabels(['Root', 'Service', 'Operation', 'Detail'], fontsize=10)

    # Grid and markers
    ax.axvline(x=360, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(360, 8.3, 'TTFT: 360ms', fontsize=9, ha='center', color='red')

    ax.axvline(x=1700, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(1700, 8.3, 'Total: 1700ms', fontsize=9, ha='center', color='blue')

    ax.grid(axis='x', alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#E3F2FD', label='Request'),
        mpatches.Patch(color='#FFECB3', label='Gateway'),
        mpatches.Patch(color='#C8E6C9', label='Cache'),
        mpatches.Patch(color='#FFCDD2', label='Inference'),
        mpatches.Patch(color='#E1BEE7', label='Processing'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tracing_visualization.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved tracing visualization to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_monitoring_dashboard()
    create_alert_overview()
    create_metrics_hierarchy()
    create_tracing_visualization()
    print("All monitoring visualization charts created successfully!")
