"""
LLM Scaling Patterns Visualization

Creates visualizations for horizontal scaling, auto-scaling behavior,
load balancing, and multi-region architectures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Output directory
OUTPUT_DIR = "/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/genai/assets/images/llmops"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_scaling_architecture_diagram():
    """Create a diagram showing horizontal scaling architecture."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Horizontal Scaling Architecture', fontsize=16, fontweight='bold',
            ha='center', va='center')

    # Global Load Balancer
    lb_box = FancyBboxPatch((5, 7.5), 4, 1, boxstyle="round,pad=0.05",
                             facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(lb_box)
    ax.text(7, 8, 'Global Load Balancer', fontsize=11, fontweight='bold',
            ha='center', va='center')

    # Auto-Scaler
    scaler_box = FancyBboxPatch((10, 4), 3, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(scaler_box)
    ax.text(11.5, 5.7, 'Auto-Scaler', fontsize=10, fontweight='bold', ha='center')
    ax.text(11.5, 5.2, '(KEDA/HPA)', fontsize=8, ha='center', color='#666')
    ax.text(11.5, 4.6, 'Queue Depth\nGPU Util\nLatency P95', fontsize=7,
            ha='center', va='center', color='#666')

    # LLM Pods
    pod_positions = [(2, 5), (4.5, 5), (7, 5)]
    for i, (x, y) in enumerate(pod_positions):
        # Pod box
        pod_box = FancyBboxPatch((x, y-0.8), 2, 1.6, boxstyle="round,pad=0.05",
                                  facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
        ax.add_patch(pod_box)
        ax.text(x+1, y+0.4, f'LLM Pod {i+1}', fontsize=9, fontweight='bold', ha='center')
        ax.text(x+1, y-0.1, '4x A100 GPU', fontsize=7, ha='center', color='#666')
        ax.text(x+1, y-0.4, 'vLLM Server', fontsize=7, ha='center', color='#666')

    # Scaling indicator (dashed pod)
    ax.add_patch(FancyBboxPatch((9.2, 5-0.8), 2, 1.6, boxstyle="round,pad=0.05",
                                 facecolor='white', edgecolor='#388E3C',
                                 linewidth=2, linestyle='--'))
    ax.text(10.2, 5+0.2, 'Pod N', fontsize=9, fontweight='bold', ha='center', color='#666')
    ax.text(10.2, 5-0.2, '(scale up)', fontsize=7, ha='center', color='#666', style='italic')

    # GPU Node Pool
    gpu_box = FancyBboxPatch((1.5, 2), 8, 1.5, boxstyle="round,pad=0.05",
                              facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2)
    ax.add_patch(gpu_box)
    ax.text(5.5, 2.75, 'GPU Node Pool (Kubernetes)', fontsize=11, fontweight='bold',
            ha='center', va='center')
    ax.text(5.5, 2.3, 'A100 80GB x 32 | Auto-provisioning enabled', fontsize=9,
            ha='center', va='center', color='#666')

    # Shared Services
    services = [
        ('Redis\nCache', 1, 0.5, '#E1BEE7'),
        ('Model\nStorage', 4, 0.5, '#B2DFDB'),
        ('Prometheus\nMetrics', 7, 0.5, '#FFECB3'),
        ('Vector\nDB', 10, 0.5, '#D1C4E9'),
    ]

    for name, x, y, color in services:
        box = FancyBboxPatch((x, y), 2, 1, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='#666', linewidth=1)
        ax.add_patch(box)
        ax.text(x+1, y+0.5, name, fontsize=8, ha='center', va='center')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#546E7A', lw=2, mutation_scale=15)

    # LB to pods
    for x, y in pod_positions:
        ax.annotate('', xy=(x+1, y+0.8), xytext=(7, 7.5),
                    arrowprops=dict(arrowstyle='->', color='#546E7A', lw=1.5,
                                   connectionstyle='arc3,rad=0', mutation_scale=12))

    # Auto-scaler arrows
    ax.annotate('', xy=(10, 5.25), xytext=(9.2, 5.25),
                arrowprops=dict(arrowstyle='->', color='#F57C00', lw=2,
                               mutation_scale=15))
    ax.annotate('', xy=(11.5, 4), xytext=(11.5, 3.5),
                arrowprops=dict(arrowstyle='<-', color='#F57C00', lw=1.5,
                               mutation_scale=12))
    ax.text(11.5, 3.3, 'metrics', fontsize=7, ha='center', color='#F57C00')

    # Pods to GPU pool
    for x, y in pod_positions:
        ax.annotate('', xy=(x+1, 3.5), xytext=(x+1, y-0.8),
                    arrowprops=dict(arrowstyle='->', color='#C2185B', lw=1.5,
                                   mutation_scale=12))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1976D2', label='Load Balancer'),
        mpatches.Patch(facecolor='#E8F5E9', edgecolor='#388E3C', label='LLM Pods'),
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='#F57C00', label='Auto-Scaler'),
        mpatches.Patch(facecolor='#FCE4EC', edgecolor='#C2185B', label='GPU Pool'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='#ccc')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scaling_architecture.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved scaling architecture diagram to {OUTPUT_DIR}")


def create_autoscaling_behavior():
    """Create a chart showing auto-scaling behavior over time."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Generate time series data
    np.random.seed(42)
    hours = np.linspace(0, 24, 288)  # 5-minute intervals

    # Traffic pattern (daily cycle with some spikes)
    base_traffic = 100 + 80 * np.sin((hours - 6) * np.pi / 12)
    spikes = np.zeros_like(hours)
    spikes[70:80] = 150  # Morning spike
    spikes[140:150] = 100  # Afternoon spike
    traffic = base_traffic + spikes + np.random.normal(0, 10, len(hours))
    traffic = np.clip(traffic, 20, 300)

    # Queue depth (follows traffic with lag)
    queue_capacity_per_pod = 30
    replicas = np.ones_like(hours) * 3  # Start with 3 replicas

    # Simulate scaling decisions
    queue_depth = np.zeros_like(hours)
    for i in range(1, len(hours)):
        incoming = traffic[i] / 60 * 5  # Requests per 5 minutes
        capacity = replicas[i-1] * queue_capacity_per_pod

        # Queue builds up if incoming > capacity
        queue_depth[i] = max(0, queue_depth[i-1] + incoming - capacity)

        # Scale up if queue > 50
        if queue_depth[i] > 50 and replicas[i-1] < 10:
            replicas[i] = min(10, replicas[i-1] + 2)
        # Scale down if queue < 10 for a while
        elif queue_depth[i] < 10 and i > 20 and np.mean(queue_depth[i-12:i]) < 15:
            replicas[i] = max(2, replicas[i-1] - 1)
        else:
            replicas[i] = replicas[i-1]

    # Top: Traffic
    ax1 = axes[0]
    ax1.fill_between(hours, traffic, alpha=0.3, color='#2196F3')
    ax1.plot(hours, traffic, color='#1565C0', linewidth=1.5)
    ax1.set_ylabel('Requests/min', fontsize=11)
    ax1.set_title('Traffic Pattern', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 350)
    ax1.grid(alpha=0.3)

    # Middle: Queue Depth
    ax2 = axes[1]
    ax2.fill_between(hours, queue_depth, alpha=0.3, color='#FF9800')
    ax2.plot(hours, queue_depth, color='#E65100', linewidth=1.5)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Scale-up threshold')
    ax2.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Scale-down threshold')
    ax2.set_ylabel('Queue Depth', fontsize=11)
    ax2.set_title('Queue Depth', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 150)
    ax2.grid(alpha=0.3)

    # Bottom: Replica Count
    ax3 = axes[2]
    ax3.step(hours, replicas, where='post', color='#4CAF50', linewidth=2)
    ax3.fill_between(hours, replicas, step='post', alpha=0.3, color='#4CAF50')
    ax3.axhline(y=2, color='#666', linestyle=':', linewidth=1, label='Min replicas')
    ax3.axhline(y=10, color='#666', linestyle=':', linewidth=1, label='Max replicas')
    ax3.set_xlabel('Hour of Day', fontsize=11)
    ax3.set_ylabel('Pod Replicas', fontsize=11)
    ax3.set_title('Auto-Scaling Response', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 12)
    ax3.set_xlim(0, 24)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)

    # Mark scaling events
    scale_ups = np.where(np.diff(replicas) > 0)[0]
    scale_downs = np.where(np.diff(replicas) < 0)[0]

    for idx in scale_ups[:5]:  # Mark first 5 scale-ups
        ax3.annotate('', xy=(hours[idx+1], replicas[idx+1]),
                    xytext=(hours[idx+1], replicas[idx+1]-0.5),
                    arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1.5))

    plt.suptitle('Auto-Scaling Behavior Over 24 Hours', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'autoscaling_behavior.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved autoscaling behavior chart to {OUTPUT_DIR}")


def create_load_balancing_strategies():
    """Create a comparison of load balancing strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    np.random.seed(42)

    strategies = [
        ('Round Robin', 0, 0),
        ('Least Connections', 0, 1),
        ('Weighted', 1, 0),
        ('LLM-Aware', 1, 1),
    ]

    for strategy, row, col in strategies:
        ax = axes[row, col]

        # Simulate load distribution across 4 servers
        servers = ['Server 1', 'Server 2', 'Server 3', 'Server 4']

        if strategy == 'Round Robin':
            # Even distribution
            loads = [25, 25, 25, 25]
            latencies = [150, 180, 200, 220]  # Varies due to different request lengths
            colors = ['#90CAF9'] * 4

        elif strategy == 'Least Connections':
            # Tries to balance but doesn't account for request duration
            loads = [22, 26, 24, 28]
            latencies = [130, 160, 150, 180]
            colors = ['#81C784'] * 4

        elif strategy == 'Weighted':
            # Accounts for server capacity
            loads = [35, 25, 25, 15]  # Server 1 has more capacity
            latencies = [120, 140, 145, 160]
            colors = ['#FFB74D'] * 4

        else:  # LLM-Aware
            # Accounts for queue depth, GPU util, etc.
            loads = [24, 26, 25, 25]
            latencies = [110, 115, 112, 118]  # Most balanced latencies
            colors = ['#CE93D8'] * 4

        # Plot bars
        x = np.arange(len(servers))
        bars = ax.bar(x, loads, color=colors, edgecolor='white', linewidth=2)

        # Add latency annotations
        for i, (bar, lat) in enumerate(zip(bars, latencies)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{lat}ms', ha='center', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(servers, fontsize=9)
        ax.set_ylabel('Load (%)', fontsize=10)
        ax.set_title(strategy, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 45)

        # Add average latency
        avg_lat = np.mean(latencies)
        ax.text(0.98, 0.95, f'Avg Latency: {avg_lat:.0f}ms',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Load Balancing Strategy Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'load_balancing_strategies.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved load balancing strategies chart to {OUTPUT_DIR}")


def create_multi_region_diagram():
    """Create a multi-region architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Multi-Region LLM Deployment', fontsize=16, fontweight='bold',
            ha='center', va='center')

    # Global DNS / CDN
    global_box = FancyBboxPatch((5, 8), 4, 1, boxstyle="round,pad=0.05",
                                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(global_box)
    ax.text(7, 8.5, 'Global DNS + CDN', fontsize=11, fontweight='bold', ha='center')

    # Regions
    regions = [
        ('US-East', 2, 5, '#E8F5E9', '35%'),
        ('EU-West', 7, 5, '#FFF3E0', '40%'),
        ('Asia-Pacific', 12, 5, '#FCE4EC', '25%'),
    ]

    for name, x, y, color, traffic in regions:
        # Region box
        region_box = FancyBboxPatch((x-1.5, y-2.5), 3, 4, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='#666', linewidth=2)
        ax.add_patch(region_box)

        # Region name
        ax.text(x, y+1.2, name, fontsize=11, fontweight='bold', ha='center')
        ax.text(x, y+0.7, f'Traffic: {traffic}', fontsize=8, ha='center', color='#666')

        # Components inside region
        components = [
            ('LB', x, y),
            ('LLM\nCluster', x, y-0.8),
            ('Cache', x-0.7, y-1.8),
            ('Storage', x+0.7, y-1.8),
        ]

        for comp_name, cx, cy in components:
            comp_box = FancyBboxPatch((cx-0.5, cy-0.3), 1, 0.6, boxstyle="round,pad=0.02",
                                       facecolor='white', edgecolor='#999', linewidth=1)
            ax.add_patch(comp_box)
            ax.text(cx, cy, comp_name, fontsize=7, ha='center', va='center')

    # Arrows from global to regions
    for name, x, y, color, traffic in regions:
        ax.annotate('', xy=(x, y+1.5), xytext=(7, 8),
                    arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2,
                                   connectionstyle='arc3,rad=0', mutation_scale=15))

    # Cross-region replication arrows
    ax.annotate('', xy=(5.5, 4.5), xytext=(3.5, 4.5),
                arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=1.5,
                               mutation_scale=10))
    ax.text(4.5, 4.2, 'replication', fontsize=7, ha='center', color='#4CAF50')

    ax.annotate('', xy=(10.5, 4.5), xytext=(8.5, 4.5),
                arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=1.5,
                               mutation_scale=10))
    ax.text(9.5, 4.2, 'replication', fontsize=7, ha='center', color='#4CAF50')

    # User locations
    users = [
        ('Users', 2, 1, '#2196F3'),
        ('Users', 7, 1, '#FF9800'),
        ('Users', 12, 1, '#E91E63'),
    ]

    for label, x, y, color in users:
        circle = Circle((x, y), 0.3, facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, 'U', fontsize=8, ha='center', va='center', color='white', fontweight='bold')
        ax.annotate('', xy=(x, y+1.2), xytext=(x, y+0.3),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, mutation_scale=10))

    # Latency annotations
    latencies = [
        ('~20ms', 2, 1.8),
        ('~30ms', 7, 1.8),
        ('~25ms', 12, 1.8),
    ]

    for text, x, y in latencies:
        ax.text(x, y, text, fontsize=8, ha='center', color='#666', style='italic')

    # Info box
    info_text = """Benefits:
- Low latency (users route to nearest region)
- High availability (failover between regions)
- Compliance (data residency requirements)
- Scale (distribute global load)"""

    ax.text(0.5, 0.5, info_text, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#ccc'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_region_diagram.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved multi-region diagram to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_scaling_architecture_diagram()
    create_autoscaling_behavior()
    create_load_balancing_strategies()
    create_multi_region_diagram()
    print("All scaling visualization charts created successfully!")
