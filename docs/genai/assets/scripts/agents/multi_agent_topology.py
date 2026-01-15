"""
Multi-Agent Communication Topology Diagram Generator

Creates visualizations of different multi-agent communication patterns:
- Hub and Spoke
- Peer-to-Peer (Mesh)
- Hierarchical
- Pipeline

Output: /docs/genai/assets/images/agents/multi_agent_topology.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "agents"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_agent_node(ax, x, y, label, color='#e3f2fd', border='#1976d2',
                    size=0.4, fontsize=9):
    """Draw a single agent node."""
    circle = Circle((x, y), size, facecolor=color, edgecolor=border, linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#333')


def draw_arrow(ax, start, end, color='#666', bidirectional=False,
               style='simple', curve=0):
    """Draw an arrow between two points."""
    if curve != 0:
        connectionstyle = f"arc3,rad={curve}"
    else:
        connectionstyle = "arc3,rad=0"

    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle='-|>' if not bidirectional else '<|-|>',
            color=color,
            lw=1.5,
            connectionstyle=connectionstyle,
            mutation_scale=12
        )
    )


def create_hub_spoke(ax, center_x, center_y):
    """Create hub and spoke topology."""
    ax.set_xlim(center_x - 2.5, center_x + 2.5)
    ax.set_ylim(center_y - 2.5, center_y + 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(center_x, center_y + 2.2, "Hub and Spoke", fontsize=12,
            ha='center', fontweight='bold', color='#1a237e')

    # Hub (orchestrator)
    draw_agent_node(ax, center_x, center_y, "Orch",
                    color='#fff3e0', border='#f57c00', size=0.5)

    # Spoke agents
    n_agents = 5
    radius = 1.5
    for i in range(n_agents):
        angle = 2 * np.pi * i / n_agents - np.pi/2
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)

        draw_agent_node(ax, x, y, f"A{i+1}")

        # Draw bidirectional arrows
        # Offset for arrow endpoints
        dx = (x - center_x) / radius
        dy = (y - center_y) / radius

        draw_arrow(ax,
                   (center_x + 0.5 * dx, center_y + 0.5 * dy),
                   (x - 0.4 * dx, y - 0.4 * dy),
                   bidirectional=True)

    # Description
    ax.text(center_x, center_y - 2.3, "Centralized control\nSingle point of coordination",
            ha='center', fontsize=8, color='#666', style='italic')


def create_peer_to_peer(ax, center_x, center_y):
    """Create peer-to-peer mesh topology."""
    ax.set_xlim(center_x - 2.5, center_x + 2.5)
    ax.set_ylim(center_y - 2.5, center_y + 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(center_x, center_y + 2.2, "Peer-to-Peer (Mesh)", fontsize=12,
            ha='center', fontweight='bold', color='#1a237e')

    # Agents in pentagon formation
    n_agents = 5
    radius = 1.3
    positions = []

    for i in range(n_agents):
        angle = 2 * np.pi * i / n_agents - np.pi/2
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        positions.append((x, y))
        draw_agent_node(ax, x, y, f"A{i+1}", color='#e8f5e9', border='#388e3c')

    # Draw connections (full mesh)
    for i, (x1, y1) in enumerate(positions):
        for j, (x2, y2) in enumerate(positions):
            if i < j:
                # Calculate offset points
                dx = (x2 - x1)
                dy = (y2 - y1)
                dist = np.sqrt(dx**2 + dy**2)
                dx, dy = dx/dist, dy/dist

                draw_arrow(ax,
                           (x1 + 0.45 * dx, y1 + 0.45 * dy),
                           (x2 - 0.45 * dx, y2 - 0.45 * dy),
                           bidirectional=True,
                           color='#81c784')

    # Description
    ax.text(center_x, center_y - 2.3, "Decentralized\nNo single point of failure",
            ha='center', fontsize=8, color='#666', style='italic')


def create_hierarchical(ax, center_x, center_y):
    """Create hierarchical tree topology."""
    ax.set_xlim(center_x - 2.5, center_x + 2.5)
    ax.set_ylim(center_y - 2.5, center_y + 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(center_x, center_y + 2.2, "Hierarchical", fontsize=12,
            ha='center', fontweight='bold', color='#1a237e')

    # Level 0: Manager
    manager_pos = (center_x, center_y + 1.3)
    draw_agent_node(ax, *manager_pos, "Mgr",
                    color='#fce4ec', border='#c2185b', size=0.45)

    # Level 1: Supervisors
    supervisor_positions = [
        (center_x - 1.2, center_y),
        (center_x + 1.2, center_y)
    ]

    for i, pos in enumerate(supervisor_positions):
        draw_agent_node(ax, *pos, f"S{i+1}",
                        color='#e8eaf6', border='#5c6bc0', size=0.4)

        # Arrow from manager
        draw_arrow(ax,
                   (manager_pos[0], manager_pos[1] - 0.5),
                   (pos[0], pos[1] + 0.45))

    # Level 2: Workers
    worker_positions = [
        (center_x - 1.8, center_y - 1.3),
        (center_x - 0.6, center_y - 1.3),
        (center_x + 0.6, center_y - 1.3),
        (center_x + 1.8, center_y - 1.3)
    ]

    for i, pos in enumerate(worker_positions):
        draw_agent_node(ax, *pos, f"W{i+1}", size=0.35, fontsize=8)

        # Arrow from supervisor
        supervisor_idx = 0 if i < 2 else 1
        sup_pos = supervisor_positions[supervisor_idx]
        draw_arrow(ax,
                   (sup_pos[0], sup_pos[1] - 0.45),
                   (pos[0], pos[1] + 0.4))

    # Description
    ax.text(center_x, center_y - 2.3, "Scalable structure\nClear responsibility chain",
            ha='center', fontsize=8, color='#666', style='italic')


def create_pipeline(ax, center_x, center_y):
    """Create pipeline topology."""
    ax.set_xlim(center_x - 2.5, center_x + 2.5)
    ax.set_ylim(center_y - 2.5, center_y + 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(center_x, center_y + 2.2, "Pipeline (Sequential)", fontsize=12,
            ha='center', fontweight='bold', color='#1a237e')

    # Agents in a line
    labels = ["Research", "Analyze", "Write", "Review"]
    colors = [
        ('#e3f2fd', '#1976d2'),
        ('#fff3e0', '#f57c00'),
        ('#e8f5e9', '#388e3c'),
        ('#fce4ec', '#c2185b')
    ]

    positions = []
    start_x = center_x - 1.5
    spacing = 1.0

    for i, (label, (color, border)) in enumerate(zip(labels, colors)):
        x = start_x + i * spacing
        y = center_y
        positions.append((x, y))
        draw_agent_node(ax, x, y, label[:3], color=color, border=border,
                        size=0.35, fontsize=8)

        # Label below
        ax.text(x, y - 0.6, label, ha='center', fontsize=7, color='#666')

    # Draw arrows between stages
    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        draw_arrow(ax, (x1 + 0.4, y1), (x2 - 0.4, y2))

    # Input arrow
    ax.annotate('', xy=(positions[0][0] - 0.4, positions[0][1]),
                xytext=(positions[0][0] - 1.0, positions[0][1]),
                arrowprops=dict(arrowstyle='-|>', color='#666', lw=1.5))
    ax.text(positions[0][0] - 1.2, positions[0][1], "Input",
            ha='right', va='center', fontsize=8, color='#666')

    # Output arrow
    ax.annotate('', xy=(positions[-1][0] + 1.0, positions[-1][1]),
                xytext=(positions[-1][0] + 0.4, positions[-1][1]),
                arrowprops=dict(arrowstyle='-|>', color='#666', lw=1.5))
    ax.text(positions[-1][0] + 1.2, positions[-1][1], "Output",
            ha='left', va='center', fontsize=8, color='#666')

    # Description
    ax.text(center_x, center_y - 2.3, "Sequential processing\nClear data flow",
            ha='center', fontsize=8, color='#666', style='italic')


def create_combined_diagram():
    """Create a combined diagram showing all topologies."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Overall title
    fig.suptitle("Multi-Agent Communication Topologies", fontsize=16,
                 fontweight='bold', y=0.98)

    # Create each topology
    create_hub_spoke(axes[0, 0], 0, 0)
    create_peer_to_peer(axes[0, 1], 0, 0)
    create_hierarchical(axes[1, 0], 0, 0)
    create_pipeline(axes[1, 1], 0, 0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_agent_topology.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_individual_diagrams():
    """Create individual diagrams for each topology."""
    topologies = [
        ('hub_spoke', create_hub_spoke),
        ('peer_to_peer', create_peer_to_peer),
        ('hierarchical', create_hierarchical),
        ('pipeline', create_pipeline)
    ]

    for name, create_func in topologies:
        fig, ax = plt.subplots(figsize=(6, 6))
        create_func(ax, 0, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'topology_{name}.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Created: topology_{name}.png")


if __name__ == "__main__":
    print("Creating multi-agent topology diagrams...")
    create_combined_diagram()
    print(f"Combined diagram saved to: {OUTPUT_DIR}/multi_agent_topology.png")
    create_individual_diagrams()
    print("Done!")
