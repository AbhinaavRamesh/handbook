#!/usr/bin/env python3
"""
Graph Visualization Generator
Generates PNG images for graph algorithm documentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

# Set the directory for output
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_fig(fig, name):
    """Save figure to the assets directory"""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {path}")

# =============================================================================
# 1. Clone Graph Visualization
# =============================================================================
def create_clone_graph_viz():
    """Visualize the clone graph process"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original graph
    ax1 = axes[0]
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 4), (2, 3), (3, 4)])
    pos = {1: (0, 1), 2: (1, 1), 3: (1, 0), 4: (0, 0)}

    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='#2196F3', node_size=800, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=14, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G, pos, ax=ax1, width=2, edge_color='gray')
    ax1.set_title('Original Graph', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Mapping visualization
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 4.5)

    ax2.text(1, 4.2, 'Hash Map: old -> new', ha='center', fontsize=12, fontweight='bold')

    for i, (old, new) in enumerate([(1, "1'"), (2, "2'"), (3, "3'"), (4, "4'")]):
        y = 3.5 - i * 0.8
        # Old node
        circle1 = patches.Circle((0.5, y), 0.25, facecolor='#2196F3', edgecolor='black', linewidth=2)
        ax2.add_patch(circle1)
        ax2.text(0.5, y, str(old), ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        # Arrow
        ax2.annotate('', xy=(1.5, y), xytext=(0.85, y),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))

        # New node
        circle2 = patches.Circle((1.8, y), 0.25, facecolor='#4CAF50', edgecolor='black', linewidth=2)
        ax2.add_patch(circle2)
        ax2.text(1.8, y, new, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    ax2.set_title('Node Mapping', fontsize=14, fontweight='bold')

    # Cloned graph
    ax3 = axes[2]
    G_clone = nx.Graph()
    G_clone.add_edges_from([(1, 2), (1, 4), (2, 3), (3, 4)])

    nx.draw_networkx_nodes(G_clone, pos, ax=ax3, node_color='#4CAF50', node_size=800, edgecolors='black', linewidths=2)
    labels_clone = {1: "1'", 2: "2'", 3: "3'", 4: "4'"}
    nx.draw_networkx_labels(G_clone, pos, labels=labels_clone, ax=ax3, font_size=12, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G_clone, pos, ax=ax3, width=2, edge_color='gray')
    ax3.set_title('Cloned Graph (Deep Copy)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    fig.suptitle('Clone Graph: Creating a Deep Copy', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'clone_graph.png')

# =============================================================================
# 2. Pacific Atlantic Water Flow Visualization
# =============================================================================
def create_pacific_atlantic_viz():
    """Visualize Pacific Atlantic water flow"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Grid with heights
    heights = np.array([
        [1, 2, 2, 3, 5],
        [3, 2, 3, 4, 4],
        [2, 4, 5, 3, 1],
        [6, 7, 1, 4, 5],
        [5, 1, 1, 2, 4]
    ])

    # Left: Show the grid with heights
    ax1 = axes[0]
    cmap = plt.cm.Blues
    im = ax1.imshow(heights, cmap=cmap, vmin=0, vmax=8)

    for i in range(5):
        for j in range(5):
            ax1.text(j, i, str(heights[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black')

    # Mark Pacific (top and left)
    ax1.axhline(y=-0.5, color='#2196F3', linewidth=8, label='Pacific')
    ax1.axvline(x=-0.5, color='#2196F3', linewidth=8)

    # Mark Atlantic (bottom and right)
    ax1.axhline(y=4.5, color='#FF5722', linewidth=8, label='Atlantic')
    ax1.axvline(x=4.5, color='#FF5722', linewidth=8)

    ax1.text(2, -0.9, 'Pacific Ocean', ha='center', fontsize=11, color='#2196F3', fontweight='bold')
    ax1.text(2, 5.3, 'Atlantic Ocean', ha='center', fontsize=11, color='#FF5722', fontweight='bold')

    ax1.set_title('Height Grid', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Right: Show cells that can reach both oceans
    ax2 = axes[1]

    # Cells that can reach both (from the problem solution)
    both_reach = [(0, 4), (1, 3), (1, 4), (2, 2), (3, 0), (3, 1), (4, 0)]

    colors = np.ones((5, 5, 3))  # White background
    for i, j in both_reach:
        colors[i, j] = [0.5, 0.8, 0.5]  # Green for cells reaching both

    ax2.imshow(colors)

    for i in range(5):
        for j in range(5):
            ax2.text(j, i, str(heights[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black')
            if (i, j) in both_reach:
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=3,
                                         edgecolor='green', facecolor='none')
                ax2.add_patch(rect)

    ax2.axhline(y=-0.5, color='#2196F3', linewidth=8)
    ax2.axvline(x=-0.5, color='#2196F3', linewidth=8)
    ax2.axhline(y=4.5, color='#FF5722', linewidth=8)
    ax2.axvline(x=4.5, color='#FF5722', linewidth=8)

    ax2.set_title('Cells Reaching Both Oceans (Green)', fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle('Pacific Atlantic Water Flow', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'pacific_atlantic.png')

# =============================================================================
# 3. Graph Valid Tree Visualization
# =============================================================================
def create_graph_valid_tree_viz():
    """Visualize valid tree vs invalid tree"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Valid tree
    ax1 = axes[0]
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4)])
    pos1 = nx.spring_layout(G1, seed=42)

    nx.draw_networkx_nodes(G1, pos1, ax=ax1, node_color='#4CAF50', node_size=700, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G1, pos1, ax=ax1, font_size=12, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G1, pos1, ax=ax1, width=2, edge_color='gray')
    ax1.set_title('Valid Tree\nn=5, edges=4\nConnected, No Cycles', fontsize=12, fontweight='bold', color='green')
    ax1.axis('off')

    # Invalid: Has cycle
    ax2 = axes[1]
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)])
    pos2 = nx.spring_layout(G2, seed=42)

    nx.draw_networkx_nodes(G2, pos2, ax=ax2, node_color='#F44336', node_size=700, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G2, pos2, ax=ax2, font_size=12, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G2, pos2, ax=ax2, width=2, edge_color='gray')

    # Highlight the cycle edges
    cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    nx.draw_networkx_edges(G2, pos2, edgelist=cycle_edges, ax=ax2, width=3, edge_color='red', style='dashed')

    ax2.set_title('Invalid: Has Cycle\nn=5, edges=5\n(edges > n-1)', fontsize=12, fontweight='bold', color='red')
    ax2.axis('off')

    # Invalid: Disconnected
    ax3 = axes[2]
    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (2, 3)])
    G3.add_node(4)
    pos3 = {0: (0, 1), 1: (1, 1), 2: (2, 0), 3: (3, 0), 4: (1.5, -1)}

    nx.draw_networkx_nodes(G3, pos3, ax=ax3, node_color='#F44336', node_size=700, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G3, pos3, ax=ax3, font_size=12, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G3, pos3, ax=ax3, width=2, edge_color='gray')

    ax3.set_title('Invalid: Disconnected\nn=5, edges=2\n(edges < n-1)', fontsize=12, fontweight='bold', color='red')
    ax3.axis('off')

    fig.suptitle('Graph Valid Tree: A valid tree has n-1 edges and is connected', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'graph_valid_tree.png')

# =============================================================================
# 4. Connected Components Visualization
# =============================================================================
def create_connected_components_viz():
    """Visualize connected components"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create graph with 3 components
    G = nx.Graph()

    # Component 1 (blue)
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])

    # Component 2 (green)
    G.add_edges_from([(3, 4), (4, 5), (5, 6), (3, 6)])

    # Component 3 (orange)
    G.add_edges_from([(7, 8)])
    G.add_node(9)  # Isolated node

    pos = {
        0: (0, 2), 1: (1, 3), 2: (1, 1),
        3: (3, 3), 4: (4, 3), 5: (4, 2), 6: (3, 2),
        7: (6, 2.5), 8: (7, 2.5),
        9: (6.5, 1)
    }

    # Color by component
    colors = ['#2196F3'] * 3 + ['#4CAF50'] * 4 + ['#FF9800'] * 2 + ['#9C27B0']

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=800, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color='gray')

    # Add component labels
    ax.text(0.5, 3.8, 'Component 1\n(nodes: 0,1,2)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#BBDEFB'))
    ax.text(3.5, 3.8, 'Component 2\n(nodes: 3,4,5,6)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#C8E6C9'))
    ax.text(6.5, 3.5, 'Component 3\n(nodes: 7,8)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#FFE0B2'))
    ax.text(6.5, 0.3, 'Component 4\n(node: 9)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#E1BEE7'))

    ax.set_title('Number of Connected Components: 4', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    save_fig(fig, 'connected_components.png')

# =============================================================================
# 5. Surrounded Regions Visualization
# =============================================================================
def create_surrounded_regions_viz():
    """Visualize surrounded regions problem"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Before
    board_before = [
        ['X', 'X', 'X', 'X'],
        ['X', 'O', 'O', 'X'],
        ['X', 'X', 'O', 'X'],
        ['X', 'O', 'X', 'X']
    ]

    # After
    board_after = [
        ['X', 'X', 'X', 'X'],
        ['X', 'X', 'X', 'X'],
        ['X', 'X', 'X', 'X'],
        ['X', 'O', 'X', 'X']
    ]

    for idx, (board, title) in enumerate([(board_before, 'Before'), (board_after, 'After')]):
        ax = axes[idx]

        for i in range(4):
            for j in range(4):
                if board[i][j] == 'X':
                    color = '#424242'
                else:
                    # Check if on border
                    if i == 0 or i == 3 or j == 0 or j == 3:
                        color = '#4CAF50'  # Green - safe (on border)
                    else:
                        color = '#F44336' if idx == 0 else '#424242'  # Red - will be captured

                rect = patches.Rectangle((j, 3-i), 1, 1, facecolor=color, edgecolor='white', linewidth=2)
                ax.add_patch(rect)
                ax.text(j + 0.5, 3-i + 0.5, board[i][j], ha='center', va='center',
                       fontsize=18, fontweight='bold', color='white')

        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    # Add legend
    fig.text(0.5, 0.02, 'Gray=X | Green=O (safe, connected to border) | Red=O (surrounded, to be captured)',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow'))

    fig.suptitle('Surrounded Regions: Capture all O not connected to border', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    save_fig(fig, 'surrounded_regions.png')

# =============================================================================
# 6. Word Ladder Visualization
# =============================================================================
def create_word_ladder_viz():
    """Visualize word ladder BFS"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Word ladder: hit -> hot -> dot -> dog -> cog
    levels = [
        ['hit'],
        ['hot'],
        ['dot', 'lot'],
        ['dog', 'log'],
        ['cog']
    ]

    positions = {}
    y_spacing = 1.5

    for level_idx, level_words in enumerate(levels):
        y = 4 - level_idx * y_spacing
        x_start = -(len(level_words) - 1) / 2
        for word_idx, word in enumerate(level_words):
            x = x_start + word_idx * 2
            positions[word] = (x, y)

    # Draw edges (transformations)
    edges = [
        ('hit', 'hot'),
        ('hot', 'dot'), ('hot', 'lot'),
        ('dot', 'dog'), ('lot', 'log'),
        ('dog', 'cog'), ('log', 'cog')
    ]

    # Highlight shortest path
    shortest_path = ['hit', 'hot', 'dot', 'dog', 'cog']
    path_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]

    for u, v in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        if (u, v) in path_edges:
            ax.annotate('', xy=(x2, y2 + 0.3), xytext=(x1, y1 - 0.3),
                       arrowprops=dict(arrowstyle='->', color='green', lw=3))
        else:
            ax.annotate('', xy=(x2, y2 + 0.3), xytext=(x1, y1 - 0.3),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.5))

    # Draw nodes
    for word, (x, y) in positions.items():
        is_path = word in shortest_path
        color = '#4CAF50' if is_path else '#E0E0E0'
        edge_color = 'green' if is_path else 'gray'

        rect = patches.FancyBboxPatch((x - 0.5, y - 0.25), 1, 0.5,
                                       boxstyle='round,pad=0.05',
                                       facecolor=color, edgecolor=edge_color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, word, ha='center', va='center', fontsize=14, fontweight='bold',
               color='white' if is_path else 'black')

    # Add level labels
    for i in range(5):
        y = 4 - i * y_spacing
        ax.text(-3.5, y, f'Level {i}', fontsize=11, va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue'))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 5)
    ax.set_title('Word Ladder: BFS finds shortest transformation\n"hit" -> "cog" in 5 steps (length = 5)',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'word_ladder.png')

# =============================================================================
# 7. Alien Dictionary Visualization
# =============================================================================
def create_alien_dictionary_viz():
    """Visualize alien dictionary topological sort"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Show word comparisons
    ax1 = axes[0]
    ax1.axis('off')

    words = ['wrt', 'wrf', 'er', 'ett', 'rftt']

    ax1.text(0.5, 0.95, 'Alien Dictionary Words:', ha='center', fontsize=14, fontweight='bold',
            transform=ax1.transAxes)

    y = 0.85
    for word in words:
        ax1.text(0.5, y, word, ha='center', fontsize=16, fontfamily='monospace',
                transform=ax1.transAxes)
        y -= 0.08

    ax1.text(0.5, 0.4, 'Deduced Order Relations:', ha='center', fontsize=14, fontweight='bold',
            transform=ax1.transAxes)

    relations = [
        ('wrt vs wrf', 't -> f'),
        ('wrf vs er', 'w -> e'),
        ('er vs ett', 'r -> t'),
        ('ett vs rftt', 'e -> r')
    ]

    y = 0.32
    for comparison, relation in relations:
        ax1.text(0.3, y, comparison, ha='center', fontsize=11, transform=ax1.transAxes)
        ax1.text(0.7, y, relation, ha='center', fontsize=11, fontweight='bold',
                color='green', transform=ax1.transAxes)
        y -= 0.07

    # Right: Show the dependency graph and topological order
    ax2 = axes[1]

    G = nx.DiGraph()
    G.add_edges_from([('t', 'f'), ('w', 'e'), ('r', 't'), ('e', 'r')])

    pos = {'w': (0, 2), 'e': (1, 2), 'r': (2, 2), 't': (3, 2), 'f': (4, 2)}

    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='#2196F3', node_size=1000, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=16, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G, pos, ax=ax2, width=2, edge_color='gray',
                          connectionstyle='arc3,rad=0.1', arrows=True, arrowsize=20)

    ax2.text(2, 0.8, 'Topological Order: w -> e -> r -> t -> f', ha='center', fontsize=14,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax2.set_title('Dependency Graph (DAG)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    fig.suptitle('Alien Dictionary: Topological Sort to Find Character Order', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'alien_dictionary.png')

# =============================================================================
# 8. Network Delay Time Visualization
# =============================================================================
def create_network_delay_viz():
    """Visualize Dijkstra's algorithm for network delay"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create weighted graph
    G = nx.DiGraph()
    edges = [(2, 1, 1), (2, 3, 1), (3, 4, 1)]  # (u, v, weight)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = {1: (0, 1), 2: (1, 2), 3: (2, 1), 4: (3, 0)}

    # Left: Graph with weights
    ax1 = axes[0]

    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='#2196F3', node_size=800, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=14, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G, pos, ax=ax1, width=2, edge_color='gray', arrows=True, arrowsize=20)

    edge_labels = {(u, v): f'{w}' for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1, font_size=12, font_weight='bold')

    # Highlight source
    nx.draw_networkx_nodes(G, pos, nodelist=[2], ax=ax1, node_color='#4CAF50', node_size=800, edgecolors='black', linewidths=3)

    ax1.set_title('Network Graph\nSource: Node 2 (green)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Right: Distance table
    ax2 = axes[1]
    ax2.axis('off')

    ax2.text(0.5, 0.95, "Dijkstra's Algorithm Result", ha='center', fontsize=14, fontweight='bold',
            transform=ax2.transAxes)

    distances = [(1, 1), (2, 0), (3, 1), (4, 2)]

    ax2.text(0.3, 0.8, 'Node', ha='center', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.7, 0.8, 'Distance from 2', ha='center', fontsize=12, fontweight='bold', transform=ax2.transAxes)

    y = 0.7
    for node, dist in distances:
        ax2.text(0.3, y, str(node), ha='center', fontsize=14, transform=ax2.transAxes)
        ax2.text(0.7, y, str(dist), ha='center', fontsize=14, transform=ax2.transAxes)
        y -= 0.1

    ax2.text(0.5, 0.2, 'Network Delay Time = max(distances) = 2', ha='center', fontsize=14,
            fontweight='bold', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    fig.suptitle('Network Delay Time: Dijkstra for Weighted Shortest Paths', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'network_delay.png')

# =============================================================================
# 9. Cheapest Flights Visualization
# =============================================================================
def create_cheapest_flights_viz():
    """Visualize cheapest flights with K stops"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create weighted graph
    G = nx.DiGraph()
    edges = [
        (0, 1, 100), (1, 2, 100), (0, 2, 500),
        (1, 3, 600), (2, 3, 200)
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = {0: (0, 2), 1: (1.5, 2), 2: (1.5, 0.5), 3: (3, 1.25)}

    # Draw all edges first
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, edge_color='lightgray', arrows=True, arrowsize=15)

    # Highlight the cheapest path with k=1: 0 -> 1 -> 2 -> 3 (but k=1 means only 1 stop)
    # With k=1: 0 -> 1 -> 2 (cost 200) or 0 -> 2 (cost 500), going to 3: 0->1->3 (700) or 0->2->3 (700)
    # Actually cheapest with k=1 to node 3: 0 -> 1 -> 3 = 100 + 600 = 700 or 0 -> 2 -> 3 = 500 + 200 = 700

    cheap_path = [(0, 1), (1, 2), (2, 3)]  # 0->1->2->3 = 400 with k=2
    nx.draw_networkx_edges(G, pos, edgelist=cheap_path, ax=ax, width=3, edge_color='green', arrows=True, arrowsize=20)

    # Draw nodes
    colors = ['#4CAF50', '#2196F3', '#2196F3', '#FF5722']  # src=green, dst=orange
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1000, edgecolors='black', linewidths=2)

    labels = {0: '0\n(src)', 1: '1', 2: '2', 3: '3\n(dst)'}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=11, font_weight='bold', font_color='white')

    # Edge labels
    edge_labels = {(u, v): f'${w}' for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=10, font_weight='bold')

    ax.text(1.5, -0.8, 'Cheapest with k=2 stops: 0 -> 1 -> 2 -> 3 = $400', ha='center', fontsize=14,
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.text(1.5, -1.3, 'Direct 0 -> 2 -> 3 = $700 (only 1 stop)', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.set_title('Cheapest Flights Within K Stops\nModified BFS/Bellman-Ford', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    save_fig(fig, 'cheapest_flights.png')

# =============================================================================
# 10. Cycle Detection Visualization
# =============================================================================
def create_cycle_detection_viz():
    """Visualize cycle detection in directed and undirected graphs"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Directed graph with cycle
    ax1 = axes[0]
    G1 = nx.DiGraph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 1)])  # 1->2->3->1 is a cycle

    pos1 = {0: (0, 1), 1: (1, 1), 2: (2, 1), 3: (1.5, 0)}

    nx.draw_networkx_nodes(G1, pos1, ax=ax1, node_color='#2196F3', node_size=800, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G1, pos1, ax=ax1, font_size=14, font_weight='bold', font_color='white')

    # Draw non-cycle edges
    nx.draw_networkx_edges(G1, pos1, edgelist=[(0, 1)], ax=ax1, width=2, edge_color='gray', arrows=True, arrowsize=20)

    # Highlight cycle
    cycle_edges = [(1, 2), (2, 3), (3, 1)]
    nx.draw_networkx_edges(G1, pos1, edgelist=cycle_edges, ax=ax1, width=3, edge_color='red',
                          arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')

    ax1.set_title('Directed Graph: Cycle Detected\n(Red edges form cycle: 1->2->3->1)\nUse DFS with recursion stack',
                 fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Right: Undirected graph with cycle
    ax2 = axes[1]
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)])  # 0-1-2-3-0 is a cycle

    pos2 = {0: (0, 1), 1: (1, 2), 2: (2, 1), 3: (1, 0), 4: (-1, 1)}

    nx.draw_networkx_nodes(G2, pos2, ax=ax2, node_color='#2196F3', node_size=800, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G2, pos2, ax=ax2, font_size=14, font_weight='bold', font_color='white')

    # Draw non-cycle edge
    nx.draw_networkx_edges(G2, pos2, edgelist=[(0, 4)], ax=ax2, width=2, edge_color='gray')

    # Highlight cycle
    cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    nx.draw_networkx_edges(G2, pos2, edgelist=cycle_edges, ax=ax2, width=3, edge_color='red')

    ax2.set_title('Undirected Graph: Cycle Detected\n(Red edges form cycle: 0-1-2-3-0)\nUse Union-Find or DFS',
                 fontsize=12, fontweight='bold')
    ax2.axis('off')

    fig.suptitle('Cycle Detection in Graphs', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'cycle_detection.png')

# =============================================================================
# Main: Generate all visualizations
# =============================================================================
if __name__ == '__main__':
    print("Generating graph visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    create_clone_graph_viz()
    create_pacific_atlantic_viz()
    create_graph_valid_tree_viz()
    create_connected_components_viz()
    create_surrounded_regions_viz()
    create_word_ladder_viz()
    create_alien_dictionary_viz()
    create_network_delay_viz()
    create_cheapest_flights_viz()
    create_cycle_detection_viz()

    print()
    print("All visualizations generated successfully!")
