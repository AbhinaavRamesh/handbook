#!/usr/bin/env python3
"""
Generate two graph algorithm visualizations:
  1. Topological Sort (Kahn's Algorithm) on a course prerequisites DAG
  2. Dijkstra's Algorithm on a weighted graph
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import deque, defaultdict
import heapq
import os

# ── Global style constants ──────────────────────────────────────────────────
BG       = "#1e1e2e"
TEXT_CLR  = "#cdd6f4"
GREEN    = "#a6e3a1"
YELLOW   = "#f9e2af"
GRAY     = "#585b70"
BLUE     = "#89b4fa"
RED      = "#f38ba8"
PEACH    = "#fab387"
EDGE_CLR = "#7f849c"
WHITE    = "#cdd6f4"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "viz")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: draw a directed graph in a matplotlib Axes
# ═══════════════════════════════════════════════════════════════════════════

def _draw_arrow(ax, p1, p2, color=EDGE_CLR, lw=1.8, node_radius=0.28,
                label=None, label_color=WHITE, connectionstyle="arc3,rad=0.0",
                arrow_head_width=0.08):
    """Draw a curved arrow between two node centres, stopping at node_radius."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = np.hypot(dx, dy)
    if dist == 0:
        return
    # shorten both ends by node_radius
    ux, uy = dx / dist, dy / dist
    start = (p1[0] + ux * node_radius, p1[1] + uy * node_radius)
    end   = (p2[0] - ux * node_radius, p2[1] - uy * node_radius)

    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle=f"->,head_width={arrow_head_width},head_length=0.06",
            color=color, lw=lw,
            connectionstyle=connectionstyle,
            shrinkA=0, shrinkB=0,
        ),
    )
    if label is not None:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        # slight offset perpendicular to the edge
        nx, ny = -uy, ux
        off = 0.13
        ax.text(mx + nx * off, my + ny * off, str(label),
                ha="center", va="center", fontsize=8, color=label_color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.85))


def _draw_node(ax, pos, label, fill=GRAY, text_color=WHITE, radius=0.28,
               fontsize=9, sublabel=None, sublabel_color=YELLOW):
    """Draw a circular node with optional sublabel below."""
    circle = plt.Circle(pos, radius, fc=fill, ec=WHITE, lw=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(pos[0], pos[1] + (0.04 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize, color=text_color,
            fontweight="bold", zorder=6)
    if sublabel is not None:
        ax.text(pos[0], pos[1] - 0.16, sublabel,
                ha="center", va="center", fontsize=7, color=sublabel_color,
                fontweight="bold", zorder=6)


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE 1 — Topological Sort (Kahn's Algorithm)
# ═══════════════════════════════════════════════════════════════════════════

def generate_topological_sort():
    # ── DAG definition ────────────────────────────────────────────────────
    nodes = ["Math101", "Calc1", "Calc2", "LinAlg", "ML",
             "CS101", "DataStruct", "Algorithms"]
    edges = [
        ("Math101", "Calc1"),
        ("Calc1", "Calc2"),
        ("Calc1", "LinAlg"),
        ("LinAlg", "ML"),
        ("CS101", "DataStruct"),
        ("DataStruct", "Algorithms"),
        ("Algorithms", "ML"),
    ]

    # Manual positions (x, y) for a nice layout
    pos = {
        "Math101":    (0.0,  1.0),
        "Calc1":      (1.5,  1.0),
        "Calc2":      (3.0,  1.6),
        "LinAlg":     (3.0,  0.4),
        "ML":         (5.0,  0.7),
        "CS101":      (0.0, -0.4),
        "DataStruct": (1.5, -0.4),
        "Algorithms": (3.0, -0.4),
    }

    # Abbreviations for display
    abbrev = {
        "Math101": "Math101",
        "Calc1": "Calc I",
        "Calc2": "Calc II",
        "LinAlg": "LinAlg",
        "ML": "ML",
        "CS101": "CS101",
        "DataStruct": "DataStr",
        "Algorithms": "Algo",
    }

    # ── Run Kahn's algorithm, capturing snapshots ─────────────────────────
    indeg = {n: 0 for n in nodes}
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1

    # Snapshot: (queue_contents, processed_so_far, indegree_copy, description)
    snapshots = []

    queue = deque()
    processed = []
    for n in nodes:
        if indeg[n] == 0:
            queue.append(n)

    snapshots.append((
        list(queue), list(processed), dict(indeg),
        "Step 0  Initialize\nEnqueue nodes with in-degree 0"
    ))

    step = 1
    while queue:
        node = queue.popleft()
        processed.append(node)
        for nb in adj[node]:
            indeg[nb] -= 1
            if indeg[nb] == 0:
                queue.append(nb)
        snapshots.append((
            list(queue), list(processed), dict(indeg),
            f"Step {step}  Process \"{abbrev[node]}\"\nReduce neighbours' in-degrees"
        ))
        step += 1
        if len(snapshots) >= 6:
            # Collect remaining
            while queue:
                node = queue.popleft()
                processed.append(node)
                for nb in adj[node]:
                    indeg[nb] -= 1
                    if indeg[nb] == 0:
                        queue.append(nb)
            snapshots.append((
                list(queue), list(processed), dict(indeg),
                f"Final  Topological Order:\n" +
                " -> ".join(abbrev[p] for p in processed)
            ))
            break
    else:
        snapshots.append((
            list(queue), list(processed), dict(indeg),
            f"Final  Topological Order:\n" +
            " -> ".join(abbrev[p] for p in processed)
        ))

    # Limit to at most 6 panels (2 rows x 3 cols)
    # Pick: 0, 1, 2, 3, one middle, and final
    if len(snapshots) > 6:
        idxs = [0, 1, 2, 3, len(snapshots) // 2, len(snapshots) - 1]
        snapshots = [snapshots[i] for i in idxs]

    n_panels = len(snapshots)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10),
                             facecolor=BG)
    fig.suptitle("Topological Sort \u2014 Kahn's Algorithm (Course Schedule)",
                 fontsize=18, fontweight="bold", color=TEXT_CLR, y=0.97)

    if nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for idx, ax in enumerate(axes_flat):
        ax.set_facecolor(BG)
        ax.set_xlim(-0.8, 5.8)
        ax.set_ylim(-1.3, 2.3)
        ax.set_aspect("equal")
        ax.axis("off")

        if idx >= n_panels:
            ax.set_visible(False)
            continue

        q_list, proc_list, indeg_snap, desc = snapshots[idx]
        proc_set = set(proc_list)
        q_set = set(q_list)

        # Draw edges
        for u, v in edges:
            ecolor = EDGE_CLR
            elw = 1.5
            if u in proc_set and v in proc_set:
                ecolor = GREEN
                elw = 2.0
            elif u in proc_set:
                ecolor = YELLOW
                elw = 1.8
            _draw_arrow(ax, pos[u], pos[v], color=ecolor, lw=elw, node_radius=0.30)

        # Draw nodes
        for n in nodes:
            if n in proc_set:
                fill = GREEN
            elif n in q_set:
                fill = YELLOW
            else:
                fill = GRAY
            tc = "#1e1e2e" if fill in (GREEN, YELLOW) else WHITE
            sublbl = f"deg={indeg_snap[n]}"
            _draw_node(ax, pos[n], abbrev[n], fill=fill, text_color=tc,
                       radius=0.30, fontsize=8, sublabel=sublbl,
                       sublabel_color=tc if fill in (GREEN, YELLOW) else PEACH)

        # Panel description
        ax.text(2.5, -1.05, desc, ha="center", va="top",
                fontsize=8.5, color=TEXT_CLR, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.35", fc="#313244", ec=EDGE_CLR, lw=0.8))

    # Legend
    legend_elements = [
        mpatches.Patch(fc=GREEN, ec=WHITE, label="Processed"),
        mpatches.Patch(fc=YELLOW, ec=WHITE, label="In Queue"),
        mpatches.Patch(fc=GRAY, ec=WHITE, label="Waiting"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=11, frameon=True, facecolor="#313244", edgecolor=EDGE_CLR,
               labelcolor=TEXT_CLR, handlelength=1.5, handleheight=1.2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    out_path = os.path.join(OUT_DIR, "topological_sort.png")
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE 2 — Dijkstra's Algorithm
# ═══════════════════════════════════════════════════════════════════════════

def generate_dijkstra():
    # ── Graph definition ──────────────────────────────────────────────────
    nodes = list("ABCDEF")
    edges = [
        ("A", "B", 4), ("A", "C", 2),
        ("B", "D", 3), ("B", "E", 1),
        ("C", "B", 1), ("C", "D", 5),
        ("D", "E", 2), ("D", "F", 1),
        ("E", "F", 3),
    ]
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))

    pos = {
        "A": (0.0, 1.0),
        "B": (1.8, 1.8),
        "C": (1.8, 0.2),
        "D": (3.6, 1.8),
        "E": (3.6, 0.2),
        "F": (5.4, 1.0),
    }

    # ── Run Dijkstra, capturing snapshots ─────────────────────────────────
    INF = float("inf")
    dist = {n: INF for n in nodes}
    prev = {n: None for n in nodes}
    dist["A"] = 0
    visited = set()
    frontier = set()

    # snapshot: (dist_copy, visited_copy, frontier_copy, current_node, desc, path_edges)
    snapshots = []

    snapshots.append((
        dict(dist), set(visited), set(frontier), None,
        "Initial State\nSource = A, all distances = \u221e",
        [],
    ))

    heap = [(0, "A")]
    frontier.add("A")

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        frontier.discard(u)

        for v, w in adj[u]:
            if v not in visited and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                frontier.add(v)
                heapq.heappush(heap, (dist[v], v))

        # Build description
        updates = []
        for v, w in adj[u]:
            if dist[v] == dist[u] + w and prev[v] == u:
                updates.append(f"{v}={dist[v]}")
            elif v not in visited:
                pass  # already had a shorter path
        desc_parts = f"Process \"{u}\" (dist={d})"
        if updates:
            desc_parts += "\nUpdate: " + ", ".join(updates)
        else:
            desc_parts += "\nNo improvements found"

        snapshots.append((
            dict(dist), set(visited), set(frontier), u,
            desc_parts, [],
        ))

    # Reconstruct shortest path A -> F
    path_nodes = []
    n = "F"
    while n is not None:
        path_nodes.append(n)
        n = prev[n]
    path_nodes.reverse()
    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

    snapshots.append((
        dict(dist), set(visited), set(), None,
        f"Shortest path A\u2192F  (cost {dist['F']})\n" +
        " \u2192 ".join(path_nodes),
        path_edges,
    ))

    # Select at most 6 snapshots
    if len(snapshots) > 6:
        # always keep first, last, and evenly spaced middle ones
        idxs = [0]
        middle = list(range(1, len(snapshots) - 1))
        step = max(1, len(middle) // 4)
        for i in range(0, len(middle), step):
            idxs.append(middle[i])
            if len(idxs) >= 5:
                break
        idxs.append(len(snapshots) - 1)
        idxs = sorted(set(idxs))[:6]
        snapshots = [snapshots[i] for i in idxs]

    n_panels = len(snapshots)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), facecolor=BG)
    fig.suptitle("Dijkstra's Algorithm \u2014 Shortest Path",
                 fontsize=18, fontweight="bold", color=TEXT_CLR, y=0.97)

    if nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for idx, ax in enumerate(axes_flat):
        ax.set_facecolor(BG)
        ax.set_xlim(-0.8, 6.2)
        ax.set_ylim(-0.6, 2.6)
        ax.set_aspect("equal")
        ax.axis("off")

        if idx >= n_panels:
            ax.set_visible(False)
            continue

        d_snap, vis_snap, front_snap, cur_node, desc, sp_edges = snapshots[idx]
        sp_edge_set = set(sp_edges)

        # Draw edges
        for u, v, w in edges:
            if (u, v) in sp_edge_set:
                ecolor = BLUE
                elw = 3.0
            elif u in vis_snap and v in vis_snap:
                ecolor = GREEN
                elw = 2.0
            else:
                ecolor = EDGE_CLR
                elw = 1.5
            _draw_arrow(ax, pos[u], pos[v], color=ecolor, lw=elw,
                        node_radius=0.30, label=w, label_color=PEACH)

        # Draw nodes
        for n in nodes:
            if sp_edges and n in [x for pair in sp_edges for x in pair]:
                fill = BLUE
            elif n in vis_snap:
                fill = GREEN
            elif n in front_snap:
                fill = YELLOW
            else:
                fill = GRAY
            tc = "#1e1e2e" if fill in (GREEN, YELLOW, BLUE) else WHITE

            dval = d_snap[n]
            sublbl = f"d={dval}" if dval != INF else "d=\u221e"
            _draw_node(ax, pos[n], n, fill=fill, text_color=tc,
                       radius=0.30, fontsize=12, sublabel=sublbl,
                       sublabel_color=tc if fill in (GREEN, YELLOW, BLUE) else PEACH)

        # Panel description
        ax.text(2.7, -0.35, desc, ha="center", va="top",
                fontsize=8.5, color=TEXT_CLR, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.35", fc="#313244", ec=EDGE_CLR, lw=0.8))

    # Legend
    legend_elements = [
        mpatches.Patch(fc=GREEN, ec=WHITE, label="Visited"),
        mpatches.Patch(fc=YELLOW, ec=WHITE, label="Frontier"),
        mpatches.Patch(fc=GRAY, ec=WHITE, label="Unvisited"),
        mpatches.Patch(fc=BLUE, ec=WHITE, label="Shortest Path"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=11, frameon=True, facecolor="#313244", edgecolor=EDGE_CLR,
               labelcolor=TEXT_CLR, handlelength=1.5, handleheight=1.2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    out_path = os.path.join(OUT_DIR, "dijkstra.png")
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating graph algorithm visualizations...")
    generate_topological_sort()
    generate_dijkstra()
    print("Done!")
