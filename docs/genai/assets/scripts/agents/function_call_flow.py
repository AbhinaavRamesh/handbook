"""
Function Call Flow Visualization

Creates a diagram showing the sequence of function calling
between user, LLM, and tools.

Output: /docs/genai/assets/images/agents/function_call_flow.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "agents"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_actor_box(ax, x, y_start, y_end, label, color, width=1.5):
    """Draw a vertical actor box (lifeline header)."""
    rect = FancyBboxPatch(
        (x - width/2, y_start - 0.4), width, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor='#333',
        linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x, y_start, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333')

    # Lifeline (dashed vertical line)
    ax.plot([x, x], [y_start - 0.4, y_end], 'k--', linewidth=1, alpha=0.5)


def draw_message(ax, x1, x2, y, label, color='#333', dashed=False):
    """Draw a horizontal message arrow."""
    style = '--' if dashed else '-'
    ax.annotate(
        '', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=1.5,
            linestyle=style,
            mutation_scale=15
        )
    )
    # Label above arrow
    mid_x = (x1 + x2) / 2
    ax.text(mid_x, y + 0.15, label, ha='center', va='bottom',
            fontsize=8, color=color, style='italic' if dashed else 'normal')


def draw_activation_box(ax, x, y_start, y_end, color='#e0e0e0', width=0.3):
    """Draw an activation box on a lifeline."""
    rect = patches.Rectangle(
        (x - width/2, y_end), width, y_start - y_end,
        facecolor=color,
        edgecolor='#666',
        linewidth=1
    )
    ax.add_patch(rect)


def draw_note(ax, x, y, text, color='#fff9c4'):
    """Draw a note box."""
    # Calculate text size for box
    lines = text.split('\n')
    max_len = max(len(line) for line in lines)
    width = max_len * 0.08 + 0.3
    height = len(lines) * 0.25 + 0.2

    rect = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='#ffc107',
        linewidth=1
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7, color='#333', family='monospace')


def create_function_call_sequence():
    """Create the main function call sequence diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-0.5, 9.5)
    ax.axis('off')

    # Title
    ax.text(6, 9.2, "Function Calling Sequence Diagram", fontsize=14,
            ha='center', fontweight='bold', color='#1a237e')

    # Actor positions
    actors = {
        'User': {'x': 1, 'color': '#e3f2fd'},
        'Application': {'x': 4, 'color': '#e8f5e9'},
        'LLM': {'x': 7, 'color': '#fff3e0'},
        'Tool': {'x': 10, 'color': '#fce4ec'}
    }

    # Draw actor boxes
    y_top = 8.5
    y_bottom = 0.5
    for name, info in actors.items():
        draw_actor_box(ax, info['x'], y_top, y_bottom, name, info['color'])

    # Sequence steps
    y = 7.8

    # Step 1: User sends query
    draw_message(ax, actors['User']['x'], actors['Application']['x'], y,
                 "1. 'What's the weather in Tokyo?'")
    y -= 0.8

    # Step 2: Application sends to LLM with tools
    draw_message(ax, actors['Application']['x'], actors['LLM']['x'], y,
                 "2. Query + Tool definitions")
    draw_note(ax, 8.5, y - 0.3, "tools=[{\n  'name': 'get_weather',\n  'params': {...}\n}]")
    y -= 1.2

    # Activation box for LLM processing
    draw_activation_box(ax, actors['LLM']['x'], y + 0.1, y - 0.4, '#fff3e0')

    # Step 3: LLM returns tool call
    draw_message(ax, actors['LLM']['x'], actors['Application']['x'], y,
                 "3. tool_call: get_weather('Tokyo')", dashed=True)
    y -= 0.8

    # Activation box for Application
    draw_activation_box(ax, actors['Application']['x'], y + 0.1, y - 0.8, '#e8f5e9')

    # Step 4: Application calls tool
    draw_message(ax, actors['Application']['x'], actors['Tool']['x'], y,
                 "4. Execute get_weather('Tokyo')")
    y -= 0.6

    # Activation box for Tool
    draw_activation_box(ax, actors['Tool']['x'], y + 0.1, y - 0.3, '#fce4ec')

    # Step 5: Tool returns result
    draw_message(ax, actors['Tool']['x'], actors['Application']['x'], y,
                 "5. {temp: 22, condition: 'sunny'}", dashed=True)
    y -= 0.8

    # Step 6: Application sends result to LLM
    draw_message(ax, actors['Application']['x'], actors['LLM']['x'], y,
                 "6. Tool result + Conversation")
    y -= 0.6

    # Activation box for LLM final processing
    draw_activation_box(ax, actors['LLM']['x'], y + 0.1, y - 0.4, '#fff3e0')

    # Step 7: LLM generates final response
    draw_message(ax, actors['LLM']['x'], actors['Application']['x'], y,
                 "7. 'It's 22°C and sunny in Tokyo'", dashed=True)
    y -= 0.8

    # Step 8: Application returns to user
    draw_message(ax, actors['Application']['x'], actors['User']['x'], y,
                 "8. Display response", dashed=True)

    # Legend
    legend_y = 1.2
    ax.text(0.5, legend_y, "Legend:", fontsize=9, fontweight='bold')
    ax.annotate('', xy=(2.5, legend_y - 0.3), xytext=(1, legend_y - 0.3),
                arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.5))
    ax.text(2.7, legend_y - 0.3, "Request", fontsize=8, va='center')
    ax.annotate('', xy=(5.5, legend_y - 0.3), xytext=(4, legend_y - 0.3),
                arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.5, linestyle='--'))
    ax.text(5.7, legend_y - 0.3, "Response", fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'function_call_flow.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_parallel_calls_diagram():
    """Create diagram showing parallel function calls."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.7, "Parallel Function Calls", fontsize=14,
            ha='center', fontweight='bold', color='#1a237e')

    # Actors
    actors = {
        'App': {'x': 2, 'color': '#e8f5e9'},
        'LLM': {'x': 5, 'color': '#fff3e0'},
        'Tool A': {'x': 8, 'color': '#e3f2fd'},
        'Tool B': {'x': 11, 'color': '#fce4ec'}
    }

    for name, info in actors.items():
        draw_actor_box(ax, info['x'], 7, 1, name, info['color'], width=1.3)

    y = 6.3

    # Step 1: Query to LLM
    draw_message(ax, actors['App']['x'], actors['LLM']['x'], y,
                 "Query requiring multiple tools")
    y -= 1

    # Step 2: LLM returns multiple tool calls
    draw_activation_box(ax, actors['LLM']['x'], y + 0.3, y - 0.3, '#fff3e0')
    ax.text(5, y - 0.5, "Returns 2 tool_calls", fontsize=8, ha='center', color='#666')
    y -= 0.8

    # Parallel calls bracket
    bracket_top = y + 0.2
    bracket_bottom = y - 0.8

    # Draw parallel indicator
    ax.plot([3, 3], [bracket_top, bracket_bottom], 'b-', linewidth=2)
    ax.plot([3, 3.2], [bracket_top, bracket_top], 'b-', linewidth=2)
    ax.plot([3, 3.2], [bracket_bottom, bracket_bottom], 'b-', linewidth=2)
    ax.text(2.8, (bracket_top + bracket_bottom)/2, "parallel",
            fontsize=8, rotation=90, ha='center', va='center', color='blue')

    # Tool A call
    draw_message(ax, actors['App']['x'], actors['Tool A']['x'], y,
                 "get_weather('Tokyo')", color='#1976d2')
    draw_activation_box(ax, actors['Tool A']['x'], y + 0.1, y - 0.4, '#e3f2fd')

    # Tool B call (same time)
    draw_message(ax, actors['App']['x'], actors['Tool B']['x'], y - 0.4,
                 "search('restaurants')", color='#c2185b')
    draw_activation_box(ax, actors['Tool B']['x'], y - 0.3, y - 0.7, '#fce4ec')

    y -= 1.2

    # Results come back
    draw_message(ax, actors['Tool A']['x'], actors['App']['x'], y,
                 "Weather result", dashed=True, color='#1976d2')
    draw_message(ax, actors['Tool B']['x'], actors['App']['x'], y - 0.3,
                 "Search results", dashed=True, color='#c2185b')

    y -= 1

    # Combined result to LLM
    draw_message(ax, actors['App']['x'], actors['LLM']['x'], y,
                 "Both tool results")
    y -= 0.8

    # Final response
    draw_activation_box(ax, actors['LLM']['x'], y + 0.2, y - 0.3, '#fff3e0')
    draw_message(ax, actors['LLM']['x'], actors['App']['x'], y - 0.3,
                 "Combined response", dashed=True)

    # Note about parallel execution
    draw_note(ax, 9.5, 1.5, "Parallel execution\nreduces latency\nwhen tools are\nindependent")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'function_call_parallel.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_error_handling_diagram():
    """Create diagram showing function call error handling."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, "Function Call Error Handling", fontsize=14,
            ha='center', fontweight='bold', color='#1a237e')

    # Flow boxes
    boxes = [
        (1, 6, "Parse\nArguments", '#e3f2fd'),
        (3.5, 6, "Validate\nParameters", '#e8f5e9'),
        (6, 6, "Execute\nTool", '#fff3e0'),
        (8.5, 6, "Return\nResult", '#e8f5e9'),
    ]

    # Error boxes
    error_boxes = [
        (2.25, 4, "Parse\nError", '#ffcdd2'),
        (4.75, 4, "Validation\nError", '#ffcdd2'),
        (7.25, 4, "Execution\nError", '#ffcdd2'),
    ]

    # Draw main flow
    for x, y, label, color in boxes:
        rect = FancyBboxPatch(
            (x - 0.7, y - 0.5), 1.4, 1,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='#333',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=9, fontweight='bold')

    # Draw arrows between main boxes
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.7
        x2 = boxes[i+1][0] - 0.7
        y = boxes[i][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='-|>', color='#388e3c', lw=2))

    # Draw error boxes
    for x, y, label, color in error_boxes:
        rect = FancyBboxPatch(
            (x - 0.6, y - 0.4), 1.2, 0.8,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='#d32f2f',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8, color='#b71c1c')

    # Draw error arrows
    error_sources = [(1, 5.5), (3.5, 5.5), (6, 5.5)]
    for i, (x, y) in enumerate(error_sources):
        ex, ey = error_boxes[i][0], error_boxes[i][1] + 0.4
        ax.annotate('', xy=(ex, ey), xytext=(x, y),
                    arrowprops=dict(arrowstyle='-|>', color='#d32f2f',
                                    lw=1.5, linestyle='--'))

    # Recovery arrow
    ax.annotate('', xy=(1.5, 2.5), xytext=(7.25, 3.6),
                arrowprops=dict(arrowstyle='-|>', color='#f57c00',
                                lw=2, connectionstyle='arc3,rad=0.3'))

    # Recovery box
    rect = FancyBboxPatch(
        (0.5, 2), 2, 1,
        boxstyle="round,pad=0.02",
        facecolor='#fff3e0',
        edgecolor='#f57c00',
        linewidth=2
    )
    ax.add_patch(rect)
    ax.text(1.5, 2.5, "Retry / Fallback\nError to LLM", ha='center', va='center',
            fontsize=9, fontweight='bold', color='#e65100')

    # Annotations
    ax.text(5, 1, "Error messages are returned to LLM for self-correction",
            ha='center', fontsize=10, style='italic', color='#666')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'function_call_error_handling.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    print("Creating function call flow diagrams...")
    create_function_call_sequence()
    print(f"Sequence diagram saved to: {OUTPUT_DIR}/function_call_flow.png")
    create_parallel_calls_diagram()
    print(f"Parallel calls diagram saved to: {OUTPUT_DIR}/function_call_parallel.png")
    create_error_handling_diagram()
    print(f"Error handling diagram saved to: {OUTPUT_DIR}/function_call_error_handling.png")
    print("Done!")
