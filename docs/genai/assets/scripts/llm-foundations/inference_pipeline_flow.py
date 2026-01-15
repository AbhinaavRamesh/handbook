#!/usr/bin/env python3
"""
End-to-End Inference Pipeline Diagram

This script creates visualizations of the complete LLM inference pipeline,
from text input through tokenization to output generation.

Output: ../images/llm-foundations/inference_*.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection

# Ensure output directory exists
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "llm-foundations"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_full_pipeline_diagram():
    """Create comprehensive inference pipeline diagram."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(8, 11.5, 'LLM Inference Pipeline', fontsize=18, fontweight='bold',
            ha='center', va='top')

    # Define stages
    stages = [
        {
            'name': '1. Input Text',
            'position': (2, 9.5),
            'size': (2.5, 1),
            'color': '#e3f2fd',
            'details': '"Hello world"'
        },
        {
            'name': '2. Tokenization',
            'position': (6, 9.5),
            'size': (2.5, 1),
            'color': '#e8f5e9',
            'details': '[15496, 995]'
        },
        {
            'name': '3. Embedding Lookup',
            'position': (10, 9.5),
            'size': (2.5, 1),
            'color': '#fff3e0',
            'details': '[768-dim vectors]'
        },
        {
            'name': '4. + Positional Encoding',
            'position': (14, 9.5),
            'size': (2.5, 1),
            'color': '#fce4ec',
            'details': 'RoPE / Learned'
        },
    ]

    # Draw first row of stages
    for stage in stages:
        x, y = stage['position']
        w, h = stage['size']

        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="round,pad=0.03",
                               facecolor=stage['color'], edgecolor='black',
                               linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.2, stage['name'], ha='center', va='center',
                fontsize=9, fontweight='bold')
        ax.text(x, y - 0.25, stage['details'], ha='center', va='center',
                fontsize=8, style='italic')

    # Arrows between first row
    for i in range(len(stages) - 1):
        x1 = stages[i]['position'][0] + stages[i]['size'][0] / 2
        x2 = stages[i + 1]['position'][0] - stages[i + 1]['size'][0] / 2
        y = stages[i]['position'][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Transformer stack (center box)
    transformer_box = FancyBboxPatch((4, 4.5), 8, 3.5,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#e8eaf6', edgecolor='#3f51b5',
                                      linewidth=3)
    ax.add_patch(transformer_box)
    ax.text(8, 7.5, 'Transformer Layers (N times)', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Components inside transformer
    components = [
        ('Self-Attention', 5.5, 6, '#bbdefb'),
        ('KV-Cache', 5.5, 5.3, '#c8e6c9'),
        ('Layer Norm', 8, 6.5, '#ffe0b2'),
        ('FFN (MLP)', 10.5, 6, '#f8bbd9'),
        ('Layer Norm', 10.5, 5.3, '#ffe0b2'),
    ]

    for name, x, y, color in components:
        rect = FancyBboxPatch((x - 1, y - 0.3), 2, 0.6,
                               boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='black',
                               linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=8)

    # Arrows inside transformer
    ax.annotate('', xy=(5.5, 5.3), xytext=(5.5, 5.7),
                arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(6.8, 6), xytext=(6.5, 6),
                arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(9.2, 6), xytext=(8.8, 6),
                arrowprops=dict(arrowstyle='->', lw=1))

    # Arrow from embeddings to transformer
    ax.annotate('', xy=(8, 8), xytext=(8, 9),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Output stages
    output_stages = [
        {
            'name': '5. LM Head',
            'position': (4, 2.5),
            'size': (2.5, 1),
            'color': '#f3e5f5',
            'details': 'Linear projection'
        },
        {
            'name': '6. Logits',
            'position': (8, 2.5),
            'size': (2.5, 1),
            'color': '#fff9c4',
            'details': '[vocab_size]'
        },
        {
            'name': '7. Sampling',
            'position': (12, 2.5),
            'size': (2.5, 1),
            'color': '#e8f5e9',
            'details': 'Top-p, Temp'
        },
    ]

    for stage in output_stages:
        x, y = stage['position']
        w, h = stage['size']

        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="round,pad=0.03",
                               facecolor=stage['color'], edgecolor='black',
                               linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.2, stage['name'], ha='center', va='center',
                fontsize=9, fontweight='bold')
        ax.text(x, y - 0.25, stage['details'], ha='center', va='center',
                fontsize=8, style='italic')

    # Arrow from transformer to output
    ax.annotate('', xy=(4, 3), xytext=(4, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Arrows between output stages
    for i in range(len(output_stages) - 1):
        x1 = output_stages[i]['position'][0] + output_stages[i]['size'][0] / 2
        x2 = output_stages[i + 1]['position'][0] - output_stages[i + 1]['size'][0] / 2
        y = output_stages[i]['position'][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Output token
    output_box = FancyBboxPatch((13.5, 2), 2, 1,
                                 boxstyle="round,pad=0.03",
                                 facecolor='#a5d6a7', edgecolor='#2e7d32',
                                 linewidth=3)
    ax.add_patch(output_box)
    ax.text(14.5, 2.5, '8. Output\nToken', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Autoregressive arrow (feedback loop)
    ax.annotate('', xy=(2, 9.5), xytext=(14.5, 1.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green',
                               connectionstyle='arc3,rad=0.3'))
    ax.text(1, 5, 'Append to\ncontext', fontsize=9, color='green',
            fontweight='bold', ha='center')

    # Complexity annotations
    complexity_text = (
        "Complexity per token:\n"
        "- Embedding: O(1)\n"
        "- Attention: O(seq_len)\n"
        "- FFN: O(1)\n"
        "- LM Head: O(vocab_size)"
    )
    ax.text(15.5, 8, complexity_text, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "inference_pipeline_full.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved full pipeline diagram to: {output_path}")


def create_kv_cache_flow():
    """Visualize how KV-cache works in generation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def draw_generation_step(ax, step, title, prompt_len=3, gen_len=0):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Token boxes
        y_tokens = 7
        token_width = 0.8
        total_tokens = prompt_len + gen_len

        for i in range(total_tokens):
            x = 1 + i * (token_width + 0.1)
            color = '#bbdefb' if i < prompt_len else '#c8e6c9'
            edge = 'red' if i == total_tokens - 1 else 'black'
            lw = 2 if i == total_tokens - 1 else 1

            rect = FancyBboxPatch((x, y_tokens - 0.3), token_width, 0.6,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor=edge, linewidth=lw)
            ax.add_patch(rect)
            ax.text(x + token_width / 2, y_tokens, f'T{i + 1}',
                    ha='center', va='center', fontsize=9)

        # Label regions
        if gen_len > 0:
            ax.text(1 + prompt_len * (token_width + 0.1) / 2, 7.5, 'Prompt',
                    ha='center', fontsize=9, color='blue')
            ax.text(1 + (prompt_len + gen_len / 2) * (token_width + 0.1), 7.5, 'Generated',
                    ha='center', fontsize=9, color='green')

        # KV Cache visualization
        y_cache = 4.5
        cache_height = 0.4

        ax.text(0.5, y_cache + 1, 'KV Cache:', fontsize=10, fontweight='bold')

        for i in range(prompt_len + gen_len):
            x = 1 + i * (token_width + 0.1)

            if step == 0:
                # First step: all computed at once
                color = '#fff59d' if i < prompt_len else 'none'
            else:
                # Later steps: cached vs new
                if i < prompt_len + gen_len - 1:
                    color = '#a5d6a7'  # Cached (green)
                else:
                    color = '#fff59d'  # New (yellow)

            if color != 'none':
                rect = FancyBboxPatch((x, y_cache - cache_height / 2), token_width, cache_height,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)

        # Legend
        if step > 0:
            ax.add_patch(FancyBboxPatch((6, 2.5), 0.5, 0.3, facecolor='#a5d6a7'))
            ax.text(6.7, 2.65, 'Cached', fontsize=8, va='center')
            ax.add_patch(FancyBboxPatch((6, 2), 0.5, 0.3, facecolor='#fff59d'))
            ax.text(6.7, 2.15, 'New', fontsize=8, va='center')

        # Computation annotation
        if step == 0:
            ax.text(5, 3, f'Compute K,V for\n{prompt_len} tokens (parallel)',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
        else:
            ax.text(5, 3, f'Compute K,V for\n1 new token only',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen'))

    draw_generation_step(axes[0], 0, 'Step 1: Process Prompt', prompt_len=3, gen_len=0)
    draw_generation_step(axes[1], 1, 'Step 2: Generate Token 4', prompt_len=3, gen_len=1)
    draw_generation_step(axes[2], 2, 'Step 3: Generate Token 5', prompt_len=3, gen_len=2)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "inference_kv_cache_flow.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved KV cache flow to: {output_path}")


def create_speculative_decoding_diagram():
    """Visualize speculative decoding process."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, 'Speculative Decoding', fontsize=16, fontweight='bold', ha='center')

    # Draft model section
    ax.add_patch(FancyBboxPatch((0.5, 5), 5, 3.5,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#e3f2fd', edgecolor='#1976d2',
                                 linewidth=2))
    ax.text(3, 8, 'Draft Model (7B)', fontsize=12, fontweight='bold', ha='center')
    ax.text(3, 7.3, 'Generate K tokens quickly', fontsize=9, ha='center', style='italic')

    # Draft tokens
    draft_tokens = ['The', 'quick', 'brown', 'fox']
    for i, token in enumerate(draft_tokens):
        rect = FancyBboxPatch((0.8 + i * 1.1, 5.7), 1, 0.5,
                               boxstyle="round,pad=0.02",
                               facecolor='#90caf9', edgecolor='black')
        ax.add_patch(rect)
        ax.text(1.3 + i * 1.1, 5.95, token, ha='center', va='center', fontsize=9)

    # Arrow to target
    ax.annotate('', xy=(6, 6.5), xytext=(5.5, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Target model section
    ax.add_patch(FancyBboxPatch((6, 5), 5, 3.5,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#fff3e0', edgecolor='#f57c00',
                                 linewidth=2))
    ax.text(8.5, 8, 'Target Model (70B)', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 7.3, 'Verify all K tokens in parallel', fontsize=9, ha='center', style='italic')

    # Verification results
    results = [('The', True), ('quick', True), ('brown', True), ('fox', False)]
    for i, (token, accepted) in enumerate(results):
        color = '#a5d6a7' if accepted else '#ef9a9a'
        symbol = 'check' if accepted else 'x'

        rect = FancyBboxPatch((6.3 + i * 1.1, 5.7), 1, 0.5,
                               boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(6.8 + i * 1.1, 5.95, token, ha='center', va='center', fontsize=9)

    # Output section
    ax.add_patch(FancyBboxPatch((11.5, 5), 2, 3.5,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#e8f5e9', edgecolor='#388e3c',
                                 linewidth=2))
    ax.text(12.5, 8, 'Output', fontsize=12, fontweight='bold', ha='center')

    # Accepted tokens + correction
    output_tokens = ['The', 'quick', 'brown', 'red']
    for i, token in enumerate(output_tokens):
        color = '#a5d6a7' if i < 3 else '#fff59d'
        rect = FancyBboxPatch((11.7, 7 - i * 0.7), 1.6, 0.5,
                               boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(12.5, 7.25 - i * 0.7, token, ha='center', va='center', fontsize=9)

    # Arrow to output
    ax.annotate('', xy=(11.5, 6.5), xytext=(11, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Annotations
    ax.text(12.5, 5.2, '3 accepted\n1 corrected',
            ha='center', fontsize=9, style='italic')

    # Speed comparison
    comparison_text = (
        "Speedup Analysis:\n"
        "- Draft: 4 tokens @ 10ms = 40ms\n"
        "- Target: 1 parallel verify = 100ms\n"
        "- Total: 140ms for 4 tokens\n"
        "- Without spec: 4 x 100ms = 400ms\n"
        "- Speedup: ~2.9x"
    )
    ax.text(7, 2.5, comparison_text, fontsize=9, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Key insight
    ax.text(7, 0.8, 'Key: Draft model proposes, target model verifies in parallel',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "inference_speculative_decoding.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved speculative decoding diagram to: {output_path}")


def create_batching_comparison():
    """Visualize static vs continuous batching."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def draw_request_timeline(ax, batching_type):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Request Slot', fontsize=11)
        ax.set_title(f'{batching_type} Batching', fontsize=12, fontweight='bold')

        if batching_type == 'Static':
            # Requests of varying lengths, all start together
            requests = [
                (0, 3, 0.8),  # (start, end, height) - Slot 1
                (0, 2, 0.8),  # Slot 2
                (0, 5, 0.8),  # Slot 3
                (0, 4, 0.8),  # Slot 4
            ]

            for i, (start, end, height) in enumerate(requests):
                rect = patches.Rectangle((start, i + 0.1), end, height,
                                          facecolor='#64b5f6', edgecolor='black',
                                          linewidth=1)
                ax.add_patch(rect)
                ax.text(start + end / 2, i + 0.5, f'R{i + 1}', ha='center', va='center')

                # Wasted time (gray)
                if end < 5:
                    waste = patches.Rectangle((end, i + 0.1), 5 - end, height,
                                               facecolor='lightgray', edgecolor='black',
                                               linewidth=1, linestyle='--', alpha=0.5)
                    ax.add_patch(waste)

            # Batch 2 starts after longest finishes
            for i, length in enumerate([2.5, 3.5]):
                rect = patches.Rectangle((5, i + 0.1), length, 0.8,
                                          facecolor='#81c784', edgecolor='black',
                                          linewidth=1)
                ax.add_patch(rect)
                ax.text(5 + length / 2, i + 0.5, f'R{i + 5}', ha='center', va='center')

            ax.axvline(x=5, color='red', linestyle='--', linewidth=2)
            ax.text(5, 5.5, 'Batch boundary\n(wait for all)', ha='center', fontsize=9, color='red')

        else:  # Continuous
            # Requests come and go dynamically
            requests = [
                (0, 3, 0, '#64b5f6'),    # R1: slot 0
                (0, 2, 1, '#64b5f6'),    # R2: slot 1
                (0, 5, 2, '#64b5f6'),    # R3: slot 2
                (2, 4, 1, '#81c784'),    # R4: slot 1 (reused!)
                (3, 6, 0, '#ffb74d'),    # R5: slot 0 (reused!)
                (4, 8, 1, '#ba68c8'),    # R6: slot 1 (reused again!)
                (0, 4, 3, '#64b5f6'),    # R4 original slot
                (4, 7, 3, '#81c784'),    # R7
            ]

            for i, (start, end, slot, color) in enumerate(requests):
                rect = patches.Rectangle((start, slot + 0.1), end - start, 0.8,
                                          facecolor=color, edgecolor='black',
                                          linewidth=1)
                ax.add_patch(rect)
                ax.text(start + (end - start) / 2, slot + 0.5, f'R{i + 1}',
                        ha='center', va='center', fontsize=9)

            # Arrows showing slot reuse
            ax.annotate('Slot freed,\nnew request\njoins', xy=(2, 1.5), xytext=(1, 4.5),
                        fontsize=8, ha='center',
                        arrowprops=dict(arrowstyle='->', color='green'))

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(['Slot 1', 'Slot 2', 'Slot 3', 'Slot 4'])
        ax.grid(True, alpha=0.3)

    draw_request_timeline(axes[0], 'Static')
    draw_request_timeline(axes[1], 'Continuous')

    # Add comparison notes
    fig.text(0.25, 0.02, 'GPU idle while waiting for longest request', ha='center',
             fontsize=9, color='red')
    fig.text(0.75, 0.02, 'Slots reused immediately, higher GPU utilization', ha='center',
             fontsize=9, color='green')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = os.path.join(OUTPUT_DIR, "inference_batching_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved batching comparison to: {output_path}")


if __name__ == "__main__":
    create_full_pipeline_diagram()
    create_kv_cache_flow()
    create_speculative_decoding_diagram()
    create_batching_comparison()
    print("All inference pipeline visualizations created successfully!")
