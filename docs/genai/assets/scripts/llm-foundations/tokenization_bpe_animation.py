#!/usr/bin/env python3
"""
BPE (Byte-Pair Encoding) Merge Process Animation

This script creates an animated GIF showing how BPE tokenization works,
visualizing the iterative merge process step by step.

Output: ../images/llm-foundations/bpe_merge_animation.gif
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# Ensure output directory exists
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "llm-foundations"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_stats(word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    """Count frequency of adjacent symbol pairs."""
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(
    word_freqs: Dict[Tuple[str, ...], int],
    pair: Tuple[str, str]
) -> Dict[Tuple[str, ...], int]:
    """Merge all occurrences of a pair."""
    new_word_freqs = {}
    replacement = pair[0] + pair[1]

    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


def run_bpe(corpus: List[str], num_merges: int) -> List[dict]:
    """Run BPE and record each step for animation."""
    # Initialize word frequencies
    word_freqs = defaultdict(int)
    for text in corpus:
        for word in text.lower().split():
            word_tuple = tuple(word) + ('_',)  # End-of-word marker
            word_freqs[word_tuple] += 1

    steps = [{
        "step": 0,
        "merge": None,
        "word_freqs": dict(word_freqs),
        "vocab": set(c for word in word_freqs for c in word)
    }]

    for i in range(num_merges):
        pairs = get_stats(word_freqs)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        word_freqs = merge_pair(word_freqs, best_pair)

        steps.append({
            "step": i + 1,
            "merge": best_pair,
            "merged_symbol": best_pair[0] + best_pair[1],
            "word_freqs": dict(word_freqs),
            "vocab": set(c for word in word_freqs for c in word),
            "pair_freq": pairs[best_pair]
        })

    return steps


def visualize_step(ax, step_data: dict, words_to_show: List[str]):
    """Visualize a single BPE step."""
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    if step_data["step"] == 0:
        title = "Initial: Character-level tokenization"
    else:
        merge = step_data["merge"]
        title = f"Step {step_data['step']}: Merge '{merge[0]}' + '{merge[1]}' → '{step_data['merged_symbol']}'"
        title += f" (freq: {step_data['pair_freq']})"

    ax.text(5, 7.5, title, ha='center', va='top', fontsize=14, fontweight='bold')

    # Show vocabulary
    vocab = sorted(step_data["vocab"], key=lambda x: (-len(x), x))
    vocab_text = "Vocabulary: " + ", ".join([f"'{v}'" for v in vocab[:15]])
    if len(vocab) > 15:
        vocab_text += f" ... (+{len(vocab) - 15} more)"
    ax.text(5, 6.8, vocab_text, ha='center', va='top', fontsize=9, style='italic')

    # Show word tokenizations
    y_pos = 5.5
    for word in words_to_show:
        word_tuple = tuple(word.lower()) + ('_',)

        # Find matching tokenization in current state
        matched_tokens = None
        for tokens, freq in step_data["word_freqs"].items():
            # Check if this is the same word
            if ''.join(tokens).replace('_', '') == word.lower():
                matched_tokens = tokens
                break

        if matched_tokens:
            # Draw the word
            ax.text(0.5, y_pos, f"'{word}':", ha='left', va='center', fontsize=11)

            # Draw token boxes
            x_pos = 2.5
            colors = plt.cm.Set3(np.linspace(0, 1, len(matched_tokens)))

            for j, (token, color) in enumerate(zip(matched_tokens, colors)):
                token_width = 0.4 + 0.35 * len(token)

                # Highlight newly merged tokens
                if step_data["step"] > 0 and token == step_data.get("merged_symbol"):
                    edgecolor = 'red'
                    linewidth = 3
                else:
                    edgecolor = 'black'
                    linewidth = 1

                rect = patches.FancyBboxPatch(
                    (x_pos, y_pos - 0.3), token_width, 0.6,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor=edgecolor, linewidth=linewidth
                )
                ax.add_patch(rect)
                ax.text(x_pos + token_width / 2, y_pos, token,
                        ha='center', va='center', fontsize=10, fontweight='bold')
                x_pos += token_width + 0.1

            y_pos -= 1.0

    # Show token count comparison
    initial_tokens = sum(len(word) + 1 for word in words_to_show)  # +1 for end marker
    current_tokens = sum(
        len(tokens) for tokens in step_data["word_freqs"].keys()
        for word in words_to_show
        if ''.join(tokens).replace('_', '') == word.lower()
    )

    ax.text(5, 0.5, f"Total tokens: {initial_tokens} → {current_tokens} "
            f"({100 * (initial_tokens - current_tokens) / initial_tokens:.0f}% reduction)",
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))


def create_bpe_animation():
    """Create the BPE animation GIF."""
    # Sample corpus
    corpus = [
        "the cat sat on the mat",
        "the cat ate the rat",
        "the rat sat on the cat",
        "the dog ran to the cat",
    ] * 10  # Repeat for frequency

    words_to_show = ["the", "cat", "sat", "ate"]

    # Run BPE
    steps = run_bpe(corpus, num_merges=10)

    # Create animation
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')

    def animate(frame):
        step_data = steps[min(frame, len(steps) - 1)]
        visualize_step(ax, step_data, words_to_show)

    # Create and save animation
    anim = FuncAnimation(
        fig, animate,
        frames=len(steps) + 3,  # Extra frames to pause at end
        interval=1500,  # 1.5 seconds per frame
        repeat=True
    )

    output_path = os.path.join(OUTPUT_DIR, "bpe_merge_animation.gif")
    anim.save(output_path, writer=PillowWriter(fps=1))
    plt.close()

    print(f"Saved BPE animation to: {output_path}")

    # Also create a static summary image
    create_static_summary(steps, words_to_show)


def create_static_summary(steps: List[dict], words_to_show: List[str]):
    """Create a static summary image showing BPE progression."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("BPE Merge Process", fontsize=16, fontweight='bold')

    key_steps = [0, 2, 4, 6, 8, len(steps) - 1]

    for ax, step_idx in zip(axes.flat, key_steps):
        if step_idx < len(steps):
            visualize_step(ax, steps[step_idx], words_to_show)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "bpe_process_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved BPE summary to: {output_path}")


if __name__ == "__main__":
    create_bpe_animation()
