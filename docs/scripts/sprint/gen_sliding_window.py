"""
Generate an animated GIF visualizing the Sliding Window technique
on the Minimum Window Substring problem.

Input:  s = "ADOBECODEBANC", t = "ABC"
Output: sliding_window.gif
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from collections import Counter
from pathlib import Path
from PIL import Image
import io
import os

# --------------- configuration ---------------
S = "ADOBECODEBANC"
T = "ABC"
BG = "#1e1e2e"
BOX_DEFAULT = "#45475a"
BOX_WINDOW = "#89b4fa"       # blue for current window
BOX_EXPAND = "#a6e3a1"       # green when right pointer expands
BOX_SHRINK = "#f38ba8"       # red when left pointer shrinks
BOX_BEST = "#f9e2af"         # gold for best window highlight
TEXT_COLOR = "#cdd6f4"
TITLE_COLOR = "#cba6f7"
ARROW_LEFT = "#f38ba8"
ARROW_RIGHT = "#a6e3a1"
TARGET_HIT = "#a6e3a1"
TARGET_MISS = "#f38ba8"
FIG_W, FIG_H = 14, 5
DPI = 100
FRAME_DURATION = 500  # ms

OUT_PATH = Path(__file__).resolve().parent.parent / "public" / "viz" / "sliding_window.gif"

# --------------- algorithm: record frames ---------------
# Each frame is a dict describing the visual state.
frames = []

def snap(left, right, phase, need, have, window_counts, best, msg,
         highlight_right=False, highlight_left_remove=False):
    """Record one frame of algorithm state."""
    frames.append(dict(
        left=left, right=right, phase=phase,
        need=dict(need), have=have,
        window_counts=dict(window_counts),
        best=tuple(best) if best[2] != float("inf") else None,
        msg=msg,
        highlight_right=highlight_right,
        highlight_left_remove=highlight_left_remove,
    ))

def run_algorithm():
    need = Counter(T)
    required = len(need)
    have = 0
    window_counts = Counter()
    left = 0
    best = [0, 0, float("inf")]  # left, right, length

    # Initial state
    snap(left, -1, "init", need, have, window_counts, best,
         "Start: find minimum window containing all of \"ABC\"")

    for right in range(len(S)):
        ch = S[right]
        window_counts[ch] += 1

        if ch in need and window_counts[ch] == need[ch]:
            have += 1

        # Frame: expand right
        valid = (have == required)
        snap(left, right, "expand", need, have, window_counts, best,
             f"Expand right -> '{ch}' (index {right})",
             highlight_right=True)

        # Shrink from left while valid
        while have == required:
            # record current valid window
            length = right - left + 1
            if length < best[2]:
                best = [left, right, length]

            snap(left, right, "valid", need, have, window_counts, best,
                 f"Valid window \"{S[left:right+1]}\" (len={length})",
                 highlight_right=False)

            # shrink
            rem = S[left]
            window_counts[rem] -= 1
            if window_counts[rem] == 0:
                del window_counts[rem]
            if rem in need and window_counts.get(rem, 0) < need[rem]:
                have -= 1

            snap(left, right, "shrink", need, have, window_counts, best,
                 f"Shrink left: remove '{rem}' (index {left})",
                 highlight_left_remove=True)
            left += 1

    # Final frame
    if best[2] != float("inf"):
        snap(best[0], best[1], "done", need, have, window_counts, best,
             f"Answer: \"{S[best[0]:best[1]+1]}\" (len={best[2]})")
    else:
        snap(0, len(S)-1, "done", need, 0, window_counts, best,
             "No valid window found")

run_algorithm()

# --------------- rendering ---------------

def render_frame(f, idx):
    """Render a single frame dict to a PIL Image."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n = len(S)
    box_w = 0.72
    box_h = 0.72
    gap = 0.08
    total_w = n * (box_w + gap) - gap
    x_start = (14 - total_w) / 2  # centre horizontally (axis coords)

    # We'll work in axis data coordinates; set limits
    ax.set_xlim(0, 14)
    ax.set_ylim(-1.0, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    left = f["left"]
    right = f["right"]
    best = f["best"]
    phase = f["phase"]

    # --- draw boxes ---
    y_box = 1.8
    for i, ch in enumerate(S):
        x = x_start + i * (box_w + gap)

        # Determine box colour
        if phase == "done" and best and best[0] <= i <= best[1]:
            color = BOX_BEST
        elif f["highlight_left_remove"] and i == left:
            color = BOX_SHRINK
        elif f["highlight_right"] and i == right:
            color = BOX_EXPAND
        elif 0 <= left and left <= i <= right:
            color = BOX_WINDOW
        else:
            color = BOX_DEFAULT

        rect = patches.FancyBboxPatch(
            (x, y_box), box_w, box_h,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#585b70", linewidth=1.5
        )
        ax.add_patch(rect)

        # letter
        txt_color = "#1e1e2e" if color in (BOX_EXPAND, BOX_BEST, BOX_SHRINK, BOX_WINDOW) else TEXT_COLOR
        ax.text(x + box_w / 2, y_box + box_h / 2, ch,
                ha="center", va="center", fontsize=15, fontweight="bold",
                color=txt_color, family="monospace")

        # index below box
        ax.text(x + box_w / 2, y_box - 0.22, str(i),
                ha="center", va="center", fontsize=8, color="#6c7086", family="monospace")

    # --- draw pointers (arrows) ---
    arrow_y = y_box + box_h + 0.18
    label_y = arrow_y + 0.38

    if left >= 0 and left < n:
        lx = x_start + left * (box_w + gap) + box_w * 0.35
        ax.annotate("", xy=(lx, y_box + box_h + 0.02), xytext=(lx, arrow_y + 0.15),
                     arrowprops=dict(arrowstyle="-|>", color=ARROW_LEFT, lw=2.2))
        ax.text(lx, label_y + 0.12, "L", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=ARROW_LEFT, family="monospace")

    if right >= 0 and right < n:
        rx = x_start + right * (box_w + gap) + box_w * 0.65
        ax.annotate("", xy=(rx, y_box + box_h + 0.02), xytext=(rx, arrow_y + 0.15),
                     arrowprops=dict(arrowstyle="-|>", color=ARROW_RIGHT, lw=2.2))
        ax.text(rx, label_y + 0.12, "R", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=ARROW_RIGHT, family="monospace")

    # --- title ---
    ax.text(7, 4.2, "Sliding Window  --  Minimum Window Substring",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=TITLE_COLOR, family="monospace")

    # --- info panel (bottom) ---
    info_y = 0.95

    # Current window string
    if left >= 0 and right >= 0 and left <= right:
        win_str = S[left:right+1]
    else:
        win_str = "--"
    ax.text(0.4, info_y, f"Window: \"{win_str}\"  (len={len(win_str) if win_str != '--' else 0})",
            fontsize=12, color=TEXT_COLOR, family="monospace", va="center")

    # Best window
    if best:
        best_str = S[best[0]:best[1]+1]
        ax.text(0.4, info_y - 0.50, f"Best:   \"{best_str}\"  (len={best[2]})",
                fontsize=12, color=BOX_BEST, family="monospace", va="center")
    else:
        ax.text(0.4, info_y - 0.50, "Best:   --",
                fontsize=12, color="#6c7086", family="monospace", va="center")

    # Characters needed display
    target_x = 7.5
    ax.text(target_x, info_y, "Target chars:", fontsize=11,
            color=TEXT_COLOR, family="monospace", va="center")
    need_full = Counter(T)
    wc = f["window_counts"]
    for j, tch in enumerate(sorted(need_full.keys())):
        cx = target_x + 1.8 + j * 1.2
        got = wc.get(tch, 0)
        req = need_full[tch]
        clr = TARGET_HIT if got >= req else TARGET_MISS
        ax.text(cx, info_y, f"{tch}:{got}/{req}", fontsize=12,
                color=clr, family="monospace", fontweight="bold", va="center")

    # Phase / message
    ax.text(7, -0.55, f"[{idx+1}/{len(frames)}]  {f['msg']}",
            ha="center", va="center", fontsize=11, color="#a6adc8",
            family="monospace", style="italic")

    plt.tight_layout(pad=0.3)

    # Convert figure to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# --------------- build GIF ---------------
def build_gif():
    print(f"Rendering {len(frames)} frames ...")
    images = []
    for i, f in enumerate(frames):
        img = render_frame(f, i)
        images.append(img)
        if (i + 1) % 5 == 0 or i == len(frames) - 1:
            print(f"  rendered frame {i+1}/{len(frames)}")

    # Make the last frame linger
    durations = [FRAME_DURATION] * len(images)
    durations[-1] = 2500  # hold final answer

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        str(OUT_PATH),
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Saved: {OUT_PATH}  ({size_kb:.0f} KB, {len(images)} frames)")


if __name__ == "__main__":
    build_gif()
