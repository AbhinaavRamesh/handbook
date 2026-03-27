"""
Generate two visualizations for the Google L4 Interview Crash Course.

1. Tier Priority Distribution — horizontal bar chart with problem counts
   and Google L4 frequency overlay.
2. Problem Difficulty Map — scatter/bubble chart placing all 32 problems
   by difficulty vs. interview frequency.

Outputs:
    ../public/viz/tier_distribution.png
    ../public/viz/problem_difficulty_map.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR.parent / "public" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

OUT_TIER = VIZ_DIR / "tier_distribution.png"
OUT_MAP = VIZ_DIR / "problem_difficulty_map.png"

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG_COLOR = "#1e1e2e"
SURFACE_COLOR = "#2a2a3d"
TEXT_COLOR = "#cdd6f4"
GRID_COLOR = "#45475a"
MUTED_TEXT = "#a6adc8"

# Tier colours (same palette used in both charts)
TIER_COLORS = {
    "Tier 1: Graphs":          "#ef4444",  # bright red
    "Tier 2: Trees":           "#f97316",  # orange
    "Tier 3: Sliding Window":  "#eab308",  # yellow
    "Tier 4: DP":              "#22c55e",  # green
    "Tier 5: HashMap/Heap/BS": "#3b82f6",  # blue
    "Tier 6: Strings":         "#a855f7",  # purple
}

TIER_SHORT = {
    "Tier 1: Graphs":          "#ef4444",
    "Tier 2: Trees":           "#f97316",
    "Tier 3: Sliding Window":  "#eab308",
    "Tier 4: DP":              "#22c55e",
    "Tier 5: HashMap/Heap/BS": "#3b82f6",
    "Tier 6: Strings":         "#a855f7",
}

# ---------------------------------------------------------------------------
# Data — Tier Priority Distribution
# ---------------------------------------------------------------------------
tiers = [
    "Tier 6: Strings",
    "Tier 5: HashMap/Heap/BS",
    "Tier 4: DP",
    "Tier 3: Sliding Window",
    "Tier 2: Trees",
    "Tier 1: Graphs",
]
problem_counts = [3, 6, 5, 4, 6, 8]
google_freq = [30, 45, 50, 60, 75, 90]  # percentages
bar_colors = [TIER_COLORS[t] for t in tiers]

# ---------------------------------------------------------------------------
# Chart 1 — Tier Priority Distribution
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 7), dpi=150)
fig1.patch.set_facecolor(BG_COLOR)
ax1.set_facecolor(SURFACE_COLOR)

y_pos = np.arange(len(tiers))
bar_height = 0.38

# Google L4 frequency bars (behind)
freq_bars = ax1.barh(
    y_pos + bar_height / 2,
    google_freq,
    height=bar_height,
    color=[c + "55" for c in bar_colors],  # translucent version
    edgecolor=[c + "99" for c in bar_colors],
    linewidth=0.8,
    label="Google L4 Frequency %",
)

# Problem count bars (front, scaled to max=100 for visual alignment)
max_problems = max(problem_counts)
scaled_counts = [c / max_problems * 100 for c in problem_counts]
count_bars = ax1.barh(
    y_pos - bar_height / 2,
    scaled_counts,
    height=bar_height,
    color=bar_colors,
    edgecolor="none",
    label="Problem Count (scaled)",
)

# Labels — problem counts inside the count bars
for i, (bar, count) in enumerate(zip(count_bars, problem_counts)):
    ax1.text(
        bar.get_width() + 2,
        bar.get_y() + bar.get_height() / 2,
        f"{count} problems",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=bar_colors[i],
        fontfamily="sans-serif",
    )

# Labels — frequency percentages inside the freq bars
for i, (bar, freq) in enumerate(zip(freq_bars, google_freq)):
    ax1.text(
        bar.get_width() - 2,
        bar.get_y() + bar.get_height() / 2,
        f"{freq}%",
        va="center",
        ha="right",
        fontsize=10,
        fontweight="bold",
        color=TEXT_COLOR,
        fontfamily="sans-serif",
    )

# "Study these first" arrow annotation
ax1.annotate(
    "",
    xy=(95, 5.6),
    xytext=(95, 0.4),
    arrowprops=dict(
        arrowstyle="->,head_width=0.5,head_length=0.3",
        color="#ef4444",
        lw=2.5,
        connectionstyle="arc3,rad=0",
    ),
)
ax1.text(
    97, 3.0,
    "Study\nthese\nfirst",
    fontsize=11,
    fontweight="bold",
    color="#ef4444",
    ha="center",
    va="center",
    fontfamily="sans-serif",
    rotation=0,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="#ef444422",
        edgecolor="#ef444466",
    ),
)

# Axes styling
ax1.set_yticks(y_pos)
ax1.set_yticklabels(tiers, fontsize=11, color=TEXT_COLOR, fontfamily="sans-serif",
                     fontweight="semibold")
ax1.set_xlim(0, 108)
ax1.set_xlabel("Relative Scale  (problem count scaled  |  Google L4 frequency %)",
               fontsize=11, color=MUTED_TEXT, fontfamily="sans-serif", labelpad=10)
ax1.set_title(
    "Problem Tier Priority \u2014 Where to Focus",
    fontsize=20,
    fontweight="bold",
    color=TEXT_COLOR,
    fontfamily="sans-serif",
    pad=18,
)

ax1.tick_params(axis="both", colors=TEXT_COLOR, labelsize=10)
ax1.grid(True, axis="x", linestyle="--", linewidth=0.4, color=GRID_COLOR, alpha=0.5)
for spine in ax1.spines.values():
    spine.set_color(GRID_COLOR)

# Legend
legend_elements = [
    mpatches.Patch(facecolor="#3b82f6", edgecolor="none", label="Problem Count"),
    mpatches.Patch(facecolor="#3b82f655", edgecolor="#3b82f699", label="Google L4 Frequency %"),
]
legend1 = ax1.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=10,
    frameon=True,
    facecolor=SURFACE_COLOR,
    edgecolor=GRID_COLOR,
    labelcolor=TEXT_COLOR,
    framealpha=0.9,
)
legend1.get_frame().set_linewidth(0.8)

fig1.tight_layout()
fig1.savefig(OUT_TIER, facecolor=fig1.get_facecolor(), edgecolor="none")
plt.close(fig1)
print(f"[1/2] Saved: {OUT_TIER}")


# =========================================================================
# Chart 2 — Problem Difficulty Map (Scatter / Bubble)
# =========================================================================

# Difficulty mapping: Easy=1, Medium=2, Hard=3 (with jitter for readability)
# Frequency: 1=Low, 2=Medium-Low, 3=Medium, 4=Medium-High, 5=High
# Size: relative time to spend (minutes)

problems = [
    # (name,                      difficulty, frequency, time, tier)
    # --- Tier 1: Graphs ---
    ("Num Islands",               2.0,  5.0, 30, "Tier 1: Graphs"),
    ("Course Schedule",           2.1,  4.8, 35, "Tier 1: Graphs"),
    ("Clone Graph",               2.0,  3.8, 25, "Tier 1: Graphs"),
    ("Pacific Atlantic",          2.2,  3.5, 30, "Tier 1: Graphs"),
    ("Word Ladder",               3.0,  4.5, 40, "Tier 1: Graphs"),
    ("Graph Valid Tree",          2.1,  3.2, 25, "Tier 1: Graphs"),
    ("Alien Dictionary",          3.1,  3.8, 40, "Tier 1: Graphs"),
    ("Network Delay",             2.3,  3.0, 30, "Tier 1: Graphs"),
    # --- Tier 2: Trees ---
    ("LCA BST",                   2.0,  4.7, 25, "Tier 2: Trees"),
    ("BT Max Path",               3.0,  4.6, 40, "Tier 2: Trees"),
    ("Validate BST",              1.2,  2.5, 15, "Tier 2: Trees"),
    ("Diameter BT",               1.3,  2.3, 15, "Tier 2: Trees"),
    ("Serialize BT",              3.1,  3.5, 35, "Tier 2: Trees"),
    ("Impl. Trie",                2.4,  2.0, 30, "Tier 2: Trees"),
    # --- Tier 3: Sliding Window ---
    ("Min Window Sub",            3.0,  4.4, 40, "Tier 3: Sliding Window"),
    ("Longest No Repeat",         2.0,  3.8, 25, "Tier 3: Sliding Window"),
    ("Sliding Win Max",           3.1,  3.2, 35, "Tier 3: Sliding Window"),
    ("Perm in String",            2.1,  3.0, 25, "Tier 3: Sliding Window"),
    # --- Tier 4: DP ---
    ("Coin Change",               2.1,  3.5, 30, "Tier 4: DP"),
    ("House Robber",              2.0,  3.3, 25, "Tier 4: DP"),
    ("LIS",                       2.5,  3.0, 30, "Tier 4: DP"),
    ("Decode Ways",               2.2,  2.8, 25, "Tier 4: DP"),
    ("Word Break",                2.4,  3.2, 30, "Tier 4: DP"),
    # --- Tier 5: HashMap/Heap/BS ---
    ("Group Anagrams",            2.0,  3.5, 20, "Tier 5: HashMap/Heap/BS"),
    ("Top K Frequent",            2.1,  3.4, 20, "Tier 5: HashMap/Heap/BS"),
    ("Merge K Sorted",            3.0,  3.2, 35, "Tier 5: HashMap/Heap/BS"),
    ("Find Peak",                 1.2,  2.2, 15, "Tier 5: HashMap/Heap/BS"),
    ("Task Scheduler",            2.3,  2.8, 25, "Tier 5: HashMap/Heap/BS"),
    ("Median Sorted",             3.2,  2.5, 35, "Tier 5: HashMap/Heap/BS"),
    # --- Tier 6: Strings ---
    ("Expressive Words",          2.5,  2.0, 25, "Tier 6: Strings"),
    ("License Key Fmt",           1.1,  1.8, 10, "Tier 6: Strings"),
    ("Longest Palindrome",        2.2,  2.5, 25, "Tier 6: Strings"),
]

fig2, ax2 = plt.subplots(figsize=(14, 10), dpi=150)
fig2.patch.set_facecolor(BG_COLOR)
ax2.set_facecolor(SURFACE_COLOR)

# Draw quadrant dividers
ax2.axvline(x=2.15, color=GRID_COLOR, linestyle="--", linewidth=1, alpha=0.5)
ax2.axhline(y=3.1, color=GRID_COLOR, linestyle="--", linewidth=1, alpha=0.5)

# Quadrant labels
quad_style = dict(fontsize=14, fontweight="bold", fontfamily="sans-serif", alpha=0.18)
ax2.text(1.35, 4.8, "Quick Wins",  color="#22c55e", ha="center", va="center", **quad_style)
ax2.text(3.0,  4.8, "Must Know",   color="#ef4444", ha="center", va="center", **quad_style)
ax2.text(1.35, 1.9, "Nice to Have", color="#a6adc8", ha="center", va="center", **quad_style)
ax2.text(3.0,  1.9, "High Value",  color="#f97316", ha="center", va="center", **quad_style)

# Plot each problem as a bubble
for name, diff, freq, time_min, tier in problems:
    color = TIER_COLORS[tier]
    size = time_min * 8  # scale for visual

    ax2.scatter(
        diff, freq,
        s=size,
        c=color,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    # Label offset to reduce overlap — nudge based on position
    dx = 0.06
    dy = 0.12
    ax2.text(
        diff + dx,
        freq + dy,
        name,
        fontsize=6.5,
        color=TEXT_COLOR,
        fontfamily="sans-serif",
        fontweight="medium",
        alpha=0.92,
        zorder=4,
    )

# Axes
ax2.set_xlim(0.7, 3.6)
ax2.set_ylim(1.4, 5.5)
ax2.set_xticks([1.0, 2.0, 3.0])
ax2.set_xticklabels(["Easy", "Medium", "Hard"], fontsize=12, color=TEXT_COLOR,
                     fontfamily="sans-serif")
ax2.set_ylabel("Google L4 Frequency", fontsize=13, color=TEXT_COLOR,
               fontfamily="sans-serif", labelpad=10)
ax2.set_xlabel("Difficulty", fontsize=13, color=TEXT_COLOR,
               fontfamily="sans-serif", labelpad=10)

# Y-axis custom labels
ax2.set_yticks([1.5, 2.5, 3.5, 4.5, 5.0])
ax2.set_yticklabels(["Low", "Med-Low", "Medium", "Med-High", "High"],
                     fontsize=10, color=TEXT_COLOR, fontfamily="sans-serif")

ax2.set_title(
    "The 32-Problem Map \u2014 Difficulty vs Frequency",
    fontsize=20,
    fontweight="bold",
    color=TEXT_COLOR,
    fontfamily="sans-serif",
    pad=18,
)

ax2.tick_params(axis="both", colors=TEXT_COLOR, labelsize=10)
ax2.grid(True, linestyle=":", linewidth=0.4, color=GRID_COLOR, alpha=0.4)
for spine in ax2.spines.values():
    spine.set_color(GRID_COLOR)

# Legend for tiers
tier_handles = [
    plt.scatter([], [], s=80, c=TIER_COLORS[t], edgecolors="white", linewidths=0.5,
                label=t)
    for t in TIER_COLORS
]
# Bubble size legend
size_handles = [
    plt.scatter([], [], s=10 * 8, c="#888888", alpha=0.5, edgecolors="white",
                linewidths=0.5, label="10 min"),
    plt.scatter([], [], s=25 * 8, c="#888888", alpha=0.5, edgecolors="white",
                linewidths=0.5, label="25 min"),
    plt.scatter([], [], s=40 * 8, c="#888888", alpha=0.5, edgecolors="white",
                linewidths=0.5, label="40 min"),
]

legend2a = ax2.legend(
    handles=tier_handles,
    loc="upper left",
    fontsize=8.5,
    frameon=True,
    facecolor=SURFACE_COLOR,
    edgecolor=GRID_COLOR,
    labelcolor=TEXT_COLOR,
    framealpha=0.92,
    title="Tier",
    title_fontsize=9,
)
legend2a.get_title().set_color(TEXT_COLOR)
legend2a.get_frame().set_linewidth(0.8)
ax2.add_artist(legend2a)

legend2b = ax2.legend(
    handles=size_handles,
    loc="lower left",
    fontsize=8.5,
    frameon=True,
    facecolor=SURFACE_COLOR,
    edgecolor=GRID_COLOR,
    labelcolor=TEXT_COLOR,
    framealpha=0.92,
    title="Time to Spend",
    title_fontsize=9,
)
legend2b.get_title().set_color(TEXT_COLOR)
legend2b.get_frame().set_linewidth(0.8)

fig2.tight_layout()
fig2.savefig(OUT_MAP, facecolor=fig2.get_facecolor(), edgecolor="none")
plt.close(fig2)
print(f"[2/2] Saved: {OUT_MAP}")

print("\nDone — both visualizations generated.")
