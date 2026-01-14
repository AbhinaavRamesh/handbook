import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(5, 1, figsize=(12, 10))
fig.suptitle('Move Zeros: Two Pointer Algorithm', fontsize=16, fontweight='bold')

states = [
    ([0, 1, 0, 3, 12], 0, 0, "Initial: read_ptr=0, write_ptr=0"),
    ([0, 1, 0, 3, 12], 1, 0, "read_ptr=1 finds 1, swap positions 0,1"),
    ([1, 0, 0, 3, 12], 2, 1, "After swap, write_ptr=1"),
    ([1, 3, 0, 0, 12], 4, 2, "read_ptr=3 finds 3, swapped to pos 1"),
    ([1, 3, 12, 0, 0], 5, 3, "Final: All non-zeros moved to front"),
]

for ax, (arr, read_ptr, write_ptr, desc) in zip(axes, states):
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    ax.set_title(desc, fontsize=11, loc='left')

    for i, val in enumerate(arr):
        color = '#ffcccc' if val == 0 else '#ccffcc'
        rect = patches.FancyBboxPatch((i, 0), 0.8, 0.8,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i + 0.4, 0.4, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(i + 0.4, -0.3, str(i), ha='center', va='center', fontsize=9, color='gray')

    if write_ptr < len(arr):
        ax.annotate('W', xy=(write_ptr + 0.4, 0), xytext=(write_ptr + 0.4, -0.6),
                   fontsize=10, ha='center', color='blue', fontweight='bold')
    if read_ptr < len(arr):
        ax.annotate('R', xy=(read_ptr + 0.4, 0.8), xytext=(read_ptr + 0.4, 1.2),
                   fontsize=10, ha='center', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/arrays/assets/move-zeros.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Generated move-zeros.png")
