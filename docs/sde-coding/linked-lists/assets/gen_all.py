import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# 1. Fast-Slow Pointers
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
NODE_COLOR = '#BBDEFB'
SLOW_COLOR = '#4CAF50'
FAST_COLOR = '#F44336'
MIDDLE_COLOR = '#FFC107'

ax1 = axes[0, 0]
ax1.set_xlim(-1, 14)
ax1.set_ylim(-1, 4)
ax1.axis('off')
ax1.set_title('Finding Middle of Linked List\nFast moves 2x, Slow finds middle', fontsize=12, fontweight='bold')

values = [1, 2, 3, 4, 5]
for i, val in enumerate(values):
    x = i * 2.2
    color = MIDDLE_COLOR if i == 2 else NODE_COLOR
    rect = FancyBboxPatch((x, 1), 1.5, 1, boxstyle='round,pad=0.05,rounding_size=0.15',
                          facecolor=color, edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x + 0.75, 1.5, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
    if i < len(values) - 1:
        arrow = FancyArrowPatch((x + 1.5, 1.5), (x + 2.2, 1.5), arrowstyle='->', mutation_scale=15, color='black', linewidth=2)
        ax1.add_patch(arrow)

ax1.text(11.3, 1.5, 'NULL', fontsize=10, fontweight='bold', color='gray')
ax1.annotate('slow', xy=(4.4 + 0.75, 2), xytext=(4.4 + 0.75, 3.2), ha='center', fontsize=11, fontweight='bold', color=SLOW_COLOR, arrowprops=dict(arrowstyle='->', color=SLOW_COLOR, lw=2))
ax1.annotate('fast (NULL)', xy=(10.5, 1.5), xytext=(10.5, 3), ha='center', fontsize=11, fontweight='bold', color=FAST_COLOR, arrowprops=dict(arrowstyle='->', color=FAST_COLOR, lw=2))
ax1.text(5.5, 0.3, 'Middle Found! (node 3)', fontsize=10, fontweight='bold', color=MIDDLE_COLOR)

ax2 = axes[0, 1]
ax2.set_xlim(-1, 14)
ax2.set_ylim(-1, 4)
ax2.axis('off')
ax2.set_title('Remove Nth Node from End (N=2)\nTwo pointers with N gap', fontsize=12, fontweight='bold')

for i, val in enumerate(values):
    x = i * 2.2
    color = '#FFCDD2' if i == 3 else NODE_COLOR
    rect = FancyBboxPatch((x, 1), 1.5, 1, boxstyle='round,pad=0.05,rounding_size=0.15', facecolor=color, edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(x + 0.75, 1.5, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
    if i < len(values) - 1:
        style = '--' if i == 2 else '-'
        arrow = FancyArrowPatch((x + 1.5, 1.5), (x + 2.2, 1.5), arrowstyle='->', mutation_scale=15, color='black', linewidth=2, linestyle=style)
        ax2.add_patch(arrow)

ax2.annotate('slow (before target)', xy=(4.4 + 0.75, 2), xytext=(4.4 + 0.75, 3.2), ha='center', fontsize=10, fontweight='bold', color=SLOW_COLOR, arrowprops=dict(arrowstyle='->', color=SLOW_COLOR, lw=2))
ax2.annotate('fast (end)', xy=(8.8 + 0.75, 2), xytext=(8.8 + 0.75, 3.2), ha='center', fontsize=10, fontweight='bold', color=FAST_COLOR, arrowprops=dict(arrowstyle='->', color=FAST_COLOR, lw=2))
ax2.text(5, 0.3, 'Gap = N = 2 nodes', fontsize=10, fontweight='bold', color='#666')

ax3 = axes[1, 0]
ax3.set_xlim(-1, 14)
ax3.set_ylim(-2, 4)
ax3.axis('off')
ax3.set_title('Palindrome Check: 1->2->2->1\nFind middle, reverse second half, compare', fontsize=12, fontweight='bold')

palindrome = [1, 2, 2, 1]
ax3.text(0, 3, 'First Half:', fontsize=10, fontweight='bold')
for i in range(2):
    x = i * 2.2
    rect = FancyBboxPatch((x, 2), 1.5, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#C8E6C9', edgecolor='black', linewidth=2)
    ax3.add_patch(rect)
    ax3.text(x + 0.75, 2.4, str(palindrome[i]), ha='center', va='center', fontsize=12, fontweight='bold')
    if i < 1:
        arrow = FancyArrowPatch((x + 1.5, 2.4), (x + 2.2, 2.4), arrowstyle='->', mutation_scale=12, color='black', linewidth=1.5)
        ax3.add_patch(arrow)

ax3.text(6, 3, 'Second Half (reversed):', fontsize=10, fontweight='bold')
for i in range(2):
    x = 6 + i * 2.2
    rect = FancyBboxPatch((x, 2), 1.5, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#BBDEFB', edgecolor='black', linewidth=2)
    ax3.add_patch(rect)
    ax3.text(x + 0.75, 2.4, str(palindrome[3-i]), ha='center', va='center', fontsize=12, fontweight='bold')
    if i < 1:
        arrow = FancyArrowPatch((x + 1.5, 2.4), (x + 2.2, 2.4), arrowstyle='->', mutation_scale=12, color='black', linewidth=1.5)
        ax3.add_patch(arrow)

ax3.text(5, -0.5, 'All match - It is a Palindrome!', ha='center', fontsize=12, fontweight='bold', color='green')

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Fast-Slow Pointer Patterns', fontsize=14, fontweight='bold')

patterns = [('Find Middle', 'slow=1x, fast=2x. When fast ends, slow is at middle'), ('Detect Cycle', 'If cycle exists, fast catches slow'), ('Kth from End', 'Move fast K steps ahead then move both'), ('Palindrome', 'Find middle, Reverse half, Compare')]
y = 9
for title, desc in patterns:
    rect = Rectangle((0.5, y - 1.8), 9, 2, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax4.add_patch(rect)
    ax4.text(5, y - 0.5, title, ha='center', va='center', fontsize=12, fontweight='bold', color='#1565C0')
    ax4.text(5, y - 1.2, desc, ha='center', va='center', fontsize=9)
    y -= 2.3

plt.tight_layout()
plt.savefig('fast_slow_pointers.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created fast_slow_pointers.png')

# 2. Merge Operations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
NODE_COLOR = '#BBDEFB'
NODE_COLOR_2 = '#C8E6C9'

ax1 = axes[0, 0]
ax1.set_xlim(-1, 16)
ax1.set_ylim(-1, 6)
ax1.axis('off')
ax1.set_title('Merge Two Sorted Lists\nCompare heads, pick smaller', fontsize=12, fontweight='bold')

ax1.text(0, 5, 'List 1:', fontsize=10, fontweight='bold', color='#1565C0')
l1 = [1, 3, 5]
for i, val in enumerate(l1):
    x = i * 2
    rect = FancyBboxPatch((x, 4), 1.3, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=NODE_COLOR, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.65, 4.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.3, 4.4), (x + 2, 4.4), arrowstyle='->', mutation_scale=12, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 3.2, 'List 2:', fontsize=10, fontweight='bold', color='#388E3C')
l2 = [2, 4, 6]
for i, val in enumerate(l2):
    x = i * 2
    rect = FancyBboxPatch((x, 2.2), 1.3, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=NODE_COLOR_2, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.65, 2.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.3, 2.6), (x + 2, 2.6), arrowstyle='->', mutation_scale=12, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 1.2, 'Merged:', fontsize=10, fontweight='bold', color='#E65100')
merged = [1, 2, 3, 4, 5, 6]
colors = [NODE_COLOR, NODE_COLOR_2, NODE_COLOR, NODE_COLOR_2, NODE_COLOR, NODE_COLOR_2]
for i, (val, col) in enumerate(zip(merged, colors)):
    x = i * 1.8
    rect = FancyBboxPatch((x, 0.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 0.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 5:
        arrow = FancyArrowPatch((x + 1.2, 0.6), (x + 1.8, 0.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax2 = axes[0, 1]
ax2.set_xlim(-1, 14)
ax2.set_ylim(-1, 6)
ax2.axis('off')
ax2.set_title('Merge K Sorted Lists\nMin-Heap for O(N log K)', fontsize=12, fontweight='bold')

lists = [[1, 5], [2, 6], [3, 7]]
list_colors = ['#FFCDD2', '#C8E6C9', '#BBDEFB']
for li, (lst, col) in enumerate(zip(lists, list_colors)):
    y = 5 - li * 1.2
    ax2.text(-0.5, y - 0.2, f'L{li+1}:', fontsize=9, fontweight='bold')
    for i, val in enumerate(lst):
        x = i * 1.5 + 0.5
        rect = FancyBboxPatch((x, y - 0.4), 1, 0.6, boxstyle='round,pad=0.03,rounding_size=0.08', facecolor=col, edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(x + 0.5, y - 0.1, str(val), ha='center', va='center', fontsize=10, fontweight='bold')

ax2.text(5, 5.5, 'Min-Heap (heads)', fontsize=10, fontweight='bold')
ax2.text(5, 0.5, 'Time: O(N log K), Space: O(K)', fontsize=9, color='gray')

ax3 = axes[1, 0]
ax3.set_xlim(-1, 14)
ax3.set_ylim(-1, 6)
ax3.axis('off')
ax3.set_title('Merge Algorithm Steps', fontsize=12, fontweight='bold')

steps = [('Step 1', '1 < 2', 'Pick 1, advance L1'), ('Step 2', '3 > 2', 'Pick 2, advance L2'), ('Step 3', '3 < 4', 'Pick 3, advance L1'), ('Step 4', '5 > 4', 'Pick 4, advance L2')]
y = 5
for step, compare, action in steps:
    rect = Rectangle((0, y - 0.8), 12, 1, facecolor='#FFF9C4', edgecolor='#FFA000', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(1.5, y - 0.3, step, fontsize=10, fontweight='bold', color='#E65100')
    ax3.text(4.5, y - 0.3, compare, fontsize=10, fontweight='bold')
    ax3.text(8, y - 0.3, action, fontsize=10, color='#666')
    y -= 1.3

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 8)
ax4.axis('off')
ax4.set_title('Merge Operations Complexity', fontsize=14, fontweight='bold')

headers = ['Operation', 'Time', 'Space']
rows = [['Merge 2 Lists', 'O(n+m)', 'O(1)'], ['Merge K (Heap)', 'O(N log K)', 'O(K)'], ['Merge K (D&C)', 'O(N log K)', 'O(log K)']]
col_widths = [4, 2.5, 2.5]
row_height = 1
start_y = 7

x = 0.5
for i, header in enumerate(headers):
    rect = Rectangle((x, start_y - row_height), col_widths[i], row_height, facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)
    ax4.add_patch(rect)
    ax4.text(x + col_widths[i]/2, start_y - row_height/2, header, ha='center', va='center', fontsize=11, fontweight='bold')
    x += col_widths[i]

for row_idx, row in enumerate(rows):
    x = 0.5
    y = start_y - (row_idx + 2) * row_height
    for col_idx, cell in enumerate(row):
        color = '#C8E6C9' if col_idx > 0 else 'white'
        rect = Rectangle((x, y), col_widths[col_idx], row_height, facecolor=color, edgecolor='black', linewidth=1)
        ax4.add_patch(rect)
        ax4.text(x + col_widths[col_idx]/2, y + row_height/2, cell, ha='center', va='center', fontsize=10)
        x += col_widths[col_idx]

plt.tight_layout()
plt.savefig('merge_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created merge_operations.png')

# 3. Arithmetic Operations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
NODE_COLOR = '#BBDEFB'
RESULT_COLOR = '#C8E6C9'

ax1 = axes[0, 0]
ax1.set_xlim(-1, 14)
ax1.set_ylim(-2, 6)
ax1.axis('off')
ax1.set_title('Add Two Numbers (LeetCode 2)\nNumbers stored in reverse order', fontsize=12, fontweight='bold')

ax1.text(0, 5.2, 'L1: 2->4->3 represents 342', fontsize=10, fontweight='bold', color='#1565C0')
l1 = [2, 4, 3]
for i, val in enumerate(l1):
    x = i * 2
    rect = FancyBboxPatch((x, 4.2), 1.3, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=NODE_COLOR, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.65, 4.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.3, 4.6), (x + 2, 4.6), arrowstyle='->', mutation_scale=12, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 3.4, 'L2: 5->6->4 represents 465', fontsize=10, fontweight='bold', color='#388E3C')
l2 = [5, 6, 4]
for i, val in enumerate(l2):
    x = i * 2
    rect = FancyBboxPatch((x, 2.4), 1.3, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#C8E6C9', edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.65, 2.8, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.3, 2.8), (x + 2, 2.8), arrowstyle='->', mutation_scale=12, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(7, 3.5, '+', fontsize=20, fontweight='bold')
ax1.plot([0, 7], [1.8, 1.8], 'k-', linewidth=2)

ax1.text(0, 1.2, 'Result: 7->0->8 represents 807', fontsize=10, fontweight='bold', color='#E65100')
result = [7, 0, 8]
for i, val in enumerate(result):
    x = i * 2
    rect = FancyBboxPatch((x, 0.2), 1.3, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=RESULT_COLOR, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.65, 0.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.3, 0.6), (x + 2, 0.6), arrowstyle='->', mutation_scale=12, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(7, 0.6, '342 + 465 = 807', fontsize=10, fontweight='bold', style='italic')

ax2 = axes[0, 1]
ax2.set_xlim(-1, 12)
ax2.set_ylim(-1, 6)
ax2.axis('off')
ax2.set_title('Carry Propagation Step by Step', fontsize=12, fontweight='bold')

steps = [('2 + 5 = 7', 'carry=0', '7'), ('4 + 6 = 10', 'carry=1', '0'), ('3 + 4 + 1 = 8', 'carry=0', '8')]
y = 5
for i, (calc, carry, result) in enumerate(steps):
    rect = Rectangle((0, y - 0.8), 10, 1, facecolor='#FFF9C4', edgecolor='#FFA000', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(0.5, y - 0.3, f'Step {i+1}:', fontsize=10, fontweight='bold', color='#E65100')
    ax2.text(3, y - 0.3, calc, fontsize=10, fontweight='bold')
    ax2.text(6.5, y - 0.3, carry, fontsize=10, color='#D32F2F', fontweight='bold')
    ax2.text(9, y - 0.3, f'digit={result}', fontsize=10, color='#388E3C', fontweight='bold')
    y -= 1.3

ax2.text(5, 0.8, 'carry = sum // 10', fontsize=10, ha='center', style='italic')
ax2.text(5, 0.3, 'digit = sum % 10', fontsize=10, ha='center', style='italic')

ax3 = axes[1, 0]
ax3.set_xlim(-1, 14)
ax3.set_ylim(-2, 6)
ax3.axis('off')
ax3.set_title('Add Two Numbers II (LeetCode 445)\nNumbers stored in normal order - Use Stacks', fontsize=12, fontweight='bold')

ax3.text(0, 5, 'L1: 7->2->4->3 = 7243', fontsize=10, fontweight='bold', color='#1565C0')
ax3.text(0, 4, 'L2: 5->6->4 = 564', fontsize=10, fontweight='bold', color='#388E3C')
ax3.text(0, 3, '7243 + 564 = 7807', fontsize=11, fontweight='bold', color='#E65100')
ax3.text(0, 2, 'Use stacks to process from right to left', fontsize=10)

ax3.text(7, 2, 'Result: 7->8->0->7', fontsize=10, fontweight='bold', color='#E65100')
result = [7, 8, 0, 7]
for i, val in enumerate(result):
    x = 7 + i * 1.5
    rect = FancyBboxPatch((x, 0.8), 1, 0.6, boxstyle='round,pad=0.03,rounding_size=0.08', facecolor=RESULT_COLOR, edgecolor='black', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(x + 0.5, 1.1, str(val), ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 3:
        arrow = FancyArrowPatch((x + 1, 1.1), (x + 1.5, 1.1), arrowstyle='->', mutation_scale=10, color='black')
        ax3.add_patch(arrow)

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 8)
ax4.axis('off')
ax4.set_title('Arithmetic Operations Summary', fontsize=14, fontweight='bold')

algorithms = [('Add Two Numbers', 'Reverse order, simple iteration', 'O(max(m,n))'), ('Add Two Numbers II', 'Normal order, use stacks', 'O(m+n)')]
y = 7
for name, approach, time in algorithms:
    rect = Rectangle((0.3, y - 1.5), 9.4, 1.8, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax4.add_patch(rect)
    ax4.text(5, y - 0.3, name, ha='center', va='center', fontsize=11, fontweight='bold', color='#1565C0')
    ax4.text(5, y - 0.8, approach, ha='center', va='center', fontsize=9)
    ax4.text(5, y - 1.2, f'Time: {time}', fontsize=9, ha='center', color='#388E3C')
    y -= 2.2

ax4.text(5, 2, 'Key: Handle carry properly, watch for different lengths', fontsize=9, ha='center', style='italic')

plt.tight_layout()
plt.savefig('arithmetic_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created arithmetic_operations.png')

# 4. Deletion Operations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.set_xlim(-1, 14)
ax1.set_ylim(-1, 5)
ax1.axis('off')
ax1.set_title('Remove Duplicates from Sorted List\n1->1->2->3->3 becomes 1->2->3', fontsize=12, fontweight='bold')

ax1.text(0, 4.2, 'Before:', fontsize=10, fontweight='bold')
before = [1, 1, 2, 3, 3]
for i, val in enumerate(before):
    x = i * 1.8
    color = '#FFCDD2' if (i == 1 or i == 4) else '#BBDEFB'
    rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=color, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 3.6), (x + 1.8, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 2, 'After:', fontsize=10, fontweight='bold')
after = [1, 2, 3]
for i, val in enumerate(after):
    x = i * 1.8
    rect = FancyBboxPatch((x, 1), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#C8E6C9', edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 1.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.2, 1.4), (x + 1.8, 1.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(7, 2.5, 'Skip nodes where\ncurr.val == curr.next.val', fontsize=10, style='italic')

ax2 = axes[0, 1]
ax2.set_xlim(-1, 14)
ax2.set_ylim(-1, 5)
ax2.axis('off')
ax2.set_title('Remove Duplicates II - Remove All Occurrences\n1->2->3->3->4->4->5 becomes 1->2->5', fontsize=12, fontweight='bold')

ax2.text(0, 4.2, 'Before:', fontsize=10, fontweight='bold')
before = [1, 2, 3, 3, 4, 4, 5]
for i, val in enumerate(before):
    x = i * 1.5
    color = '#FFCDD2' if val in [3, 4] else '#BBDEFB'
    rect = FancyBboxPatch((x, 3.2), 1, 0.8, boxstyle='round,pad=0.03,rounding_size=0.08', facecolor=color, edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(x + 0.5, 3.6, str(val), ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 6:
        arrow = FancyArrowPatch((x + 1, 3.6), (x + 1.5, 3.6), arrowstyle='->', mutation_scale=8, color='black', linewidth=1)
        ax2.add_patch(arrow)

ax2.text(0, 2, 'After:', fontsize=10, fontweight='bold')
after = [1, 2, 5]
for i, val in enumerate(after):
    x = i * 1.8
    rect = FancyBboxPatch((x, 1), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#C8E6C9', edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(x + 0.6, 1.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.2, 1.4), (x + 1.8, 1.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax2.add_patch(arrow)

ax2.text(7, 2.5, 'Use dummy node!\nSkip all duplicates', fontsize=10, style='italic')

ax3 = axes[1, 0]
ax3.set_xlim(-1, 14)
ax3.set_ylim(-1, 5)
ax3.axis('off')
ax3.set_title('Delete Node Without Head Reference\nDelete node with value 5', fontsize=12, fontweight='bold')

ax3.text(0, 4.2, 'Before:', fontsize=10, fontweight='bold')
before = [4, 5, 1, 9]
for i, val in enumerate(before):
    x = i * 2
    color = '#FFC107' if i == 1 else '#BBDEFB'
    rect = FancyBboxPatch((x, 3.2), 1.3, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=color, edgecolor='black', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(x + 0.65, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 3:
        arrow = FancyArrowPatch((x + 1.3, 3.6), (x + 2, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax3.add_patch(arrow)

ax3.text(0, 2, 'Trick: Copy next value, delete next node', fontsize=10, fontweight='bold')
ax3.text(0, 1.5, 'node.val = node.next.val', fontsize=10, color='#1565C0')
ax3.text(0, 1, 'node.next = node.next.next', fontsize=10, color='#1565C0')

ax3.text(0, 0.2, 'After:', fontsize=10, fontweight='bold')
after = [4, 1, 9]
for i, val in enumerate(after):
    x = i * 2
    rect = FancyBboxPatch((x, -0.8), 1.3, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#C8E6C9', edgecolor='black', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(x + 0.65, -0.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.3, -0.4), (x + 2, -0.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax3.add_patch(arrow)

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 8)
ax4.axis('off')
ax4.set_title('Deletion Operations Summary', fontsize=14, fontweight='bold')

ops = [('Remove Duplicates I', 'Keep one of each value', 'O(n)', 'O(1)'), ('Remove Duplicates II', 'Remove all duplicates', 'O(n)', 'O(1)'), ('Delete Without Head', 'Copy next value trick', 'O(1)', 'O(1)')]
y = 7
for name, desc, time, space in ops:
    rect = Rectangle((0.3, y - 1.5), 9.4, 1.8, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax4.add_patch(rect)
    ax4.text(5, y - 0.3, name, ha='center', va='center', fontsize=11, fontweight='bold', color='#1565C0')
    ax4.text(5, y - 0.8, desc, ha='center', va='center', fontsize=9)
    ax4.text(2.5, y - 1.2, f'Time: {time}', fontsize=9, color='#388E3C')
    ax4.text(7, y - 1.2, f'Space: {space}', fontsize=9, color='#E65100')
    y -= 2.2

plt.tight_layout()
plt.savefig('deletion_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created deletion_operations.png')

# 5. Intersection Operations
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

ax1 = axes[0]
ax1.set_xlim(-1, 16)
ax1.set_ylim(-1, 6)
ax1.axis('off')
ax1.set_title('Intersection of Two Linked Lists\nFind the node where two lists merge', fontsize=14, fontweight='bold')

# List A
ax1.text(0, 5, 'List A: 4->1->8->4->5', fontsize=10, fontweight='bold', color='#1565C0')
a_vals = [4, 1]
for i, val in enumerate(a_vals):
    x = i * 1.8
    rect = FancyBboxPatch((x, 4), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#BBDEFB', edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 4.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    arrow = FancyArrowPatch((x + 1.2, 4.4), (x + 1.8, 4.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
    ax1.add_patch(arrow)

# List B
ax1.text(0, 3, 'List B: 5->6->1->8->4->5', fontsize=10, fontweight='bold', color='#388E3C')
b_vals = [5, 6, 1]
for i, val in enumerate(b_vals):
    x = i * 1.8
    rect = FancyBboxPatch((x, 2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#C8E6C9', edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 2.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.2, 2.4), (x + 1.8, 2.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

# Intersection part
common = [8, 4, 5]
for i, val in enumerate(common):
    x = 6 + i * 1.8
    rect = FancyBboxPatch((x, 3), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#FFC107', edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 3.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.2, 3.4), (x + 1.8, 3.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

# Connecting arrows
arrow = FancyArrowPatch((3.6, 4.2), (6, 3.4), arrowstyle='->', mutation_scale=10, color='#1565C0', linewidth=1.5)
ax1.add_patch(arrow)
arrow = FancyArrowPatch((5.4, 2.6), (6, 3.2), arrowstyle='->', mutation_scale=10, color='#388E3C', linewidth=1.5)
ax1.add_patch(arrow)

ax1.text(7.8, 1.5, 'Intersection at node 8', fontsize=12, fontweight='bold', color='#E65100')

ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('Two-Pointer Algorithm', fontsize=14, fontweight='bold')

algo_text = '''Algorithm:
1. Calculate lengths of both lists
2. Align pointers by advancing longer list by difference
3. Move both pointers together until they meet

Alternative (elegant):
- pA traverses A then B
- pB traverses B then A
- They meet at intersection (or both reach NULL)

Time: O(m + n)   Space: O(1)'''

ax2.text(0.5, 5.5, algo_text, fontsize=11, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

plt.tight_layout()
plt.savefig('intersection_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created intersection_operations.png')

# 6. Advanced Operations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.set_xlim(-1, 14)
ax1.set_ylim(-2, 5)
ax1.axis('off')
ax1.set_title('Copy List with Random Pointer\nDeep copy preserving random pointers', fontsize=12, fontweight='bold')

# Original nodes
ax1.text(0, 4.2, 'Original:', fontsize=10, fontweight='bold')
vals = [7, 13, 11, 10, 1]
for i, val in enumerate(vals):
    x = i * 2.2
    rect = FancyBboxPatch((x, 3), 1.5, 0.9, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#BBDEFB', edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.75, 3.45, str(val), ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.5, 3.45), (x + 2.2, 3.45), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

# Random pointer example
ax1.annotate('', xy=(6.6 + 0.75, 3), xytext=(0 + 0.75, 3), arrowprops=dict(arrowstyle='->', color='red', lw=1.5, connectionstyle='arc3,rad=-0.3'))
ax1.text(3.5, 2.2, 'random', fontsize=9, color='red')

ax1.text(0, 1.5, 'Approaches:', fontsize=10, fontweight='bold')
ax1.text(0.5, 0.8, '1. Hash Map: O(n) space - map old to new nodes', fontsize=9)
ax1.text(0.5, 0.2, '2. Interleaving: O(1) space - weave new nodes between old', fontsize=9)
ax1.text(0.5, -0.4, '   A->A\'->B->B\'->C->C\' then separate', fontsize=9, color='#666')

ax2 = axes[0, 1]
ax2.set_xlim(-1, 12)
ax2.set_ylim(-2, 5)
ax2.axis('off')
ax2.set_title('Flatten Multilevel Doubly Linked List\nDFS-based flattening', fontsize=12, fontweight='bold')

# Level 1
ax2.text(0, 4.2, 'Level 1:', fontsize=9, color='#666')
l1 = [1, 2, 3, 4, 5, 6]
for i, val in enumerate(l1[:4]):
    x = i * 2
    rect = FancyBboxPatch((x, 3.2), 1.3, 0.7, boxstyle='round,pad=0.03,rounding_size=0.08', facecolor='#BBDEFB', edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(x + 0.65, 3.55, str(val), ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 3:
        arrow = FancyArrowPatch((x + 1.3, 3.55), (x + 2, 3.55), arrowstyle='<->', mutation_scale=8, color='black', linewidth=1)
        ax2.add_patch(arrow)

# Level 2 (child of node 3)
ax2.text(4, 2.4, 'Level 2:', fontsize=9, color='#666')
l2 = [7, 8, 9, 10]
for i, val in enumerate(l2[:3]):
    x = 4 + i * 1.5
    rect = FancyBboxPatch((x, 1.4), 1, 0.6, boxstyle='round,pad=0.03,rounding_size=0.08', facecolor='#C8E6C9', edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(x + 0.5, 1.7, str(val), ha='center', va='center', fontsize=9, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1, 1.7), (x + 1.5, 1.7), arrowstyle='<->', mutation_scale=6, color='black', linewidth=1)
        ax2.add_patch(arrow)

# Child pointer
arrow = FancyArrowPatch((4.65, 3.2), (4.5, 2), arrowstyle='->', mutation_scale=8, color='red', linewidth=1.5)
ax2.add_patch(arrow)
ax2.text(3.8, 2.6, 'child', fontsize=8, color='red')

ax2.text(0, 0.5, 'Result: 1<->2<->3<->7<->8<->9<->4<->5<->6', fontsize=9, fontweight='bold')
ax2.text(0, 0, 'Use recursion/stack to process children', fontsize=9, color='#666')

ax3 = axes[1, 0]
ax3.set_xlim(-1, 14)
ax3.set_ylim(-1, 5)
ax3.axis('off')
ax3.set_title('Odd-Even Linked List\nGroup odd-indexed then even-indexed nodes', fontsize=12, fontweight='bold')

ax3.text(0, 4.2, 'Before: 1->2->3->4->5', fontsize=10, fontweight='bold')
before = [1, 2, 3, 4, 5]
for i, val in enumerate(before):
    x = i * 1.8
    color = '#BBDEFB' if i % 2 == 0 else '#C8E6C9'
    rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=color, edgecolor='black', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(x + 0.6, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 3.6), (x + 1.8, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax3.add_patch(arrow)

ax3.text(0, 2, 'After: 1->3->5->2->4 (odd then even)', fontsize=10, fontweight='bold')
after = [1, 3, 5, 2, 4]
after_colors = ['#BBDEFB', '#BBDEFB', '#BBDEFB', '#C8E6C9', '#C8E6C9']
for i, (val, col) in enumerate(zip(after, after_colors)):
    x = i * 1.8
    rect = FancyBboxPatch((x, 1), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(x + 0.6, 1.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 1.4), (x + 1.8, 1.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax3.add_patch(arrow)

ax3.text(0, 0.2, 'Blue = odd indices, Green = even indices', fontsize=9, color='#666')

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 8)
ax4.axis('off')
ax4.set_title('Advanced Operations Summary', fontsize=14, fontweight='bold')

ops = [('Copy with Random', 'Hash map or interleaving', 'O(n)'), ('Flatten Multilevel', 'DFS/recursion', 'O(n)'), ('Odd-Even List', 'Two pointer weaving', 'O(n)')]
y = 7
for name, approach, time in ops:
    rect = Rectangle((0.3, y - 1.5), 9.4, 1.8, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax4.add_patch(rect)
    ax4.text(5, y - 0.3, name, ha='center', va='center', fontsize=11, fontweight='bold', color='#1565C0')
    ax4.text(5, y - 0.8, approach, ha='center', va='center', fontsize=9)
    ax4.text(5, y - 1.2, f'Time: {time}', fontsize=9, ha='center', color='#388E3C')
    y -= 2.2

plt.tight_layout()
plt.savefig('advanced_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created advanced_operations.png')

# 7. LRU Cache
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

ax1 = axes[0]
ax1.set_xlim(-1, 16)
ax1.set_ylim(-1, 8)
ax1.axis('off')
ax1.set_title('LRU Cache: Hash Map + Doubly Linked List\nO(1) get and put operations', fontsize=14, fontweight='bold')

# Hash map
ax1.text(0, 7, 'Hash Map (key -> node)', fontsize=11, fontweight='bold', color='#1565C0')
hash_rect = Rectangle((0, 4.5), 5, 2.2, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
ax1.add_patch(hash_rect)

keys = ['key1', 'key2', 'key3']
for i, k in enumerate(keys):
    y = 6.2 - i * 0.6
    ax1.text(0.5, y, k, fontsize=10)
    ax1.annotate('', xy=(4.5, y), xytext=(1.5, y), arrowprops=dict(arrowstyle='->', color='#666'))
    ax1.text(2, y + 0.15, 'node ref', fontsize=8, color='#666')

# Doubly linked list
ax1.text(7, 7, 'Doubly Linked List (LRU order)', fontsize=11, fontweight='bold', color='#388E3C')

# Head sentinel
ax1.text(6, 5.5, 'HEAD', fontsize=9, fontweight='bold', color='#666')
rect = FancyBboxPatch((5.5, 4.5), 1.5, 0.9, boxstyle='round,pad=0.05', facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)
ax1.add_patch(rect)

# Nodes
nodes = [('k3', 'v3'), ('k2', 'v2'), ('k1', 'v1')]
colors = ['#FFC107', '#C8E6C9', '#BBDEFB']
for i, ((k, v), col) in enumerate(zip(nodes, colors)):
    x = 7.5 + i * 2.2
    rect = FancyBboxPatch((x, 4.5), 1.8, 0.9, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.9, 4.95, f'{k}:{v}', ha='center', va='center', fontsize=9, fontweight='bold')

    # Bidirectional arrows
    if i == 0:
        arrow = FancyArrowPatch((7, 4.95), (x, 4.95), arrowstyle='<->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)
    if i < 2:
        arrow = FancyArrowPatch((x + 1.8, 4.95), (x + 2.2, 4.95), arrowstyle='<->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

# Tail sentinel
ax1.text(14, 5.5, 'TAIL', fontsize=9, fontweight='bold', color='#666')
rect = FancyBboxPatch((13.5, 4.5), 1.5, 0.9, boxstyle='round,pad=0.05', facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)
ax1.add_patch(rect)

ax1.text(7.5, 3.8, 'Most Recently Used', fontsize=9, color='#FFC107', fontweight='bold')
ax1.text(11.5, 3.8, 'Least Recently Used', fontsize=9, color='#BBDEFB', fontweight='bold')

# Arrows from hash map to nodes
ax1.annotate('', xy=(7.5 + 0.9, 5.4), xytext=(4.5, 5.3), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1, connectionstyle='arc3,rad=0.2'))

# Operations
ax1.text(0, 3, 'Operations:', fontsize=11, fontweight='bold')
ax1.text(0.5, 2.3, 'get(key): Find in hash map O(1), move to front O(1)', fontsize=10)
ax1.text(0.5, 1.6, 'put(key, val): If exists, update and move to front', fontsize=10)
ax1.text(0.5, 0.9, '              If new, add to front, evict from back if full', fontsize=10)

ax2 = axes[1]
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('LRU Cache Operations Example', fontsize=14, fontweight='bold')

ops = [
    ('put(1,1)', 'Add (1,1) to front', '[1]'),
    ('put(2,2)', 'Add (2,2) to front', '[2,1]'),
    ('get(1)', 'Access 1, move to front', '[1,2]'),
    ('put(3,3)', 'Add (3,3), capacity=2, evict 2', '[3,1]'),
    ('get(2)', 'Key 2 not found, return -1', '[3,1]'),
]

y = 5.5
for op, desc, state in ops:
    rect = Rectangle((0, y - 0.8), 15, 1, facecolor='#FFF9C4', edgecolor='#FFA000', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(1.5, y - 0.3, op, fontsize=10, fontweight='bold', color='#E65100')
    ax2.text(5, y - 0.3, desc, fontsize=10)
    ax2.text(12, y - 0.3, f'State: {state}', fontsize=10, color='#388E3C', fontweight='bold')
    y -= 1.1

plt.tight_layout()
plt.savefig('lru_cache.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created lru_cache.png')

# 8. Rotation Operations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax1 = axes[0, 0]
ax1.set_xlim(-1, 14)
ax1.set_ylim(-1, 5)
ax1.axis('off')
ax1.set_title('Rotate List Right by K (LeetCode 61)\nRotate 1->2->3->4->5 by k=2', fontsize=12, fontweight='bold')

ax1.text(0, 4.2, 'Before:', fontsize=10, fontweight='bold')
before = [1, 2, 3, 4, 5]
for i, val in enumerate(before):
    x = i * 1.8
    color = '#FFC107' if i >= 3 else '#BBDEFB'
    rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=color, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 3.6), (x + 1.8, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 2, 'After (rotate right by 2):', fontsize=10, fontweight='bold')
after = [4, 5, 1, 2, 3]
after_colors = ['#FFC107', '#FFC107', '#BBDEFB', '#BBDEFB', '#BBDEFB']
for i, (val, col) in enumerate(zip(after, after_colors)):
    x = i * 1.8
    rect = FancyBboxPatch((x, 1), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 1.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 1.4), (x + 1.8, 1.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 0.2, 'Last k nodes move to front', fontsize=10, color='#666')

ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('Rotation Algorithm', fontsize=12, fontweight='bold')

algo = '''1. Find length n, get tail
2. k = k % n (handle k > n)
3. Find (n-k)th node (new tail)
4. Connect: tail.next = head
5. Break: new_tail.next = None
6. Return new_head

Time: O(n)  Space: O(1)'''

ax2.text(0.5, 5.5, algo, fontsize=10, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

ax3 = axes[1, 0]
ax3.set_xlim(-1, 14)
ax3.set_ylim(-1, 5)
ax3.axis('off')
ax3.set_title('Step-by-Step Rotation', fontsize=12, fontweight='bold')

steps = [
    ('Step 1', 'Find length=5, tail=5'),
    ('Step 2', 'k=2%5=2, new_tail at position 3'),
    ('Step 3', 'Connect tail to head: 5->1'),
    ('Step 4', 'Break at new tail: 3->NULL'),
]
y = 4.5
for step, desc in steps:
    rect = Rectangle((0, y - 0.8), 12, 1, facecolor='#FFF9C4', edgecolor='#FFA000', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(1, y - 0.3, step, fontsize=10, fontweight='bold', color='#E65100')
    ax3.text(4, y - 0.3, desc, fontsize=10)
    y -= 1.2

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 6)
ax4.axis('off')
ax4.set_title('Edge Cases', fontsize=12, fontweight='bold')

cases = [('k = 0 or k = n', 'No rotation needed'), ('k > n', 'Use k % n'), ('Empty or single node', 'Return as-is')]
y = 5
for case, handling in cases:
    rect = Rectangle((0.5, y - 1), 9, 1.2, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax4.add_patch(rect)
    ax4.text(1, y - 0.4, case, fontsize=10, fontweight='bold', color='#1565C0')
    ax4.text(5.5, y - 0.4, handling, fontsize=10)
    y -= 1.5

plt.tight_layout()
plt.savefig('rotation_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created rotation_operations.png')

# 9. Reorder Operations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.set_xlim(-1, 14)
ax1.set_ylim(-1, 5)
ax1.axis('off')
ax1.set_title('Reorder List (LeetCode 143)\nL0->Ln->L1->Ln-1->L2->Ln-2...', fontsize=12, fontweight='bold')

ax1.text(0, 4.2, 'Before: 1->2->3->4->5', fontsize=10, fontweight='bold')
before = [1, 2, 3, 4, 5]
for i, val in enumerate(before):
    x = i * 1.8
    rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#BBDEFB', edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 3.6), (x + 1.8, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 2, 'After: 1->5->2->4->3', fontsize=10, fontweight='bold')
after = [1, 5, 2, 4, 3]
colors = ['#BBDEFB', '#C8E6C9', '#BBDEFB', '#C8E6C9', '#BBDEFB']
for i, (val, col) in enumerate(zip(after, colors)):
    x = i * 1.8
    rect = FancyBboxPatch((x, 1), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 1.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 1.4), (x + 1.8, 1.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 0.2, 'Blue=first half, Green=second half (reversed)', fontsize=9, color='#666')

ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('Reorder Algorithm (3 Steps)', fontsize=12, fontweight='bold')

algo = '''Step 1: Find middle (fast-slow)
  1->2->3 | 4->5

Step 2: Reverse second half
  1->2->3   5->4

Step 3: Merge alternating
  1->5->2->4->3

Time: O(n)  Space: O(1)'''

ax2.text(0.5, 5.5, algo, fontsize=10, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

ax3 = axes[1, 0]
ax3.set_xlim(-1, 14)
ax3.set_ylim(-1, 5)
ax3.axis('off')
ax3.set_title('Merge Step Visualization', fontsize=12, fontweight='bold')

ax3.text(0, 4.2, 'First half: 1->2->3', fontsize=10, fontweight='bold', color='#1565C0')
first = [1, 2, 3]
for i, val in enumerate(first):
    x = i * 1.8
    rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#BBDEFB', edgecolor='black', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(x + 0.6, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((x + 1.2, 3.6), (x + 1.8, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax3.add_patch(arrow)

ax3.text(7, 4.2, 'Second half (reversed): 5->4', fontsize=10, fontweight='bold', color='#388E3C')
second = [5, 4]
for i, val in enumerate(second):
    x = 7 + i * 1.8
    rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor='#C8E6C9', edgecolor='black', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(x + 0.6, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 1:
        arrow = FancyArrowPatch((x + 1.2, 3.6), (x + 1.8, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax3.add_patch(arrow)

ax3.text(0, 2, 'Interleave: take from first, then second, repeat', fontsize=10, fontweight='bold')
ax3.text(0, 1.2, '1 -> 5 -> 2 -> 4 -> 3', fontsize=12, fontweight='bold', color='#E65100')

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 6)
ax4.axis('off')
ax4.set_title('Related Problems', fontsize=12, fontweight='bold')

problems = [('Reorder List', 'Fast-slow + Reverse + Merge', 'Medium'), ('Partition List', 'Two pointers, preserve order', 'Medium'), ('Split in Parts', 'Divide into k parts', 'Medium')]
y = 5
for name, technique, diff in problems:
    rect = Rectangle((0.5, y - 1), 9, 1.2, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax4.add_patch(rect)
    ax4.text(1, y - 0.25, name, fontsize=10, fontweight='bold', color='#1565C0')
    ax4.text(4, y - 0.25, technique, fontsize=9)
    ax4.text(8.5, y - 0.25, diff, fontsize=9, color='#E65100')
    y -= 1.5

plt.tight_layout()
plt.savefig('reorder_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created reorder_operations.png')

# 10. Sort Operations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.set_xlim(-1, 14)
ax1.set_ylim(-1, 6)
ax1.axis('off')
ax1.set_title('Merge Sort for Linked List (LeetCode 148)\nO(n log n) time, O(log n) space', fontsize=12, fontweight='bold')

# Divide phase
ax1.text(0, 5.5, 'Divide Phase:', fontsize=10, fontweight='bold', color='#1565C0')
ax1.text(2, 5, '[4,2,1,3]', fontsize=10, fontweight='bold')
ax1.plot([2.8, 1.5], [4.7, 4.3], 'k-', linewidth=1)
ax1.plot([2.8, 4], [4.7, 4.3], 'k-', linewidth=1)
ax1.text(0.5, 4, '[4,2]', fontsize=9)
ax1.text(3.5, 4, '[1,3]', fontsize=9)
ax1.plot([1.2, 0.5], [3.7, 3.3], 'k-', linewidth=1)
ax1.plot([1.2, 2], [3.7, 3.3], 'k-', linewidth=1)
ax1.plot([4.2, 3.5], [3.7, 3.3], 'k-', linewidth=1)
ax1.plot([4.2, 5], [3.7, 3.3], 'k-', linewidth=1)
ax1.text(0, 3, '[4]', fontsize=9)
ax1.text(1.5, 3, '[2]', fontsize=9)
ax1.text(3, 3, '[1]', fontsize=9)
ax1.text(4.5, 3, '[3]', fontsize=9)

# Merge phase
ax1.text(7, 5.5, 'Merge Phase:', fontsize=10, fontweight='bold', color='#388E3C')
ax1.text(8, 3, '[4]', fontsize=9)
ax1.text(9.5, 3, '[2]', fontsize=9)
ax1.text(11, 3, '[1]', fontsize=9)
ax1.text(12.5, 3, '[3]', fontsize=9)
ax1.plot([8.3, 9], [3.3, 3.7], 'g-', linewidth=1)
ax1.plot([9.8, 9], [3.3, 3.7], 'g-', linewidth=1)
ax1.plot([11.3, 12], [3.3, 3.7], 'g-', linewidth=1)
ax1.plot([12.8, 12], [3.3, 3.7], 'g-', linewidth=1)
ax1.text(8.5, 4, '[2,4]', fontsize=9, color='green')
ax1.text(11.5, 4, '[1,3]', fontsize=9, color='green')
ax1.plot([9, 10.3], [4.3, 4.7], 'g-', linewidth=1)
ax1.plot([12, 10.3], [4.3, 4.7], 'g-', linewidth=1)
ax1.text(9.5, 5, '[1,2,3,4]', fontsize=10, fontweight='bold', color='green')

ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('Merge Sort Algorithm', fontsize=12, fontweight='bold')

algo = '''def sortList(head):
    if not head or not head.next:
        return head

    # Find middle
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Split
    mid = slow.next
    slow.next = None

    # Recursively sort and merge
    left = sortList(head)
    right = sortList(mid)
    return merge(left, right)'''

ax2.text(0.5, 5.8, algo, fontsize=8, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

ax3 = axes[1, 0]
ax3.set_xlim(-1, 14)
ax3.set_ylim(-1, 5)
ax3.axis('off')
ax3.set_title('Insertion Sort for Linked List (LeetCode 147)\nO(n^2) time but O(1) space', fontsize=12, fontweight='bold')

ax3.text(0, 4, 'Build sorted list one element at a time', fontsize=10, fontweight='bold')

steps = [('Start', 'sorted=[], unsorted=[4,2,1,3]'), ('Insert 4', 'sorted=[4], unsorted=[2,1,3]'), ('Insert 2', 'sorted=[2,4], unsorted=[1,3]'), ('Insert 1', 'sorted=[1,2,4], unsorted=[3]'), ('Insert 3', 'sorted=[1,2,3,4], unsorted=[]')]
y = 3.5
for step, state in steps:
    ax3.text(0, y, f'{step}: {state}', fontsize=9)
    y -= 0.7

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 6)
ax4.axis('off')
ax4.set_title('Sorting Algorithms Comparison', fontsize=12, fontweight='bold')

headers = ['Algorithm', 'Time', 'Space', 'Stable']
rows = [['Merge Sort', 'O(n log n)', 'O(log n)', 'Yes'], ['Insertion Sort', 'O(n^2)', 'O(1)', 'Yes'], ['Quick Sort', 'O(n log n)', 'O(log n)', 'No']]
col_widths = [3, 2.5, 2, 1.5]
row_height = 0.9
start_y = 5.5

x = 0.5
for i, header in enumerate(headers):
    rect = Rectangle((x, start_y - row_height), col_widths[i], row_height, facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)
    ax4.add_patch(rect)
    ax4.text(x + col_widths[i]/2, start_y - row_height/2, header, ha='center', va='center', fontsize=10, fontweight='bold')
    x += col_widths[i]

for row_idx, row in enumerate(rows):
    x = 0.5
    y = start_y - (row_idx + 2) * row_height
    for col_idx, cell in enumerate(row):
        color = '#C8E6C9' if (col_idx == 1 and 'log' in cell) else 'white'
        rect = Rectangle((x, y), col_widths[col_idx], row_height, facecolor=color, edgecolor='black', linewidth=1)
        ax4.add_patch(rect)
        ax4.text(x + col_widths[col_idx]/2, y + row_height/2, cell, ha='center', va='center', fontsize=9)
        x += col_widths[col_idx]

ax4.text(5, 1.5, 'Merge Sort is preferred for linked lists', fontsize=10, ha='center', fontweight='bold', color='#388E3C')

plt.tight_layout()
plt.savefig('sort_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created sort_operations.png')

# 11. Swap Operations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.set_xlim(-1, 14)
ax1.set_ylim(-1, 5)
ax1.axis('off')
ax1.set_title('Swap Nodes in Pairs (LeetCode 24)\nSwap every two adjacent nodes', fontsize=12, fontweight='bold')

ax1.text(0, 4.2, 'Before: 1->2->3->4', fontsize=10, fontweight='bold')
before = [1, 2, 3, 4]
for i, val in enumerate(before):
    x = i * 1.8
    color = '#BBDEFB' if i % 2 == 0 else '#C8E6C9'
    rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=color, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 3:
        arrow = FancyArrowPatch((x + 1.2, 3.6), (x + 1.8, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 2, 'After: 2->1->4->3', fontsize=10, fontweight='bold')
after = [2, 1, 4, 3]
colors = ['#C8E6C9', '#BBDEFB', '#C8E6C9', '#BBDEFB']
for i, (val, col) in enumerate(zip(after, colors)):
    x = i * 1.8
    rect = FancyBboxPatch((x, 1), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 1.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 3:
        arrow = FancyArrowPatch((x + 1.2, 1.4), (x + 1.8, 1.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 0.2, 'Each pair is swapped: (1,2)->(2,1), (3,4)->(4,3)', fontsize=9, color='#666')

ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('Recursive Solution', fontsize=12, fontweight='bold')

code = '''def swapPairs(head):
    if not head or not head.next:
        return head

    # Nodes to swap
    first = head
    second = head.next

    # Swap
    first.next = swapPairs(second.next)
    second.next = first

    return second  # New head'''

ax2.text(0.5, 5.8, code, fontsize=9, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 6)
ax3.axis('off')
ax3.set_title('Iterative Solution', fontsize=12, fontweight='bold')

code = '''def swapPairs(head):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    while prev.next and prev.next.next:
        first = prev.next
        second = prev.next.next

        # Swap
        prev.next = second
        first.next = second.next
        second.next = first

        prev = first

    return dummy.next'''

ax3.text(0.5, 5.8, code, fontsize=9, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

ax4 = axes[1, 1]
ax4.set_xlim(-1, 12)
ax4.set_ylim(-1, 5)
ax4.axis('off')
ax4.set_title('Swap Two Nodes at Positions (LeetCode 1721)', fontsize=12, fontweight='bold')

ax4.text(0, 4, 'Swap kth from beginning with kth from end', fontsize=10, fontweight='bold')
ax4.text(0, 3.3, 'For k=2 in [1,2,3,4,5]:', fontsize=10)
ax4.text(0.5, 2.6, '2nd from start = 2', fontsize=10, color='#1565C0')
ax4.text(0.5, 2, '2nd from end = 4', fontsize=10, color='#388E3C')
ax4.text(0.5, 1.3, 'Result: [1,4,3,2,5]', fontsize=10, fontweight='bold', color='#E65100')

ax4.text(0, 0.3, 'Use two pointers with k gap to find nodes', fontsize=9, color='#666')

plt.tight_layout()
plt.savefig('swap_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created swap_operations.png')

# 12. Reverse K Group
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.set_xlim(-1, 16)
ax1.set_ylim(-1, 5)
ax1.axis('off')
ax1.set_title('Reverse Nodes in K-Group (LeetCode 25)\nReverse every k nodes', fontsize=12, fontweight='bold')

ax1.text(0, 4.2, 'Before: 1->2->3->4->5 (k=2)', fontsize=10, fontweight='bold')
before = [1, 2, 3, 4, 5]
colors_before = ['#BBDEFB', '#BBDEFB', '#C8E6C9', '#C8E6C9', '#FFF9C4']
for i, (val, col) in enumerate(zip(before, colors_before)):
    x = i * 1.8
    rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 3.6), (x + 1.8, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 2, 'After: 2->1->4->3->5', fontsize=10, fontweight='bold')
after = [2, 1, 4, 3, 5]
colors_after = ['#BBDEFB', '#BBDEFB', '#C8E6C9', '#C8E6C9', '#FFF9C4']
for i, (val, col) in enumerate(zip(after, colors_after)):
    x = i * 1.8
    rect = FancyBboxPatch((x, 1), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + 0.6, 1.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 1.4), (x + 1.8, 1.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax1.add_patch(arrow)

ax1.text(0, 0.2, 'Groups: [1,2] reversed, [3,4] reversed, [5] left as-is (< k)', fontsize=9, color='#666')

ax2 = axes[0, 1]
ax2.set_xlim(-1, 16)
ax2.set_ylim(-1, 5)
ax2.axis('off')
ax2.set_title('Example with k=3', fontsize=12, fontweight='bold')

ax2.text(0, 4.2, 'Before: 1->2->3->4->5 (k=3)', fontsize=10, fontweight='bold')
before = [1, 2, 3, 4, 5]
colors_before = ['#BBDEFB', '#BBDEFB', '#BBDEFB', '#FFF9C4', '#FFF9C4']
for i, (val, col) in enumerate(zip(before, colors_before)):
    x = i * 1.8
    rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(x + 0.6, 3.6, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 3.6), (x + 1.8, 3.6), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax2.add_patch(arrow)

ax2.text(0, 2, 'After: 3->2->1->4->5', fontsize=10, fontweight='bold')
after = [3, 2, 1, 4, 5]
colors_after = ['#BBDEFB', '#BBDEFB', '#BBDEFB', '#FFF9C4', '#FFF9C4']
for i, (val, col) in enumerate(zip(after, colors_after)):
    x = i * 1.8
    rect = FancyBboxPatch((x, 1), 1.2, 0.8, boxstyle='round,pad=0.05,rounding_size=0.1', facecolor=col, edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(x + 0.6, 1.4, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    if i < 4:
        arrow = FancyArrowPatch((x + 1.2, 1.4), (x + 1.8, 1.4), arrowstyle='->', mutation_scale=10, color='black', linewidth=1.5)
        ax2.add_patch(arrow)

ax2.text(0, 0.2, 'Only first group [1,2,3] has k=3 nodes, reversed to [3,2,1]', fontsize=9, color='#666')

ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 6)
ax3.axis('off')
ax3.set_title('Algorithm Steps', fontsize=12, fontweight='bold')

steps = '''1. Count k nodes ahead
   - If less than k, return head as-is

2. Reverse k nodes
   - Standard reversal within group
   - Track prev, curr, next

3. Connect to rest
   - Original first node links to
     result of recursive call

4. Return new head of reversed group'''

ax3.text(0.5, 5.8, steps, fontsize=10, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 6)
ax4.axis('off')
ax4.set_title('Complexity & Tips', fontsize=12, fontweight='bold')

info = '''Time: O(n) - each node visited twice
Space: O(n/k) - recursion depth

Key Points:
- Use dummy node for cleaner code
- Count nodes before reversing
- Track connections carefully
- Handle remaining nodes (< k)

Interview Tip:
This is a HARD problem but builds on
basic reversal. Master reversal first!'''

ax4.text(0.5, 5.8, info, fontsize=10, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

plt.tight_layout()
plt.savefig('reverse_k_group.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Created reverse_k_group.png')

print('\nAll 12 visualizations created successfully!')
