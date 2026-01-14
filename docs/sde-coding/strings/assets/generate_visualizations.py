#!/usr/bin/env python3
"""
Generate DP table visualizations for string problems.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set the directory for saving images
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

def create_edit_distance_dp_table():
    """Create Edit Distance DP table visualization for 'horse' -> 'ros'"""
    word1 = "horse"
    word2 = "ros"
    m, n = len(word1), len(word2)

    # Build the DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create table data
    table_data = []
    for i in range(m + 1):
        row = []
        for j in range(n + 1):
            row.append(str(dp[i][j]))
        table_data.append(row)

    # Column and row labels
    col_labels = ['""'] + list(word2)
    row_labels = ['""'] + list(word1)

    # Create the table
    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=[['#E8F5E9' if i == m and j == n else '#FFFFFF'
                      for j in range(n + 1)] for i in range(m + 1)]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 2)

    # Color the header cells
    for i in range(n + 1):
        table[(0, i)].set_facecolor('#BBDEFB')
    for i in range(m + 2):
        if i > 0:
            table[(i, -1)].set_facecolor('#BBDEFB')

    ax.axis('off')
    ax.set_title('Edit Distance DP Table: "horse" -> "ros"\n(Answer: 3)',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'edit-distance-dp-table.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created edit-distance-dp-table.png")


def create_longest_palindrome_dp_table():
    """Create Longest Palindromic Substring DP table for 'babad'"""
    s = "babad"
    n = len(s)

    # Build DP table
    dp = [[False] * n for _ in range(n)]

    # All single characters are palindromes
    for i in range(n):
        dp[i][i] = True

    # Check length 2
    for i in range(n - 1):
        dp[i][i+1] = (s[i] == s[i+1])

    # Check length 3 and above
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = (s[i] == s[j]) and dp[i+1][j-1]

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create table data with T/F
    table_data = []
    for i in range(n):
        row = []
        for j in range(n):
            if j >= i:
                row.append('T' if dp[i][j] else 'F')
            else:
                row.append('-')
        table_data.append(row)

    # Labels
    col_labels = list(s)
    row_labels = list(s)

    # Create color map
    colors = []
    for i in range(n):
        row_colors = []
        for j in range(n):
            if j < i:
                row_colors.append('#F5F5F5')  # Gray for unused
            elif dp[i][j]:
                row_colors.append('#C8E6C9')  # Green for True
            else:
                row_colors.append('#FFCDD2')  # Red for False
        colors.append(row_colors)

    # Create the table
    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 2)

    # Color headers
    for i in range(n):
        table[(0, i)].set_facecolor('#BBDEFB')
    for i in range(n + 1):
        if i > 0:
            table[(i, -1)].set_facecolor('#BBDEFB')

    ax.axis('off')
    ax.set_title('Longest Palindromic Substring DP Table: "babad"\ndp[i][j] = True if s[i:j+1] is palindrome',
                 fontsize=14, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#C8E6C9', label='True (Palindrome)'),
        mpatches.Patch(facecolor='#FFCDD2', label='False'),
        mpatches.Patch(facecolor='#F5F5F5', label='Not used (j < i)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'longest-palindrome-dp-table.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created longest-palindrome-dp-table.png")


def create_regex_matching_dp_table():
    """Create Regex Matching DP table for s='aab', p='c*a*b'"""
    s = "aab"
    p = "c*a*b"
    m, n = len(s), len(p)

    # Build DP table
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Handle patterns like a*, a*b*, etc. that can match empty string
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # Zero occurrences of preceding element
                dp[i][j] = dp[i][j-2]
                # One or more occurrences
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    table_data = []
    for i in range(m + 1):
        row = []
        for j in range(n + 1):
            row.append('T' if dp[i][j] else 'F')
        table_data.append(row)

    col_labels = ['""'] + list(p)
    row_labels = ['""'] + list(s)

    colors = []
    for i in range(m + 1):
        row_colors = []
        for j in range(n + 1):
            if dp[i][j]:
                row_colors.append('#C8E6C9')
            else:
                row_colors.append('#FFCDD2')
        colors.append(row_colors)

    # Highlight the answer cell
    colors[m][n] = '#4CAF50'

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.4, 2)

    for i in range(n + 1):
        table[(0, i)].set_facecolor('#BBDEFB')
    for i in range(m + 2):
        if i > 0:
            table[(i, -1)].set_facecolor('#BBDEFB')

    ax.axis('off')
    ax.set_title('Regular Expression Matching DP Table\ns="aab", p="c*a*b" (Answer: True)',
                 fontsize=14, fontweight='bold', pad=20)

    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', label='Answer (True)'),
        mpatches.Patch(facecolor='#C8E6C9', label='Match (True)'),
        mpatches.Patch(facecolor='#FFCDD2', label='No Match (False)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'regex-matching-dp-table.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created regex-matching-dp-table.png")


def create_wildcard_matching_dp_table():
    """Create Wildcard Matching DP table for s='adceb', p='*a*b'"""
    s = "adceb"
    p = "*a*b"
    m, n = len(s), len(p)

    # Build DP table
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Handle leading *
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]

    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # * matches empty or one+ characters
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            elif p[j-1] == '?' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    table_data = []
    for i in range(m + 1):
        row = []
        for j in range(n + 1):
            row.append('T' if dp[i][j] else 'F')
        table_data.append(row)

    col_labels = ['""'] + list(p)
    row_labels = ['""'] + list(s)

    colors = []
    for i in range(m + 1):
        row_colors = []
        for j in range(n + 1):
            if dp[i][j]:
                row_colors.append('#C8E6C9')
            else:
                row_colors.append('#FFCDD2')
        colors.append(row_colors)

    colors[m][n] = '#4CAF50'

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 1.8)

    for i in range(n + 1):
        table[(0, i)].set_facecolor('#BBDEFB')
    for i in range(m + 2):
        if i > 0:
            table[(i, -1)].set_facecolor('#BBDEFB')

    ax.axis('off')
    ax.set_title('Wildcard Matching DP Table\ns="adceb", p="*a*b" (Answer: True)',
                 fontsize=14, fontweight='bold', pad=20)

    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', label='Answer (True)'),
        mpatches.Patch(facecolor='#C8E6C9', label='Match (True)'),
        mpatches.Patch(facecolor='#FFCDD2', label='No Match (False)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'wildcard-matching-dp-table.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created wildcard-matching-dp-table.png")


def create_minimum_window_visualization():
    """Create sliding window visualization for Minimum Window Substring"""
    fig, ax = plt.subplots(figsize=(14, 8))

    s = "ADOBECODEBANC"
    t = "ABC"

    # Draw the string with boxes
    box_width = 0.8
    y_pos = 0.7

    for i, char in enumerate(s):
        rect = mpatches.FancyBboxPatch(
            (i * box_width + 0.1, y_pos), box_width - 0.1, 0.15,
            boxstyle="round,pad=0.02",
            facecolor='#E3F2FD' if char in t else '#FAFAFA',
            edgecolor='#1976D2',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(i * box_width + box_width/2, y_pos + 0.075, char,
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(i * box_width + box_width/2, y_pos - 0.05, str(i),
                ha='center', va='center', fontsize=10, color='gray')

    # Draw the minimum window (indices 9-12: "BANC")
    window_start = 9
    window_end = 12
    rect = mpatches.FancyBboxPatch(
        (window_start * box_width, y_pos - 0.02),
        (window_end - window_start + 1) * box_width, 0.19,
        boxstyle="round,pad=0.02",
        facecolor='none',
        edgecolor='#4CAF50',
        linewidth=4
    )
    ax.add_patch(rect)

    # Add annotations
    ax.annotate('Minimum Window\n"BANC" (length 4)',
                xy=(10.5 * box_width, y_pos + 0.2),
                fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))

    # Add target string info
    ax.text(0.5, 0.45, f's = "{s}"', transform=ax.transAxes,
            fontsize=14, ha='center')
    ax.text(0.5, 0.35, f't = "{t}" (characters to find)', transform=ax.transAxes,
            fontsize=14, ha='center')
    ax.text(0.5, 0.25, 'Answer: "BANC" (indices 9-12)', transform=ax.transAxes,
            fontsize=14, ha='center', fontweight='bold', color='#2E7D32')

    ax.set_xlim(-0.5, len(s) * box_width + 0.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Minimum Window Substring Visualization',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'minimum-window-substring.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created minimum-window-substring.png")


def create_palindrome_partitioning_tree():
    """Create visualization for palindrome partitioning"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Tree structure for "aab"
    # Level 0: "aab"
    # Level 1: "a" + "ab", "aa" + "b"
    # Level 2: "a" + "a" + "b"

    nodes = [
        # (x, y, text, is_valid)
        (0.5, 0.9, '"aab"', True),
        (0.25, 0.65, '"a" | "ab"', False),
        (0.75, 0.65, '"aa" | "b"', True),
        (0.15, 0.4, '"a" | "a" | "b"', True),
        (0.35, 0.4, '"a" | "ab"\n(not palindrome)', False),
    ]

    # Draw nodes
    for x, y, text, is_valid in nodes:
        color = '#C8E6C9' if is_valid else '#FFCDD2'
        rect = mpatches.FancyBboxPatch(
            (x - 0.12, y - 0.06), 0.24, 0.12,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='#333333',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw edges
    edges = [
        ((0.5, 0.84), (0.25, 0.71)),
        ((0.5, 0.84), (0.75, 0.71)),
        ((0.25, 0.59), (0.15, 0.46)),
        ((0.25, 0.59), (0.35, 0.46)),
    ]

    for (x1, y1), (x2, y2) in edges:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=2))

    # Add results
    ax.text(0.5, 0.15, 'Valid Partitions:', ha='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.08, '1. ["a", "a", "b"]', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='#C8E6C9'))
    ax.text(0.5, 0.0, '2. ["aa", "b"]', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='#C8E6C9'))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.axis('off')
    ax.set_title('Palindrome Partitioning: "aab"\nBacktracking Decision Tree',
                 fontsize=16, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#C8E6C9', label='Valid (all palindromes)'),
        mpatches.Patch(facecolor='#FFCDD2', label='Invalid path')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'palindrome-partitioning.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created palindrome-partitioning.png")


def create_expand_around_center_visualization():
    """Create visualization for expand around center technique"""
    fig, ax = plt.subplots(figsize=(14, 8))

    s = "babad"
    n = len(s)

    # Draw the string
    box_width = 1.2
    y_pos = 0.7

    for i, char in enumerate(s):
        rect = mpatches.FancyBboxPatch(
            (i * box_width + 0.5, y_pos), box_width - 0.1, 0.15,
            boxstyle="round,pad=0.02",
            facecolor='#E3F2FD',
            edgecolor='#1976D2',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(i * box_width + box_width/2 + 0.45, y_pos + 0.075, char,
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(i * box_width + box_width/2 + 0.45, y_pos - 0.05, str(i),
                ha='center', va='center', fontsize=10, color='gray')

    # Show expansion from center at index 1 (for "bab")
    center = 1

    # Draw center marker
    ax.annotate('Center', xy=(center * box_width + box_width/2 + 0.45, y_pos + 0.18),
                fontsize=12, ha='center', color='#D32F2F',
                arrowprops=dict(arrowstyle='->', color='#D32F2F'))

    # Draw expansion arrows
    ax.annotate('', xy=(0.5 * box_width + 0.45, y_pos + 0.075),
                xytext=(center * box_width + 0.2, y_pos + 0.075),
                arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=3))
    ax.annotate('', xy=(2 * box_width + box_width - 0.1 + 0.45, y_pos + 0.075),
                xytext=(center * box_width + box_width + 0.45, y_pos + 0.075),
                arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=3))

    # Highlight the palindrome "bab"
    rect = mpatches.FancyBboxPatch(
        (0 * box_width + 0.45, y_pos - 0.02),
        3 * box_width, 0.19,
        boxstyle="round,pad=0.02",
        facecolor='none',
        edgecolor='#4CAF50',
        linewidth=4
    )
    ax.add_patch(rect)

    ax.text(0.5, 0.4, 'Expand Around Center at index 1:', transform=ax.transAxes,
            fontsize=14, ha='center', fontweight='bold')
    ax.text(0.5, 0.3, 's[0] == s[2]? "b" == "b" -> Yes! Expand further.',
            transform=ax.transAxes, fontsize=12, ha='center')
    ax.text(0.5, 0.2, 'Found palindrome: "bab" (length 3)',
            transform=ax.transAxes, fontsize=14, ha='center',
            fontweight='bold', color='#2E7D32')

    ax.set_xlim(0, n * box_width + 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Expand Around Center: Finding Palindrome in "babad"',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'expand-around-center.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created expand-around-center.png")


if __name__ == "__main__":
    print("Generating visualization images...")
    create_edit_distance_dp_table()
    create_longest_palindrome_dp_table()
    create_regex_matching_dp_table()
    create_wildcard_matching_dp_table()
    create_minimum_window_visualization()
    create_palindrome_partitioning_tree()
    create_expand_around_center_visualization()
    print("All visualizations generated successfully!")
