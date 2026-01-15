"""
Data Quality Funnel Visualization

This script creates visualizations of data filtering and quality assessment
pipelines for fine-tuning data preparation.

Output: data_quality_funnel.png, data_filtering_pipeline.png, data_mixing.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "fine-tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def create_data_quality_funnel():
    """Create a funnel diagram showing data filtering stages"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(6, 11.5, 'Data Quality Funnel', fontsize=16, fontweight='bold', ha='center')
    ax.text(6, 10.9, 'Filtering raw data to high-quality training examples',
            fontsize=11, ha='center', style='italic', color='gray')

    # Funnel stages (widths decrease as we go down)
    stages = [
        ('Raw Data', 100, '#e3f2fd', '1,000,000 examples'),
        ('Language Filter', 80, '#bbdefb', '800,000 examples'),
        ('Length Filter', 65, '#90caf9', '650,000 examples'),
        ('Quality Score', 40, '#64b5f6', '400,000 examples'),
        ('Toxicity Filter', 35, '#42a5f5', '350,000 examples'),
        ('Deduplication', 20, '#2196f3', '200,000 examples'),
        ('Final Dataset', 15, '#1976d2', '150,000 examples'),
    ]

    # Draw funnel
    y_pos = 10
    max_width = 10
    center_x = 6

    for i, (stage, pct, color, count) in enumerate(stages):
        width = max_width * (pct / 100)
        height = 1.2

        # Draw trapezoid for funnel effect
        if i < len(stages) - 1:
            next_width = max_width * (stages[i+1][1] / 100)
        else:
            next_width = width

        # Create trapezoid vertices
        top_left = (center_x - width/2, y_pos)
        top_right = (center_x + width/2, y_pos)
        bottom_right = (center_x + next_width/2, y_pos - height)
        bottom_left = (center_x - next_width/2, y_pos - height)

        trapezoid = Polygon(
            [top_left, top_right, bottom_right, bottom_left],
            facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(trapezoid)

        # Add text
        ax.text(center_x, y_pos - height/2, stage,
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # Add count on the right
        ax.text(center_x + max_width/2 + 0.5, y_pos - height/2, count,
                ha='left', va='center', fontsize=10, color='gray')

        # Add percentage lost
        if i > 0:
            prev_pct = stages[i-1][1]
            lost_pct = prev_pct - pct
            ax.text(center_x - max_width/2 - 0.3, y_pos - height/2, f'-{lost_pct}%',
                   ha='right', va='center', fontsize=9, color='#c62828')

        y_pos -= height + 0.1

    # Legend
    ax.text(1, 2, 'Key Insight:', fontsize=11, fontweight='bold')
    ax.text(1, 1.5, 'Start with ~1M examples', fontsize=10)
    ax.text(1, 1.1, 'End with ~150K high-quality examples', fontsize=10)
    ax.text(1, 0.7, 'Quality > Quantity for fine-tuning', fontsize=10, style='italic', color='#1976d2')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_quality_funnel.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'data_quality_funnel.png'}")


def create_filtering_pipeline():
    """Create a horizontal pipeline diagram of filtering stages"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(8, 7.5, 'Data Filtering Pipeline', fontsize=14, fontweight='bold', ha='center')

    # Pipeline stages
    stages = [
        ('Raw\nData', '#e0e0e0', '1M'),
        ('Language\nDetection', '#e3f2fd', '800K'),
        ('Length\nFilter', '#bbdefb', '650K'),
        ('Quality\nScoring', '#90caf9', '400K'),
        ('Toxicity\nFilter', '#64b5f6', '350K'),
        ('Exact\nDedup', '#42a5f5', '300K'),
        ('Semantic\nDedup', '#2196f3', '200K'),
        ('Final', '#1976d2', '150K'),
    ]

    box_width = 1.5
    box_height = 2
    start_x = 0.5
    y_center = 4

    for i, (label, color, count) in enumerate(stages):
        x = start_x + i * 1.9

        # Draw box
        box = FancyBboxPatch(
            (x, y_center - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='black' if i == len(stages)-1 else 'white',
            linewidth=2
        )
        ax.add_patch(box)

        # Add label
        ax.text(x + box_width/2, y_center + 0.2, label,
                ha='center', va='center', fontsize=9, fontweight='bold')

        # Add count
        ax.text(x + box_width/2, y_center - 0.5, count,
                ha='center', va='center', fontsize=10, color='#424242')

        # Add arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x + box_width + 0.05, y_center),
                (x + 1.85, y_center),
                arrowstyle='->',
                color='gray',
                linewidth=2,
                mutation_scale=15
            )
            ax.add_patch(arrow)

    # Add rejection annotations below
    rejections = [
        (2.4, 'Non-English: 20%'),
        (4.3, 'Too short/long: 15%'),
        (6.2, 'Low quality: 25%'),
        (8.1, 'Toxic: 5%'),
        (10.0, 'Exact duplicates: 5%'),
        (11.9, 'Semantic duplicates: 10%'),
    ]

    for x, text in rejections:
        ax.annotate(text, xy=(x, y_center - box_height/2 - 0.1),
                   xytext=(x, y_center - box_height/2 - 1.2),
                   ha='center', fontsize=8, color='#c62828',
                   arrowprops=dict(arrowstyle='->', color='#c62828', lw=1))

    # Quality score breakdown
    ax.text(6.2, 1.5, 'Quality Score Components:', fontsize=10, fontweight='bold')
    components = ['Coherence', 'Relevance', 'Completeness', 'Formatting']
    for i, comp in enumerate(components):
        ax.text(6.2 + i * 2.5, 1, f'• {comp}', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_filtering_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'data_filtering_pipeline.png'}")


def create_data_mixing_visualization():
    """Create visualization of data mixing strategies"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Pie chart data
    categories = ['Instruction\nFollowing', 'Coding', 'Math &\nReasoning',
                  'Creative\nWriting', 'Factual QA', 'Safety']
    colors = ['#1976d2', '#4caf50', '#ff9800', '#9c27b0', '#00bcd4', '#f44336']

    # Strategy 1: Proportional
    ax = axes[0]
    sizes_prop = [40, 25, 15, 10, 7, 3]
    ax.pie(sizes_prop, labels=categories, colors=colors, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 9})
    ax.set_title('Proportional to\nDataset Size', fontsize=12, fontweight='bold')

    # Strategy 2: Equal
    ax = axes[1]
    sizes_equal = [16.7, 16.7, 16.7, 16.7, 16.6, 16.6]
    ax.pie(sizes_equal, labels=categories, colors=colors, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 9})
    ax.set_title('Equal Sampling\n(Uniform)', fontsize=12, fontweight='bold')

    # Strategy 3: Task-weighted
    ax = axes[2]
    sizes_weighted = [35, 20, 20, 10, 10, 5]
    ax.pie(sizes_weighted, labels=categories, colors=colors, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 9})
    ax.set_title('Performance-Based\nWeighting', fontsize=12, fontweight='bold')
    ax.annotate('Recommended', xy=(0.5, -0.15), fontsize=10, ha='center',
                color='#2e7d32', fontweight='bold',
                xycoords='axes fraction')

    plt.suptitle('Data Mixing Strategies for Multi-Task Fine-Tuning',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_mixing.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'data_mixing.png'}")


def create_quality_dimensions():
    """Create radar chart of data quality dimensions"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    categories = ['Accuracy', 'Relevance', 'Completeness',
                  'Consistency', 'Diversity', 'Safety']
    N = len(categories)

    # Sample datasets
    datasets = {
        'Raw Data': [5, 4, 6, 3, 8, 4],
        'After Filtering': [8, 7, 8, 7, 7, 8],
        'After Augmentation': [8, 8, 8, 8, 9, 9],
    }

    colors = {
        'Raw Data': '#e0e0e0',
        'After Filtering': '#64b5f6',
        'After Augmentation': '#4caf50',
    }

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for dataset_name, scores in datasets.items():
        values = scores + scores[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=dataset_name,
                color=colors[dataset_name])
        ax.fill(angles, values, alpha=0.15, color=colors[dataset_name])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.title('Data Quality Improvement Through Pipeline', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_quality_dimensions.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'data_quality_dimensions.png'}")


def create_synthetic_data_flow():
    """Create diagram of synthetic data generation flow"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'Synthetic Data Generation Pipeline', fontsize=14, fontweight='bold', ha='center')

    # Boxes
    def draw_box(x, y, w, h, text, color):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold')

    # Seed examples
    draw_box(0.5, 4, 2.2, 1.5, 'Seed\nExamples\n(1K)', '#e3f2fd')

    # Strong model
    draw_box(3.5, 4, 2.2, 1.5, 'Strong\nModel\n(GPT-4)', '#c8e6c9')

    # Generation strategies
    draw_box(6.5, 5.5, 2.2, 1.2, 'Prompt\nVariation', '#fff9c4')
    draw_box(6.5, 4, 2.2, 1.2, 'Self-\nInstruct', '#fff9c4')
    draw_box(6.5, 2.5, 2.2, 1.2, 'Evol-\nInstruct', '#fff9c4')

    # Generated data
    draw_box(9.5, 4, 2.2, 1.5, 'Generated\nData\n(50K)', '#ffccbc')

    # Quality filter
    draw_box(9.5, 1.5, 2.2, 1.5, 'Quality\nFilter', '#e1bee7')

    # Final
    draw_box(12, 4, 1.8, 1.5, 'Final\n(30K)', '#a5d6a7')

    # Arrows
    def draw_arrow(start, end):
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='->',
            color='gray',
            linewidth=2,
            mutation_scale=15
        )
        ax.add_patch(arrow)

    draw_arrow((2.7, 4.75), (3.5, 4.75))
    draw_arrow((5.7, 4.75), (6.5, 6.1))
    draw_arrow((5.7, 4.75), (6.5, 4.6))
    draw_arrow((5.7, 4.75), (6.5, 3.1))
    draw_arrow((8.7, 6.1), (9.5, 5))
    draw_arrow((8.7, 4.6), (9.5, 4.75))
    draw_arrow((8.7, 3.1), (9.5, 4.5))
    draw_arrow((10.6, 4), (10.6, 3))
    draw_arrow((10.6, 1.5), (10.6, 0.8))
    ax.annotate('', xy=(12, 4), xytext=(10.6, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2,
                               connectionstyle='arc3,rad=-0.5'))

    # Rejection annotation
    ax.text(11.5, 1.2, '40%\nrejected', fontsize=9, color='#c62828', ha='center')

    # Notes
    ax.text(7, 0.5, 'Tip: Never use 100% synthetic data. Mix 30-50% synthetic with real data.',
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='#fff3e0', edgecolor='#ff9800'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'synthetic_data_flow.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'synthetic_data_flow.png'}")


if __name__ == "__main__":
    create_data_quality_funnel()
    create_filtering_pipeline()
    create_data_mixing_visualization()
    create_quality_dimensions()
    create_synthetic_data_flow()
    print("\nAll data quality visualizations created successfully!")
