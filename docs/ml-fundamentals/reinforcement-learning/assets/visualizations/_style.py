"""
Shared Style Module for RL Visualizations

Professional, clean design that works in both light and dark mode.
Uses neutral backgrounds with professional accent colors.

Design Principles:
- White/transparent backgrounds for dark mode compatibility
- Professional blue/indigo primary colors
- Subtle teal-green for accents only
- Gray scale for text and borders
- Thin lines (1-2px), minimal visual clutter
- No heavy saturated colored boxes
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# Professional Color Palette (Dark Mode Friendly)
# =============================================================================

COLORS = {
    # Primary colors
    'primary': '#2563eb',        # Professional blue
    'primary_light': '#3b82f6',  # Lighter blue
    'primary_dark': '#1d4ed8',   # Darker blue

    # Secondary colors
    'secondary': '#6366f1',      # Indigo
    'secondary_light': '#818cf8',
    'secondary_dark': '#4f46e5',

    # Accent (use sparingly for highlights)
    'accent': '#10b981',         # Teal-green
    'accent_light': '#34d399',
    'accent_dark': '#059669',

    # Semantic colors
    'success': '#10b981',        # Same as accent
    'warning': '#f59e0b',        # Amber
    'error': '#ef4444',          # Red
    'info': '#3b82f6',           # Blue

    # Grayscale (for text, borders, fills)
    'text': '#374151',           # Dark gray for main text
    'text_light': '#6b7280',     # Medium gray for secondary text
    'text_muted': '#9ca3af',     # Light gray for muted text
    'border': '#d1d5db',         # Light gray for borders
    'border_light': '#e5e7eb',   # Very light gray
    'fill_light': '#f3f4f6',     # Subtle fill
    'fill_very_light': '#f9fafb', # Very subtle fill

    # Backgrounds
    'background': '#ffffff',     # White (will invert in dark mode)
    'background_alt': '#f9fafb', # Slightly off-white

    # Grid
    'grid': '#e5e7eb',           # Light gray gridlines
    'grid_light': '#f3f4f6',     # Very light gridlines

    # Chart-specific colors (coordinated palette)
    'chart_1': '#2563eb',        # Blue
    'chart_2': '#6366f1',        # Indigo
    'chart_3': '#8b5cf6',        # Purple
    'chart_4': '#10b981',        # Teal
    'chart_5': '#f59e0b',        # Amber
    'chart_6': '#ef4444',        # Red

    # Component colors (for diagrams)
    'agent': '#3b82f6',          # Blue for agents
    'environment': '#6b7280',    # Gray for environment
    'state': '#6366f1',          # Indigo for states
    'action': '#2563eb',         # Blue for actions
    'reward': '#f59e0b',         # Amber for rewards
    'policy': '#8b5cf6',         # Purple for policy
    'value': '#10b981',          # Teal for value functions
    'critic': '#6366f1',         # Indigo for critic
    'actor': '#3b82f6',          # Blue for actor

    # For flowchart boxes - subtle fills with borders
    'box_fill': '#ffffff',       # White fill
    'box_fill_alt': '#f9fafb',   # Slightly off-white
    'box_border': '#d1d5db',     # Gray border
}

# =============================================================================
# Matplotlib RC Params for Professional Style
# =============================================================================

STYLE = {
    # Figure
    'figure.facecolor': 'white',
    'figure.edgecolor': 'none',
    'figure.dpi': 100,

    # Axes
    'axes.facecolor': 'white',
    'axes.edgecolor': '#d1d5db',
    'axes.linewidth': 1.0,
    'axes.labelcolor': '#374151',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.titlecolor': '#374151',
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Grid
    'axes.grid': True,
    'grid.color': '#e5e7eb',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,

    # Text
    'text.color': '#374151',
    'font.family': 'sans-serif',
    'font.size': 10,

    # Ticks
    'xtick.color': '#6b7280',
    'ytick.color': '#6b7280',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#d1d5db',
    'legend.fontsize': 9,

    # Lines
    'lines.linewidth': 1.5,
    'lines.markersize': 6,

    # Patches (for bars, boxes, etc.)
    'patch.linewidth': 1.0,
    'patch.edgecolor': '#d1d5db',

    # Savefig
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# =============================================================================
# Helper Functions
# =============================================================================

def apply_style():
    """Apply the professional style to matplotlib."""
    for key, value in STYLE.items():
        try:
            plt.rcParams[key] = value
        except KeyError:
            pass  # Skip invalid keys for older matplotlib versions


def get_box_style(style_type='default'):
    """Get consistent box styling for diagrams.

    Args:
        style_type: 'default', 'primary', 'secondary', 'accent', 'muted'

    Returns:
        dict with facecolor, edgecolor, linewidth
    """
    styles = {
        'default': {
            'facecolor': COLORS['box_fill'],
            'edgecolor': COLORS['box_border'],
            'linewidth': 1.5
        },
        'primary': {
            'facecolor': COLORS['box_fill'],
            'edgecolor': COLORS['primary'],
            'linewidth': 1.5
        },
        'secondary': {
            'facecolor': COLORS['box_fill'],
            'edgecolor': COLORS['secondary'],
            'linewidth': 1.5
        },
        'accent': {
            'facecolor': COLORS['box_fill'],
            'edgecolor': COLORS['accent'],
            'linewidth': 1.5
        },
        'muted': {
            'facecolor': COLORS['fill_light'],
            'edgecolor': COLORS['border'],
            'linewidth': 1.0
        },
        'filled_primary': {
            'facecolor': COLORS['primary'],
            'edgecolor': COLORS['primary_dark'],
            'linewidth': 1.5
        },
        'filled_secondary': {
            'facecolor': COLORS['secondary'],
            'edgecolor': COLORS['secondary_dark'],
            'linewidth': 1.5
        },
    }
    return styles.get(style_type, styles['default'])


def get_arrow_style(style_type='default'):
    """Get consistent arrow styling for diagrams.

    Args:
        style_type: 'default', 'primary', 'secondary', 'muted'

    Returns:
        dict with color, linewidth
    """
    styles = {
        'default': {
            'color': COLORS['text'],
            'linewidth': 1.5
        },
        'primary': {
            'color': COLORS['primary'],
            'linewidth': 2.0
        },
        'secondary': {
            'color': COLORS['secondary'],
            'linewidth': 1.5
        },
        'muted': {
            'color': COLORS['text_light'],
            'linewidth': 1.0
        },
        'accent': {
            'color': COLORS['accent'],
            'linewidth': 2.0
        },
    }
    return styles.get(style_type, styles['default'])


def get_chart_colors(n=6):
    """Get a list of chart colors.

    Args:
        n: Number of colors needed

    Returns:
        List of color hex codes
    """
    all_colors = [
        COLORS['chart_1'],
        COLORS['chart_2'],
        COLORS['chart_3'],
        COLORS['chart_4'],
        COLORS['chart_5'],
        COLORS['chart_6'],
    ]
    # Cycle if more colors needed
    return [all_colors[i % len(all_colors)] for i in range(n)]


def save_figure(fig, filepath, format='svg', dpi=150):
    """Save figure with consistent settings.

    Args:
        fig: matplotlib figure
        filepath: output path
        format: 'svg', 'png', 'pdf'
        dpi: resolution for raster formats
    """
    fig.savefig(
        filepath,
        format=format,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close(fig)


def create_text_box(text, style='default'):
    """Create consistent text box properties.

    Args:
        text: The text content (not used, for reference)
        style: 'default', 'highlight', 'muted'

    Returns:
        dict for bbox parameter in ax.text()
    """
    styles = {
        'default': {
            'boxstyle': 'round,pad=0.3',
            'facecolor': 'white',
            'edgecolor': COLORS['border'],
            'alpha': 0.95
        },
        'highlight': {
            'boxstyle': 'round,pad=0.3',
            'facecolor': COLORS['fill_light'],
            'edgecolor': COLORS['primary'],
            'alpha': 0.95
        },
        'muted': {
            'boxstyle': 'round,pad=0.3',
            'facecolor': COLORS['fill_very_light'],
            'edgecolor': COLORS['border_light'],
            'alpha': 0.9
        },
        'formula': {
            'boxstyle': 'round,pad=0.4',
            'facecolor': COLORS['fill_light'],
            'edgecolor': COLORS['border'],
            'alpha': 0.95
        },
    }
    return styles.get(style, styles['default'])


# =============================================================================
# Colormaps for heatmaps
# =============================================================================

def get_value_cmap():
    """Get a colormap suitable for value function visualization.

    Returns a diverging colormap from red (negative) through white to blue (positive).
    """
    from matplotlib.colors import LinearSegmentedColormap

    colors = ['#ef4444', '#fecaca', '#ffffff', '#bfdbfe', '#2563eb']
    return LinearSegmentedColormap.from_list('value', colors)


def get_sequential_cmap():
    """Get a sequential colormap for non-diverging data.

    Returns a colormap from light to primary blue.
    """
    from matplotlib.colors import LinearSegmentedColormap

    colors = ['#f9fafb', '#dbeafe', '#93c5fd', '#3b82f6', '#1d4ed8']
    return LinearSegmentedColormap.from_list('sequential', colors)


# =============================================================================
# Initialize style on import
# =============================================================================

apply_style()
