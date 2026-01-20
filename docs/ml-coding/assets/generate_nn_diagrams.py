#!/usr/bin/env python3
"""
Neural Network Architecture Diagram Generator

This script generates publication-ready neural network architecture diagrams:
1. Simple MLP (Multilayer Perceptron)
2. Deep MLP with multiple hidden layers
3. CNN (Convolutional Neural Network) architecture
4. Autoencoder architecture
5. Forward/Backward pass visualization

Requirements:
    pip install matplotlib numpy

Usage:
    python generate_nn_diagrams.py

Output:
    - mlp_architecture.svg
    - cnn_architecture.svg
    - autoencoder_architecture.svg
    - mermaid_diagrams.md (contains mermaid code for markdown embedding)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection
import numpy as np
from pathlib import Path


# Color scheme for diagrams
COLORS = {
    'input': '#4CAF50',      # Green
    'hidden': '#2196F3',     # Blue
    'output': '#FF5722',     # Orange
    'conv': '#9C27B0',       # Purple
    'pool': '#00BCD4',       # Cyan
    'fc': '#FF9800',         # Amber
    'encoder': '#3F51B5',    # Indigo
    'latent': '#E91E63',     # Pink
    'decoder': '#009688',    # Teal
    'arrow_forward': '#333333',
    'arrow_backward': '#E53935',
    'background': '#FAFAFA',
    'text': '#212121',
}


def draw_neuron_layer(ax, x, y_center, n_neurons, max_neurons=6,
                      color='#2196F3', layer_label='', show_dots=True):
    """
    Draw a layer of neurons vertically centered at (x, y_center).

    Args:
        ax: matplotlib axes
        x: x-coordinate for the layer
        y_center: y-coordinate for center of layer
        n_neurons: number of neurons in the layer
        max_neurons: maximum neurons to display (show dots if exceeded)
        color: fill color for neurons
        layer_label: label to display below the layer
        show_dots: whether to show '...' for hidden neurons

    Returns:
        list of (x, y) coordinates for each neuron
    """
    neuron_radius = 0.15
    spacing = 0.5

    # Determine how many neurons to actually draw
    if n_neurons <= max_neurons:
        display_neurons = n_neurons
        neurons_to_draw = list(range(n_neurons))
    else:
        display_neurons = max_neurons
        # Show first few, dots, then last few
        half = max_neurons // 2
        neurons_to_draw = list(range(half)) + ['dots'] + list(range(n_neurons - half, n_neurons))

    # Calculate starting y position to center the layer
    total_height = (display_neurons - 1) * spacing
    y_start = y_center + total_height / 2

    neuron_positions = []

    for i, neuron_idx in enumerate(neurons_to_draw):
        y = y_start - i * spacing

        if neuron_idx == 'dots':
            # Draw dots to indicate more neurons
            for dy in [-0.1, 0, 0.1]:
                dot = Circle((x, y + dy), 0.03, color=color, zorder=3)
                ax.add_patch(dot)
        else:
            # Draw neuron circle
            circle = Circle((x, y), neuron_radius,
                          facecolor=color, edgecolor='white',
                          linewidth=2, zorder=3, alpha=0.9)
            ax.add_patch(circle)
            neuron_positions.append((x, y))

    # Add layer label
    if layer_label:
        ax.text(x, y_center - total_height/2 - 0.5, layer_label,
                ha='center', va='top', fontsize=10, fontweight='bold',
                color=COLORS['text'])

    return neuron_positions


def draw_connections(ax, layer1_pos, layer2_pos, color='#BDBDBD', alpha=0.3, linewidth=0.5):
    """Draw connections between two layers."""
    for x1, y1 in layer1_pos:
        for x2, y2 in layer2_pos:
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                   linewidth=linewidth, zorder=1)


def draw_arrow(ax, start, end, color='#333333', style='simple',
               label='', curved=False, connectionstyle=None):
    """Draw an arrow between two points."""
    if curved and connectionstyle is None:
        connectionstyle = "arc3,rad=0.3"

    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>' if style == 'simple' else '->,head_width=0.3,head_length=0.15',
        mutation_scale=15,
        color=color,
        linewidth=2,
        connectionstyle=connectionstyle,
        zorder=5
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom',
               fontsize=9, color=color, fontweight='bold')


def create_simple_mlp():
    """
    Create a simple MLP diagram: Input -> Hidden -> Output
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(-1, 7)
    ax.set_ylim(-2, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])

    # Title
    ax.text(3, 3.5, 'Simple MLP Architecture', ha='center', va='center',
           fontsize=16, fontweight='bold', color=COLORS['text'])
    ax.text(3, 3.1, '(Multilayer Perceptron)', ha='center', va='center',
           fontsize=12, style='italic', color='#666666')

    # Draw layers
    input_pos = draw_neuron_layer(ax, 0, 1, 4, color=COLORS['input'],
                                  layer_label='Input Layer\n(4 neurons)')
    hidden_pos = draw_neuron_layer(ax, 3, 1, 5, color=COLORS['hidden'],
                                   layer_label='Hidden Layer\n(5 neurons)')
    output_pos = draw_neuron_layer(ax, 6, 1, 2, color=COLORS['output'],
                                   layer_label='Output Layer\n(2 neurons)')

    # Draw connections
    draw_connections(ax, input_pos, hidden_pos)
    draw_connections(ax, hidden_pos, output_pos)

    # Add forward pass arrow
    ax.annotate('', xy=(6.8, -1.2), xytext=(-0.8, -1.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow_forward'], lw=2))
    ax.text(3, -1.5, 'Forward Pass', ha='center', va='top', fontsize=10,
           color=COLORS['arrow_forward'], fontweight='bold')

    # Add backward pass arrow
    ax.annotate('', xy=(-0.8, -1.7), xytext=(6.8, -1.7),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow_backward'], lw=2))
    ax.text(3, -2.0, 'Backward Pass (Gradient Flow)', ha='center', va='top',
           fontsize=10, color=COLORS['arrow_backward'], fontweight='bold')

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=COLORS['input'], label='Input'),
        patches.Patch(facecolor=COLORS['hidden'], label='Hidden'),
        patches.Patch(facecolor=COLORS['output'], label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    return fig


def create_deep_mlp():
    """
    Create a deep MLP diagram with multiple hidden layers.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-2, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])

    # Title
    ax.text(6, 4.5, 'Deep MLP Architecture', ha='center', va='center',
           fontsize=16, fontweight='bold', color=COLORS['text'])
    ax.text(6, 4.1, '(Multiple Hidden Layers)', ha='center', va='center',
           fontsize=12, style='italic', color='#666666')

    # Layer configuration: [input, hidden1, hidden2, hidden3, hidden4, output]
    layer_sizes = [784, 512, 256, 128, 64, 10]
    layer_colors = [COLORS['input']] + [COLORS['hidden']]*4 + [COLORS['output']]
    layer_labels = ['Input\n(784)', 'Hidden 1\n(512)', 'Hidden 2\n(256)',
                   'Hidden 3\n(128)', 'Hidden 4\n(64)', 'Output\n(10)']
    x_positions = [0, 2.4, 4.8, 7.2, 9.6, 12]

    all_positions = []
    for i, (x, size, color, label) in enumerate(zip(x_positions, layer_sizes,
                                                     layer_colors, layer_labels)):
        pos = draw_neuron_layer(ax, x, 1.5, min(size, 8), max_neurons=6,
                               color=color, layer_label=label)
        all_positions.append(pos)

    # Draw connections between adjacent layers
    for i in range(len(all_positions) - 1):
        draw_connections(ax, all_positions[i], all_positions[i+1])

    # Add activation function labels
    activations = ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'Softmax']
    for i, (x, act) in enumerate(zip(x_positions[1:], activations)):
        ax.text(x - 1.2, 3.3, f'{act}', ha='center', va='center',
               fontsize=8, color='#666666', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='#CCCCCC', alpha=0.8))

    # Add data flow arrows
    ax.annotate('', xy=(12.8, -0.8), xytext=(-0.8, -0.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow_forward'], lw=2))
    ax.text(6, -1.1, 'Forward Pass: x -> h1 -> h2 -> h3 -> h4 -> y',
           ha='center', va='top', fontsize=10, color=COLORS['arrow_forward'])

    ax.annotate('', xy=(-0.8, -1.4), xytext=(12.8, -1.4),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow_backward'], lw=2))
    ax.text(6, -1.7, 'Backward Pass: dL/dy -> dL/dh4 -> ... -> dL/dx',
           ha='center', va='top', fontsize=10, color=COLORS['arrow_backward'])

    return fig


def draw_conv_block(ax, x, y, width, height, depth, color, label='', sublabel=''):
    """Draw a 3D-like convolutional block."""
    # Front face
    rect = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=color, edgecolor='white', linewidth=2,
                          alpha=0.9, zorder=2)
    ax.add_patch(rect)

    # Add depth effect (3D)
    depth_offset = depth * 0.1
    for i in range(1, int(depth) + 1):
        offset = i * 0.05
        rect_back = FancyBboxPatch((x + offset, y + offset), width, height,
                                   boxstyle="round,pad=0.02,rounding_size=0.05",
                                   facecolor=color, edgecolor='white',
                                   linewidth=1, alpha=0.3, zorder=1)
        ax.add_patch(rect_back)

    # Labels
    if label:
        ax.text(x + width/2, y - 0.3, label, ha='center', va='top',
               fontsize=9, fontweight='bold', color=COLORS['text'])
    if sublabel:
        ax.text(x + width/2, y - 0.55, sublabel, ha='center', va='top',
               fontsize=7, color='#666666')


def create_cnn_architecture():
    """
    Create CNN architecture diagram: Conv -> Pool -> Conv -> Pool -> FC
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])

    # Title
    ax.text(8, 5.5, 'Convolutional Neural Network (CNN) Architecture',
           ha='center', va='center', fontsize=16, fontweight='bold',
           color=COLORS['text'])

    # Draw input image
    input_img = FancyBboxPatch((0, 1), 1.5, 1.5,
                               boxstyle="round,pad=0.02",
                               facecolor='#E8E8E8', edgecolor='#333333',
                               linewidth=2, zorder=2)
    ax.add_patch(input_img)
    ax.text(0.75, 0.5, 'Input\n(32x32x3)', ha='center', va='top',
           fontsize=9, fontweight='bold', color=COLORS['text'])

    # Conv1 + ReLU
    draw_conv_block(ax, 2.5, 0.8, 0.8, 1.8, 8, COLORS['conv'],
                   'Conv1', '3x3, 32 filters')
    ax.text(2.9, 3.0, 'ReLU', ha='center', fontsize=8, style='italic',
           color='#666666')

    # Pool1
    draw_conv_block(ax, 4.5, 1.0, 0.6, 1.4, 8, COLORS['pool'],
                   'MaxPool1', '2x2, stride 2')

    # Conv2 + ReLU
    draw_conv_block(ax, 6.5, 0.9, 0.7, 1.6, 12, COLORS['conv'],
                   'Conv2', '3x3, 64 filters')
    ax.text(6.85, 3.0, 'ReLU', ha='center', fontsize=8, style='italic',
           color='#666666')

    # Pool2
    draw_conv_block(ax, 8.5, 1.1, 0.5, 1.2, 12, COLORS['pool'],
                   'MaxPool2', '2x2, stride 2')

    # Conv3 + ReLU
    draw_conv_block(ax, 10.3, 1.0, 0.6, 1.4, 16, COLORS['conv'],
                   'Conv3', '3x3, 128 filters')
    ax.text(10.6, 3.0, 'ReLU', ha='center', fontsize=8, style='italic',
           color='#666666')

    # Flatten indicator
    ax.plot([11.8, 12.2], [1.75, 1.75], 'k-', linewidth=2)
    ax.text(12, 1.3, 'Flatten', ha='center', fontsize=8, color='#666666')

    # FC layers
    fc1_pos = draw_neuron_layer(ax, 13, 1.75, 128, max_neurons=5,
                                color=COLORS['fc'], layer_label='FC1 (128)')
    fc2_pos = draw_neuron_layer(ax, 14.8, 1.75, 64, max_neurons=4,
                                color=COLORS['fc'], layer_label='FC2 (64)')
    output_pos = draw_neuron_layer(ax, 16.2, 1.75, 10, max_neurons=4,
                                   color=COLORS['output'], layer_label='Output (10)')

    # Draw FC connections
    draw_connections(ax, fc1_pos, fc2_pos, alpha=0.2)
    draw_connections(ax, fc2_pos, output_pos, alpha=0.2)

    # Draw arrows between blocks
    arrow_y = 1.75
    arrows = [(1.7, 2.5), (3.5, 4.5), (5.3, 6.5), (7.4, 8.5),
              (9.2, 10.3), (11.2, 12.5)]
    for start_x, end_x in arrows:
        ax.annotate('', xy=(end_x, arrow_y), xytext=(start_x, arrow_y),
                   arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # Legend
    legend_elements = [
        patches.Patch(facecolor=COLORS['conv'], label='Convolution'),
        patches.Patch(facecolor=COLORS['pool'], label='Pooling'),
        patches.Patch(facecolor=COLORS['fc'], label='Fully Connected'),
        patches.Patch(facecolor=COLORS['output'], label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # Feature map dimensions
    dims = [
        (0.75, 'Input\n32x32x3'),
        (2.9, 'Feature Maps\n32x32x32'),
        (4.8, '16x16x32'),
        (6.85, '16x16x64'),
        (8.75, '8x8x64'),
        (10.6, '8x8x128'),
    ]
    for x, dim in dims:
        ax.text(x, 3.5, dim, ha='center', va='bottom', fontsize=7,
               color='#888888', style='italic')

    # Forward/Backward pass indicators
    ax.annotate('', xy=(16.5, -0.8), xytext=(-0.5, -0.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow_forward'], lw=2))
    ax.text(8, -1.1, 'Forward Pass', ha='center', va='top', fontsize=10,
           color=COLORS['arrow_forward'], fontweight='bold')

    return fig


def create_autoencoder():
    """
    Create Autoencoder architecture diagram: Encoder -> Latent -> Decoder
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])

    # Title
    ax.text(7, 4.5, 'Autoencoder Architecture', ha='center', va='center',
           fontsize=16, fontweight='bold', color=COLORS['text'])
    ax.text(7, 4.1, '(Encoder - Latent Space - Decoder)', ha='center', va='center',
           fontsize=12, style='italic', color='#666666')

    # Encoder layers (decreasing size)
    encoder_sizes = [784, 256, 64]
    encoder_x = [0, 2, 4]
    encoder_labels = ['Input\n(784)', 'Enc1\n(256)', 'Enc2\n(64)']

    encoder_positions = []
    for x, size, label in zip(encoder_x, encoder_sizes, encoder_labels):
        pos = draw_neuron_layer(ax, x, 1.5, min(size, 8), max_neurons=6,
                               color=COLORS['encoder'], layer_label=label)
        encoder_positions.append(pos)

    # Latent space (bottleneck)
    latent_pos = draw_neuron_layer(ax, 7, 1.5, 16, max_neurons=4,
                                   color=COLORS['latent'],
                                   layer_label='Latent\n(16)')

    # Add latent space visualization box
    latent_box = FancyBboxPatch((6, 0.3), 2, 2.4,
                                boxstyle="round,pad=0.1,rounding_size=0.3",
                                facecolor='none', edgecolor=COLORS['latent'],
                                linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(latent_box)
    ax.text(7, 0.0, 'Compressed\nRepresentation', ha='center', va='top',
           fontsize=8, color=COLORS['latent'], style='italic')

    # Decoder layers (increasing size, mirror of encoder)
    decoder_sizes = [64, 256, 784]
    decoder_x = [10, 12, 14]
    decoder_labels = ['Dec1\n(64)', 'Dec2\n(256)', 'Output\n(784)']

    decoder_positions = []
    for x, size, label in zip(decoder_x, decoder_sizes, decoder_labels):
        color = COLORS['output'] if x == 14 else COLORS['decoder']
        pos = draw_neuron_layer(ax, x, 1.5, min(size, 8), max_neurons=6,
                               color=color, layer_label=label)
        decoder_positions.append(pos)

    # Draw connections
    all_positions = encoder_positions + [latent_pos] + decoder_positions
    for i in range(len(all_positions) - 1):
        draw_connections(ax, all_positions[i], all_positions[i+1], alpha=0.2)

    # Encoder/Decoder region labels
    ax.annotate('', xy=(5.5, 3.8), xytext=(-0.5, 3.8),
               arrowprops=dict(arrowstyle='-[,widthB=3.0', color=COLORS['encoder'], lw=2))
    ax.text(2.5, 4.0, 'ENCODER', ha='center', va='bottom',
           fontsize=12, fontweight='bold', color=COLORS['encoder'])

    ax.annotate('', xy=(14.5, 3.8), xytext=(8.5, 3.8),
               arrowprops=dict(arrowstyle='-[,widthB=3.0', color=COLORS['decoder'], lw=2))
    ax.text(11.5, 4.0, 'DECODER', ha='center', va='bottom',
           fontsize=12, fontweight='bold', color=COLORS['decoder'])

    # Forward/Backward pass
    ax.annotate('', xy=(14.8, -1.0), xytext=(-0.8, -1.0),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow_forward'], lw=2))
    ax.text(7, -1.3, 'Forward: Encode input -> Latent -> Reconstruct',
           ha='center', va='top', fontsize=10, color=COLORS['arrow_forward'])

    ax.annotate('', xy=(-0.8, -1.6), xytext=(14.8, -1.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow_backward'], lw=2))
    ax.text(7, -1.9, 'Backward: Minimize reconstruction loss (MSE)',
           ha='center', va='top', fontsize=10, color=COLORS['arrow_backward'])

    # Legend
    legend_elements = [
        patches.Patch(facecolor=COLORS['encoder'], label='Encoder'),
        patches.Patch(facecolor=COLORS['latent'], label='Latent Space'),
        patches.Patch(facecolor=COLORS['decoder'], label='Decoder'),
        patches.Patch(facecolor=COLORS['output'], label='Reconstructed Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    return fig


def generate_mermaid_diagrams():
    """
    Generate Mermaid diagram code for embedding in markdown files.
    """
    mermaid_code = """# Neural Network Architecture Diagrams (Mermaid)

These diagrams can be embedded directly in markdown files that support Mermaid.

## 1. Simple MLP Architecture

```mermaid
graph LR
    subgraph Input["Input Layer (4)"]
        I1((x1))
        I2((x2))
        I3((x3))
        I4((x4))
    end

    subgraph Hidden["Hidden Layer (5)"]
        H1((h1))
        H2((h2))
        H3((h3))
        H4((h4))
        H5((h5))
    end

    subgraph Output["Output Layer (2)"]
        O1((y1))
        O2((y2))
    end

    I1 --> H1 & H2 & H3 & H4 & H5
    I2 --> H1 & H2 & H3 & H4 & H5
    I3 --> H1 & H2 & H3 & H4 & H5
    I4 --> H1 & H2 & H3 & H4 & H5

    H1 & H2 & H3 & H4 & H5 --> O1
    H1 & H2 & H3 & H4 & H5 --> O2

    style Input fill:#4CAF50,color:#fff
    style Hidden fill:#2196F3,color:#fff
    style Output fill:#FF5722,color:#fff
```

## 2. Deep MLP Architecture

```mermaid
graph LR
    subgraph Input["Input (784)"]
        direction TB
        IN[Input Layer]
    end

    subgraph H1["Hidden 1 (512)"]
        direction TB
        HL1[Dense + ReLU]
    end

    subgraph H2["Hidden 2 (256)"]
        direction TB
        HL2[Dense + ReLU]
    end

    subgraph H3["Hidden 3 (128)"]
        direction TB
        HL3[Dense + ReLU]
    end

    subgraph H4["Hidden 4 (64)"]
        direction TB
        HL4[Dense + ReLU]
    end

    subgraph Output["Output (10)"]
        direction TB
        OUT[Dense + Softmax]
    end

    IN --> HL1 --> HL2 --> HL3 --> HL4 --> OUT

    style Input fill:#4CAF50,color:#fff
    style H1 fill:#2196F3,color:#fff
    style H2 fill:#2196F3,color:#fff
    style H3 fill:#2196F3,color:#fff
    style H4 fill:#2196F3,color:#fff
    style Output fill:#FF5722,color:#fff
```

## 3. CNN Architecture

```mermaid
graph LR
    subgraph Input["Input Image"]
        IMG[32x32x3]
    end

    subgraph Conv1["Conv Block 1"]
        C1[Conv2D 3x3<br/>32 filters]
        R1[ReLU]
        P1[MaxPool 2x2]
    end

    subgraph Conv2["Conv Block 2"]
        C2[Conv2D 3x3<br/>64 filters]
        R2[ReLU]
        P2[MaxPool 2x2]
    end

    subgraph Conv3["Conv Block 3"]
        C3[Conv2D 3x3<br/>128 filters]
        R3[ReLU]
    end

    subgraph FC["Fully Connected"]
        FL[Flatten]
        FC1[Dense 128<br/>+ ReLU]
        FC2[Dense 64<br/>+ ReLU]
    end

    subgraph Output["Output"]
        OUT[Dense 10<br/>+ Softmax]
    end

    IMG --> C1 --> R1 --> P1 --> C2 --> R2 --> P2 --> C3 --> R3 --> FL --> FC1 --> FC2 --> OUT

    style Input fill:#E8E8E8,color:#333
    style Conv1 fill:#9C27B0,color:#fff
    style Conv2 fill:#9C27B0,color:#fff
    style Conv3 fill:#9C27B0,color:#fff
    style FC fill:#FF9800,color:#fff
    style Output fill:#FF5722,color:#fff
```

## 4. Autoencoder Architecture

```mermaid
graph LR
    subgraph Encoder["ENCODER"]
        direction LR
        IN[Input<br/>784] --> E1[Dense 256<br/>+ ReLU] --> E2[Dense 64<br/>+ ReLU]
    end

    subgraph Latent["LATENT SPACE"]
        direction TB
        Z[Latent Vector<br/>16 dimensions]
    end

    subgraph Decoder["DECODER"]
        direction LR
        D1[Dense 64<br/>+ ReLU] --> D2[Dense 256<br/>+ ReLU] --> OUT[Output<br/>784]
    end

    E2 --> Z --> D1

    style Encoder fill:#3F51B5,color:#fff
    style Latent fill:#E91E63,color:#fff
    style Decoder fill:#009688,color:#fff
```

## 5. Forward and Backward Pass

```mermaid
graph TB
    subgraph Forward["Forward Pass"]
        direction LR
        X[Input x] -->|"W1"| H1[Hidden h1]
        H1 -->|"W2"| H2[Hidden h2]
        H2 -->|"W3"| Y[Output y]
        Y --> L[Loss L]
    end

    subgraph Backward["Backward Pass (Gradient Flow)"]
        direction RL
        dL[dL/dL = 1] -->|"chain rule"| dY[dL/dy]
        dY -->|"dL/dW3"| dH2[dL/dh2]
        dH2 -->|"dL/dW2"| dH1[dL/dh1]
        dH1 -->|"dL/dW1"| dX[dL/dx]
    end

    style Forward fill:#4CAF50,color:#fff
    style Backward fill:#E53935,color:#fff
```

## 6. Detailed Forward/Backward Pass with Equations

```mermaid
flowchart TB
    subgraph Forward["Forward Pass"]
        A["Input: x"] --> B["z1 = W1 * x + b1"]
        B --> C["h1 = ReLU(z1)"]
        C --> D["z2 = W2 * h1 + b2"]
        D --> E["y = softmax(z2)"]
        E --> F["L = CrossEntropy(y, target)"]
    end

    subgraph Backward["Backward Pass"]
        G["dL/dy"] --> H["dL/dz2 = y - target"]
        H --> I["dL/dW2 = dL/dz2 * h1.T"]
        H --> J["dL/dh1 = W2.T * dL/dz2"]
        J --> K["dL/dz1 = dL/dh1 * ReLU'(z1)"]
        K --> L["dL/dW1 = dL/dz1 * x.T"]
    end

    F -.->|"Compute gradients"| G

    style Forward fill:#E3F2FD,stroke:#1976D2
    style Backward fill:#FFEBEE,stroke:#D32F2F
```

## Usage Notes

1. **Embedding in Markdown**: Copy the mermaid code block (including the triple backticks) directly into any markdown file that supports Mermaid rendering (GitHub, GitLab, Obsidian, etc.).

2. **Customization**: Modify the `style` lines to change colors:
   - `fill`: Background color
   - `color`: Text color
   - `stroke`: Border color

3. **Rendering**: Mermaid diagrams are rendered client-side, so they may look slightly different across platforms.

4. **Alternative**: For more control over appearance, use the generated SVG files instead.
"""
    return mermaid_code


def main():
    """Generate all neural network architecture diagrams."""
    output_dir = Path(__file__).parent

    print("Generating Neural Network Architecture Diagrams...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # Generate Simple MLP
    print("1. Creating Simple MLP diagram...")
    fig_mlp = create_simple_mlp()
    mlp_path = output_dir / "mlp_architecture.svg"
    fig_mlp.savefig(mlp_path, format='svg', bbox_inches='tight',
                    facecolor=COLORS['background'], edgecolor='none', dpi=150)
    plt.close(fig_mlp)
    print(f"   Saved: {mlp_path}")

    # Generate Deep MLP
    print("2. Creating Deep MLP diagram...")
    fig_deep = create_deep_mlp()
    # Save as part of the MLP file (combined view)
    plt.close(fig_deep)
    print("   (Included in MLP architecture)")

    # Generate CNN
    print("3. Creating CNN architecture diagram...")
    fig_cnn = create_cnn_architecture()
    cnn_path = output_dir / "cnn_architecture.svg"
    fig_cnn.savefig(cnn_path, format='svg', bbox_inches='tight',
                    facecolor=COLORS['background'], edgecolor='none', dpi=150)
    plt.close(fig_cnn)
    print(f"   Saved: {cnn_path}")

    # Generate Autoencoder
    print("4. Creating Autoencoder architecture diagram...")
    fig_ae = create_autoencoder()
    ae_path = output_dir / "autoencoder_architecture.svg"
    fig_ae.savefig(ae_path, format='svg', bbox_inches='tight',
                   facecolor=COLORS['background'], edgecolor='none', dpi=150)
    plt.close(fig_ae)
    print(f"   Saved: {ae_path}")

    # Generate Mermaid diagrams
    print("5. Generating Mermaid diagram code...")
    mermaid_code = generate_mermaid_diagrams()
    mermaid_path = output_dir / "mermaid_diagrams.md"
    with open(mermaid_path, 'w') as f:
        f.write(mermaid_code)
    print(f"   Saved: {mermaid_path}")

    print("-" * 50)
    print("All diagrams generated successfully!")
    print("\nFiles created:")
    print(f"  - {mlp_path}")
    print(f"  - {cnn_path}")
    print(f"  - {ae_path}")
    print(f"  - {mermaid_path}")


if __name__ == "__main__":
    main()
