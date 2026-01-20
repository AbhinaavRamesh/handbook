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


def create_forward_pass_animation_frames():
    """
    Create frames for forward pass data flow animation.
    Returns a list of figures that can be saved as GIF.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    n_frames = 20
    frames = []

    # Network structure: 3 layers
    layer_x = [1, 3, 5]
    layer_sizes = [4, 5, 2]
    layer_colors = [COLORS['input'], COLORS['hidden'], COLORS['output']]

    for frame in range(n_frames):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-1, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('#FAFAFA')

        # Title
        ax.text(3, 3.5, 'Forward Pass Data Flow', ha='center', fontsize=14,
                fontweight='bold', color=COLORS['text'])

        # Draw neurons for each layer
        all_positions = []
        for layer_idx, (x, n_neurons, color) in enumerate(zip(layer_x, layer_sizes, layer_colors)):
            positions = []
            spacing = 0.6
            y_start = 1.5 + (n_neurons - 1) * spacing / 2

            for i in range(n_neurons):
                y = y_start - i * spacing
                circle = Circle((x, y), 0.2, facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3, alpha=0.9)
                ax.add_patch(circle)
                positions.append((x, y))
            all_positions.append(positions)

        # Draw connections with animation effect
        progress = frame / (n_frames - 1)

        for layer_idx in range(len(all_positions) - 1):
            for pos1 in all_positions[layer_idx]:
                for pos2 in all_positions[layer_idx + 1]:
                    # Calculate if this connection should show data flow
                    connection_progress = progress * 2 - layer_idx * 0.5
                    if connection_progress > 0 and connection_progress <= 1:
                        # Animated data packet
                        t = min(1, max(0, connection_progress))
                        packet_x = pos1[0] + t * (pos2[0] - pos1[0])
                        packet_y = pos1[1] + t * (pos2[1] - pos1[1])

                        # Draw connection line
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                               color='#BDBDBD', alpha=0.3, linewidth=0.5, zorder=1)

                        # Draw data packet
                        packet = Circle((packet_x, packet_y), 0.08,
                                       facecolor='#FF5722', edgecolor='none',
                                       zorder=4, alpha=0.8)
                        ax.add_patch(packet)
                    else:
                        # Just draw connection line
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                               color='#BDBDBD', alpha=0.3, linewidth=0.5, zorder=1)

        # Layer labels
        labels = ['Input', 'Hidden', 'Output']
        for x, label in zip(layer_x, labels):
            ax.text(x, -0.5, label, ha='center', fontsize=10, fontweight='bold',
                   color=COLORS['text'])

        # Progress indicator
        ax.text(3, -0.9, f'Step {frame + 1}/{n_frames}', ha='center', fontsize=9,
               color='#666666')

        plt.tight_layout()
        frames.append(fig)

    return frames


def create_activation_comparison():
    """
    Create 2x3 grid showing activation functions and their derivatives.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=150)
    fig.suptitle('Activation Functions and Their Derivatives', fontsize=14, fontweight='bold')

    x = np.linspace(-4, 4, 200)

    activations = {
        'Sigmoid': (lambda z: 1 / (1 + np.exp(-z)),
                    lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
        'Tanh': (lambda z: np.tanh(z),
                 lambda z: 1 - np.tanh(z)**2),
        'ReLU': (lambda z: np.maximum(0, z),
                 lambda z: (z > 0).astype(float)),
        'LeakyReLU': (lambda z: np.where(z > 0, z, 0.01 * z),
                      lambda z: np.where(z > 0, 1, 0.01)),
        'ELU': (lambda z: np.where(z > 0, z, np.exp(z) - 1),
                lambda z: np.where(z > 0, 1, np.exp(z))),
        'GELU': (lambda z: z * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3))),
                 lambda z: 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3))) +
                          z * 0.5 * (1 - np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3))**2) *
                          np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * z**2))
    }

    colors = ['#E91E63', '#9C27B0', '#2196F3', '#00BCD4', '#4CAF50', '#FF9800']

    for idx, (name, (func, deriv)) in enumerate(activations.items()):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        y = func(x)
        dy = deriv(x)

        ax.plot(x, y, color=colors[idx], linewidth=2, label=f'{name}(x)')
        ax.plot(x, dy, color=colors[idx], linewidth=2, linestyle='--', alpha=0.7,
                label=f"{name}'(x)")

        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-2, 3)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_loss_landscape():
    """
    Create 3D loss landscape surface with optimization trajectory.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Create loss surface (sum of Gaussians for interesting landscape)
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Complex loss landscape with local minima
    Z = (X**2 + Y**2) * 0.3  # Bowl shape
    Z += 2 * np.exp(-((X-1)**2 + (Y-1)**2) / 0.5)  # Local minimum
    Z += 1.5 * np.exp(-((X+1.5)**2 + (Y-0.5)**2) / 0.3)  # Another local minimum
    Z -= 3 * np.exp(-(X**2 + Y**2) / 2)  # Global minimum at origin

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)

    # Optimization trajectory (gradient descent path)
    trajectory_x = [2.5]
    trajectory_y = [2.5]
    lr = 0.1

    for _ in range(50):
        cx, cy = trajectory_x[-1], trajectory_y[-1]
        # Numerical gradient
        eps = 0.01
        def loss(px, py):
            z = (px**2 + py**2) * 0.3
            z += 2 * np.exp(-((px-1)**2 + (py-1)**2) / 0.5)
            z += 1.5 * np.exp(-((px+1.5)**2 + (py-0.5)**2) / 0.3)
            z -= 3 * np.exp(-(px**2 + py**2) / 2)
            return z

        grad_x = (loss(cx + eps, cy) - loss(cx - eps, cy)) / (2 * eps)
        grad_y = (loss(cx, cy + eps) - loss(cx, cy - eps)) / (2 * eps)

        new_x = cx - lr * grad_x
        new_y = cy - lr * grad_y

        trajectory_x.append(np.clip(new_x, -3, 3))
        trajectory_y.append(np.clip(new_y, -3, 3))

    # Calculate Z values for trajectory
    trajectory_z = []
    for tx, ty in zip(trajectory_x, trajectory_y):
        tz = (tx**2 + ty**2) * 0.3
        tz += 2 * np.exp(-((tx-1)**2 + (ty-1)**2) / 0.5)
        tz += 1.5 * np.exp(-((tx+1.5)**2 + (ty-0.5)**2) / 0.3)
        tz -= 3 * np.exp(-(tx**2 + ty**2) / 2)
        trajectory_z.append(tz + 0.1)  # Slightly above surface

    # Plot trajectory
    ax.plot(trajectory_x, trajectory_y, trajectory_z, 'r-', linewidth=2,
            label='Gradient Descent Path', zorder=5)
    ax.scatter([trajectory_x[0]], [trajectory_y[0]], [trajectory_z[0]],
               color='green', s=100, zorder=6, label='Start')
    ax.scatter([trajectory_x[-1]], [trajectory_y[-1]], [trajectory_z[-1]],
               color='red', s=100, zorder=6, label='End')

    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape with Optimization Trajectory', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')

    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    return fig


def create_learning_rate_comparison():
    """
    Create comparison of training with different learning rates.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    np.random.seed(42)
    epochs = 100

    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    # Simulate loss curves for different learning rates
    for lr, color in zip(learning_rates, colors):
        losses = []
        loss = 2.0

        for epoch in range(epochs):
            # Simulate loss decay with noise
            if lr > 0.5:
                # Unstable: oscillating loss
                loss = loss * (0.99 - lr * 0.1) + 0.1 * np.sin(epoch * lr)
                loss += np.random.normal(0, 0.05 * lr)
                loss = max(0.1, min(3, loss))
            else:
                # Stable convergence
                decay = np.exp(-lr * epoch / 20)
                noise = np.random.normal(0, 0.02)
                loss = 0.1 + 1.9 * decay + noise
                loss = max(0.05, loss)
            losses.append(loss)

        axes[0].plot(range(epochs), losses, color=color, linewidth=2,
                     label=f'lr={lr}', alpha=0.9)

    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].set_title('Training Loss vs Learning Rate', fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_ylim(0, 3)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    for lr, color in zip(learning_rates, colors):
        accuracies = []
        acc = 0.1

        for epoch in range(epochs):
            if lr > 0.5:
                # Unstable
                acc = acc + (0.9 - acc) * 0.01 + 0.05 * np.sin(epoch * lr)
                acc += np.random.normal(0, 0.02)
                acc = max(0.1, min(0.85, acc))
            else:
                # Stable improvement
                improvement = (1 - np.exp(-lr * epoch / 15))
                noise = np.random.normal(0, 0.01)
                acc = 0.1 + 0.85 * improvement + noise
                acc = min(0.98, max(0.1, acc))
            accuracies.append(acc)

        axes[1].plot(range(epochs), accuracies, color=color, linewidth=2,
                     label=f'lr={lr}', alpha=0.9)

    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Accuracy', fontsize=10)
    axes[1].set_title('Training Accuracy vs Learning Rate', fontsize=11, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Effect of Learning Rate on Training', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_weight_initialization_comparison():
    """
    Create comparison of weight initialization distributions.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=150)
    fig.suptitle('Weight Initialization Distributions', fontsize=14, fontweight='bold')

    np.random.seed(42)
    n_in, n_out = 256, 128
    n_samples = 10000

    initializations = {
        'Zeros': lambda: np.zeros(n_samples),
        'Random Normal\n(std=0.01)': lambda: np.random.randn(n_samples) * 0.01,
        'Xavier Uniform': lambda: np.random.uniform(-np.sqrt(6/(n_in+n_out)),
                                                     np.sqrt(6/(n_in+n_out)), n_samples),
        'Xavier Normal': lambda: np.random.randn(n_samples) * np.sqrt(2/(n_in+n_out)),
        'He Uniform': lambda: np.random.uniform(-np.sqrt(6/n_in),
                                                 np.sqrt(6/n_in), n_samples),
        'He Normal': lambda: np.random.randn(n_samples) * np.sqrt(2/n_in)
    }

    colors = ['#9E9E9E', '#E91E63', '#2196F3', '#00BCD4', '#4CAF50', '#FF9800']

    for idx, (name, init_func) in enumerate(initializations.items()):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        weights = init_func()

        if name == 'Zeros':
            ax.axvline(x=0, color=colors[idx], linewidth=3)
            ax.set_xlim(-0.5, 0.5)
        else:
            ax.hist(weights, bins=50, color=colors[idx], alpha=0.7,
                    edgecolor='white', linewidth=0.5, density=True)

        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Weight Value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

        # Add statistics
        if name != 'Zeros':
            stats_text = f'$\\mu$={np.mean(weights):.4f}\n$\\sigma$={np.std(weights):.4f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


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

    # Generate Forward Pass Animation (GIF)
    print("6. Creating Forward Pass Animation...")
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        frames = create_forward_pass_animation_frames()

        # Save as GIF using individual frames
        import imageio
        gif_path = output_dir / "forward_pass_animation.gif"

        # Convert figures to images
        images = []
        for fig in frames:
            fig.canvas.draw()
            # Convert to numpy array
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf)
            images.append(image)
            plt.close(fig)

        # Save as GIF
        imageio.mimsave(gif_path, images, duration=0.15, loop=0)
        print(f"   Saved: {gif_path}")
    except ImportError:
        # Fallback: save first and last frame as static images
        print("   imageio not available, saving static frames instead...")
        frames = create_forward_pass_animation_frames()
        static_path = output_dir / "forward_pass_flow.png"
        frames[-1].savefig(static_path, format='png', bbox_inches='tight',
                          facecolor='#FAFAFA', edgecolor='none', dpi=150)
        for fig in frames:
            plt.close(fig)
        print(f"   Saved: {static_path}")

    # Generate Activation Function Comparison
    print("7. Creating Activation Function Comparison...")
    fig_act = create_activation_comparison()
    act_path = output_dir / "activation_functions.png"
    fig_act.savefig(act_path, format='png', bbox_inches='tight',
                    facecolor='white', edgecolor='none', dpi=150)
    plt.close(fig_act)
    print(f"   Saved: {act_path}")

    # Generate Loss Landscape
    print("8. Creating Loss Landscape...")
    fig_loss = create_loss_landscape()
    loss_path = output_dir / "loss_landscape.png"
    fig_loss.savefig(loss_path, format='png', bbox_inches='tight',
                     facecolor='white', edgecolor='none', dpi=150)
    plt.close(fig_loss)
    print(f"   Saved: {loss_path}")

    # Generate Learning Rate Comparison
    print("9. Creating Learning Rate Comparison...")
    fig_lr = create_learning_rate_comparison()
    lr_path = output_dir / "learning_rate_comparison.png"
    fig_lr.savefig(lr_path, format='png', bbox_inches='tight',
                   facecolor='white', edgecolor='none', dpi=150)
    plt.close(fig_lr)
    print(f"   Saved: {lr_path}")

    # Generate Weight Initialization Comparison
    print("10. Creating Weight Initialization Comparison...")
    fig_weights = create_weight_initialization_comparison()
    weights_path = output_dir / "weight_initialization.png"
    fig_weights.savefig(weights_path, format='png', bbox_inches='tight',
                        facecolor='white', edgecolor='none', dpi=150)
    plt.close(fig_weights)
    print(f"   Saved: {weights_path}")

    print("-" * 50)
    print("All diagrams generated successfully!")
    print("\nFiles created:")
    print(f"  - {mlp_path}")
    print(f"  - {cnn_path}")
    print(f"  - {ae_path}")
    print(f"  - {mermaid_path}")
    print(f"  - {act_path}")
    print(f"  - {loss_path}")
    print(f"  - {lr_path}")
    print(f"  - {weights_path}")


if __name__ == "__main__":
    main()
