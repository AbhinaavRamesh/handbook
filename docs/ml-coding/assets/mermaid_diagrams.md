# Neural Network Architecture Diagrams (Mermaid)

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
