#!/usr/bin/env python3
"""
Gradient Descent Visualization Generator

This script generates animated visualizations demonstrating various aspects of
gradient descent optimization algorithms. It creates educational GIFs showing:

1. 2D contour plots with gradient descent paths
2. 3D surface plots with optimization trajectories
3. Comparison of different optimizers (SGD, Momentum, Adam)
4. Learning rate effects (too high, too low, optimal)
5. Loss curves over iterations

Author: ML Coding Reference
Date: 2024
License: MIT

Requirements:
    pip install numpy matplotlib pillow

Usage:
    python generate_gradient_descent_viz.py

    Or import and use individual functions:
        from generate_gradient_descent_viz import generate_all_visualizations
        generate_all_visualizations(output_dir='./assets/')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Tuple, List, Callable, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style for better visualizations
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


# =============================================================================
# Objective Functions
# =============================================================================

def rosenbrock(x: np.ndarray, y: np.ndarray, a: float = 1, b: float = 100) -> np.ndarray:
    """
    Rosenbrock function - a classic non-convex optimization test function.

    f(x, y) = (a - x)^2 + b(y - x^2)^2

    Global minimum at (a, a^2) = (1, 1) with f(1, 1) = 0

    Args:
        x: x-coordinate(s)
        y: y-coordinate(s)
        a: Parameter (default 1)
        b: Parameter controlling valley steepness (default 100)

    Returns:
        Function values at (x, y)
    """
    return (a - x)**2 + b * (y - x**2)**2


def rosenbrock_gradient(x: float, y: float, a: float = 1, b: float = 100) -> Tuple[float, float]:
    """
    Gradient of the Rosenbrock function.

    Args:
        x: x-coordinate
        y: y-coordinate
        a: Parameter (default 1)
        b: Parameter (default 100)

    Returns:
        Tuple of (df/dx, df/dy)
    """
    dfdx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dfdy = 2 * b * (y - x**2)
    return dfdx, dfdy


def quadratic_bowl(x: np.ndarray, y: np.ndarray,
                   a: float = 1, b: float = 10) -> np.ndarray:
    """
    Elliptical quadratic bowl function.

    f(x, y) = a*x^2 + b*y^2

    Global minimum at (0, 0) with f(0, 0) = 0
    Different scaling in x and y directions creates an elongated bowl.

    Args:
        x: x-coordinate(s)
        y: y-coordinate(s)
        a: Scaling for x^2 term
        b: Scaling for y^2 term

    Returns:
        Function values at (x, y)
    """
    return a * x**2 + b * y**2


def quadratic_bowl_gradient(x: float, y: float,
                            a: float = 1, b: float = 10) -> Tuple[float, float]:
    """
    Gradient of the quadratic bowl function.

    Args:
        x: x-coordinate
        y: y-coordinate
        a: Scaling for x^2 term
        b: Scaling for y^2 term

    Returns:
        Tuple of (df/dx, df/dy)
    """
    return 2 * a * x, 2 * b * y


def beale_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Beale function - another classic test function with a flat valley.

    f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

    Global minimum at (3, 0.5) with f(3, 0.5) = 0

    Args:
        x: x-coordinate(s)
        y: y-coordinate(s)

    Returns:
        Function values at (x, y)
    """
    term1 = (1.5 - x + x * y)**2
    term2 = (2.25 - x + x * y**2)**2
    term3 = (2.625 - x + x * y**3)**2
    return term1 + term2 + term3


def beale_gradient(x: float, y: float) -> Tuple[float, float]:
    """
    Gradient of the Beale function.

    Args:
        x: x-coordinate
        y: y-coordinate

    Returns:
        Tuple of (df/dx, df/dy)
    """
    term1 = 1.5 - x + x * y
    term2 = 2.25 - x + x * y**2
    term3 = 2.625 - x + x * y**3

    dfdx = 2 * term1 * (-1 + y) + 2 * term2 * (-1 + y**2) + 2 * term3 * (-1 + y**3)
    dfdy = 2 * term1 * x + 2 * term2 * (2 * x * y) + 2 * term3 * (3 * x * y**2)

    return dfdx, dfdy


# =============================================================================
# Optimizer Implementations
# =============================================================================

class GradientDescentOptimizer:
    """Base class for gradient descent optimizers."""

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize optimizer.

        Args:
            learning_rate: Step size for parameter updates
        """
        self.learning_rate = learning_rate
        self.path = []  # Store optimization path
        self.losses = []  # Store loss values

    def step(self, x: float, y: float, grad_x: float, grad_y: float) -> Tuple[float, float]:
        """
        Perform one optimization step.

        Args:
            x: Current x position
            y: Current y position
            grad_x: Gradient w.r.t. x
            grad_y: Gradient w.r.t. y

        Returns:
            Updated (x, y) position
        """
        raise NotImplementedError

    def reset(self):
        """Reset optimizer state."""
        self.path = []
        self.losses = []


class SGD(GradientDescentOptimizer):
    """
    Vanilla Stochastic Gradient Descent.

    Update rule: theta = theta - lr * gradient

    Simple but can oscillate in narrow valleys and get stuck in local minima.
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def step(self, x: float, y: float, grad_x: float, grad_y: float) -> Tuple[float, float]:
        """Perform vanilla gradient descent step."""
        x_new = x - self.learning_rate * grad_x
        y_new = y - self.learning_rate * grad_y
        return x_new, y_new


class MomentumSGD(GradientDescentOptimizer):
    """
    SGD with Momentum.

    Update rule:
        v = momentum * v - lr * gradient
        theta = theta + v

    Momentum helps accelerate convergence and dampens oscillations.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize Momentum SGD.

        Args:
            learning_rate: Step size
            momentum: Momentum coefficient (typically 0.9)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def step(self, x: float, y: float, grad_x: float, grad_y: float) -> Tuple[float, float]:
        """Perform momentum-accelerated gradient descent step."""
        self.velocity_x = self.momentum * self.velocity_x - self.learning_rate * grad_x
        self.velocity_y = self.momentum * self.velocity_y - self.learning_rate * grad_y

        x_new = x + self.velocity_x
        y_new = y + self.velocity_y
        return x_new, y_new

    def reset(self):
        """Reset optimizer state including velocity."""
        super().reset()
        self.velocity_x = 0.0
        self.velocity_y = 0.0


class Adam(GradientDescentOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines ideas from:
    - Momentum (exponential moving average of gradients)
    - RMSprop (exponential moving average of squared gradients)

    Update rules:
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2
        m_hat = m / (1 - beta1^t)  # Bias correction
        v_hat = v / (1 - beta2^t)  # Bias correction
        theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
    """

    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Step size
            beta1: Exponential decay rate for first moment (default 0.9)
            beta2: Exponential decay rate for second moment (default 0.999)
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_x = 0.0  # First moment (mean) for x
        self.m_y = 0.0  # First moment (mean) for y
        self.v_x = 0.0  # Second moment (variance) for x
        self.v_y = 0.0  # Second moment (variance) for y
        self.t = 0  # Time step

    def step(self, x: float, y: float, grad_x: float, grad_y: float) -> Tuple[float, float]:
        """Perform Adam optimization step."""
        self.t += 1

        # Update first moment estimates
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y

        # Update second moment estimates
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x**2
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y**2

        # Bias correction
        m_x_hat = self.m_x / (1 - self.beta1**self.t)
        m_y_hat = self.m_y / (1 - self.beta1**self.t)
        v_x_hat = self.v_x / (1 - self.beta2**self.t)
        v_y_hat = self.v_y / (1 - self.beta2**self.t)

        # Update parameters
        x_new = x - self.learning_rate * m_x_hat / (np.sqrt(v_x_hat) + self.epsilon)
        y_new = y - self.learning_rate * m_y_hat / (np.sqrt(v_y_hat) + self.epsilon)

        return x_new, y_new

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.t = 0


class RMSprop(GradientDescentOptimizer):
    """
    RMSprop optimizer.

    Maintains a moving average of squared gradients to normalize the gradient.

    Update rules:
        v = decay * v + (1 - decay) * gradient^2
        theta = theta - lr * gradient / (sqrt(v) + epsilon)
    """

    def __init__(self, learning_rate: float = 0.01, decay: float = 0.9,
                 epsilon: float = 1e-8):
        """
        Initialize RMSprop optimizer.

        Args:
            learning_rate: Step size
            decay: Decay rate for moving average (default 0.9)
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.decay = decay
        self.epsilon = epsilon
        self.v_x = 0.0
        self.v_y = 0.0

    def step(self, x: float, y: float, grad_x: float, grad_y: float) -> Tuple[float, float]:
        """Perform RMSprop optimization step."""
        self.v_x = self.decay * self.v_x + (1 - self.decay) * grad_x**2
        self.v_y = self.decay * self.v_y + (1 - self.decay) * grad_y**2

        x_new = x - self.learning_rate * grad_x / (np.sqrt(self.v_x) + self.epsilon)
        y_new = y - self.learning_rate * grad_y / (np.sqrt(self.v_y) + self.epsilon)

        return x_new, y_new

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.v_x = 0.0
        self.v_y = 0.0


# =============================================================================
# Optimization Path Generation
# =============================================================================

def run_optimization(
    optimizer: GradientDescentOptimizer,
    objective_fn: Callable,
    gradient_fn: Callable,
    start_point: Tuple[float, float],
    num_iterations: int = 100,
    tolerance: float = 1e-8
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Run gradient descent optimization and record the path.

    Args:
        optimizer: Optimizer instance
        objective_fn: Objective function f(x, y)
        gradient_fn: Gradient function returning (df/dx, df/dy)
        start_point: Initial (x, y) coordinates
        num_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (path, losses) where:
            - path: List of (x, y) coordinates
            - losses: List of function values
    """
    optimizer.reset()
    x, y = start_point

    path = [(x, y)]
    losses = [objective_fn(x, y)]

    for i in range(num_iterations):
        # Compute gradient
        grad_x, grad_y = gradient_fn(x, y)

        # Check for convergence
        if np.sqrt(grad_x**2 + grad_y**2) < tolerance:
            break

        # Update position
        x, y = optimizer.step(x, y, grad_x, grad_y)

        # Clip to reasonable bounds to prevent divergence
        x = np.clip(x, -10, 10)
        y = np.clip(y, -10, 10)

        path.append((x, y))
        losses.append(objective_fn(x, y))

    return path, losses


# =============================================================================
# Visualization Functions
# =============================================================================

def create_2d_contour_animation(
    output_path: str,
    objective_fn: Callable = quadratic_bowl,
    gradient_fn: Callable = quadratic_bowl_gradient,
    optimizer: Optional[GradientDescentOptimizer] = None,
    start_point: Tuple[float, float] = (-4.0, 4.0),
    num_iterations: int = 50,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    title: str = "Gradient Descent on 2D Surface"
):
    """
    Create animated 2D contour plot showing gradient descent path.

    Args:
        output_path: Path to save GIF
        objective_fn: Objective function
        gradient_fn: Gradient function
        optimizer: Optimizer to use (default: SGD with lr=0.05)
        start_point: Starting (x, y) coordinates
        num_iterations: Number of optimization steps
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        title: Plot title
    """
    if optimizer is None:
        optimizer = SGD(learning_rate=0.05)

    # Generate optimization path
    path, losses = run_optimization(
        optimizer, objective_fn, gradient_fn,
        start_point, num_iterations
    )

    # Create mesh for contour plot
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_fn(X, Y)

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Contour plot
    levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), 30)
    contour = ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(title)
    ax1.set_aspect('equal')

    # Mark optimal point
    ax1.plot(0, 0, 'r*', markersize=15, label='Optimal', zorder=5)

    # Initialize path line and point
    line, = ax1.plot([], [], 'r-', linewidth=2, label='Path')
    point, = ax1.plot([], [], 'ro', markersize=10, zorder=5)
    ax1.legend()

    # Loss curve
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss Over Iterations')
    ax2.set_yscale('log')
    loss_line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_xlim(0, len(losses))
    ax2.set_ylim(min(losses) * 0.9, max(losses) * 1.1)

    # Iteration counter text
    iter_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                         verticalalignment='top', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        line.set_data([], [])
        point.set_data([], [])
        loss_line.set_data([], [])
        iter_text.set_text('')
        return line, point, loss_line, iter_text

    def animate(frame):
        # Update path
        path_x = [p[0] for p in path[:frame+1]]
        path_y = [p[1] for p in path[:frame+1]]
        line.set_data(path_x, path_y)
        point.set_data([path[frame][0]], [path[frame][1]])

        # Update loss curve
        loss_line.set_data(range(frame+1), losses[:frame+1])

        # Update iteration text
        iter_text.set_text(f'Iteration: {frame}\nLoss: {losses[frame]:.4f}')

        return line, point, loss_line, iter_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(path),
        interval=100, blit=True
    )

    # Save animation
    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    plt.close()
    print(f"Saved: {output_path}")


def create_3d_surface_animation(
    output_path: str,
    objective_fn: Callable = quadratic_bowl,
    gradient_fn: Callable = quadratic_bowl_gradient,
    optimizer: Optional[GradientDescentOptimizer] = None,
    start_point: Tuple[float, float] = (-4.0, 4.0),
    num_iterations: int = 50,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    title: str = "3D Gradient Descent Visualization"
):
    """
    Create 3D surface plot with gradient descent trajectory.

    Args:
        output_path: Path to save GIF
        objective_fn: Objective function
        gradient_fn: Gradient function
        optimizer: Optimizer to use
        start_point: Starting coordinates
        num_iterations: Number of steps
        x_range: x-axis range
        y_range: y-axis range
        title: Plot title
    """
    if optimizer is None:
        optimizer = SGD(learning_rate=0.05)

    # Generate path
    path, losses = run_optimization(
        optimizer, objective_fn, gradient_fn,
        start_point, num_iterations
    )

    # Create mesh
    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = objective_fn(X, Y)

    # Set up 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

    # Initial setup
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)

    # Path line and point in 3D
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    path_z = losses

    line, = ax.plot([], [], [], 'r-', linewidth=2, label='Optimization Path')
    point, = ax.plot([], [], [], 'ro', markersize=8)
    ax.legend()

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def animate(frame):
        line.set_data(path_x[:frame+1], path_y[:frame+1])
        line.set_3d_properties(path_z[:frame+1])
        point.set_data([path_x[frame]], [path_y[frame]])
        point.set_3d_properties([path_z[frame]])

        # Rotate view
        ax.view_init(elev=30, azim=frame * 2)

        return line, point

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(path),
        interval=100, blit=False
    )

    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    plt.close()
    print(f"Saved: {output_path}")


def create_optimizer_comparison_animation(
    output_path: str,
    objective_fn: Callable = quadratic_bowl,
    gradient_fn: Callable = quadratic_bowl_gradient,
    start_point: Tuple[float, float] = (-4.0, 4.0),
    num_iterations: int = 100,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5)
):
    """
    Create animation comparing SGD, Momentum, and Adam optimizers.

    Shows how different optimizers navigate the same loss landscape.

    Args:
        output_path: Path to save GIF
        objective_fn: Objective function
        gradient_fn: Gradient function
        start_point: Starting coordinates
        num_iterations: Number of steps
        x_range: x-axis range
        y_range: y-axis range
    """
    # Define optimizers with comparable learning rates
    optimizers = {
        'SGD (lr=0.05)': SGD(learning_rate=0.05),
        'Momentum (lr=0.01, m=0.9)': MomentumSGD(learning_rate=0.01, momentum=0.9),
        'Adam (lr=0.3)': Adam(learning_rate=0.3),
    }

    colors = {'SGD (lr=0.05)': 'red',
              'Momentum (lr=0.01, m=0.9)': 'blue',
              'Adam (lr=0.3)': 'green'}

    # Generate paths for all optimizers
    paths = {}
    all_losses = {}
    for name, opt in optimizers.items():
        path, losses = run_optimization(
            opt, objective_fn, gradient_fn,
            start_point, num_iterations
        )
        paths[name] = path
        all_losses[name] = losses

    # Create mesh
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_fn(X, Y)

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Contour plot
    levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), 30)
    ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Optimizer Comparison: SGD vs Momentum vs Adam')
    ax1.set_aspect('equal')

    # Mark start and optimal
    ax1.plot(start_point[0], start_point[1], 'k^', markersize=12,
             label='Start', zorder=10)
    ax1.plot(0, 0, 'r*', markersize=15, label='Optimal', zorder=10)

    # Initialize lines and points for each optimizer
    lines = {}
    points = {}
    for name, color in colors.items():
        lines[name], = ax1.plot([], [], '-', color=color, linewidth=2,
                                label=name)
        points[name], = ax1.plot([], [], 'o', color=color, markersize=8,
                                 zorder=5)

    ax1.legend(loc='upper right', fontsize=9)

    # Loss comparison plot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss Convergence Comparison')
    ax2.set_yscale('log')

    loss_lines = {}
    max_len = max(len(losses) for losses in all_losses.values())
    ax2.set_xlim(0, max_len)

    # Calculate y-axis limits
    all_loss_values = [l for losses in all_losses.values() for l in losses]
    ax2.set_ylim(min(all_loss_values) * 0.5, max(all_loss_values) * 1.5)

    for name, color in colors.items():
        loss_lines[name], = ax2.plot([], [], '-', color=color, linewidth=2,
                                      label=name)
    ax2.legend(loc='upper right')

    # Iteration text
    iter_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                         verticalalignment='top', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        for name in colors.keys():
            lines[name].set_data([], [])
            points[name].set_data([], [])
            loss_lines[name].set_data([], [])
        iter_text.set_text('')
        return list(lines.values()) + list(points.values()) + list(loss_lines.values()) + [iter_text]

    def animate(frame):
        for name in colors.keys():
            path = paths[name]
            losses = all_losses[name]

            # Ensure we don't exceed path length
            idx = min(frame, len(path) - 1)

            path_x = [p[0] for p in path[:idx+1]]
            path_y = [p[1] for p in path[:idx+1]]
            lines[name].set_data(path_x, path_y)
            points[name].set_data([path[idx][0]], [path[idx][1]])

            loss_idx = min(frame, len(losses) - 1)
            loss_lines[name].set_data(range(loss_idx+1), losses[:loss_idx+1])

        iter_text.set_text(f'Iteration: {frame}')

        return list(lines.values()) + list(points.values()) + list(loss_lines.values()) + [iter_text]

    max_frames = max(len(path) for path in paths.values())
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=max_frames,
        interval=100, blit=True
    )

    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    plt.close()
    print(f"Saved: {output_path}")


def create_learning_rate_comparison_animation(
    output_path: str,
    objective_fn: Callable = quadratic_bowl,
    gradient_fn: Callable = quadratic_bowl_gradient,
    start_point: Tuple[float, float] = (-4.0, 4.0),
    num_iterations: int = 100,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5)
):
    """
    Create animation showing the effect of different learning rates.

    Demonstrates:
    - Too small: Slow convergence
    - Too large: Oscillation/divergence
    - Just right: Optimal convergence

    Args:
        output_path: Path to save GIF
        objective_fn: Objective function
        gradient_fn: Gradient function
        start_point: Starting coordinates
        num_iterations: Number of steps
        x_range: x-axis range
        y_range: y-axis range
    """
    # Different learning rates
    learning_rates = {
        'Too Small (lr=0.001)': 0.001,
        'Just Right (lr=0.05)': 0.05,
        'Too Large (lr=0.15)': 0.15,
    }

    colors = {
        'Too Small (lr=0.001)': 'blue',
        'Just Right (lr=0.05)': 'green',
        'Too Large (lr=0.15)': 'red',
    }

    # Generate paths
    paths = {}
    all_losses = {}
    for name, lr in learning_rates.items():
        optimizer = SGD(learning_rate=lr)
        path, losses = run_optimization(
            optimizer, objective_fn, gradient_fn,
            start_point, num_iterations
        )
        paths[name] = path
        all_losses[name] = losses

    # Create mesh
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_fn(X, Y)

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Contour plot
    levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), 30)
    ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Learning Rate Effect on Gradient Descent')
    ax1.set_aspect('equal')

    # Mark start and optimal
    ax1.plot(start_point[0], start_point[1], 'k^', markersize=12,
             label='Start', zorder=10)
    ax1.plot(0, 0, 'm*', markersize=15, label='Optimal', zorder=10)

    # Initialize lines and points
    lines = {}
    points = {}
    for name, color in colors.items():
        lines[name], = ax1.plot([], [], '-', color=color, linewidth=2,
                                label=name, alpha=0.8)
        points[name], = ax1.plot([], [], 'o', color=color, markersize=8,
                                 zorder=5)

    ax1.legend(loc='upper right', fontsize=9)

    # Loss plot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss Convergence: Learning Rate Comparison')
    ax2.set_yscale('log')

    loss_lines = {}
    max_len = max(len(losses) for losses in all_losses.values())
    ax2.set_xlim(0, max_len)

    all_loss_values = [l for losses in all_losses.values() for l in losses]
    ax2.set_ylim(min(all_loss_values) * 0.5, max(all_loss_values) * 1.5)

    for name, color in colors.items():
        loss_lines[name], = ax2.plot([], [], '-', color=color, linewidth=2)
    ax2.legend(loc='upper right')

    # Iteration text
    iter_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                         verticalalignment='top', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        for name in colors.keys():
            lines[name].set_data([], [])
            points[name].set_data([], [])
            loss_lines[name].set_data([], [])
        iter_text.set_text('')
        return list(lines.values()) + list(points.values()) + list(loss_lines.values()) + [iter_text]

    def animate(frame):
        for name in colors.keys():
            path = paths[name]
            losses = all_losses[name]

            idx = min(frame, len(path) - 1)

            path_x = [p[0] for p in path[:idx+1]]
            path_y = [p[1] for p in path[:idx+1]]
            lines[name].set_data(path_x, path_y)
            points[name].set_data([path[idx][0]], [path[idx][1]])

            loss_idx = min(frame, len(losses) - 1)
            loss_lines[name].set_data(range(loss_idx+1), losses[:loss_idx+1])

        iter_text.set_text(f'Iteration: {frame}')

        return list(lines.values()) + list(points.values()) + list(loss_lines.values()) + [iter_text]

    max_frames = max(len(path) for path in paths.values())
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=max_frames,
        interval=100, blit=True
    )

    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    plt.close()
    print(f"Saved: {output_path}")


def create_loss_curve_plot(
    output_path: str,
    objective_fn: Callable = quadratic_bowl,
    gradient_fn: Callable = quadratic_bowl_gradient,
    start_point: Tuple[float, float] = (-4.0, 4.0),
    num_iterations: int = 100
):
    """
    Create static plot showing loss curves for different optimizers.

    Args:
        output_path: Path to save PNG
        objective_fn: Objective function
        gradient_fn: Gradient function
        start_point: Starting coordinates
        num_iterations: Number of steps
    """
    optimizers = {
        'SGD': SGD(learning_rate=0.05),
        'Momentum': MomentumSGD(learning_rate=0.01, momentum=0.9),
        'RMSprop': RMSprop(learning_rate=0.1),
        'Adam': Adam(learning_rate=0.3),
    }

    colors = ['red', 'blue', 'orange', 'green']

    fig, ax = plt.subplots(figsize=(10, 6))

    for (name, opt), color in zip(optimizers.items(), colors):
        _, losses = run_optimization(
            opt, objective_fn, gradient_fn,
            start_point, num_iterations
        )
        ax.plot(losses, label=name, color=color, linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Convergence: Optimizer Comparison', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_rosenbrock_animation(
    output_path: str,
    start_point: Tuple[float, float] = (-1.5, 1.5),
    num_iterations: int = 200
):
    """
    Create animation on the challenging Rosenbrock function.

    The Rosenbrock function has a narrow curved valley that is difficult
    to navigate, making it a good test for optimizers.

    Args:
        output_path: Path to save GIF
        start_point: Starting coordinates
        num_iterations: Number of steps
    """
    optimizers = {
        'SGD': SGD(learning_rate=0.001),
        'Momentum': MomentumSGD(learning_rate=0.0005, momentum=0.9),
        'Adam': Adam(learning_rate=0.05),
    }

    colors = {'SGD': 'red', 'Momentum': 'blue', 'Adam': 'green'}

    # Generate paths
    paths = {}
    all_losses = {}
    for name, opt in optimizers.items():
        path, losses = run_optimization(
            opt, rosenbrock, rosenbrock_gradient,
            start_point, num_iterations
        )
        paths[name] = path
        all_losses[name] = losses

    # Create mesh
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Contour plot with log scale for better visualization
    levels = np.logspace(0, 3.5, 30)
    ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Rosenbrock Function: Optimizers in a Curved Valley')

    # Mark optimal point (1, 1)
    ax1.plot(1, 1, 'r*', markersize=15, label='Optimal (1, 1)', zorder=10)
    ax1.plot(start_point[0], start_point[1], 'k^', markersize=12,
             label='Start', zorder=10)

    # Initialize lines and points
    lines = {}
    points = {}
    for name, color in colors.items():
        lines[name], = ax1.plot([], [], '-', color=color, linewidth=2,
                                label=name)
        points[name], = ax1.plot([], [], 'o', color=color, markersize=8,
                                 zorder=5)

    ax1.legend(loc='upper left', fontsize=9)

    # Loss plot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Rosenbrock: Loss Convergence')
    ax2.set_yscale('log')

    loss_lines = {}
    max_len = max(len(losses) for losses in all_losses.values())
    ax2.set_xlim(0, max_len)

    all_loss_values = [l for losses in all_losses.values() for l in losses if l > 0]
    if all_loss_values:
        ax2.set_ylim(min(all_loss_values) * 0.5, max(all_loss_values) * 1.5)

    for name, color in colors.items():
        loss_lines[name], = ax2.plot([], [], '-', color=color, linewidth=2)
    ax2.legend(loc='upper right')

    iter_text = ax1.text(0.02, 0.02, '', transform=ax1.transAxes,
                         verticalalignment='bottom', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        for name in colors.keys():
            lines[name].set_data([], [])
            points[name].set_data([], [])
            loss_lines[name].set_data([], [])
        iter_text.set_text('')
        return list(lines.values()) + list(points.values()) + list(loss_lines.values()) + [iter_text]

    def animate(frame):
        for name in colors.keys():
            path = paths[name]
            losses = all_losses[name]

            idx = min(frame, len(path) - 1)

            path_x = [p[0] for p in path[:idx+1]]
            path_y = [p[1] for p in path[:idx+1]]
            lines[name].set_data(path_x, path_y)
            points[name].set_data([path[idx][0]], [path[idx][1]])

            loss_idx = min(frame, len(losses) - 1)
            loss_lines[name].set_data(range(loss_idx+1), losses[:loss_idx+1])

        iter_text.set_text(f'Iteration: {frame}')

        return list(lines.values()) + list(points.values()) + list(loss_lines.values()) + [iter_text]

    max_frames = max(len(path) for path in paths.values())
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=max_frames,
        interval=50, blit=True
    )

    anim.save(output_path, writer='pillow', fps=20, dpi=100)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_all_visualizations(output_dir: str = None):
    """
    Generate all gradient descent visualizations.

    Args:
        output_dir: Directory to save outputs (default: script directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Gradient Descent Visualizations")
    print("=" * 60)
    print(f"Output directory: {output_dir}\n")

    # 1. Basic 2D contour animation
    print("[1/6] Creating 2D contour animation...")
    create_2d_contour_animation(
        os.path.join(output_dir, 'gradient_descent_2d.gif'),
        objective_fn=quadratic_bowl,
        gradient_fn=quadratic_bowl_gradient,
        start_point=(-4.0, 4.0),
        num_iterations=50
    )

    # 2. 3D surface animation
    print("[2/6] Creating 3D surface animation...")
    create_3d_surface_animation(
        os.path.join(output_dir, 'gradient_descent_3d.gif'),
        objective_fn=quadratic_bowl,
        gradient_fn=quadratic_bowl_gradient,
        start_point=(-4.0, 4.0),
        num_iterations=50
    )

    # 3. Optimizer comparison
    print("[3/6] Creating optimizer comparison animation...")
    create_optimizer_comparison_animation(
        os.path.join(output_dir, 'gradient_descent_comparison.gif'),
        objective_fn=quadratic_bowl,
        gradient_fn=quadratic_bowl_gradient,
        start_point=(-4.0, 4.0),
        num_iterations=100
    )

    # 4. Learning rate comparison
    print("[4/6] Creating learning rate comparison animation...")
    create_learning_rate_comparison_animation(
        os.path.join(output_dir, 'learning_rate_effect.gif'),
        objective_fn=quadratic_bowl,
        gradient_fn=quadratic_bowl_gradient,
        start_point=(-4.0, 4.0),
        num_iterations=100
    )

    # 5. Loss curve plot
    print("[5/6] Creating loss curve plot...")
    create_loss_curve_plot(
        os.path.join(output_dir, 'loss_curves.png'),
        objective_fn=quadratic_bowl,
        gradient_fn=quadratic_bowl_gradient,
        start_point=(-4.0, 4.0),
        num_iterations=100
    )

    # 6. Rosenbrock function animation
    print("[6/6] Creating Rosenbrock function animation...")
    create_rosenbrock_animation(
        os.path.join(output_dir, 'rosenbrock_optimization.gif'),
        start_point=(-1.5, 1.5),
        num_iterations=200
    )

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {os.path.join(output_dir, 'gradient_descent_2d.gif')}")
    print(f"  - {os.path.join(output_dir, 'gradient_descent_3d.gif')}")
    print(f"  - {os.path.join(output_dir, 'gradient_descent_comparison.gif')}")
    print(f"  - {os.path.join(output_dir, 'learning_rate_effect.gif')}")
    print(f"  - {os.path.join(output_dir, 'loss_curves.png')}")
    print(f"  - {os.path.join(output_dir, 'rosenbrock_optimization.gif')}")


if __name__ == '__main__':
    # When run directly, generate all visualizations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_visualizations(script_dir)
