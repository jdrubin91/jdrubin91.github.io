import matplotlib.pyplot as plt
import numpy as np

def draw_neural_network(ax, layer_sizes, layer_colors=None):
    """
    Draws a neural network diagram using Matplotlib.

    Parameters:
    - ax: Matplotlib Axes object where the network will be drawn.
    - layer_sizes: List of integers, representing the number of neurons in each layer.
    - layer_colors: List of colors for each layer (optional).
    """
    n_layers = len(layer_sizes)
    v_spacing = 1.0 / (max(layer_sizes) + 1)  # Vertical spacing
    h_spacing = 1.0 / (n_layers + 1)          # Horizontal spacing with padding

    # Use default colors if none are provided
    if layer_colors is None:
        # Generate a list of colors
        cmap = plt.get_cmap('Spectral')
        layer_colors = [cmap(i) for i in np.linspace(0, 1, n_layers)]
    
    # Nodes
    for n, (layer_size, color) in enumerate(zip(layer_sizes, layer_colors)):
        layer_top = 0.5 + (layer_size - 1) * v_spacing / 2  # Center layer vertically
        for m in range(layer_size):
            x = (n + 1) * h_spacing  # Shift right for padding
            y = layer_top - m * v_spacing
            circle = plt.Circle((x, y), 0.03, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            # Add neuron labels (optional)
            # ax.text(x, y, f'{n+1},{m+1}', fontsize=6, ha='center', va='center')

    # Edges
    for n in range(n_layers - 1):
        layer_size_a = layer_sizes[n]
        layer_size_b = layer_sizes[n + 1]
        layer_top_a = 0.5 + (layer_size_a - 1) * v_spacing / 2
        layer_top_b = 0.5 + (layer_size_b - 1) * v_spacing / 2
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                x1 = (n + 1) * h_spacing
                y1 = layer_top_a - m * v_spacing
                x2 = (n + 2) * h_spacing
                y2 = layer_top_b - o * v_spacing
                line = plt.Line2D([x1, x2], [y1, y2], c='gray')
                ax.add_artist(line)

    # Adjust plot limits to ensure neurons are not cut off
    ax.set_xlim(0, (n_layers + 1) * h_spacing)
    ax.set_ylim(0, 1)
    ax.axis('off')

# Example usage
fig = plt.figure(figsize=(12, 8))
ax = fig.gca()

# Define the network architecture: [Input layer size, Hidden layer sizes..., Output layer size]
layer_sizes = [1, 5, 10, 10, 2, 1]  # Example architecture

# Define colors for each layer
layer_colors = ['gold', 'plum', 'lightgreen', 'lightgreen', 'lightcoral', 'skyblue']

draw_neural_network(ax, layer_sizes, layer_colors)

plt.savefig('neural_network.png', bbox_inches='tight', pad_inches = 0)
