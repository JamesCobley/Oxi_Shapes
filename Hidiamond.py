import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

##############################################################################
# 1. Generate the Pascal-like Diamond Structure for Any R
##############################################################################
def generate_diamond_structure(R):
    """
    Generates the universal flat diamond structure for a given R,
    where each level k has binomial(R, k) states, creating a Pascal-like diamond.
    """
    positions = {}
    edges = []
    node_labels = {}
    
    y_spacing = 1  # Vertical spacing between levels
    x_spacing = 2  # Horizontal spacing multiplier
    
    node_id = 0  # Unique node identifier
    for k in range(R + 1):
        num_nodes = int(np.math.comb(R, k))  # Number of i-states in k-manifold
        x_start = - (num_nodes - 1) * x_spacing / 2  # Centering the row
        
        for i in range(num_nodes):
            positions[node_id] = (x_start + i * x_spacing, -k * y_spacing)
            node_labels[node_id] = f"{k}-{i}"  # Label as k-state, i-th state in row
            
            # Connect to previous k-level (single-bit flips allowed)
            if k > 0:
                prev_num_nodes = int(np.math.comb(R, k - 1))
                for prev_id in range(sum(int(np.math.comb(R, j)) for j in range(k)) - prev_num_nodes,
                                     sum(int(np.math.comb(R, j)) for j in range(k))):
                    edges.append((prev_id, node_id))
            
            node_id += 1
    
    return positions, edges, node_labels

##############################################################################
# 2. Plot the Universal Diamond for Any R
##############################################################################
def plot_diamond(R):
    """
    Plots the Pascal-like diamond structure for a given R, ensuring visual clarity.
    """
    positions, edges, node_labels = generate_diamond_structure(R)
    
    G = nx.Graph()
    G.add_edges_from(edges)
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos=positions, with_labels=True, labels=node_labels, 
            node_color='lightblue', edge_color='gray', node_size=500, font_size=8)
    
    plt.title(f"Universal Flat Diamond for R = {R}")
    plt.show()

# Example visualization for R = 10
plot_diamond(R=10)
