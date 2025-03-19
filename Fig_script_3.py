# Install necessary libraries (uncomment if needed in Colab)
# !pip install networkx matplotlib seaborn

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------------------------------------------------------------
# 1) Define the i-states and adjacency for R = 3
# ------------------------------------------------------------------------------

# Generate all binary states for R=3: 000, 001, 010, 011, 100, 101, 110, 111
states = [
    "000", "001", "010", "011",
    "100", "101", "110", "111"
]

# Function to check if two states differ by exactly one bit
def single_bit_difference(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2)) == 1

# Create edges for the graph: connect nodes that differ by one bit
edges = []
for i, s1 in enumerate(states):
    for j, s2 in enumerate(states):
        if j > i and single_bit_difference(s1, s2):
            edges.append((s1, s2))

# ------------------------------------------------------------------------------
# 2) Occupancy distributions at t1 and t2 (example data)
# ------------------------------------------------------------------------------
occupancy_t1 = {
    "000": 10,  # 10%
    "001": 5,
    "010": 5,
    "011": 10,
    "100": 20,
    "101": 20,
    "110": 20,
    "111": 10
}

occupancy_t2 = {
    "000": 5,
    "001": 10,
    "010": 10,
    "011": 5,
    "100": 15,
    "101": 20,
    "110": 25,
    "111": 10
}

# ------------------------------------------------------------------------------
# 3) Build the graph in NetworkX and choose a layout
# ------------------------------------------------------------------------------
G = nx.Graph()
G.add_nodes_from(states)
G.add_edges_from(edges)

# Use a consistent layout so that panels A and B line up identically
pos = nx.spring_layout(G, seed=42)  # Change layout as needed

# ------------------------------------------------------------------------------
# 4) Helper function to draw the i-state graph with occupancy
# ------------------------------------------------------------------------------
def draw_istate_graph(ax, G, pos, occupancy_dict, title):
    # Scale node sizes by occupancy (each percentage point -> 100 units)
    node_sizes = [occupancy_dict[node] * 100 for node in G.nodes()]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    
    # Draw nodes with viridis colormap
    nx.draw_networkx_nodes(
        G, pos, ax=ax, 
        node_size=node_sizes,
        node_color=node_sizes,
        cmap=plt.cm.viridis,  # using viridis
        alpha=0.9
    )
    
    # Draw labels for the nodes
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="black")
    
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()

# ------------------------------------------------------------------------------
# 5) Build the adjacency matrix for single-bit flips
# ------------------------------------------------------------------------------
n = len(states)
adj_matrix = np.zeros((n, n), dtype=int)
for i, s1 in enumerate(states):
    for j, s2 in enumerate(states):
        if single_bit_difference(s1, s2):
            adj_matrix[i, j] = 1

# ------------------------------------------------------------------------------
# 6) Plot the three panels
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: i-State Graph at t1
draw_istate_graph(axes[0], G, pos, occupancy_t1, 
                  title="(A) i-State Graph at $t_1$")

# Panel B: i-State Graph at t2
draw_istate_graph(axes[1], G, pos, occupancy_t2, 
                  title="(B) i-State Graph at $t_2$")

# Panel C: Adjacency Matrix (Allowed Single-Bit Transitions)
sns.heatmap(adj_matrix, ax=axes[2], 
            annot=True, cbar=False, square=True, 
            xticklabels=states, yticklabels=states,
            cmap="viridis",  # using viridis
            linecolor='black', linewidths=0.5)
axes[2].set_title("(C) Adjacency Matrix (1-bit difference)")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 7) Export the figure at 300 dpi (PNG) and as vector formats (PDF/SVG)
# ------------------------------------------------------------------------------
fig.savefig("liouville_3panel.png", dpi=300, format="png")
fig.savefig("liouville_3panel.pdf", format="pdf")
fig.savefig("liouville_3panel.svg", format="svg")
