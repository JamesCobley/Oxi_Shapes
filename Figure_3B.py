import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from GraphRicciCurvature.OllivierRicci import OllivierRicci

# Generate all binary i-states for R = 3
def generate_istates(r):
    return [format(i, f'0{r}b') for i in range(2**r)]

# Create transition edges for stepwise redox transitions
def generate_edges(i_states):
    edges = []
    for state in i_states:
        for i in range(len(state)):
            new_state = list(state)
            new_state[i] = '1' if new_state[i] == '0' else '0'
            new_state = "".join(new_state)
            if new_state in i_states:
                edges.append((state, new_state))
    return edges

# Assign Ricci curvature values based on increasing resistance
def assign_ricci_curvature(edges):
    curvature = {}
    for edge in edges:
        k_start = edge[0].count('1')
        k_end = edge[1].count('1')
        
        # Custom Ricci curvature profile for deformation
        if (k_start == 0 and k_end == 1) or (k_start == 1 and k_end == 0):
            curvature[edge] = 0.2  # Low resistance
        elif (k_start == 1 and k_end == 2) or (k_start == 2 and k_end == 1):
            curvature[edge] = 0.6  # Moderate resistance
        elif (k_start == 2 and k_end == 3) or (k_start == 3 and k_end == 2):
            curvature[edge] = 1.0  # Very high resistance
        else:
            curvature[edge] = 0.5  # Default mid-range resistance
            
    return curvature

# Create i-state transition graph
i_states = generate_istates(3)
edges = generate_edges(i_states)
G = nx.Graph()
G.add_nodes_from(i_states)
G.add_edges_from(edges)

# Compute Ricci curvature using the new deformation rules
ricci_curvature = assign_ricci_curvature(edges)

# Assign Ricci curvature as an edge attribute
nx.set_edge_attributes(G, ricci_curvature, "weight")

# Apply a force-directed layout in 3D to deform the hypercube
if len(G) < 500:
    pos_2d = nx.kamada_kawai_layout(G)  # Better for small graphs
else:
    pos_2d = nx.spring_layout(G, weight="weight")  # Ensure weights are used correctly

pos = {node: (pos_2d[node][0], pos_2d[node][1], node.count('1')) for node in G.nodes()}  # k as Z-axis

# Assign colors to nodes based on k-state
k_colors = {0: "blue", 1: "green", 2: "orange", 3: "red"}
node_colors = [k_colors[state.count('1')] for state in i_states]

# Prepare 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw nodes with color mapping
for state, (x, y, z), color in zip(i_states, pos.values(), node_colors):
    ax.scatter(x, y, z, color=color, s=100)
    ax.text(x, y, z, state, fontsize=10)

# Draw edges with color mapped to Ricci curvature
for (node1, node2), curvature in ricci_curvature.items():
    x_vals = [pos[node1][0], pos[node2][0]]
    y_vals = [pos[node1][1], pos[node2][1]]
    z_vals = [pos[node1][2], pos[node2][2]]
    ax.plot(x_vals, y_vals, z_vals, color=plt.cm.viridis(curvature), linewidth=2)

# Create a legend for node colors
for k, color in k_colors.items():
    ax.scatter([], [], [], color=color, label=f"k = {k}")

ax.legend(title="k-State", loc='upper left')

# Set labels
ax.set_xlabel("X-Dimension")
ax.set_ylabel("Y-Dimension")
ax.set_zlabel("k-State (Oxidation Level)")
ax.set_title("Deformed k-Manifold: Ricci Curvature Warped Hypercube")

# Fix colorbar issue
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(ricci_curvature.values()), vmax=max(ricci_curvature.values())))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label("Ricci Curvature")

# Save image at 300 dpi
plt.savefig("/content/Deformed_k_manifold.png", dpi=300)
plt.show()
