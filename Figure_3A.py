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

# Assign Ricci curvature values based on geometric resistance
def assign_ricci_curvature(edges):
    curvature = {}
    for edge in edges:
        k_start = edge[0].count('1')
        k_end = edge[1].count('1')
        if k_start == 0 or k_end == 3:
            curvature[edge] = 0.8  # High resistance at boundaries
        elif abs(k_start - k_end) == 1:
            curvature[edge] = 0.5  # Intermediate resistance
        else:
            curvature[edge] = 0.2  # Low resistance (easy transitions)
    return curvature

# Create i-state transition graph
i_states = generate_istates(3)
edges = generate_edges(i_states)
G = nx.Graph()
G.add_nodes_from(i_states)
G.add_edges_from(edges)

# Compute Ricci curvature using Ollivier-Ricci flow
orc = OllivierRicci(G, alpha=0.5)
orc.compute_ricci_curvature()
ricci_curvature = nx.get_edge_attributes(orc.G, "ricciCurvature")

# Extract node positions in 3D space
pos = {state: (int(state[0]), int(state[1]), int(state[2])) for state in i_states}

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
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
ax.set_title("3D k-Space i-State Transitions with Ricci Curvature")

# Fix colorbar issue by linking it to the axes
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(ricci_curvature.values()), vmax=max(ricci_curvature.values())))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label("Ricci Curvature")

# Save image in /content at 300 dpi
plt.savefig("/content/Figure_3_k_space.png", dpi=300)
plt.show()
