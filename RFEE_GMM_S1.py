#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes: Pipeline Step 1 — Generate the redox diamond solution for the given fractional occupancy
using a custom C-Ricci measure (abandoning standard ORC).
Inputs: Fractional positional density data.
Outputs: Pickle solution file, plots (geometry, energy, heat, custom curvature), printouts checking the geometry.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from networkx.algorithms.components import connected_components
from mpl_toolkits.mplot3d import Axes3D

###############################################
# 1. Define Proteoform States & Allowed Transitions
###############################################
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
allowed_edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "101"), ("001", "011"),
    ("010", "110"), ("010", "011"),
    ("011", "111"),
    ("100", "110"), ("100", "101"),
    ("101", "111"),
    ("110", "111"),
]
G = nx.Graph()
G.add_nodes_from(pf_states)
G.add_edges_from(allowed_edges)

###############################################
# 2. Define Occupancy (ρ) at Each Node
###############################################
occupancy = {
    "000": 0.25,
    "001": 0.25, "010": 0.25, "100": 0.25,
    "011": 0.0,  "101": 0.0,  "110": 0.0,  "111": 0.0
}
rho_vec = np.array([occupancy[s] for s in pf_states])

###############################################
# 3. Solve a Poisson-like PDE for φ
###############################################
# We treat the allowed edges as the adjacency for the PDE.
N = len(pf_states)
A = np.zeros((N, N))
state_index = {s: i for i, s in enumerate(pf_states)}
for i, s1 in enumerate(pf_states):
    for j, s2 in enumerate(pf_states):
        if G.has_edge(s1, s2):
            A[i, j] = 1
D = np.diag(A.sum(axis=1))
L = D - A

phi_vec = np.zeros(N)
kappa = 1.0
max_iter = 1000
tol = 1e-3
damping = 0.05

for _ in range(max_iter):
    nonlin = 0.5 * kappa * rho_vec * np.exp(2 * phi_vec)
    F = L @ phi_vec + nonlin
    J = L + np.diag(kappa * rho_vec * np.exp(2 * phi_vec))
    delta_phi = np.linalg.solve(J, -F)
    # Do not update nodes with zero occupancy
    delta_phi[rho_vec == 0] = 0.0
    phi_vec += damping * delta_phi
    if np.linalg.norm(delta_phi) < tol:
        print(f"Converged after {_+1} iterations.")
        break
else:
    print("Did not converge within iteration limit.")

phi = {s: phi_vec[state_index[s]] for s in pf_states}

###############################################
# 4. Define Morse Potential
###############################################
def compute_state_coordinate(s, R=3):
    return s.count('1') / R

def morse_potential(X, D_e, a, X0):
    return D_e * (1 - np.exp(-a * (X - X0)))**2

D_e = 5.0
a_param = 2.0
X0 = 0.5

morse_energy = {}
for s in pf_states:
    X = compute_state_coordinate(s, R=3)
    morse_energy[s] = morse_potential(X, D_e, a_param, X0)

###############################################
# 5. Define a Custom C-Ricci Measure
###############################################
# We abandon standard ORC and define c_ricci as a function of occupancy + morse energy.
# Example formula (adjust as needed):
#   c_ricci(u, v) = alpha*( (rho[u]+rho[v])/2 ) - beta*( (morse[u]+morse[v])/2 ) + gamma*0
# You could incorporate heat, entropy, etc. in a real system.

alpha = 1.0
beta  = 1.0
gamma = 0.0  # Not used for now, but could incorporate heat or entropy

def c_ricci(u, v):
    occ_factor   = (occupancy[u] + occupancy[v]) / 2
    energy_factor= (morse_energy[u] + morse_energy[v]) / 2
    # Combine them:
    return alpha * occ_factor - beta * energy_factor + gamma * 0.0

# Store c_ricci in the graph edges:
for (u, v) in G.edges():
    val = c_ricci(u, v)
    G[u][v]['cRicci'] = val

###############################################
# 6. 3D Embedding: Use Occupancy as z-Coordinate
###############################################
# A simple flat layout for x-y
flat_pos = {
    "000": (0, 3),
    "001": (-2, 2),
    "010": (0, 2),
    "100": (2, 2),
    "011": (-1, 1),
    "101": (0, 1),
    "110": (1, 1),
    "111": (0, 0)
}
node_positions_3D = {}
for s in pf_states:
    x, y = flat_pos[s]
    z = occupancy[s]  # z from occupancy
    node_positions_3D[s] = (x, y, z)

###############################################
# 7. Save Everything to a Pickle
###############################################
connected = list(connected_components(G.subgraph([s for s in pf_states if occupancy[s]>0])))
betti_0 = len(connected)

solution = {
    "phi": phi,
    "phi_vector": phi_vec,
    "occupancy": occupancy,
    "rho_vector": rho_vec,
    "laplacian": L,
    "states": pf_states,
    "graph": G,
    "connected_components": connected,
    "betti_0": betti_0,
    "morse_energy": morse_energy,
    "node_positions_3D": node_positions_3D
}

with open("oxishape_solution_full.pkl", "wb") as f:
    pickle.dump(solution, f)

print("✔ Full Oxi-Shape solution saved to 'oxishape_solution_full.pkl'")
print(f"✔ Betti-0 (connected components): {betti_0}")

###############################################
# 8. Visualize: The Shape Now (Occupancy) & The C-Ricci Edges (Future)
###############################################

node_sizes = [3000 * occupancy[s] for s in pf_states]

# 8a. 2D Plot of φ or Occupancy
def plot_scalar_field_2d(title, values, filename, cmap):
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    nx.draw(
        G, flat_pos,
        with_labels=True,
        node_color=values,
        node_size=node_sizes,
        edge_color='gray',
        cmap=cmap,
        font_weight='bold',
        ax=ax
    )
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(values), vmax=max(values))),
        ax=ax,
        label=title
    )
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Plot occupancy
plot_scalar_field_2d("Occupancy", [occupancy[s] for s in pf_states], "oxishape_occupancy_2D.png", "viridis")

# 8b. 3D Plot of the Current Shape
fig = plt.figure(figsize=(10,8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

xs = [node_positions_3D[s][0] for s in pf_states]
ys = [node_positions_3D[s][1] for s in pf_states]
zs = [node_positions_3D[s][2] for s in pf_states]
sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=node_sizes, edgecolor='k')
for u, v in G.edges():
    x_vals = [node_positions_3D[u][0], node_positions_3D[v][0]]
    y_vals = [node_positions_3D[u][1], node_positions_3D[v][1]]
    z_vals = [node_positions_3D[u][2], node_positions_3D[v][2]]
    ax.plot(x_vals, y_vals, z_vals, color='gray', alpha=0.7)

for s in pf_states:
    x, y, z = node_positions_3D[s]
    ax.text(x, y, z, s, fontsize=10, weight='bold')

ax.set_title("3D Redox Diamond (Occupancy as z-axis)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Occupancy (z)")
plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="Occupancy")
plt.savefig("oxishape_3D_current.png", dpi=300, bbox_inches='tight')
plt.show()

# 8c. Plot C-Ricci on edges in 2D
def plot_c_ricci_edges(title, filename):
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    # We draw nodes in a base color, then color edges by cRicci
    c_ricci_values = [G[u][v]['cRicci'] for u, v in G.edges()]
    c_min, c_max = min(c_ricci_values), max(c_ricci_values)

    pos = flat_pos
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgray', ax=ax)
    nx.draw_networkx_labels(G, pos, labels={s:s for s in pf_states}, font_weight='bold', ax=ax)

    # Edge colormap
    edges = list(G.edges())
    edge_colors = [G[u][v]['cRicci'] for (u,v) in edges]
    norm = plt.Normalize(vmin=c_min, vmax=c_max)
    cmap = plt.cm.plasma
    # Draw edges with color
    for (u,v), ec in zip(edges, edge_colors):
        c_val = cmap(norm(ec))
        x_vals = [pos[u][0], pos[v][0]]
        y_vals = [pos[u][1], pos[v][1]]
        ax.plot(x_vals, y_vals, color=c_val, linewidth=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="C-Ricci")
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

plot_c_ricci_edges("C-Ricci Edge Values (Future)", "oxishape_cRicci_edges.png")

###############################################
# 9. Print Final Summary
###############################################

print("States and Field Values:")
for s in pf_states:
    print(f"  {s}: φ = {phi[s]:.4f}, ρ = {occupancy[s]}, E_morse = {morse_energy[s]:.4e}")

print("\nC-Ricci Edge Values:")
for (u, v) in G.edges():
    print(f"  Edge ({u},{v}): cRicci = {G[u][v]['cRicci']:.4f}")

print("\nLaplacian matrix:\n", L)
