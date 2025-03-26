#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes: Pipeline Step 1 — Generate the redox diamond solution for the given fractional occupancy
and compute a custom C-Ricci measure, ensuring:
  1) Unoccupied nodes (rho=0) are pinned at z=0.
  2) Total occupancy is normalized (sum(rho)=1) to respect volume conservation in this static snapshot.
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
# 2. Define & Normalize Occupancy (ρ) at Each Node
###############################################

# Initial occupancy (could come from data or user input)
occupancy_init = {
    "000": 0.25,
    "001": 0.25, "010": 0.25, "100": 0.25,
    "011": 0.0,  "101": 0.0,  "110": 0.0,  "111": 0.0
}

# Enforce sum(rho)=1 to respect volume conservation in this static snapshot.
sum_occ = sum(occupancy_init.values())
occupancy = {}
if sum_occ > 0:
    for s in pf_states:
        occupancy[s] = occupancy_init[s] / sum_occ
else:
    # If total occupancy was 0, we do something trivial (all zero or handle error).
    occupancy = {s: 0.0 for s in pf_states}

# Convert to a vector if needed
rho_vec = np.array([occupancy[s] for s in pf_states])

###############################################
# 3. Solve the φ Field (Poisson-like PDE) on the Allowed Graph
###############################################
N = len(pf_states)
state_index = {s: i for i, s in enumerate(pf_states)}

# Build adjacency matrix for PDE
A = np.zeros((N, N))
for i, s1 in enumerate(pf_states):
    for j, s2 in enumerate(pf_states):
        if G.has_edge(s1, s2):
            A[i, j] = 1
D = np.diag(A.sum(axis=1))
L = D - A

phi_vec = np.zeros(N)
kappa = 1.0
max_iter = 10000
tol = 1e-3
damping = 0.05

for _ in range(max_iter):
    nonlin = 0.5 * kappa * rho_vec * np.exp(2 * phi_vec)
    F = L @ phi_vec + nonlin
    J = L + np.diag(kappa * rho_vec * np.exp(2 * phi_vec))
    delta_phi = np.linalg.solve(J, -F)
    # Do not update nodes with zero occupancy
    for i, s in enumerate(pf_states):
        if occupancy[s] <= 1e-14:
            delta_phi[i] = 0.0
    phi_vec += damping * delta_phi
    if np.linalg.norm(delta_phi) < tol:
        print(f"Converged after {_+1} iterations.")
        break
else:
    print("Did not converge within iteration limit.")

phi = {s: phi_vec[state_index[s]] for s in pf_states}

###############################################
# 4. Define Morse Potential from First Principles
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
degeneracy_map = {0: 1, 1: 3, 2: 3, 3: 1}

alpha_param = 1.0  # scales occupancy
beta_param  = 1.0  # scales Morse energy
gamma_param = 1.0  # scales heat
delta_param = 1.0  # scales entropy

def c_ricci(u, v):
    # Occupancy factor: average occupancy
    occ_factor = (occupancy[u] + occupancy[v]) / 2.0
    # Energy factor: average Morse energy
    energy_factor = (morse_energy[u] + morse_energy[v]) / 2.0
    # Heat factor: 1 if node is occupied, else 0
    heat_u = 1.0 if occupancy[u] > 1e-14 else 0.0
    heat_v = 1.0 if occupancy[v] > 1e-14 else 0.0
    heat_factor = (heat_u + heat_v) / 2.0
    # Entropy factor: ln(degeneracy)
    ent_u = np.log(degeneracy_map[u.count('1')])
    ent_v = np.log(degeneracy_map[v.count('1')])
    entropy_factor = (ent_u + ent_v) / 2.0
    
    return (alpha_param * occ_factor
            - beta_param  * energy_factor
            + gamma_param * heat_factor
            + delta_param * entropy_factor)

# Compute cRicci for each edge in G
for (u, v) in G.edges():
    G[u][v]['cRicci'] = c_ricci(u, v)

###############################################
# 6. 3D Embedding: Pin unoccupied nodes at z=0
###############################################
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
    z_val = occupancy[s] if occupancy[s] > 1e-14 else 0.0
    node_positions_3D[s] = (x, y, z_val)

###############################################
# 7. Save Everything to a Pickle File
###############################################
active_nodes = [s for s in pf_states if occupancy[s] > 1e-14]
G_active = G.subgraph(active_nodes).copy()
connected = list(connected_components(G_active))
betti_0 = len(connected)

solution = {
    "phi": phi,
    "phi_vector": phi_vec,
    "occupancy": occupancy,       # normalized so sum(rho)=1
    "rho_vector": rho_vec,
    "laplacian": L,
    "states": pf_states,
    "graph": G,                   # allowed transitions graph with custom C-Ricci
    "active_subgraph": G_active,
    "connected_components": connected,
    "betti_0": betti_0,
    "cRicci": {(u, v): G[u][v]['cRicci'] for u, v in G.edges()},
    "morse_energy": morse_energy,
    "node_positions_3D": node_positions_3D
}

with open("oxishape_solution_full.pkl", "wb") as f:
    pickle.dump(solution, f)

print("✔ Full Oxi-Shape solution saved to 'oxishape_solution_full.pkl'")
print(f"✔ Betti-0 (connected components): {betti_0}")
print("Sum of occupancy after normalization:", sum(occupancy.values()))

###############################################
# 8. Visualizations
###############################################
node_sizes = [3000 * occupancy[s] for s in pf_states]

# 8a. 2D Visualization of φ
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

phi_values = [phi[s] for s in pf_states]
plot_scalar_field_2d("Scalar Field φ", phi_values, "oxishape_phi.png", "viridis")

# 8b. 3D Visualization of the Occupancy-Based Embedding
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
ax.set_title("3D Redox Diamond (Occupancy as z-axis, pinned at 0 for unoccupied)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Occupancy (z)")
plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="Occupancy")
plt.savefig("oxishape_3D.png", dpi=300, bbox_inches='tight')
plt.show()

# 8c. 2D Visualization of Custom C-Ricci on Edges
def plot_c_ricci_edges(title, filename):
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    pos = flat_pos
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgray', ax=ax)
    nx.draw_networkx_labels(G, pos, labels={s: s for s in pf_states}, font_weight='bold', ax=ax)
    edges = list(G.edges())
    c_values = [G[u][v]['cRicci'] for (u,v) in edges]
    c_min, c_max = min(c_values), max(c_values)
    norm = plt.Normalize(vmin=c_min, vmax=c_max)
    cmap = plt.cm.plasma
    for (u, v), c_val in zip(edges, c_values):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=cmap(norm(c_val)), linewidth=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Custom C-Ricci")
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

plot_c_ricci_edges("Custom C-Ricci Edge Values (Static Snapshot)", "oxishape_cRicci.png")

###############################################
# 9. Print Final Summary
###############################################
print("States and Field Values:")
for s in pf_states:
    print(f"  {s}: φ = {phi[s]:.4f}, ρ = {occupancy[s]:.4f}, E_morse = {morse_energy[s]:.4e}")
print("\nCustom C-Ricci Edge Values:")
for (u, v) in G.edges():
    print(f"  Edge ({u}, {v}): cRicci = {G[u][v]['cRicci']:.4f}")
print("\nLaplacian matrix:\n", L)
