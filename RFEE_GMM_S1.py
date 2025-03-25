#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes: Pipeline Step 1 — Generate the redox diamond solution for the given fractional occupancy
Encodes the full binomial diamond state space and allowed and barred as edges.
Inputs: Fractional positional density data.
Outputs: Pickle solution file, plots (geometry, energy, heat), printouts checking the geometry.
"""

# Core imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from networkx.algorithms.components import connected_components

###############################################
# Define Proteoform States & Allowed Transitions
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
# Define occupancy (ρ) at each node
###############################################

occupancy = {
    "000": 0.25,
    "001": 0.25, "010": 0.25, "100": 0.25,
    "011": 0.0, "101": 0.0, "110": 0.0,
    "111": 0.0
}
rho_vec = np.array([occupancy[state] for state in pf_states])

###############################################
# Compute Full Graph ORC: Include Barred Transitions
###############################################

# Build full graph including barred transitions
full_G = nx.Graph()
full_G.add_nodes_from(pf_states)
for i in range(len(pf_states)):
    for j in range(i+1, len(pf_states)):
        hd = sum(x != y for x, y in zip(pf_states[i], pf_states[j]))
        if hd == 1:
            weight = 1.0
        else:
            weight = 100.0  # High weight for barred transitions
        full_G.add_edge(pf_states[i], pf_states[j], weight=weight)

# Compute ORC on the full graph using custom weights.
orc_full = OllivierRicci(full_G.copy(), alpha=0.5, method="OTD", verbose="ERROR")
orc_full.compute_ricci_curvature()
full_G = orc_full.G

# Store computed curvature under key 'ricciCurvature'
for u, v in full_G.edges():
    full_G[u][v]['ricciCurvature'] = full_G[u][v].get('ricci', 0.0)

# Post-process ORC to incorporate occupancy difference.
# Define a scaling parameter for occupancy differences.
alpha_H = 1.0  # Adjust this parameter as needed.
for u, v in full_G.edges():
    # Compute absolute difference in occupancy between nodes u and v.
    diff = abs(occupancy[u] - occupancy[v])
    scaling_factor = 1.0 + alpha_H * diff
    full_G[u][v]['ricciCurvature'] *= scaling_factor

###############################################
# Annotate Topology (for allowed transitions graph G)
###############################################

for state in pf_states:
    k_val = state.count('1')
    G.nodes[state]['k'] = k_val
    G.nodes[state]['boundary'] = 'lower' if k_val == 0 else 'upper' if k_val == 3 else 'interior'
    G.nodes[state]['occupied'] = occupancy[state] > 0
    G.nodes[state]['geodesic_neighbors'] = [nbr for nbr in G.neighbors(state)]

active_nodes = [s for s in pf_states if occupancy[s] > 0]
G_active = G.subgraph(active_nodes).copy()
connected = list(connected_components(G_active))
betti_0 = len(connected)

###############################################
# Solve φ Field (Poisson) on the allowed graph (flat geometry)
###############################################

N = len(pf_states)
A = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if G.has_edge(pf_states[i], pf_states[j]):
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
    delta_phi[rho_vec == 0] = 0.0
    phi_vec += damping * delta_phi
    if np.linalg.norm(delta_phi) < tol:
        print(f"Converged after {_+1} iterations.")
        break
else:
    print("Did not converge within iteration limit.")

phi = {state: phi_vec[i] for i, state in enumerate(pf_states)}

# Compute Q(x) = -Δρ(x) as heat landscape from mass diffusion potential
lap_rho = L @ rho_vec
heat_vec = -lap_rho
heat_landscape = {state: heat_vec[i] for i, state in enumerate(pf_states)}

###############################################
# Derive Morse and Heat Landscapes from First Principles
###############################################
# Define a state coordinate X = k / R (with R=3 for this system)
def compute_state_coordinate(state, R=3):
    return state.count('1') / R

# Define a Morse potential function
def morse_potential(X, D_e, a, X0):
    return D_e * (1 - np.exp(-a * (X - X0)))**2

# Set Morse parameters (to be refined from first principles)
D_e = 5.0      # Dissociation energy
a_param = 2.0  # Controls the width of the potential well
X0 = 0.5       # Equilibrium coordinate

morse_energy = {}
for s in pf_states:
    X = compute_state_coordinate(s, R=3)
    morse_energy[s] = morse_potential(X, D_e, a_param, X0)

# For now, set heat landscape to zero (to be refined later)
heat_landscape = {s: 0.0 for s in pf_states}

###############################################
# 3D Embedding & Volume Conservation
###############################################
# Use a 2D flat layout (flat_pos) for x-y and assign occupancy as the z-coordinate.
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

# Create 3D node positions using flat_pos for x-y and occupancy for z.
node_positions_3D = {}
for state in pf_states:
    x, y = flat_pos[state]
    z = occupancy[state]  # z-coordinate as occupancy (or a function thereof)
    node_positions_3D[state] = (x, y, z)

# (Optional) Here you would implement a volume conservation constraint if needed.
# For now, we assume the total occupancy (mass) is fixed.

###############################################
# Save Graph Structure + φ, Morse, Heat, and 3D Positions to Pickle
###############################################
solution = {
    "phi": phi,                      # Dictionary: {state: φ value}
    "phi_vector": phi_vec,           # Array of φ values in order of pf_states
    "occupancy": occupancy,          # Original occupancy dictionary
    "rho_vector": rho_vec,           # Array of fractional occupancies
    "laplacian": L,                  # Graph Laplacian
    "states": pf_states,
    "graph": G,                      # Allowed transitions graph (for PDE)
    "full_graph": full_G,            # Full graph including barred transitions (with weighted ORC)
    "active_subgraph": G_active,
    "connected_components": connected,
    "betti_0": betti_0,
    "ricciCurvature": {(u, v): full_G[u][v]['ricciCurvature'] for u, v in full_G.edges()},
    "morse_energy": morse_energy,    # True Morse potential based on X = k/3
    "heat_landscape": heat_landscape,
    "node_positions_3D": node_positions_3D
}

with open("oxishape_solution_full.pkl", "wb") as f:
    pickle.dump(solution, f)

print("✔ Full Oxi-Shape solution saved to 'oxishape_solution_full.pkl'")
print(f"✔ Betti-0 (connected components): {betti_0}")

###############################################
# Flat Geometry Visuals: φ, Morse, Heat
###############################################

node_sizes = [3000 * occupancy[state] for state in pf_states]

def plot_scalar_field(title, values, filename, cmap):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
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

plot_scalar_field("Scalar Field φ", [phi[s] for s in pf_states], "oxishape_phi.png", "viridis")
plot_scalar_field("Morse Energy", [morse_energy[s] for s in pf_states], "oxishape_morse.png", "coolwarm")
plot_scalar_field("Heat Landscape Q(x)", [heat_landscape[s] for s in pf_states], "oxishape_heat.png", "coolwarm")

###############################################
# 3D Geometry Visual: Occupancy-Based Embedding
###############################################
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

xs = [node_positions_3D[s][0] for s in pf_states]
ys = [node_positions_3D[s][1] for s in pf_states]
zs = [node_positions_3D[s][2] for s in pf_states]
sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=node_sizes, edgecolor='k')

# Draw allowed edges in 3D (using flat_pos for x-y and occupancy for z)
for u, v in G.edges():
    x_vals = [flat_pos[u][0], flat_pos[v][0]]
    y_vals = [flat_pos[u][1], flat_pos[v][1]]
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
plt.savefig("oxishape_3D.png", dpi=300, bbox_inches='tight')
plt.show()

###############################################
# Print Final Summary
###############################################

print("States and Field Values:")
for s in pf_states:
    print(f"  {s}: φ = {phi[s]:.4f}, ρ = {occupancy[s]}, E_morse = {morse_energy[s]:.4e}, Q = {heat_landscape[s]:.4f}")

print("\nBetti-0:", betti_0)
print("Connected components:", connected)
print("\nSample Ricci curvatures:")
for (u, v), ricci in list(solution['ricciCurvature'].items())[:5]:
    print(f"  ({u}, {v}): {ricci:.4f}")
print("\nLaplacian matrix:\n", L)
