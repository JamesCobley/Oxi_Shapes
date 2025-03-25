#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes: Pipeline Step 1 — Generate the redox dimaond solution for the given fractional occupacny
Encodes the full binomial diamond state space and allowed and barred as edges
Inputs: Fractional positional density data
Outputs: Pickle soltuon file, PNG solution image with scalar, printouts checking the geometry.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from networkx.algorithms.components import connected_components

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
# 2. Define occupancy (ρ) at each node
###############################################

occupancy = {
    "000": 0.25,
    "001": 0.25, "010": 0.25, "100": 0.25,
    "011": 0.0, "101": 0.0, "110": 0.0,
    "111": 0.0
}
rho_vec = np.array([occupancy[state] for state in pf_states])

###############################################
# 3. Compute Ollivier-Ricci Curvature (on allowed G)
###############################################

orc = OllivierRicci(G.copy(), alpha=0.5, method="OTD", verbose="ERROR")
orc.compute_ricci_curvature()
G = orc.G

for u, v in G.edges():
    G[u][v]['ricciCurvature'] = G[u][v].get('ricci', 0.0)

###############################################
# 4. Annotate Topology (k-values, boundaries, etc.)
###############################################

for state in pf_states:
    k_val = state.count('1')
    G.nodes[state]['k'] = k_val
    G.nodes[state]['boundary'] = 'lower' if k_val == 0 else 'upper' if k_val == 3 else 'interior'
    G.nodes[state]['occupied'] = occupancy[state] > 0
    G.nodes[state]['geodesic_neighbors'] = [nbr for nbr in G.neighbors(state)]

active_nodes = [state for state in pf_states if occupancy[state] > 0]
G_active = G.subgraph(active_nodes).copy()
connected = list(connected_components(G_active))
betti_0 = len(connected)

###############################################
# 5. Build Laplacian & Solve φ Field
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

    # 🔐 Enforce φ = 0 update at unoccupied nodes (robust enforcement)
    unoccupied_mask = rho_vec == 0
    delta_phi[unoccupied_mask] = 0.0

    phi_vec += damping * delta_phi

    if np.linalg.norm(delta_phi) < tol:
        print(f"Converged after {_+1} iterations.")
        break
else:
    print("Did not converge within iteration limit.")

phi = {state: phi_vec[i] for i, state in enumerate(pf_states)}

###############################################
# 6. Save Graph Structure + φ to Pickle
###############################################

solution = {
    "phi": phi,
    "phi_vector": phi_vec,
    "occupancy": occupancy,
    "rho_vector": rho_vec,
    "laplacian": L,
    "states": pf_states,
    "graph": G,
    "active_subgraph": G_active,
    "connected_components": connected,
    "betti_0": betti_0,
    "orc_edge_curvatures": {
        (u, v): G[u][v]['ricciCurvature'] for u, v in G.edges()
    }
}

with open("oxishape_solution_full.pkl", "wb") as f:
    pickle.dump(solution, f)

print("✔ Full Oxi-Shape solution saved to 'oxishape_solution_full.pkl'")
print(f"✔ Betti-0 (connected components): {betti_0}")

###############################################
# 7. Flat Geometry Visual (Volume Invariant)
###############################################

flat_pos = {
    "000": (0, 3),
    "001": (-2, 2), "010": (0, 2), "100": (2, 2),
    "011": (-1, 1), "101": (0, 1), "110": (1, 1),
    "111": (0, 0)
}

node_colors = [phi[state] for state in pf_states]
node_sizes = [3000 * occupancy[state] for state in pf_states]
edge_colors = [G[u][v]['ricciCurvature'] for u, v in G.edges()]
edge_cmap = plt.cm.viridis
edge_norm = plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
nx.draw(
    G, flat_pos,
    with_labels=True,
    node_color=node_colors,
    node_size=node_sizes,
    edge_color=edge_colors,
    edge_cmap=edge_cmap,
    edge_vmin=min(edge_colors),
    edge_vmax=max(edge_colors),
    width=3,
    cmap='viridis',
    font_weight='bold',
    ax=ax
)
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))), ax=ax, label="Scalar Field φ")
plt.colorbar(plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm), ax=ax, label="Ollivier-Ricci Curvature")
plt.title("Flat Oxi-Shape: Ricci + φ Field")
plt.axis("off")
plt.savefig("oxishape_flat.png", dpi=300, bbox_inches='tight')
plt.show()

###############################################
# 8. Check the geometry is correct
###############################################

import pickle

with open("oxishape_solution_full.pkl", "rb") as f:
    solution = pickle.load(f)

# Summarize key fields
print("States and φ values:")
for state in solution['states']:
    phi_val = solution['phi'][state]
    rho_val = solution['occupancy'][state]
    print(f"  {state}: φ = {phi_val:.4f}, ρ = {rho_val}")

print("\nBetti-0:", solution['betti_0'])
print("Connected components:", solution['connected_components'])

print("\nSample Ricci curvatures:")
for (u, v), ricci in list(solution['orc_edge_curvatures'].items())[:5]:
    print(f"  ({u}, {v}): {ricci:.4f}")

print("\nLaplacian matrix:\n", solution['laplacian'])
