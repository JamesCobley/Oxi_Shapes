#!/usr/bin/env python

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from GraphRicciCurvature.OllivierRicci import OllivierRicci

###############################################
# 1. Define the R=3 i-state network (Pascal diamond)
###############################################

pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
N = len(pf_states)
state_index = {state: idx for idx, state in enumerate(pf_states)}

def hamming_distance(s1, s2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

G = nx.Graph()
G.add_nodes_from(pf_states)
for i in range(N):
    for j in range(i + 1, N):
        if hamming_distance(pf_states[i], pf_states[j]) == 1:
            G.add_edge(pf_states[i], pf_states[j])

###############################################
# 2. Define occupancy (ρ) at each node (sum to 1)
###############################################

occupancy = {
    "000": 0.25,
    "001": 0.25, "010": 0.25, "100": 0.25,
    "011": 0.0, "101": 0.0, "110": 0.0,
    "111": 0.0
}
rho_vec = np.array([occupancy[state] for state in pf_states])

###############################################
# 3. Compute Ollivier-Ricci Curvature
###############################################

orc = OllivierRicci(G.copy(), alpha=0.5, method="OTD", verbose="ERROR")
orc.compute_ricci_curvature()
G = orc.G  # Updated graph with curvature

# Standardize key name if needed
for u, v in G.edges():
    if 'ricci' in G[u][v]:
        G[u][v]['ricciCurvature'] = G[u][v]['ricci']

###############################################
# 4. Build Laplacian Matrix (Unweighted)
###############################################

A = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if G.has_edge(pf_states[i], pf_states[j]):
            A[i, j] = 1
D = np.diag(A.sum(axis=1))
L = D - A

###############################################
# 5. Solve the nonlinear Oxi-Shape equation
###############################################

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
    phi_vec += damping * delta_phi
    if np.linalg.norm(delta_phi) < tol:
        print(f"Converged after {_+1} iterations.")
        break
else:
    print("Did not converge within iteration limit.")

phi = {state: phi_vec[i] for i, state in enumerate(pf_states)}

###############################################
# 6. Visualize the Oxi-Shape with OR Curvature
###############################################

pos = {
    "000": (0, 3),
    "001": (-2, 2), "010": (0, 2), "100": (2, 2),
    "011": (-1, 1), "101": (0, 1), "110": (1, 1),
    "111": (0, 0)
}

node_colors = [phi[state] for state in pf_states]
node_sizes = [3000 * occupancy[state] for state in pf_states]

edge_colors = []
for u, v in G.edges():
    edge_colors.append(G[u][v].get('ricciCurvature', 0.0))

edge_cmap = plt.cm.viridis
node_cmap = plt.cm.viridis
edge_norm = plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
nx.draw(
    G, pos,
    with_labels=True,
    node_color=node_colors,
    node_size=node_sizes,
    edge_color=edge_colors,
    edge_cmap=edge_cmap,
    edge_vmin=min(edge_colors),
    edge_vmax=max(edge_colors),
    width=3,
    cmap=node_cmap,
    font_weight='bold',
    ax=ax
)

# Node colorbar
sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Scalar Field φ")

# Edge colorbar
sm2 = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm2.set_array([])
plt.colorbar(sm2, ax=ax, label="Ollivier-Ricci Curvature (edges)")

plt.title("Oxi-Shape with OR Curvature on R=3 Pascal Diamond (viridis)")
plt.axis("off")

# Save and show
image_filename = "oxishape_orc_pascal_diamond_viridis.png"
plt.savefig(image_filename, dpi=300, bbox_inches='tight')
plt.show()
print(f"Oxi-Shape image saved as '{image_filename}'.")

###############################################
# 7. Save the solution to a pickle file
###############################################

solution = {
    "phi": phi,
    "occupancy": occupancy,
    "laplacian": L,
    "states": pf_states,
    "graph": G,
    "orc_edge_curvatures": {
        (u, v): G[u][v].get('ricciCurvature', None) for u, v in G.edges()
    }
}

pickle_filename = "oxishape_solution_with_orc_viridis.pkl"
with open(pickle_filename, "wb") as f:
    pickle.dump(solution, f)

print(f"Oxi-Shape solution with OR curvature saved to '{pickle_filename}'.")
