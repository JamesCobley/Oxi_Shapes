#!/usr/bin/env python

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from mpl_toolkits.mplot3d import Axes3D

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
# 3. Compute Ollivier-Ricci Curvature
###############################################

orc = OllivierRicci(G.copy(), alpha=0.5, method="OTD", verbose="ERROR")
orc.compute_ricci_curvature()
G = orc.G  # Use updated graph

# Standardize Ricci key
for u, v in G.edges():
    if 'ricci' in G[u][v]:
        G[u][v]['ricciCurvature'] = G[u][v]['ricci']

###############################################
# 4. Build Graph Laplacian (Unweighted)
###############################################

A = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if G.has_edge(pf_states[i], pf_states[j]):
            A[i, j] = 1
D = np.diag(A.sum(axis=1))
L = D - A

###############################################
# 5. Solve the nonlinear field equation for φ
###############################################

phi_vec = np.zeros(N)
kappa = 1.0
max_iter = 7000
tol = 1e-6
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
# 6. Flat Graph Visualization (Volume Invariant)
###############################################

flat_pos = {
    "000": (0, 3),
    "001": (-2, 2), "010": (0, 2), "100": (2, 2),
    "011": (-1, 1), "101": (0, 1), "110": (1, 1),
    "111": (0, 0)
}

node_colors = [phi[state] for state in pf_states]
node_sizes = [3000 * occupancy[state] for state in pf_states]
edge_colors = [G[u][v].get('ricciCurvature', 0.0) for u, v in G.edges()]
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

sm1 = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
sm1.set_array([])
plt.colorbar(sm1, ax=ax, label="Scalar Field φ")

sm2 = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm2.set_array([])
plt.colorbar(sm2, ax=ax, label="Ollivier-Ricci Curvature (edges)")

plt.title("Flat Oxi-Shape: Ricci + φ Field on Pascal Diamond")
plt.axis("off")
plt.savefig("oxishape_flat_pascal.png", dpi=300, bbox_inches='tight')
plt.show()

###############################################
# 7. 2.5D Oxi-Shape Visualization (z = φ or 0)
###############################################

node_positions_3d = np.array([
    [flat_pos[state][0], flat_pos[state][1], phi[state] if occupancy[state] > 0 else 0.0]
    for state in pf_states
])
edges = [(i, j) for i in range(N) for j in range(i+1, N)
         if hamming_distance(pf_states[i], pf_states[j]) == 1]

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

for i, j in edges:
    x = [node_positions_3d[i][0], node_positions_3d[j][0]]
    y = [node_positions_3d[i][1], node_positions_3d[j][1]]
    z = [node_positions_3d[i][2], node_positions_3d[j][2]]
    ax.plot(x, y, z, color='black', alpha=0.6)

xs, ys, zs = node_positions_3d[:, 0], node_positions_3d[:, 1], node_positions_3d[:, 2]
sc = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=300, edgecolors='k')

for i, state in enumerate(pf_states):
    ax.text(xs[i], ys[i], zs[i], state, fontsize=10, weight='bold')

cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cb.set_label("Scalar Field φ (deforms z-axis)")
ax.set_title("2.5D Oxi-Shape: φ-Deformed Pascal Diamond")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z (Field deformation)")

plt.tight_layout()
plt.savefig("oxishape_pascal_2p5D.png", dpi=300, bbox_inches='tight')
plt.show()

###############################################
# 8. Save All Solution Data for Pipeline Step 2
###############################################

solution = {
    "phi": phi,
    "phi_vector": phi_vec,
    "occupancy": occupancy,
    "rho_vector": rho_vec,
    "laplacian": L,
    "states": pf_states,
    "graph": G,
    "orc_edge_curvatures": {
        (u, v): G[u][v].get('ricciCurvature', None) for u, v in G.edges()
    },
    "layout": flat_pos,
    "node_positions_3D": node_positions_3d.tolist()
}

with open("oxishape_solution_full.pkl", "wb") as f:
    pickle.dump(solution, f)

print("✔ Full solution saved to 'oxishape_solution_full.pkl'")
