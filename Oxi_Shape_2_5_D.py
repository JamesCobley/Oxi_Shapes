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
# 6. 2.5D Static Visualization (z = φ or 0)
###############################################

# Pascal diamond x-y layout
pos_2d = {
    "000": (0, 3),
    "001": (-2, 2), "010": (0, 2), "100": (2, 2),
    "011": (-1, 1), "101": (0, 1), "110": (1, 1),
    "111": (0, 0)
}

# Compute 3D coordinates: z = φ if occupied, else z = 0
node_positions_3d = np.array([
    [pos_2d[state][0], pos_2d[state][1], phi[state] if occupancy[state] > 0 else 0.0]
    for state in pf_states
])

edges = [(i, j) for i in range(N) for j in range(i+1, N)
         if hamming_distance(pf_states[i], pf_states[j]) == 1]

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# Draw edges
for i, j in edges:
    x = [node_positions_3d[i][0], node_positions_3d[j][0]]
    y = [node_positions_3d[i][1], node_positions_3d[j][1]]
    z = [node_positions_3d[i][2], node_positions_3d[j][2]]
    ax.plot(x, y, z, color='black', alpha=0.6)

# Draw nodes
xs, ys, zs = node_positions_3d[:, 0], node_positions_3d[:, 1], node_positions_3d[:, 2]
sc = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=300, edgecolors='k')

# Annotate
for i, state in enumerate(pf_states):
    ax.text(xs[i], ys[i], zs[i], state, fontsize=10, weight='bold')

# Colorbar
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

pickle_filename = "oxishape_solution_with_2p5D.pkl"
with open(pickle_filename, "wb") as f:
    pickle.dump(solution, f)

print(f"Oxi-Shape solution with φ-deformation saved to '{pickle_filename}'.")
