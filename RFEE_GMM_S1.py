#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes: Pipeline Step 1 — Compute a custom C-Ricci (from Dirichlet energy via a cotangent Laplacian),
ensure volume conservation (sum(rho)=1), pin unoccupied nodes at z=0, and build a 24-row transition table
for pipeline step 2. This version also computes an anisotropy field that penalizes transitions in high-curvature
directions (i.e. the barred transitions).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation
from networkx.algorithms.components import connected_components
from mpl_toolkits.mplot3d import Axes3D

###############################################
# 1. Define R=3 i-States & Allowed Transitions
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
# 2. Assign & Normalize Occupancy (Volume Conservation)
###############################################
occupancy_init = {
    "000": 0.25,
    "001": 0.25, "010": 0.25, "100": 0.25,
    "011": 0.0,  "101": 0.0,  "110": 0.0,  "111": 0.0
}
sum_occ = sum(occupancy_init.values())
occupancy = {}
if sum_occ > 1e-14:
    for s in pf_states:
        occupancy[s] = occupancy_init[s] / sum_occ
else:
    occupancy = {s: 0.0 for s in pf_states}

rho_vec = np.array([occupancy[s] for s in pf_states])

###############################################
# 3. Solve φ Field (Poisson-like PDE) on G
###############################################
N = len(pf_states)
state_index = {s: i for i, s in enumerate(pf_states)}

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
    # Pin nodes with zero occupancy (do not update)
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
# 4. Morse Potential from Normalized Oxidation (k/3)
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
# 5. Compute Enriched C-Ricci from Dirichlet Energy via Cotangent Laplacian
###############################################
# First, define a function to compute the cotangent Laplacian on the triangulated mesh.
def compute_cotangent_laplacian(node_xy, triangles):
    N = node_xy.shape[0]
    W = np.zeros((N, N))
    for tri in triangles:
        i, j, k = tri
        pts = node_xy[[i, j, k], :]
        # Compute edge vectors
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[0]
        v2 = pts[2] - pts[1]
        # Compute angles using dot products
        angle_i = np.arccos(np.clip(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)), -1, 1))
        angle_j = np.arccos(np.clip(np.dot(-v0, v2) / (np.linalg.norm(v0) * np.linalg.norm(v2)), -1, 1))
        angle_k = np.arccos(np.clip(np.dot(-v1, -v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        # Cotangent weights
        cot_i = 1 / np.tan(angle_i) if np.tan(angle_i) != 0 else 0
        cot_j = 1 / np.tan(angle_j) if np.tan(angle_j) != 0 else 0
        cot_k = 1 / np.tan(angle_k) if np.tan(angle_k) != 0 else 0
        # Accumulate weights for edges (j,k) opposite i, etc.
        W[j, k] += cot_i
        W[k, j] += cot_i
        W[i, k] += cot_j
        W[k, i] += cot_j
        W[i, j] += cot_k
        W[j, i] += cot_k
    # Construct Laplacian: off-diagonals = -W, diagonals = sum(W_row)
    L_t = -W
    for i in range(N):
        L_t[i, i] = np.sum(W[i, :])
    return L_t

# Constant that scales curvature (analogous to a gravitational constant)
lambda_const = 1.0

# Build a triangulated mesh of the (x,y) positions.
# (Note: node_xy is already computed in Step 6 below; here we precompute it for the Laplacian.)
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
node_xy = np.array([flat_pos[s] for s in pf_states])
tri = Delaunay(node_xy)
triangles = tri.simplices

# Compute the cotangent Laplacian on the mesh.
L_t = compute_cotangent_laplacian(node_xy, triangles)
# Compute nodewise c-Ricci from Dirichlet energy:
# Here, we take the Laplacian of the occupancy field and scale by lambda_const.
c_ricci_vec = lambda_const * (L_t @ rho_vec)

# For each state, store its c-Ricci.
c_ricci_nodes = {pf_states[i]: c_ricci_vec[i] for i in range(len(pf_states))}

# Now, assign a c-Ricci value to each edge in G as the average of its endpoints.
for (u, v) in G.edges():
    G[u][v]['cRicci'] = (c_ricci_nodes[u] + c_ricci_nodes[v]) / 2.0

# Compute anisotropy field A(x) = discrete gradient of c-Ricci.
# For each node, average the gradient (difference over distance) to its neighbors (using graph G).
A_field = {}
for s in pf_states:
    neighbors = list(G.neighbors(s))
    grad_sum = 0.0
    count = 0
    for nbr in neighbors:
        # Euclidean distance between s and neighbor nbr
        dist = np.linalg.norm(np.array(flat_pos[s]) - np.array(flat_pos[nbr]))
        if dist > 1e-6:
            grad_sum += abs(c_ricci_nodes[s] - c_ricci_nodes[nbr]) / dist
            count += 1
    A_field[s] = grad_sum / count if count > 0 else 0.0

# For each edge, define a penalty factor that penalizes transitions in regions of high anisotropy.
# The penalty factor is defined as exp(-average anisotropy on the edge).
for (u, v) in G.edges():
    penalty = np.exp(- (A_field[u] + A_field[v]) / 2.0)
    G[u][v]['penalty'] = penalty

###############################################
# 6. 3D Embedding: Pin Zero Occupancy at z=0
###############################################
node_positions_3D = {}
for s in pf_states:
    x, y = flat_pos[s]
    z_val = occupancy[s] if occupancy[s] > 1e-14 else 0.0
    node_positions_3D[s] = (x, y, z_val)

# 3D Surface plot of c-Ricci (using the nodewise c_ricci computed from the cotangent Laplacian)
z_c_ricci = np.array([c_ricci_nodes[s] for s in pf_states])
tri = Delaunay(node_xy)
triangles = tri.simplices

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
triang = Triangulation(node_xy[:, 0], node_xy[:, 1], triangles)
surf = ax.plot_trisurf(
    triang, z_c_ricci, cmap='coolwarm', edgecolor='k', linewidth=0.5, antialiased=True
)
ax.set_title("C-Ricci Curvature Surface (Oxi-Shape)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("C-Ricci (z)")
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="C-Ricci")
plt.savefig("oxishape_surface_cRicci.png", dpi=300, bbox_inches='tight')
plt.show()

# Store triangulation and z-values in solution object
solution = {}
solution["cRicci_nodewise"] = c_ricci_nodes
solution["triangulation"] = {
    "vertices": node_xy,
    "triangles": triangles.tolist(),
    "z_values": z_c_ricci.tolist()
}
# Also store anisotropy field for reference
solution["anisotropy_field"] = A_field

print("\nC-Ricci is computed as: c-Ricci(x) = λ * (cotangent Laplacian of ρ(x))")
print("Anisotropy field A(x) = discrete gradient of c-Ricci(x) penalizes high-curvature (barred) transitions.")

###############################################
# 7. Build the 24-Row Transition Table (Edges in Both Directions)
###############################################
rows = []
for (u, v) in G.edges():
    # We'll record transitions u->v and v->u, including the anisotropy penalty.
    c_val = G[u][v]['cRicci']
    penalty = G[u][v]['penalty']
    rows.append({
        "from":   u,
        "to":     v,
        "occupancy_from": occupancy[u],
        "occupancy_to":   occupancy[v],
        "k_from": u.count('1'),
        "k_to":   v.count('1'),
        "morse_from": morse_energy[u],
        "morse_to":   morse_energy[v],
        "cRicci_edge": c_val,
        "penalty": penalty
    })
    rows.append({
        "from":   v,
        "to":     u,
        "occupancy_from": occupancy[v],
        "occupancy_to":   occupancy[u],
        "k_from": v.count('1'),
        "k_to":   u.count('1'),
        "morse_from": morse_energy[v],
        "morse_to":   morse_energy[u],
        "cRicci_edge": c_val,
        "penalty": penalty
    })

df_transitions = pd.DataFrame(rows)
df_transitions.sort_values(by=["from", "to"], inplace=True)

###############################################
# 8. Save Everything (Pickle & CSV)
###############################################
active_nodes = [s for s in pf_states if occupancy[s] > 1e-14]
G_active = G.subgraph(active_nodes).copy()
connected = list(connected_components(G_active))
betti_0 = len(connected)

solution.update({
    "phi": phi,
    "phi_vector": phi_vec,
    "occupancy": occupancy,  # normalized so sum(rho)=1
    "rho_vector": rho_vec,
    "laplacian": L,
    "states": pf_states,
    "graph": G,
    "active_subgraph": G_active,
    "connected_components": connected,
    "betti_0": betti_0,
    "transition_table": df_transitions
})

with open("oxishape_solution_full.pkl", "wb") as f:
    pickle.dump(solution, f)
df_transitions.to_csv("oxi_transitions_table.csv", index=False)

print("✔ Full Oxi-Shape solution saved to 'oxishape_solution_full.pkl'")
print("✔ Transition table saved to 'oxi_transitions_table.csv'")
print(f"✔ Betti-0 (connected components): {betti_0}")
print("Sum of occupancy after normalization:", sum(occupancy.values()))

###############################################
# 9. Basic Visualizations
###############################################
node_sizes = [3000 * occupancy[s] for s in pf_states]

# 9a. 2D Visualization of φ
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

# 9b. 3D Visualization of Occupancy-Based Embedding
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

# 9c. 2D Visualization of Custom C-Ricci on Edges
def plot_c_ricci_edges(title, filename):
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    pos = flat_pos
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgray', ax=ax)
    nx.draw_networkx_labels(G, pos, labels={s: s for s in pf_states}, font_weight='bold', ax=ax)
    edges = list(G.edges())
    c_values = [G[u][v]['cRicci'] for (u, v) in edges]
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
# 10. Print Final Summary
###############################################
print("\nStates and Field Values:")
for s in pf_states:
    print(f"  {s}: φ = {phi[s]:.4f}, ρ = {occupancy[s]:.4f}, E_morse = {morse_energy[s]:.4e}")
print("\nCustom C-Ricci Edge Values:")
for (u, v) in G.edges():
    print(f"  Edge ({u}, {v}): cRicci = {G[u][v]['cRicci']:.4f}, penalty = {G[u][v]['penalty']:.4f}")

print("\nLaplacian matrix:\n", L)
print("\nSaved transition table in 'oxi_transitions_table.csv' with 24 rows (each edge in both directions).")
