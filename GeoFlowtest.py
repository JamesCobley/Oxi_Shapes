import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation
from collections import defaultdict

# ------------------------------
# 1. Define the R=3 Redox Diamond
# ------------------------------
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

flat_pos = {
    "000": (0, 3), "001": (-2, 2), "010": (0, 2), "100": (2, 2),
    "011": (-1, 1), "101": (0, 1), "110": (1, 1), "111": (0, 0)
}
node_xy = np.array([flat_pos[s] for s in pf_states])
tri = Delaunay(node_xy)
triangles = tri.simplices

# ------------------------------
# 2. Cotangent Laplacian and Ricci Flow
# ------------------------------
def compute_cotangent_laplacian(node_xy, triangles):
    N = node_xy.shape[0]
    W = np.zeros((N, N))
    for tri in triangles:
        i, j, k = tri
        pts = node_xy[[i, j, k], :]
        v0, v1, v2 = pts[1] - pts[0], pts[2] - pts[0], pts[2] - pts[1]
        angles = [
            np.arccos(np.clip(np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)), -1, 1)),
            np.arccos(np.clip(np.dot(-v0, v2)/(np.linalg.norm(v0)*np.linalg.norm(v2)), -1, 1)),
            np.arccos(np.clip(np.dot(-v1, -v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))
        ]
        cots = [1/np.tan(a) if np.tan(a) != 0 else 0 for a in angles]
        for (a, b), c in zip([(j,k),(i,k),(i,j)], cots):
            W[a,b] += c
            W[b,a] += c
    L_t = -W
    for i in range(N):
        L_t[i,i] = np.sum(W[i,:])
    return L_t

def compute_c_ricci(rho_vec, lambda_const=1.0):
    L_t = compute_cotangent_laplacian(node_xy, triangles)
    return lambda_const * (L_t @ rho_vec)

# ------------------------------
# 3. φ-Field PDE Solver
# ------------------------------
def solve_phi_field(rho_vec, G, pf_states, kappa=1.0, max_iter=5000, tol=1e-3, damping=0.05):
    N = len(pf_states)
    A = np.zeros((N, N))
    for i, s1 in enumerate(pf_states):
        for j, s2 in enumerate(pf_states):
            if G.has_edge(s1, s2):
                A[i, j] = 1
    D = np.diag(A.sum(axis=1))
    L = D - A
    phi_vec = np.zeros(N)
    for _ in range(max_iter):
        nonlin = 0.5 * kappa * rho_vec * np.exp(2 * phi_vec)
        F = L @ phi_vec + nonlin
        J = L + np.diag(kappa * rho_vec * np.exp(2 * phi_vec))
        delta_phi = np.linalg.solve(J, -F)
        phi_vec += damping * delta_phi
        if np.linalg.norm(delta_phi) < tol:
            break
    return {pf_states[i]: phi_vec[i] for i in range(N)}

# ------------------------------
# 4. Geodesic Paths
# ------------------------------
def get_all_geodesics():
    return [
        ["000", "100", "101", "111"],
        ["000", "100", "110", "111"],
        ["000", "010", "110", "111"],
        ["000", "010", "011", "111"],
        ["000", "001", "101", "111"],
        ["000", "001", "011", "111"]
    ]

# ------------------------------
# 5. Monte Carlo Engine
# ------------------------------
def monte_carlo_evolve(start_rho, steps, n_molecules, lambda_penalty=1.0):
    state_idx = {s: i for i, s in enumerate(pf_states)}
    molecule_paths = []
    rho_t = np.zeros((steps+1, len(pf_states)))
    rho_t[0] = np.array([start_rho[s] for s in pf_states])
    c_ricci_vec = compute_c_ricci(rho_t[0])

    for mol in range(n_molecules):
        x = np.random.choice(pf_states, p=rho_t[0])
        path = [x]
        for t in range(steps):
            neighbors = list(G.neighbors(x))
            probs = []
            for nbr in neighbors:
                i, j = state_idx[x], state_idx[nbr]
                delta_r = c_ricci_vec[j] - c_ricci_vec[i]
                penalty = np.exp(-lambda_penalty * delta_r)
                probs.append(penalty)
            probs = np.array(probs)
            probs /= probs.sum()
            x = np.random.choice(neighbors, p=probs)
            path.append(x)
            rho_t[t+1][state_idx[x]] += 1 / n_molecules
        molecule_paths.append(path)
    return rho_t, molecule_paths

# ------------------------------
# 6. Open-Box ML Function
# ------------------------------
def score_match(target_rho, final_rho):
    diff = np.abs(np.array([target_rho[s] for s in pf_states]) - final_rho)
    return 1 - np.mean(diff)

def simulate_supervised(start_rho, end_rho, steps, n_molecules, trials=10):
    best_score = -np.inf
    best_paths = None
    best_rho = None
    for _ in range(trials):
        rho_t, paths = monte_carlo_evolve(start_rho, steps, n_molecules)
        score = score_match(end_rho, rho_t[-1])
        if score > best_score:
            best_score = score
            best_paths = paths
            best_rho = rho_t
    return best_rho, best_paths, best_score

# ------------------------------
# 7. Define Start/End Shapes
# ------------------------------
def generate_k_distribution(k_vals):
    rho = {s: 0.0 for s in pf_states}
    groups = {
        0: ["000"],
        1: ["001", "010", "100"],
        2: ["011", "101", "110"],
        3: ["111"]
    }
    for k, weight in enumerate(k_vals):
        for s in groups[k]:
            rho[s] = weight / len(groups[k])
    return rho

# ------------------------------
# 8. Example Run
# ------------------------------
start_k = [0.25, 0.75, 0.0, 0.0]
end_k   = [0.06, 0.53, 0.33, 0.10]
start_rho = generate_k_distribution(start_k)
end_rho   = generate_k_distribution(end_k)

phi_start = solve_phi_field(np.array([start_rho[s] for s in pf_states]), G, pf_states)
phi_end   = solve_phi_field(np.array([end_rho[s] for s in pf_states]), G, pf_states)

rho_trajectory, molecule_paths, match_score = simulate_supervised(
    start_rho=start_rho,
    end_rho=end_rho,
    steps=240,
    n_molecules=100,
    trials=50
)

print(f"\nFinal matching score vs. target shape: {match_score:.4f}")

