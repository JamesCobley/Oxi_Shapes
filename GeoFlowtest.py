#!/usr/bin/env python
# coding: utf-8

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
state_idx = {s: i for i, s in enumerate(pf_states)}

# ------------------------------
# 2. Geometry and Curvature
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
# 3. Entropy Calculation
# ------------------------------
def compute_entropy(rho_vec):
    return -np.sum([p * np.log(p + 1e-14) for p in rho_vec])

# ------------------------------
# 4. Shape Generator from k-priors
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
# 5. Monte Carlo Geodesic Engine
# ------------------------------
def monte_carlo_geo_evolve(start_rho, steps, n_molecules, lambda_penalty=1.0):
    molecule_paths = []
    rho_t = np.zeros((steps+1, len(pf_states)))
    rho_t[0] = np.array([start_rho[s] for s in pf_states])
    entropy_t = [compute_entropy(rho_t[0])]
    energy_t = []

    for mol in range(n_molecules):
        x = np.random.choice(pf_states, p=rho_t[0])
        path = [x]
        for t in range(steps):
            c_ricci_vec = compute_c_ricci(rho_t[t])
            neighbors = list(G.neighbors(x))
            probs = []
            delta_energies = []
            for nbr in neighbors:
                i, j = state_idx[x], state_idx[nbr]
                delta_r = c_ricci_vec[j] - c_ricci_vec[i]
                delta_energies.append(delta_r)
                penalty = np.exp(-lambda_penalty * delta_r)
                probs.append(penalty)
            probs = np.array(probs)
            probs /= probs.sum()
            x = np.random.choice(neighbors, p=probs)
            path.append(x)
            rho_t[t+1][state_idx[x]] += 1 / n_molecules
        molecule_paths.append(path)
        entropy_t.append(compute_entropy(rho_t[min(t+1, steps)]))
    return rho_t, molecule_paths, entropy_t

# ------------------------------
# 6. Open Box Scoring & ML Inference
# ------------------------------
def score_match(target_rho, final_rho):
    diff = np.abs(np.array([target_rho[s] for s in pf_states]) - final_rho)
    return 1 - np.mean(diff)

def simulate_geoflownet(start_rho, end_rho, steps, n_molecules, trials=10):
    records = []
    for i in range(trials):
        rho_t, paths, entropy = monte_carlo_geo_evolve(start_rho, steps, n_molecules)
        score = score_match(end_rho, rho_t[-1])
        records.append((score, rho_t, paths, entropy))
    records.sort(key=lambda x: -x[0])
    return records

# ------------------------------
# 7. Run Example
# ------------------------------
if __name__ == "__main__":
    start_k = [0.25, 0.75, 0.0, 0.0]
    end_k   = [0.06, 0.53, 0.33, 0.10]
    steps = 240
    n_molecules = 100
    trials = 20

    start_rho = generate_k_distribution(start_k)
    end_rho   = generate_k_distribution(end_k)

    results = simulate_geoflownet(start_rho, end_rho, steps, n_molecules, trials)
    best_score, best_rho, best_paths, best_entropy = results[0]

    print(f"✔ Best Match Score: {best_score:.4f}")
    print("✔ Final Occupancy:")
    print(pd.Series(final_rho, index=pf_states))
    print(f"✔ ΔS = {best_entropy[-1] - best_entropy[0]:.4f}")
    print(f"✔ Trajectories recorded for {n_molecules} molecules across {steps} steps.")
