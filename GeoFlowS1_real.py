#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes + Supervised ML Pipeline.
  
Dependencies: torchdiffeq, ripser, persim
"""

# Install necessary packages (if not already installed)
!pip install torchdiffeq ripser persim

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import pandas as pd
from ripser import ripser
from persim import plot_diagrams

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###############################################################################
# Differentiable Lambda: Trainable scaling for c-Ricci
###############################################################################
class DifferentiableLambda(nn.Module):
    def __init__(self, init_val=1.0):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.tensor(np.log(init_val), dtype=torch.float32, device=device))
    def forward(self):
        return torch.exp(self.log_lambda)

# Instantiate a global differentiable lambda.
lambda_net = DifferentiableLambda(init_val=1.0).to(device)

# Global RT constant.
RT = 1.0

###############################################################################
# Define the Discrete State Space (R=3) & Its Embedding
###############################################################################
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
state_index = {s: i for i, s in enumerate(pf_states)}
num_states = len(pf_states)

# 2D embedding ("flat diamond")
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

# Precompute neighbor indices to speed up ODE loop.
neighbor_indices = {s: [state_index[nbr] for nbr in G.neighbors(s)] for s in pf_states}
# Precompute torch tensor for flat positions.
flat_pos_tensor = torch.tensor(node_xy, dtype=torch.float32, device=device)

# Visualize the state space.
plt.figure(figsize=(6,6))
nx.draw(G, pos=flat_pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, edge_color='gray')
plt.title("Discrete Proteoform State Space (R=3)")
plt.show()

###############################################################################
# Initialize Occupancy and Solve φ Field (Poisson-like PDE)
###############################################################################
# Initialize occupancy randomly.
rho_vec = np.random.rand(num_states)
rho_vec /= rho_vec.sum()
occupancy = {pf_states[i]: rho_vec[i] for i in range(num_states)}

# Build a simple Laplacian L from graph connectivity.
A = np.zeros((num_states, num_states))
for i, s1 in enumerate(pf_states):
    for j, s2 in enumerate(pf_states):
        if G.has_edge(s1, s2):
            A[i, j] = 1
D = np.diag(A.sum(axis=1))
L = D - A

phi_vec = np.zeros(num_states)
kappa = 1.0
max_iter = 10000
tol = 1e-3
damping = 0.05

for iter in range(max_iter):
    nonlin = 0.5 * kappa * rho_vec * np.exp(2 * phi_vec)
    F = L @ phi_vec + nonlin
    J = L + np.diag(kappa * rho_vec * np.exp(2 * phi_vec))
    delta_phi = np.linalg.solve(J, -F)
    for i, s in enumerate(pf_states):
        if occupancy[s] <= 1e-14:
            delta_phi[i] = 0.0
    phi_vec += damping * delta_phi
    if np.linalg.norm(delta_phi) < tol:
        print(f"φ converged after {iter+1} iterations.")
        break
else:
    print("φ did not converge within iteration limit.")

phi = {s: phi_vec[state_index[s]] for s in pf_states}

###############################################################################
# Compute Enriched C-Ricci and Anisotropy
###############################################################################
def compute_cotangent_laplacian(node_xy, triangles):
    N = node_xy.shape[0]
    W = np.zeros((N, N))
    for tri_idx in triangles:
        i, j, k = tri_idx
        pts = node_xy[[i, j, k], :]
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[0]
        v2 = pts[2] - pts[1]
        angle_i = np.arccos(np.clip(np.dot(v0, v1) / (np.linalg.norm(v0)*np.linalg.norm(v1)), -1, 1))
        angle_j = np.arccos(np.clip(np.dot(-v0, v2) / (np.linalg.norm(v0)*np.linalg.norm(v2)), -1, 1))
        angle_k = np.arccos(np.clip(np.dot(-v1, -v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))
        cot_i = 1/np.tan(angle_i) if np.tan(angle_i)!=0 else 0
        cot_j = 1/np.tan(angle_j) if np.tan(angle_j)!=0 else 0
        cot_k = 1/np.tan(angle_k) if np.tan(angle_k)!=0 else 0
        for (a, b), cval in zip([(j,k), (i,k), (i,j)], [cot_i, cot_j, cot_k]):
            W[a, b] += cval
            W[b, a] += cval
    L_t = -W
    for i in range(N):
        L_t[i,i] = np.sum(W[i,:])
    return L_t

# Precompute cotangent Laplacian in NumPy, then convert to torch.
L_np = compute_cotangent_laplacian(node_xy, triangles)
L_torch = torch.tensor(L_np, dtype=torch.float32, device=device)

# For visualization, we use a fixed lambda value (from the differentiable lambda).
lambda_value = lambda_net().detach().cpu().numpy()
c_ricci_vec = lambda_value * (L_np @ rho_vec)
c_ricci_nodes = {pf_states[i]: c_ricci_vec[i] for i in range(num_states)}

# Assign edgewise c-Ricci (average of endpoints).
for (u, v) in G.edges():
    G[u][v]['cRicci'] = (c_ricci_nodes[u] + c_ricci_nodes[v]) / 2.0

# Compute anisotropy field A(x) for each node.
A_field = {}
for s in pf_states:
    nbrs = list(G.neighbors(s))
    grad_sum = 0.0
    count = 0
    for nbr in nbrs:
        dist = np.linalg.norm(np.array(flat_pos[s]) - np.array(flat_pos[nbr]))
        if dist > 1e-6:
            grad_sum += abs(c_ricci_nodes[s] - c_ricci_nodes[nbr]) / dist
            count += 1
    A_field[s] = grad_sum / count if count > 0 else 0.0

# For each edge, assign a penalty factor.
for (u, v) in G.edges():
    penalty = np.exp(- (A_field[u] + A_field[v]) / 2.0)
    G[u][v]['penalty'] = penalty

###############################################################################
# Sheaf Theory: Stalk Initialization & Consistency Check
###############################################################################
def initialize_sheaf_stalks():
    stalks = {}
    for s in pf_states:
        stalks[s] = np.array(flat_pos[s])
    return stalks

def sheaf_consistency(stalks):
    inconsistencies = []
    for u, v in allowed_edges:
        diff = np.linalg.norm(stalks[u] - stalks[v])
        if diff > 2.5:
            inconsistencies.append((u, v, diff))
    return inconsistencies

sheaf_stalks = initialize_sheaf_stalks()
inconsistencies = sheaf_consistency(sheaf_stalks)
if inconsistencies:
    print("Sheaf inconsistencies found:", inconsistencies)
else:
    print("Sheaf stalks are consistent.")

###############################################################################
# Global Parameters
###############################################################################

MAX_MOVES_PER_STEP = 20  # Maximum number of molecule actions allowed per step
PDE_UPDATE_INTERVAL = 5
pde_step_counter = 0

###############################################################################
# ALIVE ODE: Action Limited Evolution Engine
###############################################################################

def oxi_shapes_ode_alive(t, rho):
    """
    ALIVE: Action Limited Evolution Engine
    Discrete molecule actions under capped external force, shaped by geometry.
    Geometry (c-Ricci) and anisotropy updated every N steps. Only full molecules can move.
    """
    with torch.no_grad():
        rho = rho.to(device)

        # Quantize to 100 molecules
        counts = torch.round(rho * 100)
        counts = counts.clamp(0, 100)
        counts[-1] = 100 - counts[:-1].sum()
        rho = counts / 100.0

        # PDE geometry update every N steps
        global pde_step_counter
        pde_step_counter += 1
        if pde_step_counter % PDE_UPDATE_INTERVAL == 0:
            update_c_ricci_from_rho(rho)

        inflow = torch.zeros_like(rho, device=device)
        outflow = torch.zeros_like(rho, device=device)

        # Degeneracy
        degeneracy_map = {0: 1, 1: 3, 2: 3, 3: 1}
        degeneracy = torch.tensor([
            degeneracy_map[s.count('1')] for s in pf_states
        ], dtype=torch.float32, device=device)

        # Constants
        baseline_DeltaE = 1.0
        Delta_x = 1.0
        RT = 1.0

        # Anisotropy field A(x)
        global anisotropy
        anisotropy = torch.zeros(num_states, device=device)
        for i, s in enumerate(pf_states):
            nbrs = neighbor_indices[s]
            if len(nbrs) == 0:
                continue
            grad_vals = []
            for j in nbrs:
                dist = torch.norm(flat_pos_tensor[i] - flat_pos_tensor[j])
                if dist > 1e-6:
                    grad_vals.append(torch.abs(c_ricci[i] - c_ricci[j]) / dist)
            anisotropy[i] = sum(grad_vals) / len(grad_vals) if grad_vals else 0.0

        # Randomize number of molecules allowed to act
        total_moves = np.random.randint(0, MAX_MOVES_PER_STEP + 1)

        # Get list of indices with molecules
        candidate_indices = [i for i, c in enumerate(counts) if c > 0]

        for _ in range(total_moves):
            if not candidate_indices:
                break
            i = np.random.choice(candidate_indices)
            s = pf_states[i]

            # Build transition probability list
            neighbors = neighbor_indices[s]
            if not neighbors:
                inflow[i] += 0.01
                continue

            probs = []
            for j in neighbors:
                delta_S = compute_entropy_cost(i, j, rho)
                delta_f = (baseline_DeltaE * torch.exp(rho[i]) - c_ricci[i] + delta_S) / RT
                p_ij = torch.exp(-delta_f) * torch.exp(-anisotropy[j])
                probs.append(p_ij.item())

            probs = torch.tensor(probs, dtype=torch.float32, device=device)
            if probs.sum() < 1e-8:
                inflow[i] += 0.01
                continue

            probs /= probs.sum()
            j_choice = torch.multinomial(probs, 1).item()
            target_idx = neighbor_indices[s][j_choice]

            inflow[target_idx] += 0.01
            outflow[i] += 0.01

        # Remainder of molecules stay in place
        for i in range(num_states):
            inflow[i] += (counts[i] / 100.0) - outflow[i]

        return inflow - outflow

###############################################################################
# Geometry Update
###############################################################################

def update_c_ricci_from_rho(rho):
    global c_ricci
    c_ricci = lambda_net() * (L_torch @ rho)

###############################################################################
# Entropy Cost Function
###############################################################################

def compute_entropy_cost(i, j, rho):
    baseline_DeltaE = 1.0
    mass_heat = 0.1
    reaction_heat = 0.01 * baseline_DeltaE
    conformational_cost = torch.abs(c_ricci[j])
    degeneracy_map = {0: 1, 1: 3, 2: 3, 3: 1}
    deg = degeneracy_map[pf_states[j].count('1')]
    degeneracy_penalty = 1.0 / deg
    return mass_heat + reaction_heat + conformational_cost + degeneracy_penalty

###############################################################################
# Geodesic Tracking Functions
###############################################################################

def dominant_geodesic(trajectory, geodesics):
    max_score = 0
    best_path = None
    for path in geodesics:
        score = sum([1 for s in path if s in trajectory])
        if score > max_score:
            max_score = score
            best_path = path
    return tuple(best_path) if best_path else None

def evolve_time_series_and_geodesic(rho0, t_span):
    rho_t = odeint(oxi_shapes_ode_alive, rho0, t_span)  
    dominant_path = []
    for r in rho_t:
        max_idx = torch.argmax(r).item()
        dominant_path.append(pf_states[max_idx])
    geo = dominant_geodesic(dominant_path, geodesics)
    return rho_t, geo

def geodesic_loss(predicted_final, initial, geodesics, pf_states):
    max_idx_pred = torch.argmax(predicted_final, dim=1)
    max_idx_init = torch.argmax(initial, dim=1)
    batch_loss = 0.0
    for i in range(len(max_idx_pred)):
        pred_path = [pf_states[max_idx_init[i].item()], pf_states[max_idx_pred[i].item()]]
        valid = any(set(pred_path).issubset(set(g)) for g in geodesics)
        if not valid:
            batch_loss += 1.0
    return batch_loss / len(max_idx_pred)

geodesics = [
    ["000", "100", "101", "111"],
    ["000", "100", "110", "111"],
    ["000", "010", "110", "111"],
    ["000", "010", "011", "111"],
    ["000", "001", "101", "111"],
    ["000", "001", "011", "111"]
]

from collections import Counter
geo_counter = Counter()

###############################################################################
# Data Generation (Systematic Oxi-Shape Sampling + Geodesic + Topology)
###############################################################################

def generate_systematic_initials():
    initials = []
    # (1) Single i-state occupancy
    for i in range(8):
        vec = np.zeros(8)
        vec[i] = 1.0
        initials.append(vec)

    # (2) Flat occupancy
    initials.append(np.full(8, 1.0 / 8))

    # (3) Curved within k=1 (e.g., 010 peak)
    curved_k1 = np.array([0.0, 0.15, 0.7, 0.15, 0.0, 0.0, 0.0, 0.0])
    initials.append(curved_k1 / curved_k1.sum())

    # (4) Curved within k=2 (e.g., 101 peak)
    curved_k2 = np.array([0.0, 0.0, 0.0, 0.0, 0.15, 0.7, 0.15, 0.0])
    initials.append(curved_k2 / curved_k2.sum())

    # (5) Flat in k=0 & k=1, peaked in k=2
    hybrid = np.array([0.05, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.05])
    initials.append(hybrid / hybrid.sum())

    # (6) Bell shape across k
    bell = np.array([0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.2])
    initials.append(bell / bell.sum())

    # (7) Geometric gradient (left to right in flat_pos)
    gradient = np.linspace(0.1, 0.9, 8)
    initials.append(gradient / gradient.sum())

    return initials

from ripser import ripser
from persim import plot_diagrams

# Persistent homology diagram

def persistent_diagram(rho):
    dist = np.abs(rho[:, None] - rho[None, :])
    dgms = ripser(dist, distance_matrix=True, maxdim=1)['dgms']
    return dgms

# Persistent entropy

def topological_entropy(dgm):
    if len(dgm) == 0 or len(dgm[0]) == 0:
        return 0.0
    lifespans = dgm[0][:, 1] - dgm[0][:, 0]
    probs = lifespans / np.sum(lifespans)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return entropy

from collections import Counter

def create_dataset_ODE_alive(t_span=None):
    if t_span is None:
        t_span = torch.linspace(0.0, 1.0, 100, dtype=torch.float32, device=device)

    X, Y, geos = [], [], []
    geo_counter = Counter()
    initials = generate_systematic_initials()

   for vec in initials:
    for _ in range(5):
        rho0 = torch.tensor(vec, dtype=torch.float32, device=device)
        rho_t, geopath = evolve_time_series_and_geodesic(rho0, t_span)

        final_rho = rho_t[-1]  # ← keep this line
        # final_rho = final_rho / final_rho.sum()  ← REMOVE or comment this out

        # Digital enforcement
        assert torch.allclose(final_rho * 100, torch.round(final_rho * 100), atol=1e-6), \
            f"Non-digital occupancy detected: {final_rho}"

        X.append(rho0.detach().cpu().numpy())
        Y.append(final_rho.detach().cpu().numpy())

        if geopath:
            geo_counter[geopath] += 1
            geos.append(geopath)

    print("Most traversed geodesics:")
    for path, count in geo_counter.most_common():
        print(" → ".join(path), "| Count:", count)

    return np.array(X), np.array(Y), geos

###############################################################################
# Neural Network for Learning (OxiFlowNet)
###############################################################################
class OxiNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=8):
        super(OxiNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

###############################################################################
# Training and Evaluation Functions
###############################################################################
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3,
                lambda_topo=0.5, lambda_vol=0.5, lambda_geo=0.5, geodesics=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        main_loss = mse_loss(pred, Y_train_t)
        vol_constraint = torch.mean((torch.sum(pred, dim=1) - torch.sum(Y_train_t, dim=1)) ** 2)
        support_pred = (pred > 0.05).float()
        support_true = (Y_train_t > 0.05).float()
        topo_constraint = torch.mean((support_pred - support_true) ** 2)
        geo_loss = geodesic_loss(pred, X_train_t, geodesics, pf_states)
        total_loss = main_loss + lambda_vol * vol_constraint + lambda_topo * topo_constraint + lambda_geo * geo_loss
        total_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = mse_loss(val_pred, Y_val_t)
            print(f"Epoch {epoch}/{epochs} | Total Loss: {total_loss.item():.6f} | Train: {main_loss.item():.6f} | Val: {val_loss.item():.6f}")

def evaluate_model(model, X_test, Y_test):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_test = model(X_test_t)
    mse_loss = nn.MSELoss()
    test_loss = mse_loss(pred_test, Y_test_t).item()
    return pred_test.detach().cpu().numpy(), test_loss

###############################################################################
# Main Execution: Data Generation, Training, and Evaluation
###############################################################################
if __name__ == "__main__":
    print("Generating dataset using systematic Oxi-Shape sampling...")
    t_span = torch.linspace(0.0, 1.0, 100, dtype=torch.float32, device=device)
    X, Y, geos = create_dataset_ODE_alive(t_span=t_span)
    perm = np.random.permutation(len(X))
    X, Y, targets = X[perm], Y[perm], targets[perm]
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]

    print("Building and training the neural network (OxiFlowNet)...")
    model = OxiNet(input_dim=8, hidden_dim=32, output_dim=8).to(device)
    train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3, lambda_topo=0.5, lambda_vol=0.5)

    print("\nEvaluating on validation data...")
    pred_val, val_loss = evaluate_model(model, X_val, Y_val)
    print(f"Validation Loss: {val_loss:.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'pf_states': pf_states,
        'flat_pos': flat_pos
    }, "oxinet_model.pt")
    print("✅ Trained model saved to 'oxinet_model.pt'")

    for idx in np.random.choice(len(X_val), 3, replace=False):
        init_occ = X_val[idx]
        true_final = Y_val[idx]
        pred_final = pred_val[idx]
        print("\n--- Sample ---")
        print("Initial occupancy:", np.round(init_occ, 3))
        print("True final occupancy:", np.round(true_final, 3))
        print("Predicted final occupancy:", np.round(pred_final, 3))
