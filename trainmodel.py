#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes + Supervised ML Pipeline

This script combines the Oxi‑Shapes geometric evolution with a supervised machine
learning model. It works in the following steps:
  1. Define the discrete state space (the binomial diamond for R=3) and its 2D embedding.
  2. Define the geometry functions (cotangent Laplacian, c‑Ricci, and anisotropy).
  3. Define an ODE-like evolution function that updates the occupancy distribution ρ(x)
     according to the Einstein-like Oxi‑Shapes field equation.
  4. Generate a dataset of (initial occupancy → final occupancy) pairs by simulating many trajectories.
  5. Define a PyTorch neural network that learns to predict the final occupancy given the initial occupancy.
  6. Train and evaluate the model.
"""

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

###############################################################################
# 1. Define the Discrete State Space (R=3) & Its Embedding
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

flat_pos = {
    "000": (0, 3), "001": (-2, 2), "010": (0, 2), "100": (2, 2),
    "011": (-1, 1), "101": (0, 1), "110": (1, 1), "111": (0, 0)
}
node_xy = np.array([flat_pos[s] for s in pf_states])
tri = Delaunay(node_xy)
triangles = tri.simplices

###############################################################################
# 2. Sheaf Theory Stalk Initialization
###############################################################################
def initialize_sheaf_stalks():
    # For each node (i-state), assign a local vector space
    stalks = {}
    for s in pf_states:
        stalks[s] = np.array(flat_pos[s])  # Using position as toy stalk vector
    return stalks

def sheaf_consistency(stalks):
    # Check local consistency along edges
    inconsistencies = []
    for u, v in allowed_edges:
        diff = np.linalg.norm(stalks[u] - stalks[v])
        if diff > 2.5:  # Arbitrary cutoff for inconsistency
            inconsistencies.append((u, v, diff))
    return inconsistencies

###############################################################################
# 3. Neural ODE System for Ricci-Driven Shape Evolution
###############################################################################
class DifferentiableLambda(nn.Module):
    def __init__(self, init_val=1.0):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.tensor(np.log(init_val), dtype=torch.float32))

    def forward(self):
        return torch.exp(self.log_lambda)

class OxiShapeODE(nn.Module):
    def __init__(self, node_xy, triangles, G, RT=1.0):
        super().__init__()
        self.node_xy = node_xy
        self.triangles = triangles
        self.G = G
        self.RT = RT
        self.state_index = {s: i for i, s in enumerate(pf_states)}
        self.lambda_net = DifferentiableLambda(init_val=1.0)

    def compute_cotangent_laplacian(self):
        N = self.node_xy.shape[0]
        W = np.zeros((N, N))
        for tri in self.triangles:
            i, j, k = tri
            pts = self.node_xy[[i, j, k], :]
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
        return torch.tensor(L_t, dtype=torch.float32)

    def compute_anisotropy(self, c_ricci):
        A = torch.zeros(len(pf_states))
        for i, s in enumerate(pf_states):
            nbrs = list(self.G.neighbors(s))
            grad_sum = 0.0
            for nbr in nbrs:
                j = self.state_index[nbr]
                dist = torch.norm(torch.tensor(self.node_xy[i] - self.node_xy[j], dtype=torch.float32))
                if dist > 1e-8:
                    grad_sum += torch.abs(c_ricci[i] - c_ricci[j]) / dist
            A[i] = grad_sum / len(nbrs) if nbrs else 0.0
        return A

    def compute_entropy_terms(self, rho):
        S_mass = torch.sum(-rho * torch.log(rho + 1e-12))
        S_degen = torch.tensor(np.log(len(rho)))
        # Placeholder for conformational entropy using curvature norm
        L_t = self.compute_cotangent_laplacian()
        c_ricci = torch.matmul(L_t, rho)
        S_conf = torch.sum(torch.abs(c_ricci))
        return S_mass, S_conf, S_degen

    def forward(self, t, rho):
        L_t = self.compute_cotangent_laplacian()
        lambda_val = self.lambda_net()
        c_ricci = lambda_val * torch.matmul(L_t, rho)
        A_field = self.compute_anisotropy(c_ricci)

        inflow = torch.zeros_like(rho)
        outflow = torch.zeros_like(rho)
        for i, s_i in enumerate(pf_states):
            nbrs = list(self.G.neighbors(s_i))
            for nbr in nbrs:
                j = self.state_index[nbr]
                delta_E = c_ricci[j] - c_ricci[i]
                S_term = torch.log(rho[j] + 1e-12) - torch.log(rho[i] + 1e-12)
                delta_f = delta_E + 0.5 * S_term - A_field[j]
                p_ij = torch.exp(-delta_f / self.RT)
                inflow[j] += rho[i] * p_ij
                outflow[i] += rho[i] * p_ij

        # Log entropy components (for optional use downstream)
        self.S_mass, self.S_conf, self.S_degen = self.compute_entropy_terms(rho)

        return inflow - outflow

###############################################################################
# 4. Persistent Homology — Betti Number Tracker
###############################################################################
def extract_betti_numbers(rho_snapshot, threshold=0.1):
    active_indices = np.where(rho_snapshot > threshold)[0]
    points = node_xy[active_indices]
    if len(points) < 2:
        return {'beta0': len(points), 'beta1': 0}
    diagrams = ripser(points, maxdim=1)['dgms']
    beta0 = len([pt for pt in diagrams[0] if pt[1] == np.inf])
    beta1 = len(diagrams[1])
    return {'beta0': beta0, 'beta1': beta1}

###############################################################################
# 5. Data Generation (with Energy Classifier)
###############################################################################

def compute_ricci_scalar(rho, L_t):
    return np.dot(L_t, rho)  # Ricci from discrete Laplacian

def classify_energy(ricci_scalar):
    curvature_magnitude = np.linalg.norm(ricci_scalar)
    if curvature_magnitude < 0.5:
        return 0  # low
    elif curvature_magnitude < 1.5:
        return 1  # moderate
    else:
        return 2  # high

def create_energy_classified_dataset(num_samples=1000, L_t=None):
    X, Y, labels = [], [], []
    for _ in range(num_samples):
        vec = np.random.rand(8)
        vec /= vec.sum()
        rho_start = vec
        # Simulated dynamics: small drift
        rho_end = rho_start + 0.05 * (np.random.rand(8) - 0.5)
        rho_end = np.clip(rho_end, 0, None)
        rho_end /= rho_end.sum()

        # Energy label using Ricci scalar
        if L_t is not None:
            ricci = compute_ricci_scalar(rho_end, L_t)
        else:
            ricci = rho_end - rho_start  # fallback approximation
        label = classify_energy(ricci)

        X.append(rho_start)
        Y.append(rho_end)
        labels.append(label)
    return np.array(X), np.array(Y), np.array(labels)

###############################################################################
# 6. Neural Network for Learning
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
# 7. Train and Evaluate
###############################################################################
def train_model_with_constraints(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3, lambda_topo=1.0, lambda_vol=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)

        # Core loss
        main_loss = mse_loss(pred, Y_train_t)

        # Volume constraint
        vol_constraint = torch.mean((torch.sum(pred, dim=1) - torch.sum(Y_train_t, dim=1)) ** 2)

        # Topology constraint
        support_pred = (pred > 0.05).float()
        support_true = (Y_train_t > 0.05).float()
        topo_constraint = torch.mean((support_pred - support_true) ** 2)

        # Total loss
        total_loss = main_loss + lambda_vol * vol_constraint + lambda_topo * topo_constraint
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = mse_loss(val_pred, Y_val_t)
            print(f"Epoch {epoch} | Total Loss: {total_loss.item():.6f} | Main: {main_loss.item():.6f} | Val: {val_loss.item():.6f}")

###############################################################################
# 8. Geodesic Paths
###############################################################################
geodesic_paths = [
    ["000", "100", "101", "111"],
    ["000", "100", "110", "111"],
    ["000", "010", "110", "111"],
    ["000", "010", "011", "111"],
    ["000", "001", "101", "111"],
    ["000", "001", "011", "111"],
]

###############################################################################
# 9. Main Execution (with Save + Geodesics)
###############################################################################
if __name__ == "__main__":
    print("Generating dataset using ODE-based evolution...")
    t_span = torch.linspace(0.0, 1.0, 80, dtype=torch.float32)
    X, Y = create_dataset_ODE(num_samples=2000, t_span=t_span)
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]

    # Initialize model
    model = OxiNet(input_dim=8, hidden_dim=32, output_dim=8)
    
    # Train with constraints
    train_model_with_constraints(
        model, X_train, Y_train, X_val, Y_val,
        epochs=100, lr=1e-3, lambda_topo=0.5, lambda_vol=0.5
    )

    print("\nEvaluating on validation data...")
    pred_val, val_loss = evaluate_model(model, X_val, Y_val)
    print(f"Validation Loss: {val_loss:.6f}")

    # Save model and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'geodesic_paths': geodesic_paths,
        'input_dim': 8,
        'hidden_dim': 32,
        'output_dim': 8
    }, "oxinet_model.pt")
    print("✅ Trained model saved to: oxinet_model.pt")

    print("\nTracking Betti number evolution in a few samples...")
    for idx in np.random.choice(len(X_val), 3, replace=False):
        init_occ = X_val[idx]
        print("\n--- Sample ---")
        print("Initial occupancy:", np.round(init_occ, 3))
        print("True final occupancy:", np.round(Y_val[idx], 3))
        print("Predicted final occupancy:", np.round(pred_val[idx], 3))

        betti = extract_betti_numbers(pred_val[idx], threshold=0.05)
        print(f"Betti Numbers → β₀: {betti['beta0']}  |  β₁: {betti['beta1']}")
 
