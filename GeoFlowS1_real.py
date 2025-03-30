#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes + Supervised ML Pipeline with Neural ODE PDE Solver,
Sheaf Consistency, and a Differentiable Lambda Parameter

This script performs the following steps:
  1. Defines the discrete state space (the binomial diamond for R=3) and its 2D embedding.
  2. Initializes the occupancy distribution and solves for the φ field (via a Poisson-like PDE).
  3. Computes the enriched c-Ricci (via a cotangent Laplacian) and anisotropy.
  4. Initializes sheaf stalks and checks consistency.
  5. Implements a differentiable ODE (via torchdiffeq.odeint) that evolves the occupancy distribution ρ(x)
     according to the Einstein-like Oxi-Shapes field equation. The scaling constant for curvature is now trainable
     via a DifferentiableLambda module.
  6. Generates a dataset of (initial occupancy → final occupancy) pairs using the PDE solver.
  7. Defines and trains a PyTorch neural network (OxiFlowNet) to predict final occupancy from the initial occupancy.
  8. Evaluates and saves the trained model.
  
Dependencies: torchdiffeq, ripser, persim
"""

# Install necessary packages (if running in a notebook)
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
# Differentiable Lambda: Trainable scaling for c-Ricci
###############################################################################
class DifferentiableLambda(nn.Module):
    def __init__(self, init_val=1.0):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.tensor(np.log(init_val), dtype=torch.float32))
    def forward(self):
        return torch.exp(self.log_lambda)

# Instantiate a global differentiable lambda parameter
lambda_net = DifferentiableLambda(init_val=1.0)

# Global RT constant
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

# Visualize the discrete state space
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
    # Pin nodes with near-zero occupancy using the occupancy dictionary.
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
        angle_i = np.arccos(np.clip(np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)), -1, 1))
        angle_j = np.arccos(np.clip(np.dot(-v0, v2)/(np.linalg.norm(v0)*np.linalg.norm(v2)), -1, 1))
        angle_k = np.arccos(np.clip(np.dot(-v1, -v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))
        cot_i = 1/np.tan(angle_i) if np.tan(angle_i)!=0 else 0
        cot_j = 1/np.tan(angle_j) if np.tan(angle_j)!=0 else 0
        cot_k = 1/np.tan(angle_k) if np.tan(angle_k)!=0 else 0
        for (a, b), cval in zip([(j,k), (i,k), (i,j)], [cot_i, cot_j, cot_k]):
            W[a,b] += cval
            W[b,a] += cval
    L_t = -W
    for i in range(N):
        L_t[i,i] = np.sum(W[i, :])
    return L_t

# Precompute the cotangent Laplacian (using NumPy) and convert to torch.
L_np = compute_cotangent_laplacian(node_xy, triangles)
L_torch = torch.tensor(L_np, dtype=torch.float32)
lambda_val = lambda_net().detach().cpu().numpy()  # Get current lambda value as a fixed scalar for this computation
c_ricci_vec = lambda_val * (L_np @ rho_vec)
c_ricci_nodes = {pf_states[i]: c_ricci_vec[i] for i in range(num_states)}

# Assign edgewise c-Ricci as the average of endpoint values.
for (u, v) in G.edges():
    G[u][v]['cRicci'] = (c_ricci_nodes[u] + c_ricci_nodes[v]) / 2.0

# Compute anisotropy field A(x): for each node, average the gradient over its neighbors.
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

# For each edge, assign a penalty factor based on anisotropy.
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
# Neural ODE Function for Occupancy Evolution (Differentiable PDE Solver)
###############################################################################
def oxi_shapes_ode(t, rho):
    """
    ODE function for evolving occupancy ρ (tensor of shape (num_states,)).
    Uses the differentiable lambda parameter.
    """
    # Compute c-Ricci using the trainable lambda parameter.
    c_ricci = lambda_net() * (L_torch @ rho)
    
    # Compute anisotropy field A
    A = torch.zeros(num_states)
    for i, s in enumerate(pf_states):
        nbrs = list(G.neighbors(s))
        if len(nbrs) == 0:
            A[i] = 0.0
            continue
        grad_sum = 0.0
        for nbr in nbrs:
            j = state_index[nbr]
            pos_i = torch.tensor(flat_pos[s], dtype=torch.float32)
            pos_j = torch.tensor(flat_pos[nbr], dtype=torch.float32)
            dist = torch.norm(pos_i - pos_j)
            if dist > 1e-6:
                grad_sum += torch.abs(c_ricci[i] - c_ricci[j]) / dist
        A[i] = grad_sum / len(nbrs)
    
    # Build inflow and outflow terms.
    inflow = torch.zeros_like(rho)
    outflow = torch.zeros_like(rho)
    for i, s_i in enumerate(pf_states):
        for nbr in G.neighbors(s_i):
            j = state_index[nbr]
            occ_i = rho[i]
            occ_j = rho[j]
            delta_E = c_ricci[j] - c_ricci[i]
            entropy_term = 0.0
            if occ_i > 1e-12 and occ_j > 1e-12:
                entropy_term = torch.log(occ_j + 1e-12) - torch.log(occ_i + 1e-12)
            delta_f = (delta_E + 0.5 * entropy_term - A[j]) / RT
            p_ij = torch.exp(-delta_f)
            inflow[j] += occ_i * p_ij
            outflow[i] += occ_i * p_ij
    return inflow - outflow

def evolve_oxi_shapes_pde(rho0, t_span):
    """
    Evolve initial occupancy rho0 (tensor of shape (num_states,))
    over time t_span using torchdiffeq.odeint.
    Returns the final occupancy (normalized).
    """
    rho0 = rho0 / rho0.sum()
    rho_t = odeint(oxi_shapes_ode, rho0, t_span)
    final_rho = rho_t[-1]
    final_rho = final_rho / final_rho.sum()
    return final_rho

###############################################################################
# Data Generation: Create (initial -> final) Occupancy Pairs via PDE Solver
###############################################################################
def create_dataset_ODE(num_samples=1000, t_span=None):
    if t_span is None:
        t_span = torch.linspace(0.0, 1.0, 80, dtype=torch.float32)
    X, Y = [], []
    for _ in range(num_samples):
        vec = np.random.rand(num_states)
        vec /= vec.sum()
        rho0 = torch.tensor(vec, dtype=torch.float32)
        final_rho = evolve_oxi_shapes_pde(rho0, t_span)
        X.append(rho0.detach().numpy())
        Y.append(final_rho.detach().numpy())
    return np.array(X), np.array(Y)

###############################################################################
# 7. Neural Network for Learning (OxiFlowNet)
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
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3, lambda_topo=0.5, lambda_vol=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        main_loss = mse_loss(pred, Y_train_t)
        vol_constraint = torch.mean((torch.sum(pred, dim=1) - torch.sum(Y_train_t, dim=1)) ** 2)
        support_pred = (pred > 0.05).float()
        support_true = (Y_train_t > 0.05).float()
        topo_constraint = torch.mean((support_pred - support_true) ** 2)
        total_loss = main_loss + lambda_vol * vol_constraint + lambda_topo * topo_constraint
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
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)
    with torch.no_grad():
        pred_test = model(X_test_t)
    mse_loss = nn.MSELoss()
    test_loss = mse_loss(pred_test, Y_test_t).item()
    return pred_test.numpy(), test_loss

###############################################################################
# Main Execution: Data Generation, Training, and Evaluation
###############################################################################
if __name__ == "__main__":
    print("Generating dataset using PDE-based evolution (Neural ODE)...")
    t_span = torch.linspace(0.0, 1.0, 80, dtype=torch.float32)
    X, Y = create_dataset_ODE(num_samples=2000, t_span=t_span)
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    
    print("Building and training the neural network (OxiFlowNet)...")
    model = OxiNet(input_dim=8, hidden_dim=32, output_dim=8)
    train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3, lambda_topo=0.5, lambda_vol=0.5)
    
    print("\nEvaluating on validation data...")
    pred_val, val_loss = evaluate_model(model, X_val, Y_val)
    print(f"Validation Loss: {val_loss:.6f}")
    
    # Save the trained model and metadata for later use in Pipeline Step 2.
    torch.save({
        'model_state_dict': model.state_dict(),
        'pf_states': pf_states,
        'flat_pos': flat_pos
    }, "oxinet_model.pt")
    print("✅ Trained model saved to 'oxinet_model.pt'")
    
    # Display a few sample predictions.
    for idx in np.random.choice(len(X_val), 3, replace=False):
        init_occ = X_val[idx]
        true_final = Y_val[idx]
        pred_final = pred_val[idx]
        print("\n--- Sample ---")
        print("Initial occupancy:", np.round(init_occ, 3))
        print("True final occupancy:", np.round(true_final, 3))
        print("Predicted final occupancy:", np.round(pred_final, 3))
