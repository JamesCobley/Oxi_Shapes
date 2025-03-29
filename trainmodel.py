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
!pip install torchdiffeq

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint  # optional if you want a Neural ODE solver; here we use Euler updates for simulation
import pandas as pd

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

# 2D embedding (flat diamond)
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

# Optional: visualize the state space
plt.figure(figsize=(6,6))
nx.draw(G, pos=flat_pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, edge_color='gray')
plt.title("Discrete Proteoform State Space (R=3)")
plt.show()

###############################################################################
# 2. Geometry Functions: Cotangent Laplacian, c-Ricci, and Anisotropy
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
        cot_i = 1.0/np.tan(angle_i) if np.tan(angle_i)!=0 else 0
        cot_j = 1.0/np.tan(angle_j) if np.tan(angle_j)!=0 else 0
        cot_k = 1.0/np.tan(angle_k) if np.tan(angle_k)!=0 else 0
        for (a, b), cval in zip([(j, k), (i, k), (i, j)], [cot_i, cot_j, cot_k]):
            W[a, b] += cval
            W[b, a] += cval
    L_t = -W
    for i in range(N):
        L_t[i, i] = np.sum(W[i, :])
    return L_t

def compute_c_ricci(rho_vec, lambda_const=1.0):
    L_t = compute_cotangent_laplacian(node_xy, triangles)
    return lambda_const * (L_t @ rho_vec)

def compute_anisotropy(c_ricci_vec):
    A = np.zeros(len(pf_states))
    for i, s in enumerate(pf_states):
        nbrs = list(G.neighbors(s))
        if len(nbrs)==0:
            A[i] = 0.0
            continue
        grad_sum = 0.0
        for nbr in nbrs:
            j = state_index[nbr]
            dist = np.linalg.norm(node_xy[i] - node_xy[j])
            if dist>1e-8:
                grad_sum += abs(c_ricci_vec[i] - c_ricci_vec[j]) / dist
        A[i] = grad_sum / len(nbrs)
    return A

###############################################################################
# 3. ODE-like Evolution: Euler Time-Stepping of Occupancy ρ(x)
###############################################################################
def evolve_oxi_shapes(rho_init, dt=0.01, steps=50, lambda_const=1.0, RT=1.0):
    """
    Evolve occupancy ρ(x) according to an Euler update based on the field equation:
      dρ(x_j)/dt = sum_{x_i in N(x_j)} [ρ(x_i)*exp(-Δf(i->j))] 
                   - ρ(x_j)*sum_{x_k in N(x_j)} [exp(-Δf(j->k))]
    where Δf is computed from differences in c-Ricci and anisotropy.
    """
    pf_list = list(rho_init.keys())
    N = len(pf_list)
    idx_map = {s: i for i, s in enumerate(pf_list)}
    rho_vec = np.array([rho_init[s] for s in pf_list], dtype=float)
    rho_vec /= rho_vec.sum()  # Ensure normalization

    # Record history if needed
    rho_history = []
    c_ricci_history = []

    for t in range(steps):
        rho_history.append(rho_vec.copy())
        c_ricci_vec = compute_c_ricci(rho_vec, lambda_const=lambda_const)
        c_ricci_history.append(c_ricci_vec.copy())
        A_field = compute_anisotropy(c_ricci_vec)
        inflow = np.zeros(N)
        outflow = np.zeros(N)
        for i, s_i in enumerate(pf_list):
            nbrs = list(G.neighbors(s_i))
            for nbr in nbrs:
                j = idx_map[nbr]
                # Free energy difference computed as in our field eq (simplified)
                occ_i = rho_vec[i]
                occ_j = rho_vec[j]
                delta_E = c_ricci_vec[j] - c_ricci_vec[i]
                entropy_term = 0
                if occ_i>1e-12 and occ_j>1e-12:
                    entropy_term = np.log(occ_j) - np.log(occ_i)
                # Δf includes an anisotropy penalty from neighbor j
                delta_f = delta_E + 0.5*entropy_term - A_field[j]
                delta_f /= RT
                p_ij = np.exp(-delta_f)
                inflow[j] += rho_vec[i] * p_ij
                outflow[i] += rho_vec[i] * p_ij
        new_rho = rho_vec + dt*(inflow - outflow)
        new_rho = np.maximum(new_rho, 0.0)
        new_rho /= new_rho.sum() if new_rho.sum() > 1e-12 else 1.0
        rho_vec = new_rho

    final_rho = {pf_list[i]: rho_vec[i] for i in range(N)}
    return final_rho, np.array(rho_history), np.array(c_ricci_history)

###############################################################################
# 4. Data Generation: Create Dataset of (Initial -> Final) Occupancy Pairs
###############################################################################
def random_rho_init(num_samples=1000):
    data = []
    for _ in range(num_samples):
        vec = np.random.rand(8)
        vec /= vec.sum()
        rho_dict = {s: vec[i] for i, s in enumerate(pf_states)}
        data.append(rho_dict)
    return data

def create_dataset(num_samples=1000, dt=0.01, steps=50, lambda_const=1.0, RT=1.0):
    initial_rhos = random_rho_init(num_samples)
    X, Y = [], []
    for rho_init in initial_rhos:
        final_rho, _, _ = evolve_oxi_shapes(rho_init, dt=dt, steps=steps,
                                            lambda_const=lambda_const, RT=RT)
        init_vec = np.array([rho_init[s] for s in pf_states], dtype=float)
        final_vec = np.array([final_rho[s] for s in pf_states], dtype=float)
        X.append(init_vec)
        Y.append(final_vec)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

###############################################################################
# 5. Define the Neural Network Model in PyTorch
###############################################################################
class OxiNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=8):
        super(OxiNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # Use softmax to ensure output is a probability distribution (sums to 1)
        x = torch.softmax(x, dim=1)
        return x

###############################################################################
# 6. Training and Evaluation Functions
###############################################################################
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = criterion(pred_val, Y_val_t)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}: Train Loss={loss.item():.6f}, Val Loss={val_loss.item():.6f}")

def evaluate_model(model, X_test, Y_test):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)
    with torch.no_grad():
        pred_test = model(X_test_t)
    criterion = nn.MSELoss()
    test_loss = criterion(pred_test, Y_test_t).item()
    return pred_test.numpy(), test_loss

###############################################################################
# 7. Main Execution: Generate Data, Train, Evaluate
###############################################################################
if __name__ == "__main__":
    # Generate dataset: mapping initial occupancy -> final occupancy after evolution
    print("Generating dataset...")
    X, Y = create_dataset(num_samples=2000, dt=0.01, steps=80, lambda_const=1.0, RT=1.0)
    # Shuffle and split into training and validation sets (80/20 split)
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    
    # Build the neural network model
    print("Building and training the neural network...")
    model = OxiNet(input_dim=8, hidden_dim=32, output_dim=8)
    train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3)
    
    # Evaluate on the validation set
    print("Evaluating on validation data...")
    pred_val, val_loss = evaluate_model(model, X_val, Y_val)
    print(f"Validation Loss: {val_loss:.6f}")
    
    # Display a few sample predictions
    idx_sample = np.random.choice(len(X_val), 3, replace=False)
    for idx in idx_sample:
        init_occ = X_val[idx]
        true_final = Y_val[idx]
        pred_final = pred_val[idx]
        print("\n--- Sample ---")
        print("Initial occupancy:", np.round(init_occ, 3))
        print("True final occupancy:", np.round(true_final, 3))
        print("Predicted final occupancy:", np.round(pred_final, 3))
