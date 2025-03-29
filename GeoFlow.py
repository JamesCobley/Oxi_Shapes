#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes + Supervised ML
--------------------------
1) We define a discrete ODE-like evolution of the occupancy distribution ρ(x)
   using the Einstein-like Oxi-Shapes field equation.
2) We generate a dataset by sampling multiple random initial distributions,
   evolving them, and recording (initial -> final) occupancy distributions.
3) We train a neural network in PyTorch to learn the mapping from initial
   occupancy to final occupancy, effectively "solving" the geometry problem
   via supervised ML.

Note: This code is minimal and can be extended in many ways, such as:
  - Storing intermediate geometry (c-Ricci) or anisotropy to feed as features.
  - Using more sophisticated ODE solvers.
  - Incorporating physics-informed loss terms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation

###############################################################################
# 1. Define the R=3 State Space (Binomial Diamond) & Graph
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

# 2D positions for building the triangulation
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

###############################################################################
# 2. Geometry: c-Ricci (cotangent Laplacian) & Anisotropy
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
            W[a,b] += cval
            W[b,a] += cval
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
# 3. Free Energy & ODE-like Update
###############################################################################
def compute_free_energy_diff(i, j, rho_vec, c_ricci_vec, anisotropy_vec, RT=1.0):
    # Simple example: Δf = Δ(c-Ricci) + 0.5 * Δ(log occupancy) - anisotropy penalty
    occ_i = rho_vec[i]
    occ_j = rho_vec[j]
    delta_E = c_ricci_vec[j] - c_ricci_vec[i]
    entropy_term = 0
    if occ_i>1e-12 and occ_j>1e-12:
        entropy_term = np.log(occ_j) - np.log(occ_i)
    delta_f = delta_E + 0.5*entropy_term - anisotropy_vec[j]
    return delta_f / RT

def evolve_oxi_shapes(rho_init, dt=0.01, steps=50, lambda_const=1.0, RT=1.0):
    pf_list = list(rho_init.keys())
    N = len(pf_list)
    idx_map = {s: i for i, s in enumerate(pf_list)}
    rho_vec = np.array([rho_init[s] for s in pf_list], dtype=float)
    rho_vec /= rho_vec.sum()  # ensure normalization

    for t in range(steps):
        c_ricci_vec = compute_c_ricci(rho_vec, lambda_const=lambda_const)
        A_field = compute_anisotropy(c_ricci_vec)
        inflow = np.zeros(N)
        outflow = np.zeros(N)
        for i, s_i in enumerate(pf_list):
            nbrs = list(G.neighbors(s_i))
            for nbr in nbrs:
                j = idx_map[nbr]
                delta_f = compute_free_energy_diff(i, j, rho_vec, c_ricci_vec, A_field, RT=RT)
                p_ij = np.exp(-delta_f)
                inflow[j] += rho_vec[i]*p_ij
                outflow[i] += rho_vec[i]*p_ij
        new_rho = rho_vec + dt*(inflow - outflow)
        new_rho = np.maximum(new_rho, 0.0)
        ssum = new_rho.sum()
        if ssum>1e-12:
            new_rho /= ssum
        rho_vec = new_rho
    # Return final distribution
    final_rho = {pf_list[i]: rho_vec[i] for i in range(N)}
    return final_rho

###############################################################################
# 4. Data Generation for Supervised Learning
###############################################################################
def random_rho_init(num_samples=1000):
    """
    Generate random occupancy distributions for the 8 states,
    ensuring sum(rho)=1.
    """
    data = []
    for _ in range(num_samples):
        vec = np.random.rand(8)
        vec /= vec.sum()
        rho_dict = {}
        for i, s in enumerate(pf_states):
            rho_dict[s] = vec[i]
        data.append(rho_dict)
    return data

def create_dataset(num_samples=1000, dt=0.01, steps=50, lambda_const=1.0, RT=1.0):
    """
    Creates a dataset of (initial occupancy) -> (final occupancy) pairs.
    Returns X, Y as numpy arrays of shape (num_samples, 8).
    """
    initial_rhos = random_rho_init(num_samples)
    X = []
    Y = []
    for rho_init in initial_rhos:
        final_rho = evolve_oxi_shapes(rho_init, dt=dt, steps=steps,
                                      lambda_const=lambda_const, RT=RT)
        # Convert to vectors
        init_vec = np.array([rho_init[s] for s in pf_states], dtype=float)
        final_vec = np.array([final_rho[s] for s in pf_states], dtype=float)
        X.append(init_vec)
        Y.append(final_vec)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

###############################################################################
# 5. Define a Neural Network in PyTorch
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
        # We can ensure sum=1 by softmax if we want a distribution
        # But let's just let it output any real vector and we can renormalize
        return x

###############################################################################
# 6. Training & Testing the Model
###############################################################################
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # or L1Loss, etc.

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t   = torch.tensor(Y_val, dtype=torch.float32)

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        # Optionally, renormalize output to ensure sum=1
        # For example:
        # pred = torch.softmax(pred, dim=1)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = criterion(pred_val, Y_val_t)
        if epoch%10==0:
            print(f"Epoch {epoch}/{epochs}, Loss={loss.item():.4f}, ValLoss={val_loss.item():.4f}")

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
    # 7a. Generate data
    print("Generating dataset...")
    X, Y = create_dataset(num_samples=2000, dt=0.01, steps=80, lambda_const=1.0, RT=1.0)
    # Shuffle & split
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]
    split = int(0.8*len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val,   Y_val   = X[split:], Y[split:]

    # 7b. Build & Train Model
    print("Building & training neural network...")
    model = OxiNet(input_dim=8, hidden_dim=32, output_dim=8)
    train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3)

    # 7c. Evaluate
    print("Evaluating on validation set...")
    pred_val, val_loss = evaluate_model(model, X_val, Y_val)
    print(f"Validation Loss: {val_loss:.6f}")

    # 7d. Inspect Some Predictions
    idx_sample = np.random.choice(len(X_val), 3, replace=False)
    for idx in idx_sample:
        init_occ = X_val[idx]
        true_final = Y_val[idx]
        pred_final = pred_val[idx]
        print("\n--- Sample ---")
        print("Initial occupancy:", init_occ.round(3))
        print("True final occupancy:", true_final.round(3))
        print("Predicted final occupancy:", pred_final.round(3))
        # Optionally renormalize predicted final occupancy
        ssum = pred_final.sum()
        if ssum>1e-12:
            pred_renorm = pred_final/ssum
        else:
            pred_renorm = pred_final
        print("Pred final (renorm):", pred_renorm.round(3))
