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
#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes + Supervised ML Pipeline with Persistent Homology (Betti Numbers)

This script combines the Oxi‑Shapes geometric evolution with a supervised machine
learning model and persistent homology analysis.
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
# 2. Neural ODE System for Ricci-Driven Shape Evolution
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
                delta_f = delta_E + 0.5*S_term - A_field[j]
                p_ij = torch.exp(-delta_f / self.RT)
                inflow[j] += rho[i] * p_ij
                outflow[i] += rho[i] * p_ij
        return inflow - outflow

###############################################################################
# 3. Betti Number Tracker (Persistent Homology)
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
# 4. Data Generation
###############################################################################
def random_rho_init(num_samples=1000):
    data = []
    for _ in range(num_samples):
        vec = np.random.rand(8)
        vec /= vec.sum()
        rho_dict = {s: vec[i] for i, s in enumerate(pf_states)}
        data.append(rho_dict)
    return data

def create_dataset_ODE(num_samples=1000, t_span=torch.linspace(0.0, 1.0, 80, dtype=torch.float32)):
    initial_rhos = random_rho_init(num_samples)
    X, Y = [], []
    ode_model = OxiShapeODE(node_xy=node_xy, triangles=triangles, G=G)
    for rho_init_dict in initial_rhos:
        rho_0 = torch.tensor([rho_init_dict[s] for s in pf_states], dtype=torch.float32)
        sol = odeint(ode_model, rho_0, t_span, method='dopri5')
        rho_final = sol[-1]
        X.append(rho_0.detach().numpy())
        Y.append(rho_final.detach().numpy())
    return np.array(X), np.array(Y)

###############################################################################
# 5. Neural Network for Learning
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
# 6. Train and Evaluate
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
            print(f"Epoch {epoch}: Train Loss={loss.item():.6f}, Val Loss={val_loss.item():.6f}")

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
# 7. Main Execution
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

    model = OxiNet(input_dim=8, hidden_dim=32, output_dim=8)
    train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=1e-3)

    print("Evaluating on validation data...")
    pred_val, val_loss = evaluate_model(model, X_val, Y_val)
    print(f"Validation Loss: {val_loss:.6f}")

    print("Tracking Betti number evolution in a few samples...")
    for idx in np.random.choice(len(X_val), 3, replace=False):
        init_occ = X_val[idx]
        print("\n--- Sample ---")
        print("Initial occupancy:", np.round(init_occ, 3))
        print("True final occupancy:", np.round(Y_val[idx], 3))
        print("Predicted final occupancy:", np.round(pred_val[idx], 3))

        betti = extract_betti_numbers(pred_val[idx], threshold=0.05)
        print(f"Betti Numbers → β₀: {betti['beta0']}  |  β₁: {betti['beta1']}")
