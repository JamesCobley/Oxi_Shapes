#!/usr/bin/env python
# coding: utf-8

#Installations
!pip install torch

#Imports
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation

#Binomial graph 
# R = 3 i-states
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

# Build graph
G = nx.Graph()
G.add_nodes_from(pf_states)
G.add_edges_from(allowed_edges)

# Define node positions for R = 3 redox manifold
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
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

#Geometric functions 

# ------------------------------
# 1. Cotangent Laplacian
# ------------------------------
def compute_cotangent_laplacian(node_xy, triangles):
    N = node_xy.shape[0]
    W = np.zeros((N, N))
    for tri in triangles:
        i, j, k = tri
        pts = node_xy[[i, j, k], :]
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[0]
        v2 = pts[2] - pts[1]
        angle_i = np.arccos(np.clip(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)), -1, 1))
        angle_j = np.arccos(np.clip(np.dot(-v0, v2) / (np.linalg.norm(v0) * np.linalg.norm(v2)), -1, 1))
        angle_k = np.arccos(np.clip(np.dot(-v1, -v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        cot_i = 1 / np.tan(angle_i) if np.tan(angle_i) != 0 else 0
        cot_j = 1 / np.tan(angle_j) if np.tan(angle_j) != 0 else 0
        cot_k = 1 / np.tan(angle_k) if np.tan(angle_k) != 0 else 0
        W[j, k] += cot_i
        W[k, j] += cot_i
        W[i, k] += cot_j
        W[k, i] += cot_j
        W[i, j] += cot_k
        W[j, i] += cot_k
    L_t = -W
    for i in range(N):
        L_t[i, i] = np.sum(W[i, :])
    return L_t

# ------------------------------
# 2. Compute c-Ricci
# ------------------------------
def compute_c_ricci(occupancy_dict, lambda_const=1.0):
    rho_vec = np.array([occupancy_dict[s] for s in pf_states])
    L_t = compute_cotangent_laplacian(node_xy, triangles)
    c_ricci_vec = lambda_const * (L_t @ rho_vec)
    return {pf_states[i]: c_ricci_vec[i] for i in range(len(pf_states))}

# ------------------------------
# 3. Compute Anisotropy Field
# ------------------------------
def compute_anisotropy_field(c_ricci_nodes, graph):
    A_field = {}
    for s in pf_states:
        neighbors = list(graph.neighbors(s))
        grad_sum = 0.0
        count = 0
        for nbr in neighbors:
            dist = np.linalg.norm(np.array(flat_pos[s]) - np.array(flat_pos[nbr]))
            if dist > 1e-6:
                grad_sum += abs(c_ricci_nodes[s] - c_ricci_nodes[nbr]) / dist
                count += 1
        A_field[s] = grad_sum / count if count > 0 else 0.0
    return A_field

# ------------------------------
# 4. Assign c-Ricci and Penalty to Edges
# ------------------------------
def assign_edge_curvature_and_penalty(graph, c_ricci_nodes, A_field):
    for (u, v) in graph.edges():
        graph[u][v]['cRicci'] = (c_ricci_nodes[u] + c_ricci_nodes[v]) / 2.0
        penalty = np.exp(- (A_field[u] + A_field[v]) / 2.0)
        graph[u][v]['penalty'] = penalty

class GeoFlowNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, state_dim=8):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.geodesic_layer = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, x, curvature, anisotropy):
        h = torch.relu(self.encoder(x))
        logits = self.geodesic_layer(h)
        
        # Penalty-modified logits
        penalty = self.curvature_penalty(curvature) + self.anisotropy_penalty(anisotropy)
        adjusted_logits = logits - penalty
        
        return torch.softmax(adjusted_logits, dim=-1)
    
    def curvature_penalty(self, curvature):
        return torch.tensor(curvature, dtype=torch.float32)

    def anisotropy_penalty(self, anisotropy):
        return torch.tensor(anisotropy, dtype=torch.float32)
