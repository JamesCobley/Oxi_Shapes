#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes: Pipeline Step 2 — Time Evolution Engine (Dynamic Fields Version)
Encodes the full thermo-geometric evolution model based on first principles.
This version:
  • Loads geometry, occupancy, Morse energy, and custom discrete C-Ricci from Pipeline Step 1.
  • Treats α, β, and γ as dynamic per-node fields that are updated every simulation step.
  • Uses these dynamic parameters to compute the free-energy difference (Δf) for allowed transitions
    (oxidation: k+1, reduction: k-1, or stay), ensuring that the probability simplex for each molecule sums to 1.
  • Enforces volume conservation by normalizing occupancy at each step.
Outputs: Time evolution of φ, ρ, redox state, entropy, k-bin history, Lyapunov exponent, Excel export.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import networkx as nx
import os
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay

###############################################
# 1. Load Geometry and Fields from Pipeline Step 1
###############################################
with open('oxishape_solution_full.pkl', 'rb') as f:
    oxi_shape_data = pickle.load(f)

pf_states = oxi_shape_data['states']
# For 2D triangulation, use the first two components of the 3D node positions.
node_positions_3D = oxi_shape_data['node_positions_3D']
flat_coords = np.array([(node_positions_3D[s][0], node_positions_3D[s][1]) for s in pf_states])
# Load the allowed transitions graph (with custom C-Ricci stored on edges)
G = oxi_shape_data['graph']
# Load the edge dictionary for custom C-Ricci (for hotstart use)
cRicci_edges = oxi_shape_data['cRicci']
morse_energy = oxi_shape_data['morse_energy']
occupancy = oxi_shape_data['occupancy']  # normalized occupancy (∑ρ = 1)

state_index = {s: i for i, s in enumerate(pf_states)}
index_state = {i: s for s, i in state_index.items()}

# Degeneracy map for entropy
degeneracy = {0: 1, 1: 3, 2: 3, 3: 1}
def count_ones(s):
    return s.count('1')

###############################################
# 2. Dynamic Parameter Function (Per Node)
###############################################
def dynamic_parameters(state):
    """
    Compute dynamic parameters (alpha, beta, gamma) for a given state.
    Example formulas (adjust as needed):
      alpha_dyn = alpha_base * (1 + occupancy[state])
      beta_dyn  = beta_base  * (1 + (morse_energy[state] / 5.0))
      gamma_dyn = gamma_base * (1 + |average_cRicci|)
    """
    alpha_base = 0.1
    beta_base  = 0.5
    gamma_base = 0.1
    occ = occupancy[state]
    energy = morse_energy[state]
    neighbors = list(G.neighbors(state))
    if neighbors:
        c_vals = [G[state][nbr]['cRicci'] for nbr in neighbors]
        c_avg = np.mean(c_vals)
    else:
        c_avg = 0.0
    alpha_dyn = alpha_base * (1 + occ)
    beta_dyn  = beta_base  * (1 + energy / 5.0)
    gamma_dyn = gamma_base * (1 + abs(c_avg))
    return alpha_dyn, beta_dyn, gamma_dyn

###############################################
# 3. FEM Assembly and φ Solver (Same as before)
###############################################
def fem_assemble_matrices(nodes, elements):
    num_nodes = nodes.shape[0]
    A_mat = lil_matrix((num_nodes, num_nodes))
    M_mat = lil_matrix((num_nodes, num_nodes))
    for elem in elements:
        idx = elem
        coords = nodes[idx]
        mat = np.array([[1, coords[0,0], coords[0,1]],
                        [1, coords[1,0], coords[1,1]],
                        [1, coords[2,0], coords[2,1]]])
        area = 0.5 * abs(np.linalg.det(mat))
        if area < 1e-14:
            continue
        x = coords[:,0]
        y = coords[:,1]
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        K_local = np.zeros((3,3))
        for i_local in range(3):
            for j_local in range(3):
                K_local[i_local,j_local] = (b[i_local]*b[j_local] + c[i_local]*c[j_local])/(4*area)
        M_local = (area/12.0)*np.array([[2,1,1],[1,2,1],[1,1,2]])
        for i_local, i_global in enumerate(idx):
            for j_local, j_global in enumerate(idx):
                A_mat[i_global,j_global] += K_local[i_local,j_local]
                M_mat[i_global,j_global] += M_local[i_local,j_local]
    return A_mat.tocsr(), M_mat.tocsr()

def solve_phi_and_ricci(rho, nodes, elements, alpha=1.0, gamma=0.0, phi0=None, damping=0.05, tol=1e-6, max_iter=500):
    A, M = fem_assemble_matrices(nodes, elements)
    num_nodes = len(rho)
    phi = np.zeros(num_nodes) if phi0 is None else phi0.copy()
    reg = 1e-8
    for _ in range(max_iter):
        nonlinear = 0.5 * rho * np.exp(2 * phi)
        F = A @ phi + M @ nonlinear
        J = A + M @ csr_matrix(np.diag(rho * np.exp(2 * phi))) + reg * sp.eye(num_nodes)
        delta_phi = spsolve(J, -F)
        phi += damping * delta_phi
        if np.linalg.norm(delta_phi) < tol:
            break
    M_lumped = np.array(M.sum(axis=1)).flatten()
    lap_phi = A @ phi / M_lumped
    Ricci = -2.0 * np.exp(-2 * phi) * lap_phi
    return phi, Ricci

###############################################
# 4. Lyapunov Exponent & Analysis Functions (Same as before)
###############################################
def compute_lyapunov(redox_series, dt=1.0):
    diffs = np.diff(redox_series)
    diffs[diffs == 0] = 1e-10
    log_diffs = np.log(np.abs(diffs)).reshape(-1, 1)
    t = np.arange(1, len(redox_series)).reshape(-1, 1)
    model = LinearRegression().fit(t, log_diffs)
    return model.coef_[0][0] / dt

def analyze_outputs(results):
    redox_series = np.array(results['redox_t'])
    lyap = compute_lyapunov(redox_series)
    print("Lyapunov Exponent:", lyap)
    plt.figure(figsize=(8,5))
    plt.plot(redox_series, 'o-')
    plt.xlabel("Time Steps")
    plt.ylabel("Global Redox State (%)")
    plt.title("Redox Evolution")
    plt.grid(True)
    plt.savefig("global_redox_evolution.png", dpi=300)
    plt.show()

###############################################
# 5. Hot-Start Function for φ from Pipeline Step 1 (Same as before)
###############################################
def load_phi0_from_step1(filename='oxishape_solution_full.pkl', pf_states=pf_states, num_nodes=None):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            sol = pickle.load(f)
        print('Loaded φ₀ from Step 1 solution.')
        phi_dict = sol.get('phi', {})
        if num_nodes is None:
            num_nodes = len(pf_states)
        phi0 = np.array([phi_dict[s] for s in pf_states[:num_nodes]])
        return phi0
    print('[Info] No Step 1 φ₀ found. Using zeros.')
    if num_nodes is None:
        num_nodes = len(pf_states)
    return np.zeros(num_nodes)

###############################################
# 6. Main Evolution Engine for Oxi-Shapes (Dynamic Fields Version)
###############################################
def simulate_forward(alpha, beta, rho_init, steps=240, total_molecules=100, gamma=0.0, use_hotstart=True):
    """
    At each step, for each state, dynamic parameters (alpha, beta, gamma) are derived using dynamic_parameters.
    Transitions (oxidation, reduction, stay) are computed from free-energy differences that incorporate these parameters.
    The occupancy is updated and normalized (volume conservation), and φ is recalculated.
    """
    # Initial occupancy vector
    rho_vec = np.array([rho_init.get(s, 0.0) for s in pf_states])
    # Use flat 2D coordinates from node_positions_3D (first two components)
    node_coords_local = np.array([(node_positions_3D[s][0], node_positions_3D[s][1]) for s in pf_states])
    elements = Delaunay(node_coords_local).simplices
    num_nodes = len(pf_states)
    
    # Hotstart for φ
    phi0 = load_phi0_from_step1(num_nodes=num_nodes) if use_hotstart else None
    phi, _ = solve_phi_and_ricci(rho_vec, node_coords_local, elements, alpha=alpha, gamma=gamma, phi0=phi0)
    
    # Initialize heat vector and histories
    heat_vec = np.zeros(num_nodes)
    history_rho = [rho_vec.copy()]
    redox_init = np.dot([count_ones(s)/3 for s in pf_states], rho_vec) * 100
    history_redox = [redox_init]
    history_entropy = [-np.sum([p*np.log2(p) for p in rho_vec if p > 0])]
    
    # Main time evolution loop
    for t in range(steps):
        new_rho = np.zeros_like(rho_vec)
        for i, state in enumerate(pf_states):
            # Derive dynamic parameters for the current state
            alpha_dyn, beta_dyn, gamma_dyn = dynamic_parameters(state)
            current_k = count_ones(state)
            transitions = []
            # Option: stay in the same state (free-energy difference = 0)
            transitions.append((state, "stay", 0.0))
            # Allowed transitions: oxidation (k+1) or reduction (k-1)
            for neighbor in list(G.neighbors(state)):
                neighbor_k = count_ones(neighbor)
                if neighbor_k == current_k + 1:
                    move_type = "oxidation"
                elif neighbor_k == current_k - 1:
                    move_type = "reduction"
                else:
                    continue  # Skip barred transitions
                j = state_index[neighbor]
                delta_phi = phi[j] - phi[i]
                q = -delta_phi
                # Optionally accumulate heat (here we add q to neighbor's heat)
                heat_vec[j] += q
                # Entropy term based on degeneracy
                g_i = degeneracy[current_k]
                g_j = degeneracy[neighbor_k]
                entropy_term = np.log(g_j / g_i) if g_i > 0 else 0.0
                # Get custom cRicci from Pipeline Step 1 for this edge
                cRicci_term = cRicci_edges.get((state, neighbor), cRicci_edges.get((neighbor, state), 0.0))
                RT = 0.001987 * 310.15
                delta_f = (delta_phi * np.exp(alpha_dyn * rho_vec[i])
                           - beta_dyn * cRicci_term
                           + entropy_term
                           + gamma_dyn * heat_vec[i])
                transitions.append((neighbor, move_type, delta_f))
            # Compute Boltzmann weights for all transitions from this state
            weights = np.array([np.exp(-tf/RT) for (_, _, tf) in transitions])
            if np.sum(weights) < 1e-12:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights /= np.sum(weights)
            # Redistribute occupancy from state i according to the computed probabilities
            for (target, mtype, tf), w in zip(transitions, weights):
                j = state_index[target]
                new_rho[j] += rho_vec[i] * w
        
        total_occ = np.sum(new_rho)
        if total_occ < 1e-12:
            print(f"[Warning] Occupancy vanished at step {t}. Breaking.")
            break
        # Normalize occupancy to conserve volume
        rho_vec = new_rho / total_occ
        
        try:
            phi, _ = solve_phi_and_ricci(rho_vec, node_coords_local, elements, alpha=alpha, gamma=gamma, phi0=phi)
        except Exception as e:
            print(f"[Warning] φ solver failed at step {t} with error: {e}. Using last φ.")
        
        history_rho.append(rho_vec.copy())
        redox = np.dot([count_ones(s)/3 for s in pf_states], rho_vec) * 100
        history_redox.append(redox)
        entropy = -np.sum([p*np.log2(p) for p in rho_vec if p > 0])
        history_entropy.append(entropy)
    
    return {
        'rho_t': history_rho,
        'redox_t': history_redox,
        'entropy_t': history_entropy,
        'final_phi': phi,
        'final_heat': heat_vec,
        'coords': flat_coords,  # 2D coordinates for visualization
        'pf_states': pf_states
    }

###############################################
# 7. Save Simulation Results Function (Same as before)
###############################################
def save_simulation_results(results, filename_prefix='oxi_step2_result'):
    with open(f'{filename_prefix}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Simulation saved as {filename_prefix}.pkl")
    df_k = pd.DataFrame([
        {f'k={k}': sum(r[i] for i, s in enumerate(results['pf_states']) if count_ones(s)==k)
         for k in range(4)}
        for r in results['rho_t']
    ])
    df_k.index.name = 'Time_Step'
    df_k.to_excel(f'{filename_prefix}_k_history.xlsx')
    print(f'k-bin time series exported to {filename_prefix}_k_history.xlsx')

###############################################
# 8. Main Script Execution
###############################################
if __name__ == '__main__':
    # Use initial occupancy from Pipeline Step 1 (should already be normalized)
    rho_start = {s: occupancy.get(s, 0.0) for s in pf_states}
    
    sim_result = simulate_forward(alpha=0.1, beta=0.5, rho_init=rho_start, steps=240, gamma=0.0, use_hotstart=True)
    save_simulation_results(sim_result, filename_prefix='oxi_step2_result')
    
    redox_series = np.array(sim_result['redox_t'])
    if np.any(np.isnan(redox_series)):
        print("[Error] Redox series contains NaN values. Simulation may be unstable.")
    else:
        analyze_outputs(sim_result)
    
    final_rho = sim_result['rho_t'][-1]
    molecule_counts = {s: int(round(r * 100)) for s, r in zip(sim_result['pf_states'], final_rho)}
    k_bin_counts = {k: 0 for k in range(4)}
    for s, count in molecule_counts.items():
        k_bin_counts[count_ones(s)] += count
    total_molecules = sum(molecule_counts.values())
    print('\nFinal State Summary:')
    print(f'Total Molecules: {total_molecules}')
    print('Molecules per k-bin:')
    for k in sorted(k_bin_counts):
        print(f'  k={k}: {k_bin_counts[k]}')
    print('Molecules per i-state:')
    for s in sim_result['pf_states']:
        print(f'  {s}: {molecule_counts[s]}')
    
    print(f'Shannon Entropy (final): {sim_result["entropy_t"][-1]:.4f}')
    lyap = compute_lyapunov(sim_result["redox_t"])
    print(f'Lyapunov Exponent: {lyap:.6f}')
    
    fisher_metric = np.sum((np.diff(sim_result['rho_t'], axis=0)**2))
    print(f'Fisher Information Metric (trajectory): {fisher_metric:.6f}')
