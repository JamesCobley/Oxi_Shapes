#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes: Pipeline Step 2 — Time Evolution Engine (Updated)
This version derives dynamic alpha, beta, gamma for each coordinate (state) based on its starting point,
and uses these to compute per-molecule transition probabilities (P oxidation, P reduction, P stay)
that sum to 1 for each molecule.
Outputs: φ(x,t), updated ρ(x,t), redox signal, k-bin evolution, entropy, Lyapunov exponent, Excel export.
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

pf_states   = oxi_shape_data['states']
# For 2D triangulation, we use the first two coordinates of the 3D node positions.
node_positions_3D = oxi_shape_data['node_positions_3D']
flat_coords = np.array([ (node_positions_3D[s][0], node_positions_3D[s][1]) for s in pf_states ])
G = oxi_shape_data['graph']      # allowed transitions graph with custom cRicci stored in edges
cRicci_edges = oxi_shape_data['cRicci']  # Dictionary {(u,v): cRicci value}
morse_energy = oxi_shape_data['morse_energy']
occupancy = oxi_shape_data['occupancy']   # normalized occupancy (sum = 1)

state_index = {s: i for i, s in enumerate(pf_states)}
index_state = {i: s for s, i in state_index.items()}

# Degeneracy map remains the same
degeneracy = {0: 1, 1: 3, 2: 3, 3: 1}
def count_ones(s):
    return s.count('1')

###############################################
# 2. Function to Derive Dynamic Parameters per State
###############################################
def dynamic_parameters(state):
    """
    Derive dynamic parameters (alpha, beta, gamma) for a given state based on its starting values.
    For example, you might let:
      - alpha_dyn = alpha_base * (1 + occupancy(state))
      - beta_dyn  = beta_base  * (1 + (morse_energy[state] / 5.0))   # 5.0 is a reference energy
      - gamma_dyn = gamma_base * (1 + cRicci_value)  (or simply gamma_base)
    Adjust the formulas as needed.
    """
    # Base values (could be derived from calibration)
    alpha_base = 1.0
    beta_base  = 0.5
    gamma_base = 0.8
    occ = occupancy[state]
    energy = morse_energy[state]
    # For cRicci, average over edges from state
    neighbor_cricci = [G[state][nbr]['cRicci'] for nbr in G.neighbors(state)]
    if neighbor_cricci:
        c_avg = np.mean(neighbor_cricci)
    else:
        c_avg = 0.0
    # Example dynamic update:
    alpha_dyn = alpha_base * (1 + occ)
    beta_dyn  = beta_base  * (1 + energy/5.0)
    gamma_dyn = gamma_base * (1 + abs(c_avg))
    return alpha_dyn, beta_dyn, gamma_dyn

###############################################
# 3. Modified Simulation Loop: Compute Transition Probabilities per State
###############################################
def simulate_forward(alpha, beta, rho_init, steps=240, total_molecules=100, gamma=0.0, use_hotstart=True):
    """
    For each state, dynamically derive alpha, beta, gamma.
    Then compute transition probabilities for oxidation (k+1), reduction (k-1), and stay.
    Each molecule's probability simplex sums to 1.
    """
    # Initial occupancy from input (should be same as loaded occupancy)
    rho_vec = np.array([rho_init.get(s, 0.0) for s in pf_states])
    # Use flat 2D coordinates from node_positions_3D (first two components)
    node_coords_local = np.array([ (node_positions_3D[s][0], node_positions_3D[s][1]) for s in pf_states ])
    elements = Delaunay(node_coords_local).simplices
    num_nodes = len(pf_states)
    
    # Hotstart for φ if desired
    phi0 = load_phi0_from_step1(num_nodes=num_nodes) if use_hotstart else None
    phi, _ = solve_phi_and_ricci(rho_vec, node_coords_local, elements, alpha=alpha, gamma=gamma, phi0=phi0)
    
    # Initialize heat vector and history arrays
    heat_vec = np.zeros(num_nodes)
    history_rho   = [rho_vec.copy()]
    history_redox = [np.dot([count_ones(s)/3 for s in pf_states], rho_vec) * 100]
    history_entropy = [-np.sum([p*np.log2(p) for p in rho_vec if p>0])]
    
    # For each step, update occupancy based on transitions computed per state
    for t in range(steps):
        new_rho = np.zeros_like(rho_vec)
        for i, state in enumerate(pf_states):
            # Get dynamic parameters for this state
            alpha_dyn, beta_dyn, gamma_dyn = dynamic_parameters(state)
            current_k = count_ones(state)
            # Define three potential moves: stay, oxidation (k+1), reduction (k-1)
            transitions = []
            # Stay option: no change in state => delta_f = 0
            transitions.append((state, "stay", 0.0))
            # Allowed neighbor moves:
            for neighbor in list(G.neighbors(state)):
                neighbor_k = count_ones(neighbor)
                if neighbor_k == current_k + 1:
                    move_type = "oxidation"
                elif neighbor_k == current_k - 1:
                    move_type = "reduction"
                else:
                    # If neighbor is not exactly one level different, skip (barred transition)
                    continue
                j = state_index[neighbor]
                # Compute delta_phi (difference in φ)
                delta_phi = phi[j] - phi[i]
                # Compute a heat term: for simplicity, use -delta_phi
                q = -delta_phi
                # Update heat vector for neighbor (this could be accumulated over transitions)
                # (Here, we do not accumulate over states to avoid double counting)
                # Compute an entropy term based on degeneracy difference:
                g_i = degeneracy[current_k]
                g_j = degeneracy[neighbor_k]
                entropy_term = np.log(g_j / g_i) if g_i>0 else 0.0
                # Get the custom cRicci for this edge (from Pipeline Step 1)
                cRicci_term = cRicci_edges.get((state, neighbor), cRicci_edges.get((neighbor, state), 0.0))
                # Compute free-energy difference delta_f using dynamic parameters:
                # Note: The functional form below is illustrative.
                RT = 0.001987 * 310.15
                delta_f = (delta_phi * np.exp(alpha_dyn * rho_vec[i])
                           - beta_dyn * cRicci_term
                           + entropy_term
                           + gamma_dyn * heat_vec[i])
                transitions.append((neighbor, move_type, delta_f))
            # Now compute Boltzmann weights for all transitions (including stay)
            # We require that the probabilities sum to 1 for the state
            weights = np.array([np.exp(-tf/RT) for (_, _, tf) in transitions])
            if np.sum(weights) < 1e-12:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights /= np.sum(weights)
            # Distribute occupancy from state i according to these probabilities:
            for (target, mtype, tf), w in zip(transitions, weights):
                j = state_index[target]
                new_rho[j] += rho_vec[i] * w
        
        # Enforce volume conservation by normalizing new_rho so that sum(new_rho)=1
        total_occ = np.sum(new_rho)
        if total_occ < 1e-12:
            print(f"[Warning] Occupancy vanished at step {t}. Breaking.")
            break
        rho_vec = new_rho / total_occ
        
        # Optionally update φ by re-solving PDE (if desired to capture new geometry)
        try:
            phi, _ = solve_phi_and_ricci(rho_vec, node_coords_local, elements, alpha=alpha, gamma=gamma, phi0=phi)
        except Exception as e:
            print(f"[Warning] φ solver failed at step {t} with error: {e}. Using last φ.")
        
        history_rho.append(rho_vec.copy())
        redox = np.dot([count_ones(s)/3 for s in pf_states], rho_vec) * 100
        history_redox.append(redox)
        entropy = -np.sum([p*np.log2(p) for p in rho_vec if p>0])
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
# 5. Hot-Start Function for φ from Pipeline Step 1
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
# 6. Save Simulation Results
###############################################
def save_simulation_results(results, filename_prefix='oxi_step2_result'):
    with open(f'{filename_prefix}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Simulation saved as {filename_prefix}.pkl")
    df_k = pd.DataFrame([
        {f'k={k}': sum(r[i] for i, s in enumerate(results['pf_states']) if count_ones(s) == k)
         for k in range(4)}
        for r in results['rho_t']
    ])
    df_k.index.name = 'Time_Step'
    df_k.to_excel(f'{filename_prefix}_k_history.xlsx')
    print(f'k-bin time series exported to {filename_prefix}_k_history.xlsx')

###############################################
# 7. Lyapunov Exponent & Analysis Function
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
# 8. Main Script Execution
###############################################
if __name__ == '__main__':
    # Use initial occupancy from Pipeline Step 1
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
