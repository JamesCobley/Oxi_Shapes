#!/usr/bin/env python
# coding: utf-8

"""
Oxi-Shapes: Pipeline Step 2 — Time Evolution Engine
Encodes the full thermo-geometric evolution model
Inputs: alpha, beta, rho_init
Outputs: phi(x,t), R(x,t), Q(x,t), rho(x,t), k-bin evolution, entropy, redox signal, Lyapunov exponent, Excel export
"""

# Core imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay
import os

###############################################
# 1. Define R=3 i-State Space and Graph Topology
###############################################
pf_states = ['000', '001', '010', '011', '100', '101', '110', '111']
state_index = {s: i for i, s in enumerate(pf_states)}
index_state = {i: s for s, i in state_index.items()}

def count_ones(s):
    return s.count('1')

# Define allowed transitions (Hamming-1)
def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

G = nx.Graph()
G.add_nodes_from(pf_states)
for i in range(len(pf_states)):
    for j in range(i + 1, len(pf_states)):
        if hamming_distance(pf_states[i], pf_states[j]) == 1:
            G.add_edge(pf_states[i], pf_states[j])

# Degeneracy for each k-value
degeneracy = {0: 1, 1: 3, 2: 3, 3: 1}

###############################################
# 2. Geometry: Discrete Layout for R=3 State Space
###############################################
coords_dict = {
    '000': (0.0,  1.0),
    '001': (-1.0, 0.0),
    '010': (0.0,  0.0),
    '100': (1.0,  0.0),
    '011': (-1.0, -1.0),
    '101': (0.0,  -1.0),
    '110': (1.0,  -1.0),
    '111': (0.0,  -2.0)
}
node_coords = np.array([coords_dict[s] for s in pf_states])
triangulation = Delaunay(node_coords)

###############################################
# 3. FEM Assembly for Field Equation (φ)
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
                K_local[i_local, j_local] = (b[i_local]*b[j_local] + c[i_local]*c[j_local])/(4*area)
        M_local = (area/12.0)*np.array([[2,1,1],
                                        [1,2,1],
                                        [1,1,2]])
        for i_local, i_global in enumerate(idx):
            for j_local, j_global in enumerate(idx):
                A_mat[i_global, j_global] += K_local[i_local, j_local]
                M_mat[i_global, j_global] += M_local[i_local, j_local]
    return A_mat.tocsr(), M_mat.tocsr()

###############################################
# 4. Scalar Field φ Solver with Ricci Curvature
###############################################
def solve_phi_and_ricci(rho, nodes, elements,
                        alpha=1.0, gamma=0.0, phi0=None,
                        damping=0.05, tol=1e-6, max_iter=500):
    A, M = fem_assemble_matrices(nodes, elements)
    num_nodes = len(rho)
    phi = np.zeros(num_nodes) if phi0 is None else phi0.copy()
    for _ in range(max_iter):
        nonlinear = 0.5 * rho * np.exp(2 * phi)
        F = A @ phi + M @ nonlinear
        J = A + M @ csr_matrix(np.diag(rho * np.exp(2 * phi)))
        delta_phi = spsolve(J, -F)
        phi += damping * delta_phi
        if np.linalg.norm(delta_phi) < tol:
            break
    M_lumped = np.array(M.sum(axis=1)).flatten()
    lap_phi = A @ phi / M_lumped
    Ricci = -2.0 * np.exp(-2 * phi) * lap_phi
    return phi, Ricci

###############################################
# 5. Save Simulation Results
###############################################
def save_simulation_results(results, filename_prefix='oxi_step2_result'):
    with open(f'{filename_prefix}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f'Simulation saved as {filename_prefix}.pkl')

    # Example k-bin time series
    df_k = pd.DataFrame([
        {f'k={k}': sum(r[i] for i, s in enumerate(results['pf_states']) if count_ones(s) == k)
         for k in range(4)}
        for r in results['rho_t']
    ])
    df_k.index.name = 'Time_Step'
    df_k.to_excel(f'{filename_prefix}_k_history.xlsx')
    print(f'k-bin time series exported to {filename_prefix}_k_history.xlsx')

###############################################
# 6. Lyapunov Exponent & Additional Analysis
###############################################
def compute_lyapunov(redox_series, dt=1.0):
    diffs = np.diff(redox_series)
    diffs[diffs == 0] = 1e-10
    log_diffs = np.log(np.abs(diffs)).reshape(-1, 1)
    t = np.arange(1, len(redox_series)).reshape(-1, 1)
    model = LinearRegression().fit(t, log_diffs)
    return model.coef_[0][0] / dt

def analyze_outputs(results):
    rho_t = results['rho_t']
    pf_states_local = results['pf_states']
    redox_series = np.array(results['redox_t'])
    final_rho = rho_t[-1]

    # k-bin distribution
    k_bin = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for i, s in enumerate(pf_states_local):
        k_bin[count_ones(s)] += final_rho[i]

    print("Final k-Bin Distribution:")
    for k in sorted(k_bin):
        print(f"k={k}: {k_bin[k]*100:.2f}%")

    # Lyapunov exponent
    lyap = compute_lyapunov(redox_series)
    print("Lyapunov Exponent:", lyap)

    # Plot redox evolution
    plt.figure(figsize=(8,5))
    plt.plot(redox_series, 'o-')
    plt.xlabel("Time Steps")
    plt.ylabel("Global Redox State (%)")
    plt.title("Redox Evolution")
    plt.grid(True)
    plt.savefig("global_redox_evolution.png", dpi=300)
    plt.show()

###############################################
# 7. Optional: Hot-start φ₀ from Pipeline Step 1
###############################################
def load_phi0_from_step1(filename='oxishape_solution_full.pkl', pf_states=pf_states, num_nodes=None):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            sol = pickle.load(f)
        print('Loaded φ₀ from Step 1 solution.')
        phi_dict = sol.get('phi', {})
        # If num_nodes isn't provided, use all states.
        if num_nodes is None:
            num_nodes = len(pf_states)
        # Build an array in the order of pf_states (or a subset if num_nodes is given)
        phi0 = np.array([phi_dict[s] for s in pf_states[:num_nodes]])
        return phi0
    print('[Info] No Step 1 φ₀ found. Using zeros.')
    if num_nodes is None:
        num_nodes = len(pf_states)
    return np.zeros(num_nodes)


###############################################
# 8. Main Evolution Engine for Oxi-Shapes
###############################################
def simulate_forward(alpha, beta, rho_init,
                     steps=240, total_molecules=100, gamma=0.0,
                     use_hotstart=True):
    # Convert dict occupancy to array
    rho_vec = np.array([rho_init.get(s, 0.0) for s in pf_states])
    node_coords_local = np.array([coords_dict[s] for s in pf_states])
    elements = Delaunay(node_coords_local).simplices
    num_nodes = len(pf_states)

    # Hotstart for φ
    phi0 = load_phi0_from_step1(num_nodes=num_nodes) if use_hotstart else None
    phi, R_vals = solve_phi_and_ricci(rho_vec, node_coords_local,
                                      elements, alpha=alpha, gamma=gamma,
                                      phi0=phi0)

    heat_vec = np.zeros(num_nodes)
    history_rho = [rho_vec.copy()]
    history_redox = [np.dot([count_ones(s)/3 for s in pf_states], rho_vec) * 100]
    history_entropy = [-np.sum([p*np.log2(p) for p in rho_vec if p > 0])]

    # Time evolution
    for t in range(steps):
        new_rho = np.zeros_like(rho_vec)
        for i, state in enumerate(pf_states):
            neighbors = list(G.neighbors(state))
            dE = []
            for neighbor in neighbors:
                j = state_index[neighbor]
                delta_phi = phi[j] - phi[i]
                q = -delta_phi
                heat_vec[j] += q
                g_i = degeneracy[count_ones(state)]
                g_j = degeneracy[count_ones(neighbor)]
                entropy_term = np.log(g_j / g_i) if g_i > 0 else 0.0

                # Example free-energy difference
                delta_f = (delta_phi * np.exp(alpha * rho_vec[i])
                           - beta * R_vals[i]
                           + entropy_term
                           + gamma * heat_vec[i])
                dE.append(np.exp(-delta_f / (0.001987 * 310.15)))

            if len(dE) > 0:
                dE = np.array(dE)
                dE /= np.sum(dE)
                for idx, neighbor in enumerate(neighbors):
                    j = state_index[neighbor]
                    new_rho[j] += rho_vec[i] * dE[idx]
            else:
                # No neighbors
                new_rho[i] += rho_vec[i]

        # Normalize occupancy
        total_occ = np.sum(new_rho)
        if total_occ < 1e-12:
            print(f"[Warning] Occupancy vanished at step {t}. Breaking.")
            break
        rho_vec = new_rho / total_occ

        # Recompute φ, R
        try:
            phi, R_vals = solve_phi_and_ricci(rho_vec, node_coords_local,
                                              elements, alpha=alpha,
                                              gamma=gamma, phi0=phi)
        except:
            print(f"[Warning] Ricci solver failed at step {t}. Using last φ.")

        # Track time series
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
        'final_ricci': R_vals,
        'final_heat': heat_vec,
        'coords': coords_dict,
        'pf_states': pf_states
    }

###############################################
# 9. Main Script Execution
###############################################
if __name__ == '__main__':
    # Example initial distribution
    rho_start = {
        '000': 0.25,
        '001': 0.25,
        '010': 0.25,
        '100': 0.25,
        '011': 0.0,
        '101': 0.0,
        '110': 0.0,
        '111': 0.0
    }

    # Run the simulation
    sim_result = simulate_forward(alpha=0.1, beta=0.5, rho_init=rho_start)
    save_simulation_results(sim_result, filename_prefix='oxi_step2_result')

    # Analyze outputs
    analyze_outputs(sim_result)

    # Extended result printout
    final_rho = sim_result['rho_t'][-1]
    molecule_counts = {s: int(round(r * 100)) for s, r in zip(sim_result['pf_states'], final_rho)}
    k_bin_counts = {k: 0 for k in range(4)}
    for s, count_ in molecule_counts.items():
        k = count_ones(s)
        k_bin_counts[k] += count_

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

    # Example "Fisher Information" measure on the occupancy trajectory
    # (just a placeholder example)
    fisher_metric = np.sum((np.diff(sim_result['rho_t'], axis=0)**2))
    print(f'Fisher Information Metric (trajectory): {fisher_metric:.6f}')
