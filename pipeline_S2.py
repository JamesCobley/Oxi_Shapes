#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
import pandas as pd
import networkx as nx
import pickle
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

###############################################
# Load continuous PDE solution from pipeline step 1
###############################################
with open("pde_solution.pkl", "rb") as f:
    sol = pickle.load(f)
phi_cont = sol["phi"]
R_cont = sol["R_vals"]
occupancy_cont = sol["occ_vector"]

###############################################
# 1. Define Discrete i-States for R=3 (8 states)
###############################################
pf_states = ["000", "001", "010", "011", "100", "101", "110", "111"]

def count_ones(s):
    return s.count('1')

# Define coordinates for each discrete state (reflecting the k–manifold ordering)
coords_dict = {
    "000": (0.0,  1.0),   # k = 0
    "001": (-1.0, 0.0),   # k = 1
    "010": (0.0,  0.0),   # k = 1
    "100": (1.0,  0.0),   # k = 1
    "011": (-1.0, -1.0),  # k = 2
    "101": (0.0,  -1.0),  # k = 2
    "110": (1.0,  -1.0),  # k = 2
    "111": (0.0,  -2.0)   # k = 3
}
node_index = {s: i for i, s in enumerate(pf_states)}
discrete_nodes = np.array([coords_dict[s] for s in pf_states])

###############################################
# 2. Build the Discrete Domain for FEM (8 nodes)
###############################################
nodes = discrete_nodes.copy()  # shape (8, 2)
num_nodes = nodes.shape[0]
triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################
# 3. Initialize Discrete Population at the k–Manifold Level
###############################################
total_molecules = 100.0
pop_dict = {s: 0.0 for s in pf_states}
pop_dict["000"] = 25.0  
# Distribute 75 molecules equally among k=1 states (states with one '1')
for s in pf_states:
    if count_ones(s) == 1:
        pop_dict[s] = 75.0 / 3.0
# For k>=2, set to zero.
for s in pf_states:
    if count_ones(s) >= 2:
        pop_dict[s] = 0.0

def get_occ_vector_discrete(pop_dict):
    occ = np.zeros(len(pf_states))
    for s, cnt in pop_dict.items():
        occ[node_index[s]] = cnt / total_molecules
    return occ

###############################################
# 4. FEM Assembly & Adaptive PDE Solver (Discrete Domain)
###############################################
def fem_assemble_matrices(nodes, elements):
    num_nodes = nodes.shape[0]
    A_mat = sp.lil_matrix((num_nodes, num_nodes))
    M_mat = sp.lil_matrix((num_nodes, num_nodes))
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

# Modified PDE solver: now optionally returns the final delta norm (residual).
def solve_pde_for_occupancy(nodes, elements, occ_vec, max_iter=150, tol=1e-1, damping=0.05, phi0=None, return_res=False):
    A_mat, M_mat = fem_assemble_matrices(nodes, elements)
    num_nodes = nodes.shape[0]
    if phi0 is None:
        phi = np.zeros(num_nodes)
    else:
        phi = phi0.copy()
    kappa_target = 1.0
    num_steps = 20
    kappa_values = np.linspace(0, kappa_target, num_steps+1)
    last_delta = None
    for kappa in kappa_values[1:]:
        for it in range(max_iter):
            nonlin = occ_vec * np.exp(2*phi)
            F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
            reg = 1e-12
            J = A_mat + M_mat.dot(sp.diags(occ_vec * np.exp(2*phi))) + reg*sp.eye(num_nodes)
            delta_phi = spla.spsolve(J, -F)
            phi += damping * delta_phi
            delta_norm = np.linalg.norm(delta_phi)
            if delta_norm < tol:
                last_delta = delta_norm
                break
    M_lumped = np.array(M_mat.sum(axis=1)).flatten()
    lap_phi = A_mat.dot(phi) / M_lumped
    R_curv = -2.0 * np.exp(-2*phi) * lap_phi
    if return_res:
        return phi, R_curv, last_delta
    return phi, R_curv

###############################################
# 5. Warm Start Setup: Use the loaded PDE solution as the initial guess
###############################################
# We use the first num_nodes entries from the continuous solution.
phi_warm = phi_cont[:num_nodes]

###############################################
# 6. (Optional) Re-solve the PDE for the baseline occupancy on the discrete domain.
###############################################
occ_vector = get_occ_vector_discrete(pop_dict)
phi_baseline, R_baseline = solve_pde_for_occupancy(nodes, elements, occ_vector)
print("Discrete PDE baseline solved.")

###############################################
# 7. New Rule-Based Monte Carlo Update (No External Perturbation)
###############################################
def new_allowed_k(state):
    k_val = count_ones(state)
    if k_val == 0:
        return [s for s in pf_states if count_ones(s)==1]
    elif k_val == 1:
        return [s for s in pf_states if count_ones(s) in [0,2]]
    elif k_val == 2:
        return [s for s in pf_states if count_ones(s) in [1,3]]
    elif k_val == 3:
        return [s for s in pf_states if count_ones(s)==2]
    else:
        return []

alpha_field = 0.1
beta_field = 1.0
DeltaE_flip = 5.0

def new_field_delta_f_discrete(state_i, state_j, occ_vec, R_vals, external_weight):
    i_ndx = node_index[state_i]
    energy_term = DeltaE_flip * np.exp(alpha_field * occ_vec[i_ndx])
    curvature_term = - beta_field * R_vals[i_ndx]
    return energy_term + curvature_term + external_weight

def new_mc_probabilities_discrete(state, occ_vec, R_vals, external_weight):
    allowed_states = new_allowed_k(state)
    p_list = []
    for s_target in allowed_states:
        df = new_field_delta_f_discrete(state, s_target, occ_vec, R_vals, external_weight)
        p_list.append(np.exp(- df/(0.001987 * 310.15)))
    p_list = np.array(p_list)
    p_stay = max(0, 1 - np.sum(p_list))
    full_p = np.concatenate(([p_stay], p_list))
    full_p = full_p / np.sum(full_p)
    return full_p, allowed_states

# Initialize the discrete molecule states.
molecule_states = []
for s, cnt in pop_dict.items():
    molecule_states += [s] * int(round(cnt))
molecule_states = np.array(molecule_states)

state_history = [molecule_states.copy()]
global_redox_history = []

def global_redox(states):
    k_vals = np.array([count_ones(s) for s in states])
    return np.mean(k_vals) / 3 * 100

global_redox_history.append(global_redox(molecule_states))

def get_occ_vector_discrete_from_states(states):
    counts = {s: np.sum(states == s) for s in pf_states}
    occ = np.zeros(len(pf_states))
    for s, cnt in counts.items():
        occ[node_index[s]] = cnt / total_molecules
    return occ

###############################################
# 8. Lyapunov Exponent Calculation Function
###############################################
def compute_lyapunov(time_series, dt=1.0):
    diffs = np.diff(time_series)
    diffs[diffs == 0] = 1e-10  # avoid log(0)
    log_diffs = np.log(np.abs(diffs)).reshape(-1, 1)
    t = np.arange(1, len(time_series)).reshape(-1, 1)
    model = LinearRegression().fit(t, log_diffs)
    lyap = model.coef_[0][0] / dt
    return lyap

###############################################
# 9. Monte Carlo Simulation Loop (e.g., 240 steps, No External Force)
###############################################
num_steps_mc = 240
tol = 1e-1  # convergence tolerance
max_extra_iterations = 20  # maximum extra iterations every 10 steps

for t in range(1, num_steps_mc+1):
    external_weight = 0.0  # No external force.
    occ_vec = get_occ_vector_discrete_from_states(molecule_states)
    
    # Update PDE solution with warm-start and obtain convergence metric.
    phi_warm, R_vals, res_norm = solve_pde_for_occupancy(nodes, elements, occ_vec, phi0=phi_warm, return_res=True)
    
    # Every 10 steps, enforce additional convergence.
    if t % 10 == 0:
        extra_iter = 0
        while (res_norm is None or res_norm > tol) and extra_iter < max_extra_iterations:
            phi_warm, R_vals, res_norm = solve_pde_for_occupancy(nodes, elements, occ_vec, phi0=phi_warm, return_res=True)
            extra_iter += 1
        if res_norm is None or res_norm > tol:
            print(f"Warning: PDE did not fully converge at step {t} (residual: {res_norm})")
    
    # Monte Carlo update of molecule states.
    new_states = []
    for state in molecule_states:
        full_p, allowed_states = new_mc_probabilities_discrete(state, occ_vec, R_vals, external_weight)
        outcome = np.random.choice(len(full_p), p=full_p)
        if outcome == 0:
            new_states.append(state)
        else:
            new_states.append(allowed_states[outcome - 1])
    molecule_states = np.array(new_states)
    state_history.append(molecule_states.copy())
    global_redox_history.append(global_redox(molecule_states))

###############################################
# 10. Compute Lyapunov Exponent from Global Redox Time Series
###############################################
global_redox_array = np.array(global_redox_history)
lyapunov_exponent = compute_lyapunov(global_redox_array, dt=1.0)
print("Estimated Lyapunov Exponent:", lyapunov_exponent)

###############################################
# 11. Final Discrete Outputs: k-Bin Distribution, Global Redox, Shannon Entropy
###############################################
final_distribution = {s: np.sum(molecule_states == s) for s in pf_states}
final_total = sum(final_distribution.values())
k_bin = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
for s, cnt in final_distribution.items():
    k_bin[count_ones(s)] += cnt

print("Final k-Bin Distribution after", num_steps_mc, "steps:")
for k_val in sorted(k_bin.keys()):
    frac = (k_bin[k_val] / final_total)*100 if final_total > 0 else 0.0
    print(f"k={k_val}: {k_bin[k_val]:.2f} molecules ({frac:.2f}%)")
print("Total molecules at start:", total_molecules)
print("Total molecules at end:  ", final_total)
print("Global Redox State (final):", global_redox_history[-1], "%")

def shannon_entropy_discrete(states):
    tot = len(states)
    entropy = 0.0
    for s in pf_states:
        p = np.sum(states == s) / tot
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

entropy_initial = shannon_entropy_discrete(state_history[0])
entropy_final = shannon_entropy_discrete(molecule_states)
print("Shannon entropy at start:", entropy_initial)
print("Shannon entropy at end:  ", entropy_final)

###############################################
# 12. Plot Global Redox Evolution Over Time
###############################################
plt.figure(figsize=(8,5))
plt.plot(global_redox_history, 'o-')
plt.xlabel("Time Steps")
plt.ylabel("Global Redox State (%)")
plt.title("Evolution of Global Redox State Over 240 Steps (No External Force)")
plt.grid(True)
plt.savefig("global_redox_evolution.png", dpi=300)
plt.show()

###############################################
# 13. Record Aggregated k-Manifold History to Excel
###############################################
agg_history = []
for counts in state_history:
    agg = {}
    for s, cnt in {s: np.sum(counts==s) for s in pf_states}.items():
        k_val = count_ones(s)
        agg[k_val] = agg.get(k_val, 0) + cnt
    agg_history.append(agg)
df_k = pd.DataFrame(agg_history)
df_k.index.name = 'Time_Step'
df_k.columns = ['k=' + str(k) for k in sorted(df_k.columns)]

with pd.ExcelWriter("simulation_results.xlsx") as writer:
    df_k.to_excel(writer, sheet_name="k_manifold_counts")
print("Monte Carlo aggregated k-manifold history saved to 'simulation_results.xlsx'.")
