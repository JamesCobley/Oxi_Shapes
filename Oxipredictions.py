import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
import pandas as pd
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

###############################################
# 1. Define Discrete i-States for R=3 (8 states)
###############################################
pf_states = ["000", "001", "010", "011", "100", "101", "110", "111"]

def count_ones(s):
    return s.count('1')

# Coordinates for the 8 discrete states (diamond layout)
coords_dict = {
    "000": (0.0,  1.0),   # k=0
    "001": (-1.0, 0.0),   # k=1
    "010": (0.0,  0.0),   # k=1
    "100": (1.0,  0.0),   # k=1
    "011": (-1.0, -1.0),  # k=2
    "101": (0.0,  -1.0),  # k=2
    "110": (1.0,  -1.0),  # k=2
    "111": (0.0,  -2.0)   # k=3
}

node_index = {s: i for i, s in enumerate(pf_states)}
discrete_nodes = np.array([coords_dict[s] for s in pf_states])

###############################################
# 2. Build the Discrete Domain for FEM (8 nodes)
###############################################
# In the discrete version, we work solely with the 8 nodes.
nodes = discrete_nodes.copy()
num_nodes = nodes.shape[0]
triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################
# 3. Allowed Transitions (from Table 4)
###############################################
allowed_map = {
    "000": ["000", "100", "010", "001"],
    "001": ["001", "101", "011", "000"],
    "010": ["010", "110", "000", "011"],
    "011": ["011", "111", "001", "010"],
    "100": ["100", "000", "110", "101"],
    "101": ["101", "001", "111", "100"],
    "110": ["110", "010", "100", "111"],
    "111": ["111", "011", "101", "110"]
}

###############################################
# 4. Initialize Discrete Population (100 molecules)
###############################################
total_molecules = 100.0
pop_dict = {s: 0.0 for s in pf_states}
pop_dict["000"] = 0.25 * total_molecules
for s in pf_states:
    if count_ones(s) == 1:
        pop_dict[s] = (0.75 * total_molecules) / 3.0
for s in pf_states:
    if count_ones(s) >= 2:
        pop_dict[s] = 0.0

def get_occ_vector_discrete(pop_dict):
    occ = np.zeros(len(pf_states))
    for s, cnt in pop_dict.items():
        occ[node_index[s]] = cnt / total_molecules
    return occ

###############################################
# 5. FEM Assembly & PDE Solver for φ and Ricci Curvature
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

def solve_pde_for_occupancy(nodes, elements, occ_vec, max_iter=150, tol=1e-1, damping=0.1):
    A_mat, M_mat = fem_assemble_matrices(nodes, elements)
    num_nodes = nodes.shape[0]
    phi = np.zeros(num_nodes)
    kappa_target = 1.0
    num_cont_steps = 5
    kappa_values = np.linspace(0, kappa_target, num_cont_steps+1)
    for kappa in kappa_values[1:]:
        for it in range(max_iter):
            nonlin = occ_vec * np.exp(2*phi)
            F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
            reg = 1e-12  # regularization
            J = A_mat + M_mat.dot(sp.diags(occ_vec * np.exp(2*phi))) + reg*sp.eye(num_nodes)
            delta_phi = spla.spsolve(J, -F)
            phi += damping * delta_phi
            if np.linalg.norm(delta_phi) < tol:
                break
    M_lumped = np.array(M_mat.sum(axis=1)).flatten()
    lap_phi = A_mat.dot(phi) / M_lumped
    R_curv = -2.0 * np.exp(-2*phi) * lap_phi
    return phi, R_curv

# Solve PDE for the initial occupancy.
occ_vector = get_occ_vector_discrete(pop_dict)
phi, R_vals = solve_pde_for_occupancy(nodes, elements, occ_vector)

# Plot initial Oxi-Shape
z = phi - occ_vector
triang_plot = mtri.Triangulation(nodes[:,0], nodes[:,1], elements)
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
R_norm = (R_vals - R_vals.min())/(R_vals.max()-R_vals.min()+1e-14)
facecolors = plt.cm.viridis(R_norm)
surf = ax.plot_trisurf(triang_plot, z, cmap='viridis', shade=True,
                         edgecolor='none', antialiased=True, linewidth=0.2,
                         alpha=0.9, facecolors=facecolors)
ax.set_title("Initial Oxi-Shape (Discrete Domain, R=3)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("z = φ - occ")
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(R_vals)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')
plt.savefig("initial_oxishape.png", dpi=300)
plt.show()
print("Initial PDE solution complete. Oxi-Shape saved as 'initial_oxishape.png'.")

###############################################
# 6. Discrete Monte Carlo Update (Conservative, Per Molecule)
###############################################
# Represent the 100 molecules as a list of their discrete state (strings).
molecule_states = []
for s, cnt in pop_dict.items():
    molecule_states += [s] * int(round(cnt))
molecule_states = np.array(molecule_states)  # length should be 100

# Field equation parameters (for discrete updates)
alpha = 0.1
beta_c = 1.0
DeltaE_flip = 5.0
def field_delta_f_discrete(state_i, state_j, occ_vec, R_vals, external_weight):
    i_ndx = node_index[state_i]
    energy_term = DeltaE_flip * np.exp(alpha * occ_vec[i_ndx])
    curvature_term = - beta_c * R_vals[i_ndx]
    return energy_term + curvature_term + external_weight

def mc_probabilities_discrete(state, occ_vec, R_vals, external_weight):
    allowed = allowed_map[state]
    p_list = []
    for s_target in allowed:
        df = field_delta_f_discrete(state, s_target, occ_vec, R_vals, external_weight)
        p_list.append(np.exp(- df/(0.001987 * 310.15)))
    p_list = np.array(p_list)
    p_stay = max(0, 1 - np.sum(p_list))
    full_p = np.concatenate(([p_stay], p_list))
    full_p = full_p / np.sum(full_p)
    return full_p, allowed

state_history = [molecule_states.copy()]
global_redox_history = []

def global_redox(states):
    k_vals = np.array([count_ones(s) for s in states])
    return np.mean(k_vals) / 3 * 100

global_redox_history.append(global_redox(molecule_states))

# For discrete PDE re-solution, we use the current counts per state.
def get_occ_vector_discrete_from_states(states):
    counts = {s: np.sum(states==s) for s in pf_states}
    occ = np.zeros(len(pf_states))
    for s, cnt in counts.items():
        occ[node_index[s]] = cnt / total_molecules
    return occ

num_steps_mc = 10
for t in range(1, num_steps_mc+1):
    external_weight = 0.1 if t < 5 else 0.0
    occ_vec = get_occ_vector_discrete_from_states(molecule_states)
    # Re-solve PDE for discrete occupancy.
    phi, R_vals = solve_pde_for_occupancy(nodes, elements, occ_vec)
    new_states = []
    for state in molecule_states:
        full_p, allowed = mc_probabilities_discrete(state, occ_vec, R_vals, external_weight)
        outcome = np.random.choice(len(full_p), p=full_p)
        if outcome == 0:
            new_states.append(state)
        else:
            new_states.append(allowed[outcome - 1])
    molecule_states = np.array(new_states)
    state_history.append(molecule_states.copy())
    global_redox_history.append(global_redox(molecule_states))

###############################################
# 7. Final Discrete Outputs: k-Bin Distribution, Global Redox, Shannon Entropy
###############################################
final_distribution = {s: np.sum(molecule_states==s) for s in pf_states}
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

def shannon_entropy_discrete(states):
    tot = len(states)
    entropy = 0.0
    for s in pf_states:
        p = np.sum(states==s) / tot
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

entropy_initial = shannon_entropy_discrete(state_history[0])
entropy_final = shannon_entropy_discrete(molecule_states)
print("Shannon entropy at start:", entropy_initial)
print("Shannon entropy at end:  ", entropy_final)
print("Global Redox State (final):", global_redox_history[-1], "%")

###############################################
# 8. Plot Global Redox Evolution Over Time
###############################################
plt.figure(figsize=(8,5))
plt.plot(global_redox_history, 'o-')
plt.xlabel("Time Steps")
plt.ylabel("Global Redox State (%)")
plt.title("Evolution of Global Redox State Over 10 Steps")
plt.grid(True)
plt.savefig("global_redox_evolution.png", dpi=300)
plt.show()

###############################################
# 9. Save Final Oxi-Shape (Discrete PDE Solution) as 300 DPI PNG
###############################################
final_occ = get_occ_vector_discrete_from_states(molecule_states)
phi_final, _ = solve_pde_for_occupancy(nodes, elements, final_occ)
z_final = phi_final - final_occ
triang_plot = mtri.Triangulation(nodes[:,0], nodes[:,1], elements)
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
R_norm_final = (R_vals - R_vals.min())/(R_vals.max()-R_vals.min()+1e-14)
facecolors_final = plt.cm.viridis(R_norm_final)
surf = ax.plot_trisurf(triang_plot, z_final, cmap='viridis', shade=True,
                         edgecolor='none', antialiased=True, linewidth=0.2, alpha=0.9,
                         facecolors=facecolors_final)
ax.set_title("Final Oxi-Shape (Discrete PDE) after 10 Steps")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("z = φ - occ")
mappable_final = plt.cm.ScalarMappable(cmap='viridis')
mappable_final.set_array(R_vals)
fig.colorbar(mappable_final, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')
for s in pf_states:
    i = node_index[s]
    ax.text(nodes[i,0], nodes[i,1], z_final[i], s, fontsize=12, color='k', weight='bold')
plt.savefig("final_oxishape.png", dpi=300)
plt.show()

###############################################
# 10. Record Monte Carlo State History to Excel
###############################################
discrete_history = []
for states in state_history:
    counts = {s: np.sum(states==s) for s in pf_states}
    discrete_history.append(counts)
df_states = pd.DataFrame(discrete_history)
df_states.index.name = 'Time_Step'
df_states.columns.name = 'i_state'

agg_history = []
for counts in discrete_history:
    agg = {}
    for s, cnt in counts.items():
        k_val = count_ones(s)
        agg[k_val] = agg.get(k_val, 0) + cnt
    agg_history.append(agg)
df_k = pd.DataFrame(agg_history)
df_k.index.name = 'Time_Step'
df_k.columns = ['k=' + str(k) for k in sorted(df_k.columns)]

with pd.ExcelWriter("simulation_results.xlsx") as writer:
    df_states.to_excel(writer, sheet_name="i_state_counts")
    df_k.to_excel(writer, sheet_name="k_manifold_counts")
print("Monte Carlo state history saved to 'simulation_results.xlsx'.")
