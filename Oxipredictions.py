import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from scipy.interpolate import griddata

###############################################################################
# 1. Enumerate the 8 i-states (R=3) and assign diamond coordinates
###############################################################################
pf_states = ["000", "001", "010", "011", "100", "101", "110", "111"]

def count_ones(s):
    return s.count('1')

# Diamond layout (discrete):
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
nodes = np.array([coords_dict[s] for s in pf_states])
num_nodes = len(nodes)

# Build a Delaunay triangulation over these 8 nodes
triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################################################
# 2. Allowed/Barred Transitions from Table 4
###############################################################################
# Each i-state has a list of allowed transitions. Barred transitions get large cost.
allowed_map = {
    "000": ["000","100","010","001"], 
    "001": ["001","101","011","000"],
    "010": ["010","110","000","011"],
    "011": ["011","111","001","010"],
    "100": ["100","000","110","101"],
    "101": ["101","001","111","100"],
    "110": ["110","010","100","111"],
    "111": ["111","011","101","110"]
}

###############################################################################
# 3. Initial Population: 10,000 molecules, distributed as per Table 4
###############################################################################
# For example, 25% in "000" (k=0), 75% in k=1 states, 0% in k=2,3.
total_molecules = 10000.0
pop_dict = {s: 0.0 for s in pf_states}
pop_dict["000"] = 0.25 * total_molecules
for s in pf_states:
    if count_ones(s) == 1:
        pop_dict[s] = (0.75 * total_molecules)/3.0
for s in pf_states:
    if count_ones(s) >= 2:
        pop_dict[s] = 0.0

def get_occ_vector(pop_dict):
    occ = np.zeros(num_nodes)
    for s, cnt in pop_dict.items():
        i = node_index[s]
        occ[i] = cnt / total_molecules
    return occ

###############################################################################
# 4. FEM Assembly & PDE Solver for φ -> Ricci Curvature
###############################################################################
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
                K_local[i_local,j_local] = (b[i_local]*b[j_local]+c[i_local]*c[j_local])/(4*area)
        M_local = (area/12.0)*np.array([[2,1,1],
                                        [1,2,1],
                                        [1,1,2]])
        for i_local, i_global in enumerate(idx):
            for j_local, j_global in enumerate(idx):
                A_mat[i_global, j_global]+=K_local[i_local,j_local]
                M_mat[i_global, j_global]+=M_local[i_local,j_local]
    return A_mat.tocsr(), M_mat.tocsr()

A_mat, M_mat = fem_assemble_matrices(nodes, elements)

def solve_pde_for_occupancy(occ_vec, max_iter=150, tol=1e-1, damping=0.1):
    phi = np.zeros(num_nodes)
    kappa_target = 1.0
    num_cont_steps = 5
    kappa_values = np.linspace(0, kappa_target, num_cont_steps+1)
    for kappa in kappa_values[1:]:
        for it in range(max_iter):
            nonlin = occ_vec * np.exp(2*phi)
            F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
            J = A_mat + M_mat.dot(sp.diags(occ_vec*np.exp(2*phi)))
            delta_phi = spla.spsolve(J, -F)
            phi += damping*delta_phi
            if np.linalg.norm(delta_phi) < tol:
                break
    M_lumped = np.array(M_mat.sum(axis=1)).flatten()
    lap_phi = A_mat.dot(phi)/M_lumped
    R_curv = -2.0 * np.exp(-2*phi)*lap_phi
    return phi, R_curv

###############################################################################
# 5. Field Equation & Transition Probability
###############################################################################
# We'll define oxidation cost = +37 kcal/mol, reduction cost = -37 kcal/mol.
# If i->k is not in allowed_map[i], we assign a huge cost.
def delta_E_flip(i_state, k_state, E_ox=37.0, E_red=-37.0):
    # Compare bits
    # If i_state == k_state, return 0 (stay).
    if i_state == k_state:
        return 0.0
    # Find the bit that differs
    for idx in range(len(i_state)):
        if i_state[idx] != k_state[idx]:
            if i_state[idx]=='0' and k_state[idx]=='1':
                return E_ox
            elif i_state[idx]=='1' and k_state[idx]=='0':
                return E_red
    return 0.0

alpha = 1.0
beta_c = 1.0
gamma = 0.5  # external perturbation

def delta_f(i_state, k_state, R_vals, occ_vec):
    # If k_state not in allowed_map[i_state], return huge cost => barred
    if k_state not in allowed_map[i_state]:
        return 1e6
    i_ndx = node_index[i_state]
    dE = delta_E_flip(i_state, k_state)
    rho_i = occ_vec[i_ndx]
    energy_term = dE * np.exp(alpha*rho_i)
    curvature_term = - beta_c * R_vals[i_ndx]
    return energy_term + curvature_term + gamma

def transition_probability(df):
    kB = 0.001987
    T_sim = 310.15
    p = np.exp(-df/(kB*T_sim))
    return min(1.0, p)

###############################################################################
# 6. Monte Carlo Update Over 240 Steps
###############################################################################
num_steps_mc = 240
pop_dict_initial = pop_dict.copy()
pop_history = []
global_redox_history = []

def shannon_entropy(popd):
    tot = sum(popd.values())
    ent = 0.0
    for cnt in popd.values():
        if cnt>0:
            p = cnt/tot
            ent -= p*np.log2(p)
    return ent

def compute_global_redox(popd):
    # Weighted average k-value => sum(k_i * pop_i)/(total * 3)*100
    tot = sum(popd.values())
    ksum = 0.0
    for s, cnt in popd.items():
        k_val = count_ones(s)
        ksum += k_val*cnt
    if tot>0:
        return (ksum/(tot*3.0))*100.0
    else:
        return 0.0

for step in range(num_steps_mc):
    # Build occ_vec from pop_dict
    occ_vec = get_occ_vector(pop_dict)
    # Solve PDE => phi, R
    phi, R_vals = solve_pde_for_occupancy(occ_vec)
    
    # Two-phase flux approach
    flux_in = {s:0.0 for s in pf_states}
    flux_out= {s:0.0 for s in pf_states}
    for i_state, cnt_i in pop_dict.items():
        if cnt_i<=0:
            continue
        # Attempt transitions to each allowed neighbor + "stay"
        # Actually from table: i_state can go to allowed_map[i_state].
        # We'll also consider i_state->i_state as "stay" => df=0 if we want
        # but your table includes "000"->"000" as an allowed transition. 
        for k_state in allowed_map[i_state]:
            df_ik = delta_f(i_state, k_state, R_vals, occ_vec)
            p_ik = transition_probability(df_ik)
            flux = cnt_i * p_ik
            flux_out[i_state]+=flux
            flux_in[k_state]+=flux
    # Update population
    new_pop = {}
    for s in pf_states:
        new_pop[s] = pop_dict[s] - flux_out[s] + flux_in[s]
        if new_pop[s]<0:
            new_pop[s]=0.0
    pop_dict = new_pop
    pop_history.append(pop_dict.copy())
    
    # Global redox
    redox = compute_global_redox(pop_dict)
    global_redox_history.append(redox)

###############################################################################
# 7. Final Outputs
###############################################################################
final_pop = pop_dict.copy()
final_total = sum(final_pop.values())
k_bin = {0:0.0, 1:0.0, 2:0.0, 3:0.0}
for s, cnt in final_pop.items():
    k_val = count_ones(s)
    k_bin[k_val]+=cnt

print("Final distribution after", num_steps_mc, "steps:")
for k_val in sorted(k_bin.keys()):
    frac = (k_bin[k_val]/final_total)*100 if final_total>0 else 0
    print(f"k={k_val}: {k_bin[k_val]:.2f} molecules ({frac:.2f}%)")

print("Total molecules at start:", sum(pop_dict_initial.values()))
print("Total molecules at end:  ", final_total)
entropy_initial = shannon_entropy(pop_dict_initial)
entropy_final = shannon_entropy(final_pop)
print("Shannon entropy at start:", entropy_initial)
print("Shannon entropy at end:  ", entropy_final)
print("Global Redox State (final):", compute_global_redox(final_pop), "%")

# Plot global redox evolution
plt.figure(figsize=(8,5))
plt.plot(global_redox_history, 'o-')
plt.xlabel("Time Steps")
plt.ylabel("Global Redox State (%)")
plt.title("Evolution of Global Redox Over 240 Steps (Discrete R=3 System)")
plt.grid(True)
plt.savefig("global_redox_evolution.png", dpi=300)
plt.show()

###############################################################################
# 8. Save Final Oxi-Shape (z = φ - occ) at Step 240
###############################################################################
# We re-solve PDE with final occupancy distribution => new phi, R
occ_vec_final = get_occ_vector(final_pop)
phi_final, R_final = solve_pde_for_occupancy(occ_vec_final)
z_final = phi_final - occ_vec_final

triang_plot = mtri.Triangulation(nodes[:,0], nodes[:,1], elements)
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

R_norm_final = (R_final - R_final.min())/(R_final.max()-R_final.min()+1e-14)
facecolors_final = plt.cm.viridis(R_norm_final)

surf = ax.plot_trisurf(triang_plot, z_final, cmap='viridis',
                         shade=True, edgecolor='none',
                         antialiased=True, linewidth=0.2, alpha=0.9,
                         facecolors=facecolors_final)

ax.set_title("Final Oxi-Shape (Discrete R=3) at t=240")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("z = φ - occupancy")

mappable_final = plt.cm.ScalarMappable(cmap='viridis')
mappable_final.set_array(R_final)
fig.colorbar(mappable_final, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')

for s in pf_states:
    i = node_index[s]
    ax.text(nodes[i,0], nodes[i,1], z_final[i], s, fontsize=12, color='k')

plt.savefig("final_oxi_shape.png", dpi=300)
plt.show()

###############################################################################
# 9. Identify Geodesics from "000" to "111" in the Final Deformed Diamond
###############################################################################
G_final = nx.Graph()
for i in range(num_nodes):
    G_final.add_node(i, pos=(nodes[i,0], nodes[i,1], z_final[i]))
# Edges from triangulation
for elem in elements:
    for i_local in range(3):
        for j_local in range(i_local+1, 3):
            n1 = elem[i_local]
            n2 = elem[j_local]
            pos1 = np.array([nodes[n1,0], nodes[n1,1], z_final[n1]])
            pos2 = np.array([nodes[n2,0
