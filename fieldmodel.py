import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import scipy.sparse as sp
import scipy.sparse.linalg as spla

###############################################################################
# 1. Define the Diamond Domain for R = 3 (8 i-states)
###############################################################################
# Manually define the 8 i-states and assign coordinates.
pf_states = ["000", "001", "010", "011", "100", "101", "110", "111"]

def count_ones(s):
    return s.count('1')

# Define coordinates in a diamond:
# k=0: "000" at (0, 1)
# k=1: "001" at (-1, 0), "010" at (0, 0), "100" at (1, 0)
# k=2: "011" at (-1, -1), "101" at (0, -1), "110" at (1, -1)
# k=3: "111" at (0, -2)
coords_dict = {
    "000": (0.0,  1.0),
    "001": (-1.0, 0.0),
    "010": (0.0,  0.0),
    "100": (1.0,  0.0),
    "011": (-1.0, -1.0),
    "101": (0.0,  -1.0),
    "110": (1.0,  -1.0),
    "111": (0.0,  -2.0)
}

# Build node mapping: each state maps to a unique node index (0 to 7)
node_index = {s: i for i, s in enumerate(pf_states)}

# Build nodes array from these 8 coordinates (no extra points)
nodes = np.array([coords_dict[s] for s in pf_states])

# Build Delaunay triangulation on these 8 nodes
triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################################################
# 2. Define Allowed Transitions (from your table)
###############################################################################
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

###############################################################################
# 3. Set the Initial Population (10,000 molecules)
###############################################################################
# For R=3, there are 4 k-manifolds:
# k=0: only "000"
# k=1: "001", "010", "100"
# k=2: "011", "101", "110"
# k=3: "111"
# Use the empirical initial condition: K₀ = 25%, K₁ = 75%, K₂ = 0%, K₃ = 0%
pop_dict = {s: 0.0 for s in pf_states}
total_molecules = 10000.0
pop_dict["000"] = 0.25 * total_molecules  # 25% in k=0
# k=1: equally distribute 75% among three states
for s in pf_states:
    if count_ones(s) == 1:
        pop_dict[s] = (0.75 * total_molecules) / 3.0
# k=2 and k=3 remain 0
for s in pf_states:
    if count_ones(s) >= 2:
        pop_dict[s] = 0.0

###############################################################################
# 4. FEM Assembly & PDE Solver (with placeholder Ricci Flow)
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
                K_local[i_local, j_local] = (b[i_local]*b[j_local] + c[i_local]*c[j_local])/(4*area)
        M_local = (area/12.0)*np.array([[2,1,1],
                                        [1,2,1],
                                        [1,1,2]])
        for i_local, i_global in enumerate(idx):
            for j_local, j_global in enumerate(idx):
                A_mat[i_global, j_global] += K_local[i_local, j_local]
                M_mat[i_global, j_global] += M_local[i_local, j_local]
    return A_mat.tocsr(), M_mat.tocsr()

def solve_pde_for_occupancy(nodes, elements, occupancy_vec, max_iter=150, tol=1e-1, damping=0.1):
    """
    Solve the nonlinear PDE:
      A*φ + (κ/2)*M*(ρ * exp(2φ)) = 0
    using continuation (κ ramped from 0 to 1) and damped Newton-Raphson.
    Then compute Ricci curvature as:
      R = -2 * exp(-2φ) * (A*φ / M_lumped)
    This is a placeholder solver.
    """
    A_mat, M_mat = fem_assemble_matrices(nodes, elements)
    num_nodes = nodes.shape[0]
    phi = np.zeros(num_nodes)
    kappa_target = 1.0
    num_cont_steps = 5
    kappa_values = np.linspace(0, kappa_target, num_cont_steps+1)
    for kappa in kappa_values[1:]:
        for it in range(max_iter):
            nonlin = occupancy_vec * np.exp(2*phi)
            F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
            J = A_mat + M_mat.dot(sp.diags(occupancy_vec * np.exp(2*phi)))
            delta_phi = spla.spsolve(J, -F)
            phi += damping * delta_phi
            if np.linalg.norm(delta_phi) < tol:
                break
    M_lumped = np.array(M_mat.sum(axis=1)).flatten()
    lap_phi = A_mat.dot(phi) / M_lumped
    R_curv = -2.0 * np.exp(-2*phi) * lap_phi
    return phi, R_curv

def update_metric(g, dt=1.0):
    """
    Update the metric g using a simplified Ricci flow.
    g_new = g - 2*dt*(d^2/dk^2 ln(g))
    """
    g = np.array(g, dtype=float)
    ln_g = np.log(g + 1e-12)
    n = len(g)
    R_flow = np.zeros(n)
    for i in range(1, n-1):
        R_flow[i] = - (ln_g[i+1] - 2*ln_g[i] + ln_g[i-1])
    R_flow[0] = R_flow[1]
    R_flow[-1] = R_flow[-2]
    g_new = g - 2 * dt * R_flow
    g_new = np.maximum(g_new, 1e-6)
    return g_new, R_flow

###############################################################################
# 5. Define the Field Equation and Transition Probability
###############################################################################
# Field eq: Δf(i→k) = ΔE_flip - β * R(i)
# We use the Ricci curvature from the PDE solver at the donor state's node.
beta_c = 1.0
DeltaE_flip = 1.0

def delta_f(i, k, R_vals):
    if k not in allowed_map[i]:
        return 1e6
    if i == k:
        return 0.0
    i_ndx = node_index[i]
    R_i = R_vals[i_ndx]
    return DeltaE_flip - beta_c * R_i

def transition_probability(df):
    kB = 0.001987
    T_sim = 310.15
    return min(1.0, np.exp(- df/(kB*T_sim)))

###############################################################################
# 6. Monte Carlo Population Update (Conserving Total Population)
###############################################################################
def update_population(pop_dict, R_vals):
    flux_in = {s: 0.0 for s in pop_dict}
    flux_out = {s: 0.0 for s in pop_dict}
    
    for i_state, cnt in pop_dict.items():
        if cnt <= 0:
            continue
        for k_state in allowed_map[i_state]:
            df_ik = delta_f(i_state, k_state, R_vals)
            p_ik = transition_probability(df_ik)
            flux = cnt * p_ik
            flux_out[i_state] += flux
            flux_in[k_state] += flux
    new_pop = {}
    for s in pop_dict:
        new_pop[s] = pop_dict[s] - flux_out[s] + flux_in[s]
        if new_pop[s] < 0:
            new_pop[s] = 0.0
    return new_pop

###############################################################################
# 7. Run the Dynamic Simulation for 240 Steps
###############################################################################
num_steps = 240
history = []

# Initialize occupancy vector for the 8 nodes (normalized to fraction of total population)
occ_vector = np.zeros(len(pf_states))
for s, cnt in pop_dict.items():
    occ_vector[node_index[s]] = cnt / total_molecules

# Initialize metric g for the diamond (one per node), starting with 1.
g = np.ones(len(pf_states))

for t in range(num_steps):
    # Update occupancy vector from current population
    occ_vector = np.zeros(len(pf_states))
    for s, cnt in pop_dict.items():
        occ_vector[node_index[s]] = cnt / total_molecules
    # Solve PDE for φ and Ricci curvature on the 8 nodes (diamond)
    phi, R_vals = solve_pde_for_occupancy(nodes, elements, occ_vector)
    # Update metric g via Ricci flow (on the 8 nodes)
    g, R_flow = update_metric(g, dt=1.0)
    
    # Update population using the field equation and current R_vals.
    pop_dict = update_population(pop_dict, R_vals)
    
    history.append(pop_dict.copy())

###############################################################################
# 8. Compute Final Distribution by k-Manifold and Global Redox
###############################################################################
final_total = sum(pop_dict.values())
k_bin = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
for s, cnt in pop_dict.items():
    k_bin[count_ones(s)] += cnt

print("Final distribution after", num_steps, "steps:")
for k_val in sorted(k_bin.keys()):
    frac = (k_bin[k_val]/final_total)*100 if final_total > 0 else 0.0
    print(f"k={k_val}: {k_bin[k_val]:.2f} molecules ({frac:.2f}%)")

# Compute global redox state: weighted average oxidation = (sum(k * pop))/ (total*3)*100
global_redox = (sum(count_ones(s)*cnt for s, cnt in pop_dict.items()) / (final_total * r)) * 100
print(f"Global Redox State: {global_redox:.2f}%")

###############################################################################
# 9. Plot Global Redox Evolution Over Time
###############################################################################
global_redox_history = []
for pop in history:
    tot = sum(pop.values())
    ox = sum(count_ones(s)*pop[s] for s in pop)
    redox = (ox/(tot*r))*100 if tot > 0 else 0
    global_redox_history.append(redox)

plt.figure(figsize=(8,5))
plt.plot(global_redox_history, 'o-')
plt.xlabel("Time Steps")
plt.ylabel("Global Redox State (%)")
plt.title("Evolution of Global Redox State Over 240 Steps")
plt.grid(True)
plt.show()
