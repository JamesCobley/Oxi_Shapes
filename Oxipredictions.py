import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point
import networkx as nx
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# 1. Define the Discrete i-States for R=3 and their Coordinates (Diamond Layout)
###############################################################################
# We have 8 i-states (from "000" to "111").
pf_states = ["000", "001", "010", "011", "100", "101", "110", "111"]

def count_ones(s):
    return s.count('1')

# Manually assign coordinates for the discrete states:
# "000": (0, 1)        [k=0]
# "001": (-1, 0)       [k=1]
# "010": (0, 0)        [k=1]
# "100": (1, 0)        [k=1]
# "011": (-1, -1)      [k=2]
# "101": (0, -1)       [k=2]
# "110": (1, -1)       [k=2]
# "111": (0, -2)       [k=3]
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

# Build mapping and discrete nodes array (8 nodes)
node_index = {s: i for i, s in enumerate(pf_states)}
discrete_nodes = np.array([coords_dict[s] for s in pf_states])

###############################################################################
# 2. Build the Continuous Diamond Domain
###############################################################################
# We define the diamond by its 4 corners. For example:
# Corners: A (k=0): (0,0); B (k=1): (1,1); C (k=2): (0,2); D (k=3): (-1,1)
corners = np.array([
    [0.0, 0.0],    # k=0
    [1.0, 1.0],    # k=1
    [0.0, 2.0],    # k=2
    [-1.0, 1.0]    # k=3
])
corner_occupancies = np.array([0.25, 0.75, 0.0, 0.0])  # starting occupancy

# Create a shapely polygon
diamond_poly = Polygon(corners)

# Generate random internal points in the diamond.
def random_points_in_polygon(n, polygon):
    minx, miny, maxx, maxy = polygon.bounds
    pts = []
    while len(pts) < n:
        rx = np.random.uniform(minx, maxx)
        ry = np.random.uniform(miny, maxy)
        if polygon.contains(Point(rx, ry)):
            pts.append([rx, ry])
    return np.array(pts)

num_internal_points = 300
internal_points = random_points_in_polygon(num_internal_points, diamond_poly)

# Combine the 4 corners and the internal points.
nodes = np.vstack((corners, internal_points))
num_nodes = nodes.shape[0]

# Build Delaunay triangulation on the continuous domain.
triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################################################
# 3. Interpolate Occupancy onto Continuous Nodes from Discrete Corner Values
###############################################################################
def diamond_interpolate_occupancy(px, py, corners, corner_occ):
    # Divide diamond into two triangles: T1: A-B-C, T2: A-C-D.
    A = corners[0]
    B = corners[1]
    C = corners[2]
    D = corners[3]
    p = np.array([px, py])
    
    def tri_area(pts):
        mat = np.array([[1, pts[0,0], pts[0,1]],
                        [1, pts[1,0], pts[1,1]],
                        [1, pts[2,0], pts[2,1]]])
        return 0.5 * abs(np.linalg.det(mat))
    
    def barycentric_weight(p, tri_coords, tri_occ):
        area_full = tri_area(tri_coords)
        weights = []
        for i in range(3):
            sub_tri = np.vstack((p, tri_coords[(i+1)%3], tri_coords[(i+2)%3]))
            area_sub = tri_area(sub_tri)
            weights.append(area_sub/(area_full+1e-14))
        weights = np.array(weights)
        return np.sum(weights*tri_occ)
    
    from shapely.geometry import Polygon, Point
    T1 = Polygon([A, B, C])
    T2 = Polygon([A, C, D])
    p_sh = Point(px, py)
    if T1.contains(p_sh) or T1.touches(p_sh):
        tri_coords = np.array([A, B, C])
        tri_occ = np.array([corner_occ[0], corner_occ[1], corner_occ[2]])
        return barycentric_weight(p, tri_coords, tri_occ)
    elif T2.contains(p_sh) or T2.touches(p_sh):
        tri_coords = np.array([A, C, D])
        tri_occ = np.array([corner_occ[0], corner_occ[2], corner_occ[3]])
        return barycentric_weight(p, tri_coords, tri_occ)
    else:
        return 0.0

occupancy = np.zeros(num_nodes)
for i in range(num_nodes):
    px, py = nodes[i]
    occupancy[i] = diamond_interpolate_occupancy(px, py, corners, corner_occupancies)

###############################################################################
# 4. FEM Assembly & Nonlinear PDE Solver for φ and Ricci Curvature
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

A_mat, M_mat = fem_assemble_matrices(nodes, elements)

phi = np.zeros(num_nodes)
max_iter = 150
tol = 1e-1
damping = 0.1
kappa_target = 1.0
num_cont_steps = 5
kappa_values = np.linspace(0, kappa_target, num_cont_steps+1)

for kappa in kappa_values[1:]:
    print(f"Continuation: kappa = {kappa:.3f}")
    for it in range(max_iter):
        nonlin = occupancy * np.exp(2*phi)
        F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
        reg = 1e-12  # regularization to avoid singularity
        J = A_mat + M_mat.dot(sp.diags(occupancy*np.exp(2*phi))) + reg*sp.eye(num_nodes)
        delta_phi = spla.spsolve(J, -F)
        phi += damping * delta_phi
        if np.linalg.norm(delta_phi) < tol:
            print(f"  Converged in {it} iterations at kappa = {kappa:.3f}")
            break
    else:
        print(f"  Did NOT converge at kappa = {kappa:.3f}")

# Compute Ricci curvature: R = -2*exp(-2*phi)*(A*phi / M_lumped)
M_lumped = np.array(M_mat.sum(axis=1)).flatten()
lap_phi = A_mat.dot(phi) / M_lumped
R_curv = -2.0 * np.exp(-2*phi) * lap_phi

###############################################################################
# 5. Compute and Plot the Initial Oxi-Shape
###############################################################################
z = phi - occupancy
triang_plot = mtri.Triangulation(nodes[:,0], nodes[:,1], elements)
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
R_norm = (R_curv - R_curv.min())/(R_curv.max()-R_curv.min()+1e-14)
facecolors = plt.cm.viridis(R_norm)
surf = ax.plot_trisurf(triang_plot, z, cmap='viridis', shade=True,
                         edgecolor='none', antialiased=True, linewidth=0.2, alpha=0.9,
                         facecolors=facecolors)
ax.set_title("Initial Oxi-Shape on Diamond Domain (R=3)")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("z = φ - occupancy")
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(R_curv)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')
plt.savefig("initial_oxishape.png", dpi=300)
plt.show()
print("Initial PDE solution complete. Oxi-Shape saved as 'initial_oxishape.png'.")

###############################################################################
# 6. Convert Discrete Population to Continuous Occupancy Vector
###############################################################################
# We have a discrete population for 8 i-states in pop_dict.
# Now, we want to assign each continuous node (304 nodes) an occupancy value.
# We do this by assigning each continuous node to the nearest discrete i-state,
# then distributing the discrete population evenly among those continuous nodes.
discrete_occ = np.zeros(len(pf_states))
for s in pf_states:
    discrete_occ[node_index[s]] = pop_dict[s]  # discrete population for state s

# For each continuous node, find the nearest discrete node (from the 8)
continuous2discrete = np.zeros(num_nodes, dtype=int)
for i in range(num_nodes):
    dists = np.linalg.norm(nodes[i] - discrete_nodes, axis=1)
    continuous2discrete[i] = np.argmin(dists)

# Count how many continuous nodes map to each discrete node.
counts = np.zeros(len(pf_states))
for i in continuous2discrete:
    counts[i] += 1

# Build continuous occupancy vector: for each continuous node, occupancy = (discrete_occ for its state)/count.
occ_vector_cont = np.zeros(num_nodes)
for i in range(num_nodes):
    d = continuous2discrete[i]
    if counts[d] > 0:
        occ_vector_cont[i] = discrete_occ[d] / counts[d]
    else:
        occ_vector_cont[i] = 0.0

###############################################################################
# 7. Monte Carlo Simulation (10 Steps) with External Oxidation Force
###############################################################################
# We will run 10 steps. For the first 5 steps, external weight γ = 0.1 (favor oxidation).
# After that, γ = 0.0.
alpha = 0.1
beta_c = 1.0
DeltaE_flip = 5.0
def field_delta_f(i, j, R_vals, occ_vec, external_weight):
    # i and j are continuous node indices.
    # Allow transition if there is an edge in connectivity graph (we use G below).
    # Use donor node i for occupancy and curvature.
    energy_term = DeltaE_flip * np.exp(alpha * occ_vec[i])
    curvature_term = - beta_c * R_vals[i]
    return energy_term + curvature_term + external_weight

def mc_transition_probability(df):
    kB = 0.001987
    T_sim = 310.15
    return min(1.0, np.exp(- df/(kB*T_sim)))

# Build connectivity graph from the Delaunay triangulation.
G = nx.Graph()
for elem in elements:
    for i in range(3):
        for j in range(i+1, 3):
            G.add_edge(elem[i], elem[j])

# Initialize continuous population vector: use occ_vector_cont (which was computed above)
pop_vec = occ_vector_cont * total_molecules

num_steps_mc = 10
pop_history = []
global_redox_history = []

# For global redox, assign each continuous node a discrete "k" from its nearest discrete node.
node_k = np.array([continuous2discrete[i] for i in range(num_nodes)])
# However, our discrete k is given by count_ones(s) for the corresponding pf_state.
# So we create an array of k-values for the 8 discrete states:
discrete_k = np.array([count_ones(s) for s in pf_states])
node_k = np.array([discrete_k[continuous2discrete[i]] for i in range(num_nodes)])

for t in range(num_steps_mc):
    external_weight = 0.1 if t < 5 else 0.0
    occ_vec = pop_vec / total_molecules
    # Re-solve PDE for current occupancy:
    phi, R_vals = solve_pde_for_occupancy(nodes, elements, occ_vec)
    # Monte Carlo update: for each node, distribute flux to its neighbors.
    flux_in = np.zeros(num_nodes)
    flux_out = np.zeros(num_nodes)
    for i in range(num_nodes):
        if pop_vec[i] <= 0:
            continue
        for j in G.neighbors(i):
            df = field_delta_f(i, j, R_vals, occ_vec, external_weight)
            p = mc_transition_probability(df)
            flux = pop_vec[i] * p
            flux_out[i] += flux
            flux_in[j] += flux
    pop_vec = pop_vec - flux_out + flux_in
    pop_history.append(pop_vec.copy())
    redox = (np.sum(node_k * pop_vec) / (total_molecules * 3)) * 100
    global_redox_history.append(redox)

###############################################################################
# 8. Final Outputs: k-Bin Distribution, Global Redox, Shannon Entropy
###############################################################################
final_pop = pop_vec.copy()
final_total = np.sum(final_pop)
k_bin = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
# We assign each continuous node to a discrete k by its nearest discrete node.
for i in range(num_nodes):
    k_val = discrete_k[continuous2discrete[i]]
    k_bin[k_val] += final_pop[i]

print("Final k-Bin Distribution after", num_steps_mc, "steps:")
for k_val in sorted(k_bin.keys()):
    frac = (k_bin[k_val] / final_total) * 100 if final_total > 0 else 0.0
    print(f"k={k_val}: {k_bin[k_val]:.2f} molecules ({frac:.2f}%)")
print("Total molecules at start:", total_molecules)
print("Total molecules at end:  ", final_total)

def shannon_entropy(pop_array):
    tot = np.sum(pop_array)
    entropy = 0.0
    for p in pop_array:
        if p > 0:
            frac = p / tot
            entropy -= frac * np.log2(frac)
    return entropy

entropy_initial = shannon_entropy(pop_history[0])
entropy_final = shannon_entropy(final_pop)
print("Shannon entropy at start:", entropy_initial)
print("Shannon entropy at end:  ", entropy_final)
print("Global Redox State (final):", global_redox_history[-1], "%")

###############################################################################
# 9. Plot Global Redox Evolution Over Time
###############################################################################
plt.figure(figsize=(8,5))
plt.plot(global_redox_history, 'o-')
plt.xlabel("Time Steps")
plt.ylabel("Global Redox State (%)")
plt.title("Evolution of Global Redox State Over 10 Steps")
plt.grid(True)
plt.savefig("global_redox_evolution.png", dpi=300)
plt.show()

###############################################################################
# 10. Save Final Oxi-Shape as 300 DPI PNG
###############################################################################
phi_final, _ = solve_pde_for_occupancy(nodes, elements, pop_vec/total_molecules)
z_final = phi_final - (pop_vec/total_molecules)
triang_plot = mtri.Triangulation(nodes[:,0], nodes[:,1], elements)
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
R_norm_final = (R_vals - R_vals.min())/(R_vals.max()-R_vals.min()+1e-14)
facecolors_final = plt.cm.viridis(R_norm_final)
surf = ax.plot_trisurf(triang_plot, z_final, cmap='viridis', shade=True,
                         edgecolor='none', antialiased=True, linewidth=0.2, alpha=0.9,
                         facecolors=facecolors_final)
ax.set_title("Final Oxi-Shape on Diamond Domain (t=10)")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("z = φ - occ")
mappable_final = plt.cm.ScalarMappable(cmap='viridis')
mappable_final.set_array(R_vals)
fig.colorbar(mappable_final, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')
# Label the 4 corners
corner_labels = ["k=0","k=1","k=2","k=3"]
for i, label in enumerate(corners):
    xL, yL = corners[i]
    zL = griddata(nodes, z_final, np.array([[xL, yL]]), method='linear')[0]
    ax.text(xL, yL, zL, f" {corner_labels[i]}", fontsize=12, color='k', weight='bold')
plt.savefig("final_oxishape.png", dpi=300)
plt.show()

###############################################################################
# 11. Record Monte Carlo Population History to Excel
###############################################################################
# We record the discrete population (per pf_state) at each MC step.
# First, we convert each continuous pop_vec into a discrete distribution using the mapping:
discrete_history = []
for pop in pop_history:
    discrete_pop = {s: 0.0 for s in pf_states}
    for i in range(num_nodes):
        s = pf_states[continuous2discrete[i]]
        discrete_pop[s] += pop[i]
    discrete_history.append(discrete_pop)

# Build a DataFrame: rows = time steps, columns = i-states.
df = pd.DataFrame(discrete_history)
df.index.name = 'Time_Step'
df.columns.name = 'i_state'

# Also aggregate by k-manifold.
def i_to_k(i_state):
    return i_state.count('1')

agg_history = []
for step_pop in discrete_history:
    agg = {}
    for s, count in step_pop.items():
        k_val = i_to_k(s)
        agg[k_val] = agg.get(k_val, 0) + count
    agg_history.append(agg)
df_k = pd.DataFrame(agg_history)
df_k.index.name = 'Time_Step'
df_k.columns = ['k=' + str(k) for k in sorted(df_k.columns)]

with pd.ExcelWriter("simulation_results.xlsx") as writer:
    df.to_excel(writer, sheet_name="i_state_counts")
    df_k.to_excel(writer, sheet_name="k_manifold_counts")

print("Monte Carlo population history saved to 'simulation_results.xlsx'.")

###############################################################################
# 12. Identify Geodesic on the Final Deformed Diamond
###############################################################################
G_final = nx.Graph()
for i in range(num_nodes):
    G_final.add_node(i, pos=(nodes[i,0], nodes[i,1], z_final[i]))
for elem in elements:
    for i in range(3):
        for j in range(i+1, 3):
            n1 = elem[i]
            n2 = elem[j]
            pos1 = np.array([nodes[n1,0], nodes[n1,1], z_final[n1]])
            pos2 = np.array([nodes[n2,0], nodes[n2,1], z_final[n2]])
            dist = np.linalg.norm(pos1-pos2)
            G_final.add_edge(n1, n2, weight=dist)

def nearest_node(point, nodes):
    dists = np.linalg.norm(nodes - point, axis=1)
    return np.argmin(dists)

node_A = nearest_node(corners[0], nodes)  # nearest to k=0
node_D = nearest_node(corners[3], nodes)  # nearest to k=3
try:
    geodesic_nodes = nx.shortest_path(G_final, source=node_A, target=node_D, weight="weight")
    print("Geodesic path from corner k=0 to k=3 (node indices):", geodesic_nodes)
except Exception as e:
    print("Error computing geodesic:", e)
