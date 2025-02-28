import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

###############################################################################
# 1. Define the Rectangular Domain for R=10 (11 k-states)
#    Domain: x in [0,1] and y in [0,10]
###############################################################################
# Corners of the rectangle:
A = np.array([0.0, 0.0])   # bottom-left, corresponds to k = 0
B = np.array([1.0, 0.0])   # bottom-right
C = np.array([1.0, 10.0])  # top-right, corresponds to k = 10
D = np.array([0.0, 10.0])  # top-left

# We'll generate internal points uniformly in the rectangle.
def random_points_in_rectangle(n, x_range=(0,1), y_range=(0,10)):
    xs = np.random.uniform(x_range[0], x_range[1], n)
    ys = np.random.uniform(y_range[0], y_range[1], n)
    return np.vstack((xs, ys)).T

num_internal_points = 300
internal_points = random_points_in_rectangle(num_internal_points)

# Combine the corner vertices with the internal points.
vertices = np.vstack((A, B, C, D))
nodes = np.vstack((vertices, internal_points))
num_nodes = nodes.shape[0]

# Create Delaunay triangulation for the domain.
triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################################################
# 2. Define Occupancy as a Function of y via Linear Interpolation
#    For R=10, we have 11 k-states: k = 0, 1, …, 10.
#    Provide an experimental occupancy array (length 11). 
###############################################################################
# Example occupancy distribution for k = 0..10:
occupancy_states = np.array([0.83, 0.00, 0.0, 0.00, 0.00, 
                             0.00, 0.00, 0.0, 0.0, 0.0, 0.17])
# The vertical coordinate y runs from 0 to 10. Use np.interp to define occupancy.
def occupancy_from_y(y, occ_states):
    k_values = np.linspace(0, 10, len(occ_states))
    return np.interp(y, k_values, occ_states)

occupancy = np.zeros(num_nodes)
for i in range(num_nodes):
    # Here, occupancy depends only on y
    occupancy[i] = occupancy_from_y(nodes[i,1], occupancy_states)

###############################################################################
# 3. Assemble the FEM Matrices (Stiffness A and Mass M)
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
        area = 0.5 * np.abs(np.linalg.det(mat))
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
        M_local = (area/12.0) * np.array([[2,1,1],
                                          [1,2,1],
                                          [1,1,2]])
        for i_local, i_global in enumerate(idx):
            for j_local, j_global in enumerate(idx):
                A_mat[i_global, j_global] += K_local[i_local,j_local]
                M_mat[i_global, j_global] += M_local[i_local,j_local]
    return A_mat.tocsr(), M_mat.tocsr()

A_mat, M_mat = fem_assemble_matrices(nodes, elements)

###############################################################################
# 4. Solve the Nonlinear PDE Using Continuation and Damped Newton-Raphson
#    We solve: A φ + (κ/2) M (ρ e^(2φ)) = 0, with κ ramping from 0 to 1.
###############################################################################
phi = np.zeros(num_nodes)  # initial guess for φ
max_iter = 120
tol = 1e-2
damping = 0.2

kappa_target = 1.0
num_steps = 5
kappa_values = np.linspace(0, kappa_target, num_steps+1)

for kappa in kappa_values[1:]:
    print(f"Continuation step: kappa = {kappa:.3f}")
    for it in range(max_iter):
        nonlin = occupancy * np.exp(2*phi)
        F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
        J = A_mat + M_mat.dot(sp.diags(occupancy * np.exp(2*phi)))
        delta_phi = spla.spsolve(J, -F)
        phi += damping * delta_phi
        if np.linalg.norm(delta_phi) < tol:
            print(f"  Newton converged in {it} iterations for kappa = {kappa:.3f}")
            break
    else:
        print(f"  Newton did NOT converge for kappa = {kappa:.3f}")

###############################################################################
# 5. Compute Ricci Curvature
#    Using the formula R = -2 e^(-2φ) Δφ, where Δφ is approximated using a lumped mass matrix.
###############################################################################
M_lumped = np.array(M_mat.sum(axis=1)).flatten()
lap_phi = A_mat.dot(phi) / M_lumped
R = -2.0 * np.exp(-2*phi) * lap_phi

###############################################################################
# 6. Density-Induced "Gravity": Define z = φ - occupancy
###############################################################################
z = phi - occupancy

###############################################################################
# 7. Segment the Rectangle Horizontally into 11 Bands (k = 0 to 10)
#    The domain's y-range is [0, 10]. The boundaries between bands are at y = 1,2,...,9.
###############################################################################
band_boundaries = np.linspace(0, 10, 12)  # 12 values => 11 bands
# The centroid of each band is at x=0.5 and y = (lower+upper)/2.
band_centroids = []
for i in range(len(band_boundaries)-1):
    y_lower = band_boundaries[i]
    y_upper = band_boundaries[i+1]
    mid_y = (y_lower + y_upper) / 2.0
    band_centroids.append([0.5, mid_y])
band_centroids = np.array(band_centroids)

###############################################################################
# 8. Plot the Deformed Shape with Ricci Curvature Coloring and k-State Labels
###############################################################################
triang_plot = tri.Triangulation(nodes[:,0], nodes[:,1], elements)
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

R_norm = (R - R.min())/(R.max()-R.min()+1e-10)
facecolors = plt.cm.viridis(R_norm)

surf = ax.plot_trisurf(triang_plot, z, cmap='viridis',
                         shade=True, edgecolor='none',
                         antialiased=True, linewidth=0.2, alpha=0.9,
                         facecolors=facecolors)

ax.set_title("Cdc20 0-min")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis (oxidation state)")
ax.set_zlabel("z = φ - occupancy")

mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(R)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')

# Label each band centroid with K0, K1, ... K10.
k_labels = [f"K₍{i}₎" for i in range(11)]
for i, label in enumerate(k_labels):
    cx, cy = band_centroids[i]
    # Interpolate z at the centroid for proper 3D placement.
    z_centroid = griddata(nodes, z, np.array([[cx, cy]]), method='linear')[0]
    ax.text(cx, cy, z_centroid, f" {label}", fontsize=12, color='r', weight='bold')
plt.savefig("cdc20.png", dpi=300)
plt.show()
