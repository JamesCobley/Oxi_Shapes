#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as tri
import pickle
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

###############################################################################
# 1. Define the Square Domain for R=3 (4 k-states)
#    Corners:
#      A = (0,0)  -> K₀
#      B = (1,0)  -> K₁
#      C = (1,1)  -> K₂
#      D = (0,1)  -> K₃
###############################################################################
A = np.array([0.0, 0.0])  # K₀
B = np.array([1.0, 0.0])  # K₁
C = np.array([1.0, 1.0])  # K₂
D = np.array([0.0, 1.0])  # K₃

# Stack vertices and set example occupancies at each corner
vertices = np.vstack((A, B, C, D))
vertex_occupancies = np.array([0.25, 0.75, 0.0, 0.0])  # Example fractional occupancy values

# Generate internal points uniformly in the square [0,1]×[0,1]
def random_points_in_square(n):
    return np.random.rand(n, 2)

num_internal_points = 200
internal_points = random_points_in_square(num_internal_points)

# Combine vertices and internal points
nodes = np.vstack((vertices, internal_points))
num_nodes = nodes.shape[0]

# Build the Delaunay triangulation over the nodes
triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################################################
# 2. Bilinear Interpolation for Occupancy
#    For a point (x,y) in [0,1]²:
#      ρ(x,y) = (1-x)(1-y)*ρ_A + x(1-y)*ρ_B + x*y*ρ_C + (1-x)*y*ρ_D
###############################################################################
def bilinear_occupancy(x, y, occ):
    wA = (1 - x) * (1 - y)
    wB = x * (1 - y)
    wC = x * y
    wD = (1 - x) * y
    return wA*occ[0] + wB*occ[1] + wC*occ[2] + wD*occ[3]

occupancy = np.zeros(num_nodes)
for i in range(num_nodes):
    x_i, y_i = nodes[i]
    occupancy[i] = bilinear_occupancy(x_i, y_i, vertex_occupancies)

###############################################################################
# 3. Assemble FEM Matrices (Stiffness A and Mass M)
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

###############################################################################
# 4. Solve the Nonlinear PDE Using Continuation and Damped Newton-Raphson
#    We solve: A·φ + (κ/2)*M*(ρ e^(2φ)) = 0, with κ ramped from 0 to 1.
###############################################################################
phi = np.zeros(num_nodes)
max_iter = 500
tol = 1e-1
damping = 0.05

kappa_target = 1.0
num_steps = 20
kappa_values = np.linspace(0, kappa_target, num_steps+1)

for kappa in kappa_values[1:]:
    print(f"Continuation: kappa = {kappa:.3f}")
    for it in range(max_iter):
        nonlin = occupancy * np.exp(2*phi)
        F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
        J = A_mat + M_mat.dot(sp.diags(occupancy * np.exp(2*phi)))
        delta_phi = spla.spsolve(J, -F)
        phi += damping * delta_phi
        if np.linalg.norm(delta_phi) < tol:
            print(f"  Converged in {it} iterations at kappa = {kappa:.3f}")
            break
    else:
        print(f"  Did NOT converge at kappa = {kappa:.3f}")

###############################################################################
# 5. Compute Ricci Curvature
#    Approximate Δφ using a lumped mass matrix (row sum of M)
#    R = -2 e^(-2φ) Δφ
###############################################################################
M_lumped = np.array(M_mat.sum(axis=1)).flatten()
lap_phi = A_mat.dot(phi) / M_lumped
R = -2.0 * np.exp(-2*phi) * lap_phi

###############################################################################
# 6. Density-Induced "Gravity": z = φ - occupancy
###############################################################################
z = phi - occupancy

###############################################################################
# 7. Plot the Deformed Square with Ricci Curvature Coloring and Vertex Labels
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

ax.set_title("GAPDH control")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("z = φ - occupancy")

mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(R)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')

# Label the four corners as K₀, K₁, K₂, K₃
corner_labels = ["K₀", "K₁", "K₂", "K₃"]
for i, label in enumerate(corner_labels):
    xL, yL = vertices[i]
    zL = griddata(nodes, z, np.array([[xL, yL]]), method='linear')[0]
    ax.text(xL, yL, zL, f" {label}", fontsize=12, color='k', weight='bold')
plt.savefig("oxishape_continuation.png", dpi=300)
plt.show()

#Save the soltution for the second pipeline step
solution = {"phi": phi, "R_vals": R_vals, "occ_vector": occ_vector}
with open("pde_solution.pkl", "wb") as f:
    pickle.dump(solution, f)
print("PDE solution saved to 'pde_solution.pkl'.")
