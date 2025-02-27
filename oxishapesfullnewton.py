import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

###############################################################################
# 1. Define the Triangle Domain (k-manifolds) and Generate Mesh Points
###############################################################################

# Vertices arranged vertically: k₀ at the base, k₁ mid, k₂ top.
A = np.array([0.0, 0.0])   # k₀
B = np.array([0.5, 1.0])   # k₁
C = np.array([0.0, 2.0])   # k₂

vertices = np.array([A, B, C])
vertex_occupancies = np.array([0.3, 0.3, 0.4])  # fractional occupancies

def random_points_in_triangle(A, B, C, n):
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    return A + np.outer(r1, (B - A)) + np.outer(r2, (C - A))

num_internal_points = 150
internal_points = random_points_in_triangle(A, B, C, num_internal_points)
nodes = np.vstack((vertices, internal_points))
num_nodes = nodes.shape[0]

triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################################################
# 2. Compute Occupancy at Each Node via Barycentric Interpolation
###############################################################################

def barycentric_coords(P, A, B, C):
    T = np.vstack((B - A, C - A)).T
    v = P - A
    sol = np.linalg.solve(T, v)
    wB, wC = sol
    wA = 1 - wB - wC
    return np.array([wA, wB, wC])

occupancy = np.zeros(num_nodes)
for i in range(num_nodes):
    w = barycentric_coords(nodes[i], A, B, C)
    occupancy[i] = w[0]*vertex_occupancies[0] + w[1]*vertex_occupancies[1] + w[2]*vertex_occupancies[2]

###############################################################################
# 3. Assemble the FEM Matrices (Stiffness and Mass)
###############################################################################

def fem_assemble_matrices(nodes, elements):
    num_nodes = nodes.shape[0]
    A_mat = sp.lil_matrix((num_nodes, num_nodes))
    M_mat = sp.lil_matrix((num_nodes, num_nodes))
    
    for elem in elements:
        indices = elem
        coords = nodes[indices]
        mat = np.array([[1, coords[0,0], coords[0,1]],
                        [1, coords[1,0], coords[1,1]],
                        [1, coords[2,0], coords[2,1]]])
        area = 0.5 * np.abs(np.linalg.det(mat))
        if area <= 0:
            continue
        x = coords[:,0]
        y = coords[:,1]
        b_coeff = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c_coeff = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        K_local = np.zeros((3,3))
        for i_local in range(3):
            for j_local in range(3):
                K_local[i_local, j_local] = (b_coeff[i_local]*b_coeff[j_local] + c_coeff[i_local]*c_coeff[j_local])/(4*area)
        M_local = (area/12.0)*np.array([[2,1,1],
                                        [1,2,1],
                                        [1,1,2]])
        for i_local, i_global in enumerate(indices):
            for j_local, j_global in enumerate(indices):
                A_mat[i_global, j_global] += K_local[i_local, j_local]
                M_mat[i_global, j_global] += M_local[i_local, j_local]
    return A_mat.tocsr(), M_mat.tocsr()

A_mat, M_mat = fem_assemble_matrices(nodes, elements)

###############################################################################
# 4. Solve the Nonlinear PDE Using Continuation and Damped Newton-Raphson
#    We want to solve: A·φ + (κ/2)*M*(ρ exp(2φ)) = 0, with κ increasing from 0 to 1.
###############################################################################

phi = np.zeros(num_nodes)  # initial guess
max_iter = 100
tol = 1e-1
damping = 0.1  # smaller damping factor helps stability

# Continuation: increase κ gradually
kappa_target = 1.0
num_steps = 10
kappa_values = np.linspace(0, kappa_target, num_steps+1)

for kappa in kappa_values[1:]:
    print(f"Continuation step: kappa = {kappa:.3f}")
    for iteration in range(max_iter):
        nonlinear_term = occupancy * np.exp(2*phi)
        F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlinear_term)
        J = A_mat + M_mat.dot(sp.diags(occupancy * np.exp(2*phi)))
        delta_phi = spla.spsolve(J, -F)
        phi += damping * delta_phi
        if np.linalg.norm(delta_phi) < tol:
            print(f"  Newton converged in {iteration} iterations for kappa = {kappa:.3f}.")
            break
    else:
        print(f"  Newton did not converge for kappa = {kappa:.3f}.")
        # Optionally: you can break out or adjust damping here.

###############################################################################
# 5. Compute the Ricci Curvature at Each Node
#    R = -2 exp(-2φ) Δφ, with Δφ approximated using a lumped mass matrix.
###############################################################################

M_lumped = np.array(M_mat.sum(axis=1)).flatten()
lap_phi = A_mat.dot(phi) / M_lumped
R = -2.0 * np.exp(-2*phi) * lap_phi

###############################################################################
# 6. Apply the Density-Induced "Gravity" Effect
#    Define final vertical coordinate: z = φ - occupancy.
###############################################################################

z = phi - occupancy

###############################################################################
# 7. Compute Segmentation Lines (Horizontal Cuts) to Divide the Triangle
###############################################################################

y_min = np.min(nodes[:,1])
y_max = np.max(nodes[:,1])
y_seg1 = y_min + (y_max - y_min) / 3.0
y_seg2 = y_min + 2*(y_max - y_min) / 3.0

def horizontal_intersection(y_line, vertices):
    pts = []
    for i in range(3):
        v1 = vertices[i]
        v2 = vertices[(i+1)%3]
        if (v1[1] - y_line) * (v2[1] - y_line) <= 0 and v1[1] != v2[1]:
            t = (y_line - v1[1]) / (v2[1] - v1[1])
            pts.append([v1[0] + t*(v2[0]-v1[0]), y_line])
    return np.array(pts)

seg_line1 = horizontal_intersection(y_seg1, vertices)
seg_line2 = horizontal_intersection(y_seg2, vertices)
z_seg1 = griddata(nodes, z, seg_line1, method='linear')
z_seg2 = griddata(nodes, z, seg_line2, method='linear')

###############################################################################
# 8. Plot the Deformed Triangle Colored by Ricci Curvature, with Segmentation
###############################################################################

triang_plot = tri.Triangulation(nodes[:,0], nodes[:,1], elements)
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

R_norm = (R - R.min())/(R.max()-R.min() + 1e-10)
facecolors = plt.cm.viridis(R_norm)

surf = ax.plot_trisurf(triang_plot, z, cmap='viridis',
                         shade=True, edgecolor='none',
                         antialiased=True, linewidth=0.2, alpha=0.9,
                         facecolors=facecolors)

ax.set_title("Deformed k-Manifold Triangle (Continuation & Damped Newton)")
ax.set_xlabel("X (manifold axis)")
ax.set_ylabel("Y (manifold axis)")
ax.set_zlabel("Deformation z = φ - occupancy")

mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(R)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')

vertex_labels = ["K₀", "K₁", "K₂"]
for i, label in enumerate(vertex_labels):
    x_label, y_label = vertices[i][0], vertices[i][1]
    z_label = griddata(nodes, z, np.array([[x_label, y_label]]), method='linear')[0]
    ax.text(x_label, y_label, z_label, f" {label}", fontsize=12, color='k', weight='bold')

if seg_line1.shape[0] == 2:
    ax.plot(seg_line1[:,0], seg_line1[:,1], z_seg1, color='k', lw=2, label='k₀/k₁ boundary')
if seg_line2.shape[0] == 2:
    ax.plot(seg_line2[:,0], seg_line2[:,1], z_seg2, color='k', lw=2, label='k₁/k₂ boundary')

plt.legend()
plt.savefig("oxishape_continuation.png", dpi=300)
plt.show()
