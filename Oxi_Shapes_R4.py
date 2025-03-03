import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from matplotlib.path import Path

###############################################################################
# 1. Define the Pentagonal Domain for R=4 (5 k-states)
###############################################################################
n_vertices = 5
center = np.array([0.5, 0.5])
radius = 0.5
# Compute vertices of a regular pentagon (starting at the top)
angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False) + np.pi/2  
vertices = np.column_stack((center[0] + radius*np.cos(angles),
                            center[1] + radius*np.sin(angles)))
vertex_labels = ["K₀", "K₁", "K₂", "K₃", "K₄"]

# Example fractional occupancy values at the pentagon's vertices
vertex_occupancies = np.array([0.25, 0.50, 0.75, 0.0, 0.0])  

###############################################################################
# Generate internal points uniformly within the pentagon
###############################################################################
def random_points_in_polygon(n, polygon):
    # Compute bounding box of polygon
    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    poly_path = Path(polygon)
    points = []
    while len(points) < n:
        p = np.random.rand(2) * (max_x - min_x) + [min_x, min_y]
        if poly_path.contains_point(p):
            points.append(p)
    return np.array(points)

num_internal_points = 200
internal_points = random_points_in_polygon(num_internal_points, vertices)

# Combine vertices and internal points
nodes = np.vstack((vertices, internal_points))
num_nodes = nodes.shape[0]

# Build the Delaunay triangulation over the nodes
triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################################################
# 2. Interpolate Occupancy onto All Nodes using griddata
###############################################################################
# Use the occupancy values at the vertices and interpolate over the entire domain.
occupancy = griddata(vertices, vertex_occupancies, nodes, method='linear')
# For any nodes where interpolation fails, fall back to the nearest value.
mask = np.isnan(occupancy)
if np.any(mask):
    occupancy[mask] = griddata(vertices, vertex_occupancies, nodes[mask], method='nearest')

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
# 4. Solve the Nonlinear PDE Using Continuation and Damped Newton–Raphson
#    We solve: A·φ + 0.5*M*(occupancy*exp(2φ)) = 0, with κ ramped from 0 to 1.
###############################################################################
phi = np.zeros(num_nodes)
max_iter = 150
tol = 1e-1
damping = 0.1

kappa_target = 1.0
num_steps = 5
kappa_values = np.linspace(0, kappa_target, num_steps+1)

for kappa in kappa_values[1:]:
    print(f"Continuation: kappa = {kappa:.3f}")
    for it in range(max_iter):
        nonlin = occupancy * np.exp(2*phi)
        F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
        # Jacobian: A + M*diag(occupancy*exp(2φ))
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
#    R = -2*exp(-2φ)*Δφ
###############################################################################
M_lumped = np.array(M_mat.sum(axis=1)).flatten()
lap_phi = A_mat.dot(phi) / M_lumped
R = -2.0 * np.exp(-2*phi) * lap_phi

###############################################################################
# 6. Density-Induced "Gravity": Compute z = φ - occupancy
###############################################################################
z = phi - occupancy

###############################################################################
# 7. Plot the Deformed Pentagonal Domain with Ricci Curvature Coloring
###############################################################################
triang_plot = tri.Triangulation(nodes[:,0], nodes[:,1], elements)
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

# Normalize Ricci curvature for coloring
R_norm = (R - R.min())/(R.max() - R.min() + 1e-10)
facecolors = plt.cm.viridis(R_norm)

surf = ax.plot_trisurf(triang_plot, z, cmap='viridis', shade=True,
                         edgecolor='none', antialiased=True, linewidth=0.2,
                         alpha=0.9, facecolors=facecolors)

ax.set_title("Oxi-Shape for a Protein with 4 Cysteines")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("z = φ - occupancy")

mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(R)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')

# Label the pentagon vertices
for i, label in enumerate(vertex_labels):
    xL, yL = vertices[i]
    zL = griddata(nodes, z, np.array([[xL, yL]]), method='linear')[0]
    ax.text(xL, yL, zL, f" {label}", fontsize=12, color='k', weight='bold')

plt.savefig("oxishape_R4.svg", dpi=300)
plt.show()
