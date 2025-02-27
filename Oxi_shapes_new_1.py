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

# Redefine vertices so that the progression is vertical:
# A = (0,0) for k₀ (base, high occupancy), 
# B = (0.5,1.0) for k₁, 
# C = (0,2) for k₂ (top, low occupancy)
A = np.array([0.0, 0.0])   # k₀
B = np.array([0.5, 1.0])   # k₁
C = np.array([0.0, 2.0])   # k₂

vertices = np.array([A, B, C])
vertex_occupancies = np.array([0.3, 0.3, 0.4])  # high occupancy at the base, low at the top

# Function to generate n uniformly random points inside triangle ABC using barycentrics
def random_points_in_triangle(A, B, C, n):
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    mask = r1 + r2 > 1  # reflect points outside the triangle
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    points = A + np.outer(r1, (B - A)) + np.outer(r2, (C - A))
    return points

num_internal_points = 150
internal_points = random_points_in_triangle(A, B, C, num_internal_points)

# Combine vertices and internal points into nodes
nodes = np.vstack((vertices, internal_points))
num_nodes = nodes.shape[0]

# Build the mesh connectivity using Delaunay triangulation
triangulation = Delaunay(nodes)
elements = triangulation.simplices

###############################################################################
# 2. Compute Occupancy at Each Node via Barycentric Interpolation
###############################################################################

def barycentric_coords(P, A, B, C):
    T = np.vstack((B - A, C - A)).T  # 2x2 matrix
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
# 3. Assemble the FEM System for the PDE
#    We solve: Δφ = - (κ/2)·ρ·exp(2φ)
###############################################################################

def fem_assemble_system(nodes, elements, occupancy, kappa=1.0):
    num_nodes = nodes.shape[0]
    A_mat = sp.lil_matrix((num_nodes, num_nodes))
    b_vec = np.zeros(num_nodes)
    
    for elem in elements:
        indices = elem
        coords = nodes[indices]  # shape: 3x2
        
        # Compute area of the triangle
        mat = np.array([
            [1, coords[0,0], coords[0,1]],
            [1, coords[1,0], coords[1,1]],
            [1, coords[2,0], coords[2,1]]
        ])
        area = 0.5 * np.abs(np.linalg.det(mat))
        if area == 0:
            continue
        
        # Compute gradients for the linear basis functions
        x = coords[:,0]
        y = coords[:,1]
        b_coeff = np.array([y[1]-y[2],
                            y[2]-y[0],
                            y[0]-y[1]])
        c_coeff = np.array([x[2]-x[1],
                            x[0]-x[2],
                            x[1]-x[0]])
        K_local = np.zeros((3,3))
        for i_local in range(3):
            for j_local in range(3):
                K_local[i_local, j_local] = (b_coeff[i_local]*b_coeff[j_local] + c_coeff[i_local]*c_coeff[j_local])/(4*area)
        
        # Assemble local stiffness into global matrix
        for i_local, i_global in enumerate(indices):
            for j_local, j_global in enumerate(indices):
                A_mat[i_global, j_global] += K_local[i_local, j_local]
        
        # Assemble load vector (source term) using average occupancy
        rho_avg = np.mean(occupancy[indices])
        for i_local, i_global in enumerate(indices):
            b_vec[i_global] += - (kappa/2) * rho_avg * area / 3.0
    return A_mat, b_vec

A_mat, b_vec = fem_assemble_system(nodes, elements, occupancy, kappa=1.0)

###############################################################################
# 4. Solve the Nonlinear PDE Using Newton-Raphson
#    We solve: A·φ - b - exp(2φ)*occupancy = 0
###############################################################################

phi = np.zeros(num_nodes)  # initial guess for φ
max_iter = 100
tol = 1e-6

for iteration in range(max_iter):
    F = A_mat @ phi - b_vec - np.exp(2*phi)*occupancy
    J = A_mat - sp.diags(2*np.exp(2*phi)*occupancy)
    delta_phi = spla.spsolve(J.tocsr(), -F)
    phi += delta_phi
    if np.linalg.norm(delta_phi) < tol:
        print(f"Newton converged in {iteration} iterations.")
        break
else:
    print("Newton did not converge.")

###############################################################################
# 5. Compute the Ricci Curvature at Each Node
#    For a conformally flat metric g = exp(2φ)δ, the scalar curvature is: 
#         R = -2·exp(-2φ)·Δφ,
#    where we approximate Δφ with A·φ.
###############################################################################

R = -2.0 * np.exp(-2*phi) * (A_mat @ phi)

###############################################################################
# 6. Apply the Density-Induced "Gravity" Effect
#    Here, higher occupancy deforms the shape downward.
#    We define the final vertical coordinate as:
#         z = φ - occupancy
###############################################################################

z = phi - occupancy

###############################################################################
# 7. Compute Segmentation Lines (Horizontal Cuts) to Divide the Triangle
#    Since there is a linear progression from k₀ (base) to k₂ (top), we
#    compute horizontal segmentation curves at y = y_min + 1/3*(y_max-y_min)
#    and y = y_min + 2/3*(y_max-y_min), which will separate k₀/k₁ and k₁/k₂.
###############################################################################

y_min = np.min(nodes[:,1])
y_max = np.max(nodes[:,1])
y_seg1 = y_min + (y_max - y_min) / 3.0
y_seg2 = y_min + 2*(y_max - y_min) / 3.0

def horizontal_intersection(y_line, vertices):
    """
    For a given horizontal line y = y_line and a triangle defined by vertices,
    compute the intersection points (there will be two).
    """
    pts = []
    # Triangle edges: (v0,v1), (v1,v2), (v2,v0)
    for i in range(3):
        v1 = vertices[i]
        v2 = vertices[(i+1)%3]
        # Check if the horizontal line crosses the edge:
        if (v1[1] - y_line) * (v2[1] - y_line) <= 0 and v1[1] != v2[1]:
            t = (y_line - v1[1]) / (v2[1] - v1[1])
            x_intersect = v1[0] + t*(v2[0]-v1[0])
            pts.append([x_intersect, y_line])
    return np.array(pts)

# Use the triangle's vertices (A, B, C) to compute intersections.
seg_line1 = horizontal_intersection(y_seg1, vertices)
seg_line2 = horizontal_intersection(y_seg2, vertices)

# Interpolate z-values along these segmentation lines using griddata from FEM nodes.
z_seg1 = griddata(nodes, z, seg_line1, method='linear')
z_seg2 = griddata(nodes, z, seg_line2, method='linear')

###############################################################################
# 8. Plot the Deformed Triangle Colored by Ricci Curvature, with Segmentation
###############################################################################

# Create a triangulation for plotting the FEM mesh
triang_plot = tri.Triangulation(nodes[:,0], nodes[:,1], elements)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Normalize Ricci curvature for coloring
R_norm = (R - R.min())/(R.max()-R.min() + 1e-10)
facecolors = plt.cm.viridis(R_norm)

# Plot the deformed surface with z = (φ - occupancy) and color by R.
surf = ax.plot_trisurf(triang_plot, z, cmap='viridis',
                         shade=True, edgecolor='none',
                         antialiased=True, linewidth=0.2, alpha=0.9,
                         facecolors=facecolors)

ax.set_title("Deformed k-Manifold Triangle (Density-Induced Well & Segmentation)")
ax.set_xlabel("X (manifold axis)")
ax.set_ylabel("Y (manifold axis)")
ax.set_zlabel("Deformation z = φ - occupancy")

# Add a colorbar for Ricci curvature
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(R)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')

# Label the vertices with their k-manifold tags.
vertex_labels = ["K₀", "K₁", "K₂"]
for i, label in enumerate(vertex_labels):
    x_label, y_label = vertices[i][0], vertices[i][1]
    # Interpolate z at the vertex location
    z_label = griddata(nodes, z, np.array([[x_label, y_label]]), method='linear')[0]
    ax.text(x_label, y_label, z_label, f" {label}", fontsize=12, color='k', weight='bold')

# Overlay segmentation lines (horizontal cuts)
if seg_line1.shape[0] == 2:
    ax.plot(seg_line1[:,0], seg_line1[:,1], z_seg1, color='k', lw=2, label='k₀/k₁ boundary')
if seg_line2.shape[0] == 2:
    ax.plot(seg_line2[:,0], seg_line2[:,1], z_seg2, color='k', lw=2, label='k₁/k₂ boundary')

plt.legend()
plt.savefig("oxishape_segmented.png", dpi=300)
plt.show()
