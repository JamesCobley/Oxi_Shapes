import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
# 1. Define Triangle & Occupancy
##############################################################################
vertices = np.array([
    [0.0, 0.0],  # k=0
    [1.0, 1.0],  # k=1
    [2.0, 0.0]   # k=2
])

# Larger occupancy differences for a more dramatic shape
occupancies_vertices = np.array([0.0, 1.0, 0.8])

Nx, Ny = 100, 100
x_vals = np.linspace(0, 2, Nx)
y_vals = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x_vals, y_vals)

def barycentric_interpolate(px, py, v0, v1, v2, f0, f1, f2):
    """ Interpolate f in the triangle (v0,v1,v2). Return np.nan if outside. """
    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2
    denom = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
    w0 = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / denom
    w1 = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / denom
    w2 = 1 - w0 - w1
    if (w0 < 0) or (w1 < 0) or (w2 < 0):
        return np.nan
    return w0*f0 + w1*f1 + w2*f2

v0, v1, v2 = vertices
f0, f1, f2 = occupancies_vertices

occupancy_grid = np.zeros_like(X)
mask = np.zeros_like(X, dtype=bool)

for i in range(Ny):
    for j in range(Nx):
        px, py = X[i,j], Y[i,j]
        val = barycentric_interpolate(px, py, v0, v1, v2, f0, f1, f2)
        occupancy_grid[i,j] = val
        mask[i,j] = np.isnan(val)

# PDE source term S = +kappa * occupancy
kappa = 5.0   # Increase for bigger amplitude
S_grid = kappa * occupancy_grid
S_grid[mask] = 0.0  # outside triangle => no source

##############################################################################
# 2. Solve Poisson: Δz = S with Neumann-like boundary
##############################################################################
z = np.zeros_like(X)  # initial guess
dx = x_vals[1] - x_vals[0]
dy = y_vals[1] - y_vals[0]

max_iter = 5000
tolerance = 1e-5

def apply_neumann_boundary(z_array):
    """ Approximate Neumann by copying interior values to boundary. """
    # top/bottom edges
    for j in range(Nx):
        z_array[0, j]   = z_array[1, j]     # y=0 boundary
        z_array[Ny-1,j] = z_array[Ny-2, j]  # y=1 boundary
    # left/right edges
    for i in range(Ny):
        z_array[i, 0]   = z_array[i, 1]
        z_array[i, Nx-1] = z_array[i, Nx-2]

for it in range(max_iter):
    # apply boundary first
    apply_neumann_boundary(z)
    max_diff = 0.0
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            # If outside the triangle, skip or treat as boundary
            if mask[i,j]:
                continue
            # Gauss-Seidel update for Poisson in 2D:
            # Δz = (z[i+1,j] + z[i-1,j] + z[i,j+1] + z[i,j-1] - 4*z[i,j]) / (dx^2)
            # we want Δz = S => z_new = avg_of_neighbors - (dx^2/4)*S
            # assuming dx ~ dy for simplicity
            z_new = (z[i+1,j] + z[i-1,j] + z[i,j+1] + z[i,j-1])/4.0 - (dx**2/4.0)*S_grid[i,j]
            diff = abs(z_new - z[i,j])
            if diff > max_diff:
                max_diff = diff
            z[i,j] = z_new
    if max_diff < tolerance:
        print(f"Converged at iteration {it} with max diff={max_diff}")
        break

##############################################################################
# 3. Visualize the Floating Shape
##############################################################################
import numpy.ma as ma
z_masked = ma.masked_array(z, mask=mask)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, z_masked, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('X (k-manifold axis)')
ax.set_ylabel('Y (k-manifold axis)')
ax.set_zlabel('z(x,y)')
ax.set_title('Poisson-Based Deformation with Neumann-like Boundary')
plt.colorbar(surf, shrink=0.5, aspect=10, label='z value')

# Mark corners
corner_z = []
for (cx, cy) in vertices:
    j = int(round((cx - x_vals[0]) / dx))
    i = int(round((cy - y_vals[0]) / dy))
    corner_z.append(z[i,j])
corner_z = np.array(corner_z)
ax.scatter(vertices[:,0], vertices[:,1], corner_z, color='r', s=80, label='Vertices')
ax.legend()
plt.savefig("/content/oxishape.png", dpi=300)
plt.show()
