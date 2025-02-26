import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

##############################################################################
# 1. Define the Triangle Representing the k-Basis
##############################################################################
vertices = np.array([
    [0.0, 0.0],  # v0, k=0
    [1.0, 1.0],  # v1, k=1
    [2.0, 0.0]   # v2, k=2
])

# Empirical Positional Density (Occupancy)
occupancies_vertices = np.array([0.4, 0.8, 0.2])  # Match empirical data

##############################################################################
# 2. Create a Grid Over the Triangle for Interpolation
##############################################################################
Nx, Ny = 150, 150  # Increase resolution for smoothness
x_vals = np.linspace(0, 2, Nx)
y_vals = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x_vals, y_vals)

def barycentric_interpolate(px, py, v0, v1, v2, f0, f1, f2):
    """Barycentric interpolation for function values inside a triangle."""
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

# Unpack vertices and occupancy values
v0, v1, v2 = vertices
f0, f1, f2 = occupancies_vertices

occupancy_grid = np.zeros_like(X)
mask = np.zeros_like(X, dtype=bool)

for i in range(Ny):
    for j in range(Nx):
        px, py = X[i, j], Y[i, j]
        occ_val = barycentric_interpolate(px, py, v0, v1, v2, f0, f1, f2)
        occupancy_grid[i, j] = occ_val
        mask[i, j] = np.isnan(occ_val)

# Mask out points outside the triangle
occupancy_masked = np.ma.masked_where(mask, occupancy_grid)

##############################################################################
# 3. Compute the Conformal Factor and Ricci-like Curvature
##############################################################################
alpha = 10.0  # Strength of geometric deformation (tunable)
phi_grid = 0.5 * alpha * occupancy_grid  # Conformal factor

# Compute discrete Laplacian (Ricci curvature)
dx = x_vals[1] - x_vals[0]
dy = y_vals[1] - y_vals[0]
lap_phi = np.full_like(phi_grid, np.nan)

for i in range(1, Ny-1):
    for j in range(1, Nx-1):
        if mask[i,j] or mask[i, j-1] or mask[i, j+1] or mask[i-1, j] or mask[i+1, j]:
            continue
        d2x = (phi_grid[i, j+1] - 2*phi_grid[i, j] + phi_grid[i, j-1]) / (dx**2)
        d2y = (phi_grid[i+1, j] - 2*phi_grid[i, j] + phi_grid[i-1, j]) / (dy**2)
        lap_phi[i, j] = d2x + d2y

# Ricci-like curvature approximation: R(x) ≈ -(n-1) α Δρ(x)
n = 3
R_grid = -(n - 1) * alpha * lap_phi

##############################################################################
# 4. Compute Free Energy from Ricci Curvature
##############################################################################
baseline = 1.5  # Baseline energy level
gamma = 1.0     # Scaling factor for curvature
F_grid = baseline - gamma * R_grid

# Smooth the Free Energy Surface
F_grid_filled = np.copy(F_grid)
F_grid_filled[np.isnan(F_grid_filled)] = baseline
sigma = 10.0  # Increase for a smoother shape
F_grid_smoothed = gaussian_filter(F_grid_filled, sigma=sigma)
F_masked_smoothed = np.ma.masked_where(np.isnan(F_grid), F_grid_smoothed)

##############################################################################
# 5. Compute Geodesic Paths (Vector Flow Field)
##############################################################################
gradient_x, gradient_y = np.gradient(-F_grid_smoothed, dx, dy)

##############################################################################
# 6. Plot the Final 3D Oxi-Shape with Geodesic Paths
##############################################################################
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, Y, F_masked_smoothed, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('X (k-manifold axis)')
ax.set_ylabel('Y (manifold axis)')
ax.set_zlabel('Free Energy F')
ax.set_title('Oxi-Shapes: Geometrically Informed Proteoform Transitions')
plt.colorbar(surf, shrink=0.5, aspect=10, label='F Value')


plt.savefig("/content/oxishapesnew.png", dpi=300)
ax.legend()
plt.show()
