import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------------------------
# 1. Define the Single Triangle for k=0,1,2 in 2D
# ------------------------------------------------------------------------------
# Corners of the triangle:
#   (0,0) -> k=0
#   (1,1) -> k=1
#   (2,0) -> k=2
corners = np.array([
    [0.0, 0.0],  # k=0
    [1.0, 1.0],  # k=1
    [2.0, 0.0]   # k=2
])

# Occupancy (positional density) at each corner
occupancy_corners = np.array([0.2, 0.5, 0.3])

# ------------------------------------------------------------------------------
# 2. Create a Grid and Interpolate Occupancy on the Triangle
# ------------------------------------------------------------------------------
Nx, Ny = 60, 60
x_vals = np.linspace(0, 2, Nx)
y_vals = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x_vals, y_vals)

def barycentric_interpolate(px, py, 
                            x1, y1, x2, y2, x3, y3, 
                            f1, f2, f3):
    # Calculate barycentric coordinates
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    w1 = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / denom
    w2 = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / denom
    w3 = 1 - w1 - w2
    if (w1 < 0) or (w2 < 0) or (w3 < 0):
        return np.nan
    return w1*f1 + w2*f2 + w3*f3

# Unpack corners and occupancy values
(x0, y0), (x1, y1), (x2, y2) = corners
f0, f1_occ, f2 = occupancy_corners

occupancy_grid = np.zeros_like(X)
for i in range(Ny):
    for j in range(Nx):
        occupancy_grid[i,j] = barycentric_interpolate(X[i,j], Y[i,j],
                                                        x0, y0, x1, y1, x2, y2,
                                                        f0, f1_occ, f2)

# ------------------------------------------------------------------------------
# 3. Compute a Conformal Factor & Ricci-like Curvature
# ------------------------------------------------------------------------------
alpha = 1.0
n = 3  # For a 3D analogy
phi_grid = 0.5 * alpha * occupancy_grid

dx = x_vals[1] - x_vals[0]
dy = y_vals[1] - y_vals[0]

lap_phi = np.full_like(phi_grid, np.nan)
for i in range(1, Ny-1):
    for j in range(1, Nx-1):
        if np.isnan(phi_grid[i,j]):
            continue
        d2x = (phi_grid[i, j+1] - 2*phi_grid[i,j] + phi_grid[i, j-1]) / (dx**2)
        d2y = (phi_grid[i+1, j] - 2*phi_grid[i,j] + phi_grid[i-1, j]) / (dy**2)
        lap_phi[i,j] = d2x + d2y

# Compute Ricci-like curvature: For small phi, R ~ -(n-1)*2*lap_phi
R_grid = -(n - 1) * 2.0 * lap_phi

# ------------------------------------------------------------------------------
# 4. Define a Morse-like Free Energy F from Ricci curvature
# ------------------------------------------------------------------------------
baseline = 1.0
gamma = 1.0
F_grid = baseline - gamma * R_grid  # F = baseline - gamma*R

# ------------------------------------------------------------------------------
# 5. Plot Only the Final 3D Surface (X, Y, F)
# ------------------------------------------------------------------------------
# Mask the grid so that points outside the triangle (NaN values) are not plotted.
F_masked = np.ma.masked_invalid(F_grid)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, F_masked, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('X (k-manifold axis)')
ax.set_ylabel('Y (manifold axis)')
ax.set_zlabel('Free Energy F')
ax.set_title('Oxi-Shapes: 3D Deformation of k-Triangle via Morse-like Energy')
plt.colorbar(surf, shrink=0.5, aspect=10, label='F Value')

# Optionally, mark the original corner points
corner_F = []
for (cx, cy) in corners:
    # Find nearest grid indices
    j = int(round((cx - x_vals[0]) / dx))
    i = int(round((cy - y_vals[0]) / dy))
    corner_F.append(F_grid[i, j])
corners_F = np.array(corner_F)
ax.scatter(corners[:,0], corners[:,1], corners_F, color='r', s=80, label='Original i-states')
ax.legend()
plt.savefig("/content/oxishape.png", dpi=300)
plt.show()
