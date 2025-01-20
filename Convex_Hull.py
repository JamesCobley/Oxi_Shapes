import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Generate example k-space manifold data
k_states = np.linspace(0, 10, 11)  # Oxidation levels
population_sizes = np.linspace(1000, 8000, 11)  # Population size
ricci_curvature = np.linspace(0, 0.5, 11)  # Ricci curvature values

# Create 3D meshgrid
X, Y = np.meshgrid(k_states, population_sizes)
Z = np.tile(ricci_curvature, (len(population_sizes), 1))

# Convert meshgrid into point cloud for convex hull
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

# Add small perturbation to avoid collinear/singular input
points += np.random.normal(scale=0.01, size=points.shape)

# Compute Convex Hull safely
hull = ConvexHull(points)

# Plot 3D Convex Hull
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Scatter original points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="viridis", s=10, alpha=0.7)

# Plot Convex Hull Faces
for simplex in hull.simplices:
    hull_vertices = points[simplex]
    poly = Poly3DCollection([hull_vertices], color="lightblue", alpha=0.3)
    ax.add_collection3d(poly)

# Labels
ax.set_title("Convex Hull Representation of k-Space Manifold")
ax.set_xlabel("k-State (Oxidation Level)")
ax.set_ylabel("Population Size")
ax.set_zlabel("Ricci Curvature")
plt.savefig("/content/convex_geo.png", dpi=300)
plt.show()
