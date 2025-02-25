import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# --------------------------
# 1. Discrete Data: For a 2-cysteine protein:
# k-manifolds: k = 0, 1, 2
# Occupancy (volume weight): 30%, 30%, 40%
# Oxidation state (for R=2): 0%, 50%, 100%
# --------------------------
k_data = np.array([0, 1, 2], dtype=float)
occ_data = np.array([0.30, 0.30, 0.40])
ox_data = k_data * (100/2)  # [0, 50, 100]

# --------------------------
# 2. Compute a discrete "curvature" approximation
# Use finite differences to approximate the second derivative of occupancy.
dx = 1.0
first_deriv = np.gradient(occ_data, dx)
second_deriv = np.gradient(first_deriv, dx)
n = 3  # dimension for our analogy
alpha = 1.0  # scaling constant for occupancy-to-metric
# Approximate Ricci curvature: R ~ -(n-1)*alpha*(d²(occ)/dk²)
R_data = -(n - 1) * alpha * second_deriv

# --------------------------
# 3. Define a simple free energy potential F at each discrete point:
# F = ΔE * e^(α * occupancy) * Δx - β * R
# For simplicity, assume ΔE = 1, Δx = 1, β = 1.
Delta_E = 1.0
beta = 1.0
F_data = Delta_E * np.exp(alpha * occ_data) * dx - beta * R_data

print("Discrete F values:", F_data)

# --------------------------
# 4. Interpolate the discrete data to form a smooth curve.
# We interpolate k, occupancy, oxidation, and F.
t_data = np.linspace(0, 1, len(k_data))
t_fine = np.linspace(0, 1, 100)

interp_k = interp1d(t_data, k_data, kind='quadratic')
interp_occ = interp1d(t_data, occ_data, kind='quadratic')
interp_ox = interp1d(t_data, ox_data, kind='quadratic')
interp_F = interp1d(t_data, F_data, kind='quadratic')

k_fine = interp_k(t_fine)
occ_fine = interp_occ(t_fine)
ox_fine = interp_ox(t_fine)
F_fine = interp_F(t_fine)

# For visualization, define our 3D curve as before:
# x-axis: k_fine, y-axis: ox_fine, z-axis: R_fine (we also interpolate R_data)
interp_R = interp1d(t_data, R_data, kind='quadratic')
R_fine = interp_R(t_fine)

curve = np.vstack((k_fine, ox_fine, R_fine)).T

# --------------------------
# 5. Compute Tangents and Normals for Tube Extrusion
# --------------------------
tangents = np.gradient(curve, axis=0)
tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]

normals = []
binormals = []
for T in tangents:
    arbitrary = np.array([0, 0, 1])
    if np.allclose(T, arbitrary, atol=1e-3):
        arbitrary = np.array([0, 1, 0])
    N = np.cross(T, arbitrary)
    N = N / np.linalg.norm(N)
    B = np.cross(T, N)
    B = B / np.linalg.norm(B)
    normals.append(N)
    binormals.append(B)
normals = np.array(normals)
binormals = np.array(binormals)

# --------------------------
# 6. Use the Free Energy Potential F_fine to modulate the tube's radius.
# For instance, let tube radius = scale * (1 / (1 + F)) or similar.
# Here we choose a modulation so that lower free energy (stable) gives a larger radius.
base_radius = 0.2
gamma = 1.0  # modulation strength
radii = base_radius * (1 + gamma / (1 + F_fine))  # adjust as desired

# --------------------------
# 7. Create the tube: for each point along the curve, create a circle in the plane defined by the normal and binormal.
theta = np.linspace(0, 2*np.pi, 20)
tube_x = []
tube_y = []
tube_z = []

for i in range(len(curve)):
    center = curve[i]
    N = normals[i]
    B = binormals[i]
    r = radii[i]
    circle_points = np.array([center + r * (np.cos(t)*N + np.sin(t)*B) for t in theta])
    tube_x.append(circle_points[:, 0])
    tube_y.append(circle_points[:, 1])
    tube_z.append(circle_points[:, 2])

tube_x = np.array(tube_x)
tube_y = np.array(tube_y)
tube_z = np.array(tube_z)

# --------------------------
# 8. Plot the 3D tube with volume
# --------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(tube_x, tube_y, tube_z, facecolors=plt.cm.viridis(tube_z / np.max(tube_z)), rstride=1, cstride=1, alpha=0.8, edgecolor='none')
ax.plot(k_fine, ox_fine, R_fine, 'r-', linewidth=2, label='Centerline')
ax.set_xlabel('k-Manifold (bit-flip state)')
ax.set_ylabel('Oxidation (%)')
ax.set_zlabel('Ricci Curvature')
ax.set_title('Oxi-Shapes: 3D Tube with Free Energy Modulation')
ax.legend()
plt.savefig("/content/oxishape.png", dpi=300)
plt.show()
