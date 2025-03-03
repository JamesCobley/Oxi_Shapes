import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as tri
from scipy.spatial import Delaunay
import io

# ------------------------------
# 1) Diamond Domain & Points
# ------------------------------

def in_diamond(x, y):
    """Return True if (x, y) is inside the diamond |x| + |y| <= 1."""
    return (abs(x) + abs(y)) <= 1.0

def generate_diamond_points(n_points=400):
    """
    Generate n_points random points in the diamond |x| + |y| <= 1,
    plus the 4 corner vertices.
    """
    pts = []
    # Random approach: bounding box [-1,1]x[-1,1]
    # Keep only those inside the diamond
    while len(pts) < n_points:
        x = 2.0*np.random.rand() - 1.0
        y = 2.0*np.random.rand() - 1.0
        if in_diamond(x, y):
            pts.append([x, y])
    # Add corners so the triangulation covers entire diamond
    pts.append([0.0, 1.0])   # top
    pts.append([1.0, 0.0])   # right
    pts.append([0.0, -1.0])  # bottom
    pts.append([-1.0, 0.0])  # left
    return np.array(pts)

def assign_occupancy_diamond(nodes, R, occupancy_values):
    """
    Assign occupancy to each node by mapping y -> which k-row it belongs to.
    We'll define row k in the band:
        y in [1 - 2*(k+1)/R,  1 - 2*k/R)
    so top row (k=0) near y=1, bottom row (k=R) near y=-1.
    """
    occ = np.zeros(nodes.shape[0])
    for i, (x, y) in enumerate(nodes):
        # Map y -> k
        # fraction from top: frac = (1 - y)/2
        # row = int( frac * R ) ...
        # but let's do a direct band approach
        # We'll loop over k=0..R and check if y is in that band's range
        found_k = False
        for k in range(R):
            y_top = 1.0 - 2.0*(k)/R
            y_bot = 1.0 - 2.0*(k+1)/R
            if (y <= y_top) and (y > y_bot):
                occ[i] = occupancy_values[k]
                found_k = True
                break
        # If not found in [0..R-1], it must be row R
        if not found_k:
            occ[i] = occupancy_values[R]
    return occ

# ------------------------------
# 2) FEM Assembly & PDE Solve
# ------------------------------

def fem_assemble_matrices(nodes, elements):
    """
    Assemble the FEM stiffness (A) and mass (M) matrices using linear triangular elements.
    """
    num_nodes = nodes.shape[0]
    A_mat = sp.lil_matrix((num_nodes, num_nodes))
    M_mat = sp.lil_matrix((num_nodes, num_nodes))
    
    for elem in elements:
        idx = elem
        coords = nodes[idx]
        mat = np.array([[1, coords[0, 0], coords[0, 1]],
                        [1, coords[1, 0], coords[1, 1]],
                        [1, coords[2, 0], coords[2, 1]]])
        area = 0.5 * abs(np.linalg.det(mat))
        if area < 1e-14:
            continue
        x = coords[:, 0]
        y = coords[:, 1]
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

def solve_pde(nodes, elements, occupancy, max_iter=150, tol=1e-2, damping=0.1, kappa_target=1.0, num_steps=5):
    """
    Solve the nonlinear PDE using a damped Newton–Raphson with continuation in kappa.
    PDE: A·φ + 0.5 * M*(occupancy * exp(2φ)) = 0
    """
    A_mat, M_mat = fem_assemble_matrices(nodes, elements)
    phi = np.zeros(nodes.shape[0])
    kappa_values = np.linspace(0, kappa_target, num_steps+1)
    convergence_info = []
    
    for kappa in kappa_values[1:]:
        step_converged = False
        for it in range(max_iter):
            nonlin = occupancy * np.exp(2*phi)
            F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
            # Jacobian
            J = A_mat + M_mat.dot(sp.diags(occupancy * np.exp(2*phi)))
            delta_phi = spla.spsolve(J, -F)
            phi += damping * delta_phi
            if np.linalg.norm(delta_phi) < tol:
                convergence_info.append(f"Converged at kappa={kappa:.3f}, iter={it+1}")
                step_converged = True
                break
        if not step_converged:
            convergence_info.append(f"Did NOT converge at kappa={kappa:.3f}")
    return phi, A_mat, M_mat, convergence_info

def compute_ricci(phi, A_mat, M_mat):
    """
    Approximate Ricci: R = -2 exp(-2φ) Δφ
    with Δφ from a lumped mass approach.
    """
    M_lumped = np.array(M_mat.sum(axis=1)).flatten()
    lap_phi = A_mat.dot(phi) / M_lumped
    R = -2.0 * np.exp(-2*phi) * lap_phi
    return R

# ------------------------------
# 3) Streamlit Application
# ------------------------------

def run_app():
    st.title("Oxi-Shapes in a 2D Diamond Domain")
    st.write("""
    This version uses a finer 2D mesh inside the diamond |x| + |y| ≤ 1.
    Each row k is assigned a fractional occupancy, creating a piecewise-constant 
    function across horizontal 'bands'. We then solve the PDE and compute the 
    Ricci curvature on a more refined triangulation.
    """)
    
    # Inputs
    R = st.number_input("Number of cysteines (R)", min_value=1, max_value=10, value=4, step=1)
    st.write("Enter fractional occupancy for each k in [0..R]:")
    occ_vals = []
    for k in range(R+1):
        occ = st.number_input(f"Occupancy (k={k})", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        occ_vals.append(occ)
    occ_vals = np.array(occ_vals)
    
    # PDE parameters
    st.subheader("PDE & Newton Settings")
    n_points = st.slider("Number of random points in diamond", 100, 3000, 600, step=100)
    max_iter = st.number_input("Max Newton iterations", min_value=10, max_value=1000, value=150)
    tol = st.number_input("Newton tolerance", min_value=1e-4, max_value=1e-1, value=1e-2, format="%.4f")
    damping = st.number_input("Damping factor", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
    kappa_target = st.number_input("kappa_target", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    num_steps = st.number_input("Continuation steps", min_value=1, max_value=20, value=5, step=1)
    
    if st.button("Compute Oxi-Shape"):
        with st.spinner("Generating points & solving PDE..."):
            # 1) Generate random points in diamond
            nodes = generate_diamond_points(n_points)
            # 2) Assign occupancy in horizontal bands
            occupancy = assign_occupancy_diamond(nodes, R, occ_vals)
            # 3) Triangulate
            elements = Delaunay(nodes).simplices
            # 4) Solve PDE
            phi, A_mat, M_mat, conv_info = solve_pde(
                nodes, elements, occupancy,
                max_iter=int(max_iter),
                tol=tol,
                damping=damping,
                kappa_target=kappa_target,
                num_steps=int(num_steps)
            )
            # 5) Ricci curvature
            R_curv = compute_ricci(phi, A_mat, M_mat)
            z = phi - occupancy  # vertical displacement
            
        # Display convergence info
        st.subheader("Convergence Info")
        for msg in conv_info:
            st.write(msg)
        
        # 6) Plot
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")
        
        triang_plot = tri.Triangulation(nodes[:,0], nodes[:,1], elements)
        # Average curvature per triangle
        tvals = np.mean(R_curv[triang_plot.triangles], axis=1)
        # Map to [0,1]
        tnorm = (tvals - tvals.min()) / (tvals.max() - tvals.min() + 1e-12)
        facecolors = plt.cm.viridis(tnorm)
        
        surf = ax.plot_trisurf(triang_plot, z, cmap="viridis",
                               edgecolor="none", alpha=0.9, facecolors=facecolors)
        ax.set_title("Oxi-Shape in a 2D Diamond")
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("z = phi - occupancy")
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap="viridis")
        sm.set_array(tvals)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, label="Ricci Curvature")
        
        st.pyplot(fig)
        
        # Download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button("Download (300 dpi PNG)", data=buf, file_name="oxishape_diamond.png", mime="image/png")

# Actually run the app
if __name__ == "__main__":
    run_app()
