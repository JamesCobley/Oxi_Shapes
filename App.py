import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as tri
from scipy.spatial import Delaunay
import io

# ------------------------------
# 1) Generate a Structured Grid in a Diamond
# ------------------------------

def generate_diamond_grid(num_points_per_axis=50):
    """
    Generate a structured grid of points in the square [-1,1]x[-1,1]
    and then filter to keep only those inside the diamond: |x|+|y| <= 1.
    """
    x_vals = np.linspace(-1, 1, num_points_per_axis)
    y_vals = np.linspace(-1, 1, num_points_per_axis)
    xx, yy = np.meshgrid(x_vals, y_vals)
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    # Filter: keep only points with |x|+|y| <= 1
    diamond_pts = pts[np.abs(pts[:, 0]) + np.abs(pts[:, 1]) <= 1.0]
    return diamond_pts

# ------------------------------
# 2) Assign Occupancy to Nodes Based on k-Domains
# ------------------------------

def assign_occupancy_diamond(nodes, R, occupancy_values):
    """
    Assign occupancy to each node by determining which horizontal band (k-domain)
    the node belongs to. We define row k by:
         y in [1 - 2*(k+1)/R,  1 - 2*k/R)
    so that k = 0 corresponds to near the top (y ≈ 1) and k = R to the bottom (y ≈ -1).
    """
    occ = np.zeros(nodes.shape[0])
    for i, (x, y) in enumerate(nodes):
        found = False
        for k in range(R):
            y_top = 1.0 - 2.0 * (k) / R
            y_bot = 1.0 - 2.0 * (k + 1) / R
            if (y <= y_top) and (y > y_bot):
                occ[i] = occupancy_values[k]
                found = True
                break
        if not found:
            occ[i] = occupancy_values[R]
    return occ

# ------------------------------
# 3) FEM Assembly & PDE Solve Functions
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
    Solve the nonlinear PDE using a damped Newton–Raphson method with continuation in kappa.
    PDE: A·φ + 0.5 * M*(occupancy * exp(2φ)) = 0
    Returns: φ, A_mat, M_mat, and a list of convergence messages.
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
            # Jacobian: A + M * diag(occupancy * exp(2φ))
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
    Approximate Ricci curvature: R = -2 exp(-2φ) Δφ,
    where Δφ is approximated using a lumped mass matrix.
    """
    M_lumped = np.array(M_mat.sum(axis=1)).flatten()
    lap_phi = A_mat.dot(phi) / M_lumped
    R = -2.0 * np.exp(-2*phi) * lap_phi
    return R

def nearest_node_value(x, y, coords, values):
    """
    Find the value at the node in coords that is closest to (x, y).
    """
    d = np.sum((coords - np.array([x, y]))**2, axis=1)
    i_min = np.argmin(d)
    return values[i_min]

def generate_oxishape(R, occupancy_values, grid_res, newton_params):
    """
    Generate the Oxi‑Shape for a protein with R cysteines (hence R+1 k‑states)
    using a structured grid over the diamond.
    newton_params is a dictionary with keys: max_iter, tol, damping, kappa_target, num_steps.
    """
    # 1) Generate a structured grid in the diamond
    nodes = generate_diamond_grid(num_points_per_axis=grid_res)
    # 2) Assign occupancy based on horizontal bands (k-domains)
    occupancy = assign_occupancy_diamond(nodes, R, occupancy_values)
    # 3) Triangulate the grid nodes
    elements = Delaunay(nodes).simplices
    # 4) Solve PDE
    phi, A_mat, M_mat, conv_info = solve_pde(
        nodes, elements, occupancy,
        max_iter=newton_params["max_iter"],
        tol=newton_params["tol"],
        damping=newton_params["damping"],
        kappa_target=newton_params["kappa_target"],
        num_steps=newton_params["num_steps"]
    )
    R_curv = compute_ricci(phi, A_mat, M_mat)
    z = phi - occupancy  # vertical displacement
    return nodes, elements, z, R_curv, conv_info

# ------------------------------
# 4) Streamlit Application
# ------------------------------

def run_app():
    st.title("Oxi‑Shapes Generator (Structured Diamond Domain)")
    st.write("""
    This app generates Oxi‑Shapes for cysteine proteoforms using a theorem‑derived field equation.
    Instead of random points, we use a structured grid over the diamond \(|x| + |y| \le 1\), 
    so that the k‑domains (horizontal bands) are in predictable positions.
    """)
    
    # Protein and occupancy inputs
    R = st.number_input("Number of cysteines (R)", min_value=1, max_value=10, value=4, step=1)
    st.write("Enter fractional occupancy for each k‑state (from k = 0 to k = R):")
    occ_vals = []
    for k in range(int(R) + 1):
        occ = st.number_input(f"Occupancy (k={k})", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key=f"occ_{k}")
        occ_vals.append(occ)
    occ_vals = np.array(occ_vals)
    
    # Grid resolution (structured)
    grid_res = st.slider("Grid resolution (points per axis)", min_value=20, max_value=200, value=50, step=5)
    
    # PDE & Newton parameters
    st.subheader("PDE & Newton Settings")
    max_iter = st.number_input("Max Newton iterations", min_value=10, max_value=1000, value=150)
    tol = st.number_input("Newton tolerance", min_value=1e-4, max_value=1e-1, value=1e-2, format="%.4f")
    damping = st.number_input("Damping factor", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
    kappa_target = st.number_input("kappa_target", min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f")
    num_steps = st.number_input("Continuation steps", min_value=1, max_value=20, value=5, step=1)
    
    newton_params = {
        "max_iter": int(max_iter),
        "tol": tol,
        "damping": damping,
        "kappa_target": kappa_target,
        "num_steps": int(num_steps)
    }
    
    if st.button("Generate Oxi‑Shape"):
        with st.spinner("Generating grid & solving PDE..."):
            nodes, elements, z, R_curv, conv_info = generate_oxishape(int(R), occ_vals, grid_res, newton_params)
        # Display convergence info
        st.subheader("Newton Convergence Information")
        for msg in conv_info:
            st.write(msg)
        
        # Plot the results in 3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        
        triang_plot = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
        # Compute average Ricci curvature per triangle for face coloring
        tvals = np.mean(R_curv[triang_plot.triangles], axis=1)
        tnorm = (tvals - tvals.min()) / (tvals.max() - tvals.min() + 1e-12)
        facecolors = plt.cm.viridis(tnorm)
        
        surf = ax.plot_trisurf(triang_plot, z, cmap="viridis",
                               edgecolor="none", alpha=0.9, facecolors=facecolors)
        ax.set_title("Oxi‑Shape (Structured Diamond)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("z = φ - occupancy")
        sm = plt.cm.ScalarMappable(cmap="viridis")
        sm.set_array(tvals)
        fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, label="Ricci Curvature")
        
        # Label each k-domain.
        # We define each horizontal band based on y: for row k, band is [1-2*(k+1)/R, 1-2*k/R)
        for k_label in range(int(R)+1):
            y_top = 1.0 - 2.0*(k_label)/R
            y_bot = 1.0 - 2.0*(k_label+1)/R
            y_mid = (y_top + y_bot)/2.0
            # Find z value near (0, y_mid) using nearest neighbor search in the grid
            z_mid = nearest_node_value(0.0, y_mid, nodes, z)
            ax.text(0.0, y_mid, z_mid, f"k={k_label}", color="red", fontsize=10, weight="bold")
        
        st.pyplot(fig)
        
        # Provide download option for a 300 dpi PNG image.
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button("Download Image (300 dpi PNG)", data=buf,
                           file_name="oxishape_structured.png", mime="image/png")

if __name__ == "__main__":
    run_app()
