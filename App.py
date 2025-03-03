import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from matplotlib.path import Path
import io

# ------------------------------
# 1. Define the Polygon Domain for R Cysteines (R+1 k-states)
# ------------------------------

def regular_polygon_vertices(R):
    """
    For a protein with R cysteines, there are R+1 k-states.
    Generate the vertices of a regular (R+1)-gon centered at (0.5, 0.5)
    with a given radius.
    """
    n_vertices = R + 1
    center = np.array([0.5, 0.5])
    radius = 0.5
    # Start at the top and go around
    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False) + np.pi/2
    vertices = np.column_stack((center[0] + radius*np.cos(angles),
                                center[1] + radius*np.sin(angles)))
    return vertices

# Generate vertex labels, e.g. K₀, K₁, ..., K_R
def generate_vertex_labels(R):
    n_labels = R + 1
    labels = []
    for k in range(n_labels):
        labels.append(f"K₍{k}₎")
    return labels

# ------------------------------
# 2. Generate Internal Points Uniformly Within the Polygon
# ------------------------------

def random_points_in_polygon(n, polygon):
    """
    Generate n random points inside the given polygon.
    """
    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    poly_path = Path(polygon)
    points = []
    while len(points) < n:
        p = np.random.rand(2) * (max_x - min_x) + [min_x, min_y]
        if poly_path.contains_point(p):
            points.append(p)
    return np.array(points)

# ------------------------------
# 3. FEM Assembly & PDE Solver Functions
# ------------------------------

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

def solve_pde(nodes, elements, occupancy, max_iter=400, tol=1e-2, damping=0.05, kappa_target=1.0, num_steps=5):
    num_nodes = nodes.shape[0]
    A_mat, M_mat = fem_assemble_matrices(nodes, elements)
    phi = np.zeros(num_nodes)
    kappa_values = np.linspace(0, kappa_target, num_steps+1)
    conv_info = []
    
    for kappa in kappa_values[1:]:
        for it in range(max_iter):
            nonlin = occupancy * np.exp(2*phi)
            F = A_mat.dot(phi) + 0.5 * M_mat.dot(nonlin)
            J = A_mat + M_mat.dot(sp.diags(occupancy * np.exp(2*phi)))
            delta_phi = spla.spsolve(J, -F)
            phi += damping * delta_phi
            if np.linalg.norm(delta_phi) < tol:
                conv_info.append(f"Converged at kappa = {kappa:.3f} in {it+1} iterations.")
                break
        else:
            conv_info.append(f"Did NOT converge at kappa = {kappa:.3f}.")
    return phi, A_mat, M_mat, conv_info

def compute_ricci(phi, A_mat, M_mat):
    M_lumped = np.array(M_mat.sum(axis=1)).flatten()
    lap_phi = A_mat.dot(phi) / M_lumped
    R = -2.0 * np.exp(-2*phi) * lap_phi
    return R

# ------------------------------
# 4. Streamlit App: Vertex Model Version
# ------------------------------

def run_app():
    st.title("Oxi‑Shapes Generator (Vertex Model)")
    st.write("""
    This app generates Oxi‑Shapes for cysteine proteoforms using a theorem‑derived field equation.
    The domain is defined by a regular polygon (a (R+1)-gon), corresponding to the k‑states (from K₀ to K_R).
    Internal points are generated uniformly within the polygon, and the PDE is solved using a damped Newton–Raphson method.
    """)
    
    # Protein and occupancy inputs
    R = st.number_input("Enter the number of cysteines (R)", min_value=1, max_value=10, value=4, step=1)
    vertex_count = int(R) + 1
    st.write(f"Enter the fractional occupancy for each k‑state (vertex, total {vertex_count}):")
    vertex_occ = []
    default_occ = [0.20, 0.50, 0.30]  # default for R=2; adjust for higher R
    for k in range(vertex_count):
        # Use 0.0 as default if not provided in default_occ
        default_val = default_occ[k] if k < len(default_occ) else 0.0
        occ = st.number_input(f"Occupancy for K₍{k}₎", min_value=0.0, max_value=1.0, value=default_val, step=0.05, key=f"vertex_occ_{k}")
        vertex_occ.append(occ)
    vertex_occ = np.array(vertex_occ)
    
    # Grid parameters
    num_internal_points = st.slider("Number of internal points", 50, 1000, 200, step=50)
    
    # PDE & Newton parameters
    st.subheader("PDE & Newton Settings")
    max_iter = st.number_input("Max Newton iterations", min_value=100, max_value=1000, value=400, step=50)
    tol = st.number_input("Newton tolerance", min_value=1e-4, max_value=1e-1, value=1e-2, format="%.4f")
    damping = st.number_input("Damping factor", min_value=0.01, max_value=1.0, value=0.05, step=0.01, format="%.2f")
    kappa_target = st.number_input("Kappa target", min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f")
    num_steps = st.number_input("Continuation steps", min_value=1, max_value=20, value=5, step=1)
    
    newton_params = {
        "max_iter": int(max_iter),
        "tol": tol,
        "damping": damping,
        "kappa_target": kappa_target,
        "num_steps": int(num_steps)
    }
    
    if st.button("Generate Oxi‑Shape"):
        with st.spinner("Generating internal points & solving PDE..."):
            # 1. Define the polygon (regular (R+1)-gon)
            vertices = regular_polygon_vertices(int(R))
            vertex_labels = generate_vertex_labels(int(R))
            
            # 2. Generate internal points uniformly within the polygon
            internal_points = random_points_in_polygon(num_internal_points, vertices)
            nodes = np.vstack((vertices, internal_points))
            
            # 3. Interpolate occupancy onto all nodes using griddata
            occupancy = griddata(vertices, vertex_occ, nodes, method='linear')
            mask = np.isnan(occupancy)
            if np.any(mask):
                occupancy[mask] = griddata(vertices, vertex_occ, nodes[mask], method='nearest')
            
            # 4. Build Delaunay triangulation over nodes
            elements = Delaunay(nodes).simplices
            
            # 5. Solve the PDE
            phi, A_mat, M_mat, conv_info = solve_pde(nodes, elements, occupancy,
                                                      max_iter=newton_params["max_iter"],
                                                      tol=newton_params["tol"],
                                                      damping=newton_params["damping"],
                                                      kappa_target=newton_params["kappa_target"],
                                                      num_steps=newton_params["num_steps"])
            R_curv = compute_ricci(phi, A_mat, M_mat)
            z = phi - occupancy
            
        # Display Newton convergence info
        st.subheader("Newton Convergence Information")
        for msg in conv_info:
            st.write(msg)
        
        # 6. Plot the deformed polygon with Ricci curvature coloring
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111, projection='3d')
        
        triang_plot = tri.Triangulation(nodes[:,0], nodes[:,1], elements)
        # For face coloring, compute average Ricci curvature for each triangle
        tri_vals = np.mean(R_curv[triang_plot.triangles], axis=1)
        norm_vals = (tri_vals - tri_vals.min()) / (tri_vals.max() - tri_vals.min() + 1e-10)
        facecolors = plt.cm.viridis(norm_vals)
        
        surf = ax.plot_trisurf(triang_plot, z, cmap='viridis', shade=True,
                                 edgecolor='none', antialiased=True, linewidth=0.2,
                                 alpha=0.9, facecolors=facecolors)
        ax.set_title("Oxi‑Shape for Protein with {} Cysteines".format(int(R)))
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("z = φ - occupancy")
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(tri_vals)
        fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, label='Ricci Curvature')
        
        # 7. Label the polygon vertices with k-state labels
        for i, label in enumerate(vertex_labels):
            xL, yL = vertices[i]
            # Interpolate to get z at the vertex
            zL = griddata(nodes, z, np.array([[xL, yL]]), method='linear')[0]
            ax.text(xL, yL, zL, f" {label}", fontsize=12, color='k', weight='bold')
        
        st.pyplot(fig)
        
        # 8. Provide a download button for the 300 dpi PNG image
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", dpi=300)
        buf.seek(0)
        st.download_button("Download Image (300 dpi SVG)", data=buf,
                           file_name="oxishape_vertex.png", mime="image/png")

if __name__ == "__main__":
    run_app()

