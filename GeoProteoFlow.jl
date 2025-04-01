# --- Device Setup ---
using Flux
using CUDA
using Meshes
using GeometryBasics
using LinearAlgebra
using StatsBase

if CUDA.has_cuda()
    device = gpu
    println("Using device: GPU")
else
    device = cpu
    println("Using device: CPU")
end

# --- Differentiable Lambda: Trainable scaling for c-Ricci ---
# We store log(λ) for stability
log_lambda = param([log(1.0f0)])  # Flux param makes it trainable

# Accessor function
function get_lambda()
    return exp(log_lambda[1])  # scalar λ value
end

# --- Global RT constant ---
const RT = 1.0f0

################################################################################
# Step 1: Define the flat 2D proteoform lattice with ρ(x) -> z(x)
################################################################################

# Define the proteoform states and their flat (x, y) positions
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
flat_pos = Dict(
    "000" => Point3(0.0, 3.0, 0.0),
    "001" => Point3(-2.0, 2.0, 0.0),
    "010" => Point3(0.0, 2.0, 0.0),
    "100" => Point3(2.0, 2.0, 0.0),
    "011" => Point3(-1.0, 1.0, 0.0),
    "101" => Point3(0.0, 1.0, 0.0),
    "110" => Point3(1.0, 1.0, 0.0),
    "111" => Point3(0.0, 0.0, 0.0)
)

# Define neighbor connections (edges by Hamming distance 1)
allowed_edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "101"), ("001", "011"),
    ("010", "110"), ("010", "011"),
    ("011", "111"),
    ("100", "110"), ("100", "101"),
    ("101", "111"), ("110", "111")
]

state_index = Dict(s => i for (i, s) in enumerate(pf_states))
num_states = length(pf_states)

################################################################################
# Step 2: Define occupancy ρ(x) and build curved 3D mesh with z = ρ(x)
################################################################################

function lift_to_3d_mesh(rho::Vector{Float64})
    @assert length(rho) == num_states
    rho = rho ./ sum(rho)  # Ensure volume conservation
    lifted_points = [Point3(flat_pos[s].x, flat_pos[s].y, rho[state_index[s]]) for s in pf_states]

    # Manually define triangles for the R=3 diamond based on connectivity
    triangles = [
        Triangle(1, 2, 3), Triangle(1, 3, 4), Triangle(2, 5, 3),
        Triangle(3, 5, 6), Triangle(3, 6, 7), Triangle(4, 3, 7),
        Triangle(5, 8, 6), Triangle(6, 8, 7)
    ]

    mesh = SimpleMesh(lifted_points, triangles)
    return mesh
end

################################################################################
# Step 3: Compute cotangent Laplacian ∆_T over 3D mesh
################################################################################

function cotangent_laplacian(mesh::SimpleMesh)
    N = length(mesh.points)
    L = zeros(Float64, N, N)
    A = zeros(Float64, N)

    for tri in mesh.connectivity
        i, j, k = tri.indices
        p1, p2, p3 = mesh.points[i], mesh.points[j], mesh.points[k]

        # Edges
        u = p2 - p1
        v = p3 - p1
        w = p3 - p2

        # Angles
        angle_i = acos(clamp(dot(u, v) / (norm(u) * norm(v)), -1, 1))
        angle_j = acos(clamp(dot(-u, w) / (norm(u) * norm(w)), -1, 1))
        angle_k = acos(clamp(dot(-v, -w) / (norm(v) * norm(w)), -1, 1))

        # Cotangent weights
        cot_i = 1 / tan(angle_i)
        cot_j = 1 / tan(angle_j)
        cot_k = 1 / tan(angle_k)

        for (a, b, cot) in ((j,k,cot_i), (i,k,cot_j), (i,j,cot_k))
            L[a,b] -= cot
            L[b,a] -= cot
            L[a,a] += cot
            L[b,b] += cot
        end
    end

    return L
end

################################################################################
# Step 4: Compute c-Ricci = λ ⋅ ∆_T ρ(x)
################################################################################

function compute_c_ricci(rho::Vector{Float64}, λ::Float64 = 1.0)
    mesh = lift_to_3d_mesh(rho)
    L = cotangent_laplacian(mesh)
    rho_norm = rho ./ sum(rho)  # Ensure volume is normalized
    c_ricci = λ .* (L * rho_norm)
    return c_ricci
end

# Example run:
rho_example = rand(num_states)
c_ricci_out = compute_c_ricci(rho_example, 1.0)
println("c-Ricci curvature:", round.(c_ricci_out; digits=4))

################################################################################
# Step 5: Compute anisotropy field A(x) = ∇_T [C-Ricci(x)]
################################################################################

function compute_anisotropy(mesh::SimpleMesh, c_ricci::Vector{Float64})
    A_field = zeros(Float64, num_states)

    for i in 1:num_states
        p_i = mesh.points[i]
        grad_vals = Float64[]
        for j in 1:num_states
            if i == j
                continue
            end
            p_j = mesh.points[j]
            d = norm(p_j - p_i)
            if d > 1e-6
                push!(grad_vals, abs(c_ricci[i] - c_ricci[j]) / d)
            end
        end
        A_field[i] = isempty(grad_vals) ? 0.0 : mean(grad_vals)
    end
    return A_field
end
