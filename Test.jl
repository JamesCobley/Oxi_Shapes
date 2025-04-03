# ╔═╡ Install required packages (only needs to run once per session)
using Pkg
Pkg.activate(".")  # Optional: activate project environment
Pkg.add(["Flux", "CUDA", "Meshes", "GeometryBasics", "LinearAlgebra",
         "StatsBase", "DifferentialEquations", "Ripserer",
         "Distances", "Makie"])

# --- Imports ---
using Flux
using CUDA
using LinearAlgebra
using StatsBase
using GeometryBasics: Point3

# === Lambda parameter (trainable) ===
λ_container = Ref(log(1.0f0))         # log(λ) for stable optimization
ps_lambda = Flux.Params([λ_container])

function get_lambda()
    return Float64(exp(λ_container[]))  # Convert back to λ
end

# === Proteoform state lattice (Pascal diamond) ===
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]

# Coordinates in (x, y) for each i-state
flat_pos = Dict(
    "000" => Point3( 0.0, 3.0, 0.0),
    "001" => Point3(-2.0, 2.0, 0.0),
    "010" => Point3( 0.0, 2.0, 0.0),
    "100" => Point3( 2.0, 2.0, 0.0),
    "011" => Point3(-1.0, 1.0, 0.0),
    "101" => Point3( 0.0, 1.0, 0.0),
    "110" => Point3( 1.0, 1.0, 0.0),
    "111" => Point3( 0.0, 0.0, 0.0)
)

state_index = Dict(s => i for (i, s) in enumerate(pf_states))
num_states = length(pf_states)

# Allowed edges (bitwise Hamming-1 transitions)
allowed_edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "011"), ("001", "101"),
    ("010", "011"), ("010", "110"),
    ("100", "110"), ("100", "101"),
    ("011", "111"), ("101", "111"), ("110", "111")
]

# === Graph Laplacian constructor ===
function build_graph_laplacian(pf_states, allowed_edges)
    n = length(pf_states)
    idx_map = Dict(s => i for (i, s) in enumerate(pf_states))
    L = zeros(Float64, n, n)

    for (u, v) in allowed_edges
        i, j = idx_map[u], idx_map[v]
        L[i, j] = -1
        L[j, i] = -1
        L[i, i] += 1
        L[j, j] += 1
    end

    return L  # row-sum is zero → volume conserving
end

# === Ricci curvature computation ===
function compute_c_ricci(rho::Vector{Float64}; λ::Float64 = get_lambda())
    @assert abs(sum(rho) - 1.0) < 1e-6 "ρ(x) must be normalized for volume conservation"
    L = build_graph_laplacian(pf_states, allowed_edges)
    return λ .* (L * rho)
end

# === Example usage ===
rho_example = rand(num_states)
rho_example ./= sum(rho_example)  # normalize for volume conservation

c_ricci = compute_c_ricci(rho_example)
println("✅ c-Ricci curvature (volume-conserving): ", round.(c_ricci; digits=4))
