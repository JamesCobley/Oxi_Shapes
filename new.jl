# ============================================================================
# 1. Load packages
# ============================================================================

using LinearAlgebra
using SparseArrays
using GeometryBasics
using Graphs
using Statistics: mean

# ============================================================================
# Part A: Define the Geometry 
# ============================================================================

function lift_to_z_plane(rho, pf_states, flat_pos)
    return [Point3(flat_pos[s][1], flat_pos[s][2], -rho[i]) for (i, s) in enumerate(pf_states)]
end

function compute_R(points3D, flat_pos, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    R = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        p0, p3 = flat_pos[s], points3D[i]
        neighbors = [v for (u, v) in edges if u == s] ∪ [u for (u, v) in edges if v == s]
        for n in neighbors
            j, q0, q3 = idx[n], flat_pos[n], points3D[idx[n]]
            d0 = norm([p0[1] - q0[1], p0[2] - q0[2]])
            d3 = norm(p3 - q3)
            R[i] += d3 - d0
        end
    end
    return R
end

function compute_c_ricci_dirichlet(R, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    C_R = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        neighbors = [v for (u, v) in edges if u == s] ∪ [u for (u, v) in edges if v == s]
        for n in neighbors
            j = idx[n]
            C_R[i] += (R[i] - R[j])^2
        end
    end
    return C_R
end

function compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = Dict(s => Int[] for s in pf_states)
    for (u, v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end
    anisotropy = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        grad_vals = [
            abs(C_R_vals[i] - C_R_vals[j]) / norm([
                flat_pos[s][1] - flat_pos[pf_states[j]][1],
                flat_pos[s][2] - flat_pos[pf_states[j]][2]
            ]) for j in neighbor_indices[s] if norm([
                flat_pos[s][1] - flat_pos[pf_states[j]][1],
                flat_pos[s][2] - flat_pos[pf_states[j]][2]
            ]) > 1e-6
        ]
        anisotropy[i] = isempty(grad_vals) ? 0.0 : mean(grad_vals)
    end
    return anisotropy
end

function initialize_sheaf_stalks(flat_pos, pf_states)
    Dict(s => [flat_pos[s][1], flat_pos[s][2]] for s in pf_states)
end

function sheaf_consistency(stalks, edges; threshold=2.5)
    [(u, v, norm(stalks[u] .- stalks[v])) for (u, v) in edges if norm(stalks[u] .- stalks[v]) > threshold]
end

struct GeoGraphStruct
    pf_states::Vector{String}
    flat_pos::Dict{String, Tuple{Float64, Float64}}
    edges::Vector{Tuple{String, String}}
    idx_map::Dict{String, Int}
    points3D::Vector{Point3}
    R_vals::Vector{Float64}
    C_R_vals::Vector{Float64}
    anisotropy::Vector{Float64}
    adjacency::Matrix{Float32}
    stalks::Dict{String, Vector{Float64}}
    sheaf_inconsistencies::Vector{Tuple{String, String, Float64}}
end

function build_GeoGraphStruct(ρ::Vector{Float64}, pf_states, flat_pos, edges)
    idx_map = Dict(s => i for (i, s) in enumerate(pf_states))
    points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
    R_vals = compute_R(points3D, flat_pos, pf_states, edges)
    C_R_vals = compute_c_ricci_dirichlet(R_vals, pf_states, edges)
    anisotropy = compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)

    g = SimpleDiGraph(length(pf_states))
    for (u, v) in edges
        add_edge!(g, idx_map[u], idx_map[v])
    end
    A = Float32.(adjacency_matrix(g))

    stalks = initialize_sheaf_stalks(flat_pos, pf_states)
    inconsistencies = sheaf_consistency(stalks, edges)

    return GeoGraphStruct(
        pf_states, flat_pos, edges, idx_map,
        points3D, R_vals, C_R_vals, anisotropy, A,
        stalks, inconsistencies
    )
end

pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
flat_pos = Dict(
    "000" => (0.0, 3.0), "001" => (-2.0, 2.0), "010" => (0.0, 2.0),
    "100" => (2.0, 2.0), "011" => (-1.0, 1.0), "101" => (0.0, 1.0),
    "110" => (1.0, 1.0), "111" => (0.0, 0.0)
)
edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "011"), ("001", "101"), ("010", "011"), ("010", "110"),
    ("100", "110"), ("100", "101"), ("011", "111"), ("101", "111"), ("110", "111")
]

function update_geometry_from_rho(ρ::Vector{Float64}, geo::GeoGraphStruct)
    pf_states = geo.pf_states
    flat_pos = geo.flat_pos
    edges = geo.edges

    points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
    R_vals = compute_R(points3D, flat_pos, pf_states, edges)
    C_R_vals = compute_c_ricci_dirichlet(R_vals, pf_states, edges)
    anisotropy_vals = compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)

    return points3D, R_vals, C_R_vals, anisotropy_vals
end

# === Build initial geometry and test ===
ρ0 = fill(1.0 / length(pf_states), length(pf_states))
geo = build_GeoGraphStruct(ρ0, pf_states, flat_pos, edges)

println("✅ GeoGraphStruct successfully built.")
println("Adjacency:\n", geo.adjacency)
println("C_R_vals (initial): ", geo.C_R_vals)
println("Sheaf inconsistencies: ", geo.sheaf_inconsistencies)

# === Test an update
ρ1 = [0.2, 0.1, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1]
pts, R, C_R, aniso = update_geometry_from_rho(ρ1, geo)
println("Updated C_R_vals: ", C_R)
println("Updated Anisotropy: ", aniso)

# ============================================================================
# Part B: Custom GNN for Predicting Updated Occupancy from Curvature
# ============================================================================

# === Define the GNN model ===
geo_brain_model = Chain(
    Dense(1, 16, relu),   # Input: scalar curvature → hidden layer
    Dense(16, 1)          # Output: predicted occupancy (per node)
)

# === Define the update function using the geometry ===
function GNN_update_custom(ρ_t::Vector{Float64}, model, geo::GeoGraphStruct)
    # Recompute curvature features from current ρ
    _, _, C_R_vals, _ = update_geometry_from_rho(ρ_t, geo)

    # Reshape C_R_vals into input format for the model (8×1)
    x = reshape(Float32.(C_R_vals), :, 1)

    # Run forward pass through model
    ρ_pred = model(x) |> relu

    # Flatten and normalize
    ρ_vec = vec(ρ_pred)
    ρ_vec ./= sum(ρ_vec)

    return ρ_vec
end
