# ============================================================================
# 1. Load packages
# ============================================================================

using LinearAlgebra
using SparseArrays
using GeometryBasics
using Graphs
using Statistics: mean
using Flux
using Flux: Dense, relu

# ============================================================================
# Part A: Define the Geometry 
# ============================================================================

function lift_to_z_plane(rho::Vector{Float32}, pf_states, flat_pos)
    return [Point3(Float32(flat_pos[s][1]), Float32(flat_pos[s][2]), -rho[i]) for (i, s) in enumerate(pf_states)]
end

function compute_R(points3D, flat_pos, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    R = zeros(Float32, length(pf_states))
    for (i, s) in enumerate(pf_states)
        p0, p3 = flat_pos[s], points3D[i]
        neighbors = [v for (u, v) in edges if u == s] ∪ [u for (u, v) in edges if v == s]
        for n in neighbors
            j, q0, q3 = idx[n], flat_pos[n], points3D[idx[n]]
            d0 = norm(Float32[p0[1] - q0[1], p0[2] - q0[2]])
            d3 = norm(p3 - q3)
            R[i] += d3 - d0
        end
    end
    return R
end

function compute_c_ricci_dirichlet(R::Vector{Float32}, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    C_R = zeros(Float32, length(pf_states))
    for (i, s) in enumerate(pf_states)
        neighbors = [v for (u, v) in edges if u == s] ∪ [u for (u, v) in edges if v == s]
        for n in neighbors
            j = idx[n]
            C_R[i] += (R[i] - R[j])^2
        end
    end
    return C_R
end

function compute_anisotropy(C_R_vals::Vector{Float32}, pf_states, flat_pos, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = Dict(s => Int[] for s in pf_states)
    for (u, v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end
    anisotropy = zeros(Float32, length(pf_states))
    for (i, s) in enumerate(pf_states)
        grad_vals = Float32[
            abs(C_R_vals[i] - C_R_vals[j]) / norm(Float32[
                flat_pos[s][1] - flat_pos[pf_states[j]][1],
                flat_pos[s][2] - flat_pos[pf_states[j]][2]
            ]) for j in neighbor_indices[s] if norm(Float32[
                flat_pos[s][1] - flat_pos[pf_states[j]][1],
                flat_pos[s][2] - flat_pos[pf_states[j]][2]
            ]) > 1e-6
        ]
        anisotropy[i] = isempty(grad_vals) ? 0.0f0 : mean(grad_vals)
    end
    return anisotropy
end

function initialize_sheaf_stalks(flat_pos, pf_states)
    Dict(s => [Float32(flat_pos[s][1]), Float32(flat_pos[s][2])] for s in pf_states)
end

function sheaf_consistency(stalks, edges; threshold=2.5f0)
    [(u, v, norm(stalks[u] .- stalks[v])) for (u, v) in edges if norm(stalks[u] .- stalks[v]) > threshold]
end

struct GeoGraphStruct
    pf_states::Vector{String}
    flat_pos::Dict{String, Tuple{Float64, Float64}}
    edges::Vector{Tuple{String, String}}
    idx_map::Dict{String, Int}
    points3D::Vector{Point3}
    R_vals::Vector{Float32}
    C_R_vals::Vector{Float32}
    anisotropy::Vector{Float32}
    adjacency::Matrix{Float32}
    stalks::Dict{String, Vector{Float32}}
    sheaf_inconsistencies::Vector{Tuple{String, String, Float32}}
end

function build_GeoGraphStruct(ρ::Vector{Float32}, pf_states, flat_pos, edges)
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

function update_geometry_from_rho(ρ::Vector{Float32}, geo::GeoGraphStruct)
    pf_states = geo.pf_states
    flat_pos = geo.flat_pos
    edges = geo.edges

    points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
    R_vals = compute_R(points3D, flat_pos, pf_states, edges)
    C_R_vals = compute_c_ricci_dirichlet(R_vals, pf_states, edges)
    anisotropy_vals = compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)

    return points3D, R_vals, C_R_vals, anisotropy_vals
end

# === Initialize and test ===
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

# ============================================================================
# Part B: Custom GNN for Predicting Updated Occupancy from Curvature
# ============================================================================

# === Define the GNN model ===
geo_brain_model = Chain(
    Dense(1, 16, relu),   # Input: scalar curvature → hidden layer
    Dense(16, 1)          # Output: predicted occupancy (per node)
)

function GNN_update_custom(ρ_t::Vector{Float32}, model, geo::GeoGraphStruct)
    _, _, C_R_vals, _ = update_geometry_from_rho(ρ_t, geo)
    x_input = transpose(reshape(Float32.(C_R_vals), :, 1))  # (1, 8)

    ρ_pred = model(x_input) |> relu
    ρ_vec = vec(ρ_pred)

    total = sum(ρ_vec)
    if total == 0.0f0 || isnan(total)
        ρ_vec .= 1.0f0 / length(ρ_vec)  # fallback to uniform
    else
        ρ_vec ./= total
    end

    return ρ_vec
end

# ============================================================================
# Part C: oxi_shapes_alive!
# ============================================================================

function compute_entropy_cost(i::Int, j::Int, C_R_vals::Vector{Float32}, pf_states::Vector{String})
    baseline_DeltaE = 1.0f0
    mass_heat = 0.1f0
    reaction_heat = 0.01f0 * baseline_DeltaE
    conformational_cost = abs(C_R_vals[j])

    degeneracy_map = Dict(0 => 1, 1 => 3, 2 => 3, 3 => 1)
    deg = degeneracy_map[count(c -> c == '1', pf_states[j])]
    degeneracy_penalty = 1.0f0 / deg

    return mass_heat + reaction_heat + conformational_cost + degeneracy_penalty
end

function oxi_shapes_alive!(
    ρ::Vector{Float32},
    pf_states::Vector{String},
    flat_pos::Dict{String, Tuple{Float64, Float64}},
    edges::Vector{Tuple{String, String}};
    max_moves::Int = 10
)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = Dict(s => Int[] for s in pf_states)
    for (u, v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end

    # Step 1: Integer molecule approximation
    counts = round.(Int, ρ .* 100)
    counts[end] = 100 - sum(counts[1:end-1])
    ρ .= Float32.(counts) ./ 100

    # Step 2: Update curvature
    points3D, R_vals, C_R_vals, anisotropy_vals = update_geometry_from_rho(ρ, build_GeoGraphStruct(ρ, pf_states, flat_pos, edges))

    inflow = zeros(Float32, length(pf_states))
    outflow = zeros(Float32, length(pf_states))

    total_moves = rand(0:max_moves)
    candidate_indices = findall(x -> x > 0, counts)

    for _ in 1:total_moves
        isempty(candidate_indices) && break
        i = rand(candidate_indices)
        s = pf_states[i]
        nbrs = neighbor_indices[s]
        isempty(nbrs) && continue

        # Build transition probabilities
        probs = Float32[]
        for j in nbrs
            ΔS = compute_entropy_cost(i, j, C_R_vals, pf_states)
            Δf = exp(ρ[i]) - C_R_vals[i] + ΔS
            p = exp(-Δf) * exp(-anisotropy_vals[j])
            push!(probs, p)
        end

        if sum(probs) < 1e-8f0
            inflow[i] += 0.01f0
            continue
        end

        probs ./= sum(probs)
        chosen = sample(nbrs, Weights(probs))
        inflow[chosen] += 0.01f0
        outflow[i] += 0.01f0
    end

    for i in eachindex(pf_states)
        inflow[i] += (counts[i] / 100.0f0) - outflow[i]
    end

    # Step 3: Clamp, normalize, threshold
    inflow .= max.(inflow, 0.0f0)
    if sum(inflow) > 0
        inflow ./= sum(inflow)
    end

    for i in eachindex(inflow)
        if inflow[i] > 0.0f0 && inflow[i] < 0.01f0
            inflow[i] = 0.01f0
        end
    end

    inflow .= max.(inflow, 0.0f0)
    inflow ./= sum(inflow)
    ρ .= inflow
end

# ============================================================================
# Part D: Recurrent GNN Rollout (Pure model-based prediction)
# ============================================================================

"""
    GNN_recurrent_step!(ρ_t, model, geo)

Perform a single step GNN update:
- Uses geometry-derived curvature from `ρ_t`
- Predicts next state `ρ̂ₜ₊₁`
"""
function GNN_recurrent_step!(ρ_t::Vector{Float32}, model, geo::GeoGraphStruct)
    _, _, C_R_vals, _ = update_geometry_from_rho(ρ_t, geo)
    x_input = transpose(reshape(C_R_vals, :, 1))  # (1, 8)

    ρ_pred = model(x_input) |> relu
    ρ_vec = vec(ρ_pred)

    total = sum(ρ_vec)
    if total == 0.0f0 || isnan(total)
        ρ_vec .= 1.0f0 / length(ρ_vec)  # fallback to uniform
    else
        ρ_vec ./= total
    end

    ρ_t .= ρ_vec  # Update in-place
end

"""
    rollout_GNN(ρ0, model, geo, T)

Run pure GNN rollout for T steps starting from `ρ0`
Returns matrix of shape (T+1, N)
"""
function rollout_GNN(ρ0::Vector{Float32}, model, geo::GeoGraphStruct, T::Int)
    N = length(ρ0)
    history = zeros(Float32, T+1, N)
    ρ_t = copy(ρ0)
    history[1, :] .= ρ_t

    for t in 1:T
        GNN_recurrent_step!(ρ_t, model, geo)
        history[t+1, :] .= ρ_t
    end

    return history
end
