#Add installations

# Imports

using GeometryBasics: Point2, Point3
using Interpolations
using Random
using StatsBase: sample, Weights
using LinearAlgebra
using Statistics
using ComplexityMeasures
using Distributions
using Distances
using DelimitedFiles
using Flux
using BSON
using BSON: @save  
using CUDA
 
# Manifold and functions

# Set device dynamically: use GPU if available, otherwise CPU
device(x) = CUDA.functional() ? gpu(x) : x

# === Proteoform Setup ===
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

# === Geometry Update ===
function lift_to_z_plane(rho, pf_states, flat_pos)
    return [Point3(flat_pos[s][1], flat_pos[s][2], -rho[i]) for (i, s) in enumerate(pf_states)]
end

function compute_R(points3D, flat_pos, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    R = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        p0 = flat_pos[s]; p3 = points3D[i]
        neighbors = [v for (u, v) in edges if u == s]
        append!(neighbors, [u for (u, v) in edges if v == s])
        for n in neighbors
            j = idx[n]; q0 = flat_pos[n]; q3 = points3D[j]
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
        neighbors = [v for (u, v) in edges if u == s]
        append!(neighbors, [u for (u, v) in edges if v == s])
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
        nbrs = neighbor_indices[s]
        grad_vals = Float64[]
        for j in nbrs
            dist = norm([
                flat_pos[s][1] - flat_pos[pf_states[j]][1],
                flat_pos[s][2] - flat_pos[pf_states[j]][2]
            ])
            if dist > 1e-6
                push!(grad_vals, abs(C_R_vals[i] - C_R_vals[j]) / dist)
            end
        end
        anisotropy[i] = isempty(grad_vals) ? 0.0 : sum(grad_vals) / length(grad_vals)
    end
    return anisotropy
end

function update_geometry_from_rho(ρ, pf_states, flat_pos, edges)
    points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
    R_vals = compute_R(points3D, flat_pos, pf_states, edges)
    C_R_vals = compute_c_ricci_dirichlet(R_vals, pf_states, edges)
    anisotropy_vals = compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)
    return points3D, R_vals, C_R_vals, anisotropy_vals
end

# === Sheaf Setup ===
function initialize_sheaf_stalks(flat_pos, pf_states)
    stalks = Dict{String, Vector{Float64}}()
    for s in pf_states
        stalks[s] = [flat_pos[s][1], flat_pos[s][2]]
    end
    return stalks
end

function sheaf_consistency(stalks, edges; threshold=2.5)
    inconsistencies = []
    for (u, v) in edges
        diff = norm(stalks[u] .- stalks[v])
        if diff > threshold
            push!(inconsistencies, (u, v, diff))
        end
    end
    return inconsistencies
end

function compute_entropy_cost(i, j, C_R_vals, pf_states)
    baseline_DeltaE = 1.0
    mass_heat = 0.1
    reaction_heat = 0.01 * baseline_DeltaE
    conformational_cost = abs(C_R_vals[j])
    degeneracy_map = Dict(0 => 1, 1 => 3, 2 => 3, 3 => 1)
    deg = degeneracy_map[count(c -> c == '1', pf_states[j])]
    degeneracy_penalty = 1.0 / deg
    return mass_heat + reaction_heat + conformational_cost + degeneracy_penalty
end
