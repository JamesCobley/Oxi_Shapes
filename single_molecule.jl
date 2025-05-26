using StaticArrays
using LinearAlgebra
using Graphs

# =============================================================================
# Geometric object 1: The GeoGraphReal (modal discrete states graph)
# =============================================================================

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

struct GeoGraphReal2
    pf_states::Vector{String}
    flat_pos::Dict{String, Tuple{Float64, Float64}}
    edges::Vector{Tuple{String, String}}

    n::Int
    flat_x::Vector{Float32}
    flat_y::Vector{Float32}
    neighbors::Vector{Vector{Int}}
    d0::Vector{Vector{Float32}}
    edges_idx::Vector{Tuple{Int, Int}}
    adjacency::Matrix{Float32}

    R_vals::Vector{Float32}
    anisotropy::Vector{SVector{3, Float32}}  # a 3D vector per node

    cys_indices::Vector{Int}                  # cysteine residue indices for Omega coords
end

function GeoGraphReal2(pf_states, flat_pos, edges, cys_indices)
    n = length(pf_states)
    idx_map = Dict(s => i for (i, s) in enumerate(pf_states))
    fx = Float32[flat_pos[s][1] for s in pf_states]
    fy = Float32[flat_pos[s][2] for s in pf_states]
    eidx = [(idx_map[u], idx_map[v]) for (u, v) in edges]
    nbrs = [Int[] for _ in 1:n]
    for (i, j) in eidx
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end
    d0 = [Float32[sqrt((fx[i] - fx[j])^2 + (fy[i] - fy[j])^2) for j in nbrs[i]] for i in 1:n]
    g = SimpleGraph(n)
    for (i, j) in eidx
        add_edge!(g, i, j)
    end
    A = Float32.(adjacency_matrix(g))
    Rbuf = zeros(Float32, n)
    anis = [SVector{3, Float32}(0f0, 0f0, 0f0) for _ in 1:n]

    return GeoGraphReal2(pf_states, flat_pos, edges, n, fx, fy, nbrs, d0, eidx, A, Rbuf, anis, cys_indices)
end


function update_geometry!(G::GeoGraphReal2, rho::Vector{Float32}; eps::Float32=1e-3)
    n = G.n
    # Reset R_vals
    fill!(G.R_vals, 0f0)

    # Compute discrete Laplacian (Ricci curvature proxy) R_i = sum neighbors (rho_j - rho_i)
    @inbounds for i in 1:n
        ri = rho[i]
        for j in G.neighbors[i]
            G.R_vals[i] += rho[j] - ri
        end
    end

    # Compute anisotropy vectors A_i (directional bias)
    @inbounds for i in 1:n
        pi = SVector(G.flat_x[i], G.flat_y[i], 0f0)
        bundle = SVector{3, Float32}(0f0, 0f0, 0f0)
        for j in G.neighbors[i]
            pj = SVector(G.flat_x[j], G.flat_y[j], 0f0)
            ΔR = G.R_vals[i] - G.R_vals[j]
            dir = pi - pj
            normdir = norm(dir)
            if normdir > 1e-6
                bundle += (ΔR / normdir^2) * dir
            end
        end
        n_nbrs = length(G.neighbors[i])
        G.anisotropy[i] = n_nbrs > 0 ? bundle / n_nbrs : bundle
    end

    # Optional: check volume and shape constraints (not needed for rho with one 1 and rest 0)
    violated = Int[]
    for i in 1:n
        vol = rho[i] + sum(rho[j] for j in G.neighbors[i])
        expected_vol = (1 + length(G.neighbors[i])) / n
        vol_ok = abs(vol - expected_vol) ≤ eps
        shape_ok = abs(G.R_vals[i]) ≤ eps * (1.0f0 + norm(G.anisotropy[i]))
        if !(vol_ok && shape_ok)
            push!(violated, i)
        end
    end

    return violated
end

# Morse function based on Ricci, anisotropy, and deformation
function compute_morse_function(G::GeoGraphReal2, Omega_coords::Dict{String, Vector{SVector{3, Float64}}}, reactant::Reactant)
    n = G.n
    f = zeros(Float64, n)
    for i in 1:n
        R = G.R_vals[i]
        A = norm(G.anisotropy[i])

        # Use RMSD between the state's own Omega and itself = 0
        D = 0.0

        # Optional: reactant coupling energy (still geometric)
        shape = Omega_coords[G.pf_states[i]]
        C_react = reactant_coupling_energy(shape, G.cys_indices, reactant)

        f[i] = R + A + D + C_react
    end
    return f
end


# Morse saddle barrier = max(f(i), f(j)) + small constant
function compute_saddle_points(G::GeoGraphReal2, f::Vector{Float64}; barrier=0.1)
    saddles = Dict{Tuple{Int,Int}, Float64}()
    for (i, j) in G.edges_idx
        key = i < j ? (i,j) : (j,i)
        saddles[key] = max(f[i], f[j]) + barrier
    end
    return saddles
end

# =============================================================================
# Redox real shape Omega loader and RMSD
# =============================================================================

function load_ca_trace_and_cysteines(pdb_path::String)
    coords = SVector{3, Float64}[]
    resnames = String[]

    open(pdb_path, "r") do io
        for line in eachline(io)
            startswith(line, "ATOM") || continue
            atom_name = strip(line[13:16])
            resname   = strip(line[18:20])
            if atom_name == "CA"
                x = parse(Float64, line[31:38])
                y = parse(Float64, line[39:46])
                z = parse(Float64, line[47:54])
                push!(coords, SVector(x, y, z))
                push!(resnames, resname)
            end
        end
    end

    cys_indices = findall(x -> x == "CYS", resnames)
    return coords, cys_indices
end

function rmsd(c1::Vector{SVector{3, Float64}}, c2::Vector{SVector{3, Float64}})
    sqrt(sum(norm(c1[i] - c2[i])^2 for i in 1:length(c1)) / length(c1))
end

# Simplified deformation energy between two coordinate sets
function real_deformation_energy(coords1::Vector{SVector{3, Float64}}, coords2::Vector{SVector{3, Float64}})
    rmsd(coords1, coords2)
end

# =============================================================================
# Reactant structure type and example reactants
# =============================================================================

struct Reactant
    name::String
    coords::Vector{SVector{3, Float64}}  # simplified shape points
    interaction_radius::Float64           # effective radius for interaction decay
end

# Example reductant and oxidant shapes (simple 3-point models)
reductant = Reactant(
    "reductant",
    [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)],
    5.0
)

oxidant = Reactant(
    "oxidant",
    [SVector(0.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0), SVector(0.0, -1.0, 0.0)],
    5.0
)

# =============================================================================
# Reactant coupling energy calculation
# =============================================================================

# Total geometric coupling (curvature + anisotropy + deformation + saddle)
function total_coupling_with_morse(
    x_i::Int, x_j::Int,
    G::GeoGraphReal2,
    Omega_coords::Dict{String, Vector{SVector{3, Float64}}},
    reactant::Reactant,
    saddles::Dict{Tuple{Int,Int}, Float64}
)
    R_val = G.R_vals[x_j]
    A_val = norm(G.anisotropy[x_j])
    C_real = rmsd(Omega_coords[G.pf_states[x_i]], Omega_coords[G.pf_states[x_j]])
    C_react = reactant_coupling_energy(Omega_coords[G.pf_states[x_j]], G.cys_indices, reactant)
    key = x_i < x_j ? (x_i, x_j) : (x_j, x_i)
    M_barrier = get(saddles, key, 0.0)
    return R_val + A_val + C_real + C_react + M_barrier
end

# Optional: quantum-style rate equation
function transition_rate(x_i::Int, x_j::Int, G::GeoGraphReal2, Omega_coords, reactant, saddles; α=1.0)
    ΔF = total_coupling_with_morse(x_i, x_j, G, Omega_coords, reactant, saddles)
    return exp(-2α * ΔF)
end

# =============================================================================
# Total coupling function incorporating all terms
# =============================================================================

function total_coupling_with_morse(
    x_i::Int, x_j::Int,
    G::GeoGraphReal2,
    Omega_coords::Dict{String, Vector{SVector{3, Float64}}},
    reactant::Reactant,
    saddles::Dict{Tuple{Int,Int}, Float64}
)
    R_val = G.R_vals[x_j]
    A_val = norm(G.anisotropy[x_j])
    C_real = real_deformation_energy(Omega_coords[G.pf_states[x_i]], Omega_coords[G.pf_states[x_j]])
    C_react = reactant_coupling_energy(Omega_coords[G.pf_states[x_j]], G.cys_indices, reactant)
    # Get Morse saddle barrier if edge exists; else 0
    key = x_i < x_j ? (x_i, x_j) : (x_j, x_i)
    M_barrier = haskey(saddles, key) ? saddles[key] : 0.0
    return R_val + A_val + C_real + C_react + M_barrier
end

# =============================================================================
# Example usage
# =============================================================================

pdb_path = "/content/AF-P04406-F1-model_v4.pdb"
coords, cys_indices = load_ca_trace_and_cysteines(pdb_path)

# Omega_coords for each pf_state (simplified: all same coords now)
Omega_coords = Dict{String, Vector{SVector{3, Float64}}}()
for state in pf_states
    Omega_coords[state] = coords  # Replace with deformed Ω if available
end

G = GeoGraphReal2(pf_states, flat_pos, edges, cys_indices)

rho = zeros(Float32, length(pf_states))
rho[findfirst(==( "000"), pf_states)] = 1.0f0

println("Testing occupancy in state: 000")
violated_nodes = update_geometry!(G, rho; eps=1f-3)
println("Violated nodes after update: ", violated_nodes)
println("R_vals after update: ", G.R_vals)
println("Anisotropy vectors after update: ", G.anisotropy)

# -- Morse analysis --
morse_vals = compute_morse_function(G, Omega_coords, oxidant)
saddle_points = compute_saddle_points(G, morse_vals; barrier=0.1)

# Define neighbors of "000"
neighbors = ["001", "010", "100"]
idx_000 = findfirst(==( "000"), pf_states)

# Prepare to store results
results = Dict{String, Tuple{Float64, Float64}}()  # state => (coupling, rate)

for nbr in neighbors
    idx_nbr = findfirst(==(nbr), pf_states)
    coupling = total_coupling_with_morse(idx_000, idx_nbr, G, Omega_coords, oxidant, saddle_points)
    rate = transition_rate(idx_000, idx_nbr, G, Omega_coords, oxidant, saddle_points)
    results[nbr] = (coupling, rate)
end

println("\n--- Transition Summary from 000 ---")
for (state, (coupling, rate)) in results
    println("000 → $state: Coupling = $(round(coupling, digits=4)), Rate = $(round(rate, digits=6))")
end

# Sort to find the most likely
most_likely = argmax(rate for (_, (_, rate)) in results)
sorted = sort(collect(results), by = x -> x[2][2], rev = true)

println("\nMost likely transition from 000: 000 → $(sorted[1][1])")
