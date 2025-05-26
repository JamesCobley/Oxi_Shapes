using StaticArrays
using LinearAlgebra
using Graphs

# -- Your Working Standalone Redox Geometry Model --

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

function OmegaReal1(pdb_path::String, istate::BitVector)
    coords, cys_indices = load_ca_trace_and_cysteines(pdb_path)
    # No deformation: use coords directly
    deformed = copy(coords)
    energy = 0.0  # No deformation, so zero energy
    return OmegaReal1(coords, cys_indices, istate, deformed, energy)
end


function rmsd(c1::Vector{SVector{3, Float64}}, c2::Vector{SVector{3, Float64}})
    sqrt(sum(norm(c1[i] - c2[i])^2 for i in 1:length(c1)) / length(c1))
end

mutable struct OmegaReal1
    coords::Vector{SVector{3, Float64}}         # original real shape
    cys_indices::Vector{Int}                    # positions of CYS
    redox_state::BitVector                      # redox i-state
    deformed::Vector{SVector{3, Float64}}       # Ω(x)
    deformation_energy::Float64                 # ||Ω(x) - coords||
end

function OmegaReal1(pdb_path::String, istate::BitVector)
    coords, cys_indices = load_ca_trace_and_cysteines(pdb_path)
    deformed = generate_Omega_x(coords, cys_indices, istate)
    energy = rmsd(coords, deformed)
    return OmegaReal1(coords, cys_indices, istate, deformed, energy)
end

# -- Modal graph states and edges --

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

# -- Integrated GeoGraphReal struct --

mutable struct GeoGraphReal1
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

    points3D::Vector{SVector{3, Float32}}       # 2D pos + height (modal)
    R_vals::Vector{Float32}                      # Ricci curvature at each node
    anisotropy::Vector{SVector{3, Float32}}     # Residue-level directional functional

    omega_states::Vector{OmegaReal1}             # The real proteoform shape at each modal node
end

function GeoGraphReal1(pf_states, flat_pos, edges, pdb_path::String)
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

    pts3D = [SVector{3, Float32}(fx[i], fy[i], 0f0) for i in 1:n]
    Rbuf = zeros(Float32, n)
    anis = [SVector{3, Float32}(0f0, 0f0, 0f0) for _ in 1:n]

    # Load all OmegaReal1 redox states ONCE here
    omega_states = Vector{OmegaReal1}(undef, n)
    for (i, state_str) in enumerate(pf_states)
        istate = BitVector([c == '1' for c in state_str])
        omega_states[i] = OmegaReal1(pdb_path, istate)
    end

    return GeoGraphReal1(pf_states, flat_pos, edges, n, fx, fy, nbrs, d0, eidx, A, pts3D, Rbuf, anis, omega_states)
end

# -- Update geometry and anisotropy --

function compute_ricci_curvature(G::GeoGraphReal, rho::Vector{Float32})
    R = zeros(Float32, G.n)
    for i in 1:G.n
        sum_neigh = 0f0
        for j in G.neighbors[i]
            sum_neigh += rho[j] - rho[i]
        end
        R[i] = sum_neigh
    end
    return R
end

function update_real_geometry!(G::GeoGraphReal, rho::Vector{Float32}; eps::Float32=1f-3)
    violated = Int[]

    # Update 3D points height for visualization only (optional)
    @inbounds for i in 1:G.n
        G.points3D[i] = SVector{3, Float32}(G.flat_x[i], G.flat_y[i], -rho[i])
    end

    # Compute Ricci curvature as graph Laplacian on rho
    G.R_vals = compute_ricci_curvature(G, rho)

    # Compute anisotropy A(x) as before (residue-level directional functional)
    @inbounds for i in 1:G.n
        pi = G.points3D[i]
        A_vec = SVector{3, Float32}(0f0, 0f0, 0f0)
        for j in G.neighbors[i]
            shape_i = G.omega_states[i].deformed
            shape_j = G.omega_states[j].deformed
            diff = 0.0f0
            for k in 1:length(shape_i)
                diff += norm(shape_i[k] - shape_j[k])^2
            end
            diff = sqrt(diff / length(shape_i))

            dir = pi - G.points3D[j]
            if norm(dir) > 1e-6
                A_vec += (diff / norm(dir)^2) * dir
            end
        end
        n_neighbors = length(G.neighbors[i])
        G.anisotropy[i] = n_neighbors > 0 ? A_vec / n_neighbors : A_vec
    end

    # Check violations as before
    for i in 1:G.n
        vol = rho[i] + sum(rho[j] for j in G.neighbors[i])
        expected_vol = (1 + length(G.neighbors[i])) / G.n
        vol_ok = abs(vol - expected_vol) ≤ eps
        shape_ok = abs(G.R_vals[i]) ≤ eps * (1.0f0 + norm(G.anisotropy[i]))
        if !(vol_ok && shape_ok)
            push!(violated, i)
        end
    end

    return violated
end

function one_hot_rho(pf_states, target_state)
    n = length(pf_states)
    rho = zeros(Float32, n)
    idx = findfirst(==(target_state), pf_states)
    if isnothing(idx)
        error("Target state $target_state not found in pf_states")
    end
    rho[idx] = 1.0f0
    return rho
end
rho = one_hot_rho(pf_states, "000")

# =================
# Example test code
# =================

function test_single_state(pdb_path::String, target_state::String)
    G = GeoGraphReal1(pf_states, flat_pos, edges, pdb_path)
    rho = one_hot_rho(pf_states, target_state)

    println("Testing occupancy in state: $target_state at index $(findfirst(==(target_state), pf_states))")
    println("Before update:")
    println("R_vals = ", G.R_vals)
    println("anisotropy = ", G.anisotropy)

    # Use Float32 literal for eps
    violated = update_real_geometry!(G, rho; eps=1f-3)

    println("\nAfter update:")
    println("R_vals = ", G.R_vals)
    println("anisotropy = ", G.anisotropy)
    println("Violated nodes: ", violated)
end

# Run:
test_single_state("/content/AF-P04406-F1-model_v4.pdb", "000")
