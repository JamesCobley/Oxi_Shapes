using LinearAlgebra
using StaticArrays
using Graphs

# =============================================================================
# Load coordinates and identify cysteines from PDB
# =============================================================================
function load_ca_trace_and_cysteines(pdb_path::String)
    coords = SVector{3, Float64}[]
    cys_indices = Int[]

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
                if resname == "CYS"
                    push!(cys_indices, length(coords))  # aligned index
                end
            end
        end
    end

    return coords, cys_indices
end

println("Length of coords: ", length(coords))
println("cys_indices: ", cys_indices)

# =============================================================================
# Normalize occupancy volume
# =============================================================================
function normalize_volume!(ρ::Vector{Float32})
    total = sum(ρ)
    if total > 0f0
        ρ ./= total
    end
    return ρ
end

function dirichlet_energy(coords::Vector{SVector{3, Float64}})
    n = length(coords)
    center = sum(coords) / n
    return sum(norm(coord - center)^2 for coord in coords)
end


# =============================================================================
# Graph + Geometry Structure
# =============================================================================
struct GeoGraphReal1
    pf_states::Vector{String}
    pf_states_map::Dict{String, Int}
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
    anisotropy::Vector{SVector{3, Float32}}

    cys_indices::Vector{Int}
end

function GeoGraphReal1(pf_states, flat_pos, edges, cys_indices)
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

    return GeoGraphReal1(pf_states, idx_map, flat_pos, edges, n, fx, fy, nbrs, d0, eidx, A, Rbuf, anis, cys_indices)
end

# =============================================================================
# Compute Dirichlet Field
# =============================================================================
function compute_dirichlet_field(G::GeoGraphReal1)
    n = G.n
    field = zeros(Float64, n)
    for i in 1:n
        for j in G.neighbors[i]
            Δρ = G.R_vals[i] - G.R_vals[j]
            field[i] += 0.5 * Δρ^2
        end
    end
    return field
end

# =============================================================================
# Bitwise A(x) vector + Dirichlet Energy
# =============================================================================
function bitwise_Ax(state::String, G::GeoGraphReal1, Omega_coords::Dict{String, Vector{SVector{3, Float64}}})
    A = Dict{Int, SVector{3, Float64}}()
    E_bit = Dict{Int, Float64}()
    bits = collect(state)

    coords = get(Omega_coords, state, nothing)
    if coords === nothing || length(coords) != 3
        @warn "Missing or invalid coords for state $state; using fallback"
        return Dict(i => SVector(0.0, 0.0, 0.0) for i in 1:3), Dict(i => 0.0 for i in 1:3)
    end

    idx = G.pf_states_map[state]
    R_self = G.R_vals[idx]

    # Define 3D coordinates for the current state using Oxi-Shape embedding (z = -R)
    coords3D = [
        SVector(coord[1], coord[2], -R_self) for coord in coords
    ]
    center3D = sum(coords3D) / 3

    for i in 1:3
        flipped = copy(bits)
        flipped[i] = bits[i] == '0' ? '1' : '0'
        flipped_state = join(flipped)

        ΔR = 0.0
        sign = 0.0
        if haskey(G.pf_states_map, flipped_state)
            R_flipped = G.R_vals[G.pf_states_map[flipped_state]]
            ΔR = abs(R_flipped - R_self)
            sign = signbit(R_flipped - R_self) ? -1.0 : 1.0
        end

        vec3D = coords3D[i] - center3D
        dir_vec = iszero(norm(vec3D)) ? SVector(0.0, 0.0, 0.0) : normalize(vec3D)
        A[i] = sign * ΔR * dir_vec
    end

    # Dirichlet energy field from global R(x)
    dirichlet_field = compute_dirichlet_field(G)
    global_E = dirichlet_field[idx]

    # Energy weighting by 3D displacement from center
    dists = [norm(coords3D[k] - center3D) for k in 1:3]
    total_dist = sum(dists) + 1e-6
    weights = dists ./ total_dist
    for k in 1:3
        E_bit[k] = weights[k] * global_E
    end

    return A, E_bit
end

# =============================================================================
# Main Geometry Update
# =============================================================================
function update_geometry!(G::GeoGraphReal1, rho::Vector{Float32}, Omega_coords::Dict{String, Vector{SVector{3, Float64}}}; eps::Float32=1e-3)
    n = G.n

    # Compute graph Laplacian curvature: R(x) = ∑ [ρ(y) - ρ(x)]
    A = G.adjacency
    D = diagm(0 => sum(A, dims=2)[:])
    L = D - A
    G.R_vals = L * rho

    # Compute anisotropy using Oxi-Shape fibre bundle: A(x) = (1/N) ∑ [R(x) - R(y)] * (x - y) in 3D
    @inbounds for i in 1:n
        xi2D = SVector(G.flat_x[i], G.flat_y[i])
        zi = -G.R_vals[i]  # vertical deformation from curvature
        xi = SVector{3, Float32}(xi2D[1], xi2D[2], zi)

        bundle = SVector{3, Float32}(0f0, 0f0, 0f0)
        for j in G.neighbors[i]
            xj2D = SVector(G.flat_x[j], G.flat_y[j])
            zj = -G.R_vals[j]
            xj = SVector{3, Float32}(xj2D[1], xj2D[2], zj)

            ΔR = G.R_vals[i] - G.R_vals[j]
            bundle += ΔR * (xi - xj)
        end
        n_nbrs = length(G.neighbors[i])
        G.anisotropy[i] = n_nbrs > 0 ? bundle / n_nbrs : bundle
    end

    # Optional: check for shape/volume violations
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

# =============================================================================
# Deformation Sim: Sulfenic Acid
# =============================================================================
function simulate_sulfenic_deformation(coords::Vector{SVector{3, Float64}}, oxidized_idxs::Vector{Int})
    displaced = copy(coords)
    for idx in oxidized_idxs
        normal = normalize(displaced[idx])
        displaced[idx] += 0.5 * normal
    end
    return displaced
end

# =============================================================================
# Build Geometry for All States
# =============================================================================
function build_state_geometry(pf_states, coords, cys_indices)
    Omega_coords = Dict{String, Vector{SVector{3, Float64}}}()
    for state in pf_states
        bits = collect(state)
        cys_coords = [coords[i] for i in cys_indices]
        oxidized_idxs = findall(x -> x == '1', bits)
        displaced = copy(cys_coords)
        for i in oxidized_idxs
            normal = normalize(displaced[i])
            displaced[i] += 0.5 * normal
        end
        Omega_coords[state] = displaced
    end
    return Omega_coords
end

println("\nΩ(x) Coordinate Map:")
for state in pf_states
    coords = Omega_coords[state]
    println("State $state:")
    for (i, coord) in enumerate(coords)
        println("  Cys $i → (x=$(round(coord[1], digits=3)), y=$(round(coord[2], digits=3)), z=$(round(coord[3], digits=3)))")
    end
end

# =============================================================================
# Define Reactant: Hydrogen Peroxide (H2O2)
# =============================================================================
h2o2 = Reactant(
    "H2O2",
    [
        SVector(0.000,  0.000,  0.000),   # O1
        SVector(1.480,  0.000,  0.000),   # O2
        SVector(-0.320, 0.890,  0.780),   # H1
        SVector(1.800, -0.890, -0.780)    # H2
    ],
    6.0  # Effective interaction radius in Å
)

h2o2_coords = [
    SVector(0.000,  0.000,  0.000),   # O1
    SVector(1.480,  0.000,  0.000),   # O2
    SVector(-0.320, 0.890,  0.780),   # H1
    SVector(1.800, -0.890, -0.780)    # H2
]

E_D_h2o2 = dirichlet_energy(h2o2_coords)
println("Dirichlet Energy of Reactant (H₂O₂): ", round(E_D_h2o2, digits=4))

# =============================================================================
# Delta chem
# =============================================================================
function estimate_sasa(coords::Vector{SVector{3, Float64}}, cys_indices::Vector{Int}; cutoff=5.0)
    sasa = zeros(Float64, length(cys_indices))
    for (i, cys_idx) in enumerate(cys_indices)
        center = coords[cys_idx]
        sasa_count = 0
        for (j, atom) in enumerate(coords)
            if j == cys_idx
                continue
            end
            if norm(atom - center) < cutoff
                sasa_count += 1
            end
        end
        sasa[i] = 1.0 / (sasa_count + 1e-3)  # more crowded = less exposed
    end
    return sasa
end

# Example: pKa values for 3 cysteines
pKa_values = [5.57, 9.93, 10.29]  # Must match order of cys_indices

function compute_delta_chem(sasa::Vector{Float64}, pKa::Vector{Float64})
    Δ_chem = Float64[]
    for i in 1:length(sasa)
        # Lower pKa and higher SASA = higher chemical reactivity
        push!(Δ_chem, sasa[i] * (12.0 - pKa[i]))  # 12 as an arbitrary max scale
    end
    return Δ_chem
end

function dirichlet_energy_shape_only(coords::Vector{SVector{3, Float64}}; cutoff=8.0)
    E = 0.0
    for i in 1:length(coords), j in i+1:length(coords)
        d = norm(coords[i] - coords[j])
        if d < cutoff
            E += (1.0 - 1.0)^2  # constant field — so energy = 0
        end
    end
    return E  # trivially zero
end

function shape_energy(coords::Vector{SVector{3, Float64}}; cutoff=8.0)
    E = 0.0
    for i in 1:length(coords), j in i+1:length(coords)
        d = norm(coords[i] - coords[j])
        if d < cutoff
            E += d^2
        end
    end
    return E
end

function local_dirichlet_energy(coords::Vector{SVector{3, Float64}}, cys_indices::Vector{Int}; cutoff=8.0)
    energies = zeros(length(cys_indices))
    for (i, cys_idx) in enumerate(cys_indices)
        cys_coord = coords[cys_idx]
        E = 0.0
        for (j, atom) in enumerate(coords)
            if j == cys_idx
                continue
            end
            dist = norm(atom - cys_coord)
            if dist < cutoff
                E += dist^2
            end
        end
        energies[i] = E
    end
    return energies
end


# =============================================================================
# Example Inputs and Simulation
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

# Load your structure
pdb_path = "/content/AF-P04406-F1-model_v4.pdb"
coords, cys_indices = load_ca_trace_and_cysteines(pdb_path)
Omega_coords = build_state_geometry(pf_states, coords, cys_indices)
G = GeoGraphReal1(pf_states, flat_pos, edges, cys_indices)

rho = zeros(Float32, length(pf_states))
rho[findfirst(==("000"), pf_states)] = 1.0f0
normalize_volume!(rho)

violated_nodes = update_geometry!(G, rho; eps=1f-3)

println("Violated nodes after update: ", violated_nodes)
println("\nR(x) values:")
for (state, idx) in zip(pf_states, 1:length(pf_states))
    println("State $state → R = $(round(G.R_vals[idx], digits=4))")
end

println("\nA(x) anisotropy magnitudes:")
for (state, idx) in zip(pf_states, 1:length(pf_states))
    A_mag = norm(G.anisotropy[idx])
    println("State $state → |A| = $(round(A_mag, digits=4))")
end

# Step 7: Compute and print bitwise A(x) and internal energy for each proteoform state
println("\nBitwise A(x) vectors and Dirichlet energy:")
for state in pf_states
    A_bitwise, E_bit = bitwise_Ax(state, G, Omega_coords)
    println("State $state:")
    for i in 1:3
        vec = get(A_bitwise, i, SVector(0.0, 0.0, 0.0))
        energy = get(E_bit, i, 0.0)
        println("  Cys $i → A = $(round.(vec, digits=4)), E = $(round(energy, digits=4))")
    end
end

coords, cys_indices = load_ca_trace_and_cysteines(pdb_path)
cys_ED = local_dirichlet_energy(coords, cys_indices)

println("\nLocal Dirichlet Energy of Each Cys (Real Structure):")
for (i, E) in enumerate(cys_ED)
    println("  Cys $(i) → E = $(round(E, digits=4))")
end
