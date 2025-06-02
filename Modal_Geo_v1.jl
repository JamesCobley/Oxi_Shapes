using LinearAlgebra
using StaticArrays
using Graphs

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
pdb_path = "/content/AF-P04406-F1-model_v4.pdb"
coords, cys_indices = load_ca_trace_and_cysteines(pdb_path)
println("Length of coords: ", length(coords))
println("cys_indices: ", cys_indices)

# =============================================================================
# Define Reactant: Hydrogen Peroxide (H2O2)
# =============================================================================
struct Reactant
    name::String
    coords::Vector{SVector{3, Float64}}
    radius::Float64  # effective interaction radius or similar
end

h2o2 = Reactant(
    "H2O2",
    [
        SVector(0.000,  0.000,  0.000),   # O1
        SVector(1.480,  0.000,  0.000),   # O2
        SVector(-0.320, 0.890,  0.780),   # H1
        SVector(1.800, -0.890, -0.780)    # H2
    ],
    6.0
)

E_D_h2o2 = dirichlet_energy(h2o2.coords)
println("Dirichlet Energy of Reactant (H₂O₂): ", round(E_D_h2o2, digits=4))
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

# Cross-Dirichlet energy between a proteoform geometry and a reactant
function real_coupling_energy(
    Omega_coords::Vector{SVector{3, Float64}},
    state::String,
    cys_indices::Vector{Int},
    reactant::Reactant
)
    bits = collect(state)
    E_total = 0.0
    for (local_idx, bit) in enumerate(bits)
        if bit == '0'
            # Get the Cys coordinate for this site
            cys_coord = Omega_coords[local_idx]
            # Compute coupling: deformational Dirichlet energy with reactant
            E_total += dirichlet_energy([cys_coord; reactant.coords...])
        end
    end
    return E_total
end

# =============================================================================
# Graph + Geometry Structure
# =============================================================================
mutable struct GeoGraphReal1
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
    E_D_vals::Vector{Float32}  # ← New field for Dirichlet energy
end

function GeoGraphReal1(
    pf_states::Vector{String},
    flat_pos::Dict{String, Tuple{Float64, Float64}},
    edges::Vector{Tuple{String, String}},
    cys_indices::Vector{Int},
    E_D_vals::Vector{Float32}  # <- pass it in
)
    n = length(pf_states)
    pf_states_map = Dict(s => i for (i, s) in enumerate(pf_states))
    flat_x = Float32[flat_pos[s][1] for s in pf_states]
    flat_y = Float32[flat_pos[s][2] for s in pf_states]

    neighbors = [Int[] for _ in 1:n]
    d0 = [Float32[] for _ in 1:n]
    adjacency = zeros(Float32, n, n)
    edges_idx = Tuple{Int, Int}[]

    for (s1, s2) in edges
        i, j = pf_states_map[s1], pf_states_map[s2]
        push!(neighbors[i], j)
        push!(neighbors[j], i)
        push!(d0[i], 1.0f0)
        push!(d0[j], 1.0f0)
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
        push!(edges_idx, (i, j))
    end

    R_vals = zeros(Float32, n)
    anisotropy = [SVector{3, Float32}(0.0, 0.0, 0.0) for _ in 1:n]

    return GeoGraphReal1(
        pf_states, pf_states_map, flat_pos, edges,
        n, flat_x, flat_y, neighbors, d0,
        edges_idx, adjacency, R_vals, anisotropy,
        cys_indices, E_D_vals
    )
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

    # 👉 Add Dirichlet energy values for each state
    E_D_vals = Dict{String, Float64}()
    for state in pf_states
        E_D_vals[state] = dirichlet_energy(Omega_coords[state])
    end
    E_D_vec = Float32[E_D_vals[state] for state in pf_states]

    return Omega_coords, E_D_vec
end

# =============================================================================
# Define Reactant: Hydrogen Peroxide (H2O2)
# =============================================================================
struct Reactant
    name::String
    coords::Vector{SVector{3, Float64}}
    radius::Float64  # effective interaction radius or similar
end

h2o2 = Reactant(
    "H2O2",
    [
        SVector(0.000,  0.000,  0.000),   # O1
        SVector(1.480,  0.000,  0.000),   # O2
        SVector(-0.320, 0.890,  0.780),   # H1
        SVector(1.800, -0.890, -0.780)    # H2
    ],
    6.0
)

E_D_h2o2 = dirichlet_energy(h2o2.coords)
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

# Load structure and build modal state geometry

# Load PDB structure first
pdb_path = "/content/AF-P04406-F1-model_v4.pdb"
coords, cys_indices = load_ca_trace_and_cysteines(pdb_path)

# Build modal geometry and compute Dirichlet energy per state
Omega_coords, E_D_vec = build_state_geometry(pf_states, coords, cys_indices)

println("\nΩ(x) Coordinate Map with Dirichlet Energy:")
for (i, state) in enumerate(pf_states)
    coords = Omega_coords[state]
    E_D = E_D_vec[i]
    println("State $state:")
    for (j, coord) in enumerate(coords)
        println("  Cys $j → (x=$(round(coord[1], digits=3)), y=$(round(coord[2], digits=3)), z=$(round(coord[3], digits=3)))")
    end
    println("  E_D = $(round(E_D, digits=4))")
end

# Build graph
G = GeoGraphReal1(pf_states, flat_pos, edges, cys_indices, E_D_vec)

# Define occupancy
rho = zeros(Float32, length(pf_states))
rho[findfirst(==("000"), pf_states)] = 1.0f0
normalize_volume!(rho)

# Update geometry
update_geometry!(G, rho, Omega_coords; eps = 0.001f0)  # ✅ Float32

println("\nA(x) anisotropy magnitudes and curvature:")
for (state, idx) in zip(pf_states, 1:length(pf_states))
    A_mag = norm(G.anisotropy[idx])
    κ = G.R_vals[idx]  # ✅ R(x) = curvature
    println("State $state → |A| = $(round(A_mag, digits=4)), κ = $(round(κ, digits=4))")
end

# Bitwise outputs
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

# Dirichlet energy of real structure
coords, cys_indices = load_ca_trace_and_cysteines(pdb_path)
cys_ED = local_dirichlet_energy(coords, cys_indices)
println("\nLocal Dirichlet Energy of Each Cys (Real Structure):")
for (i, E) in enumerate(cys_ED)
    println("  Cys $(i) → E = $(round(E, digits=4))")
end

# -----------------------------------------------------------------------------
# Dirichlet Shape Coupling Between Modal States and Reactant (H₂O₂)
# -----------------------------------------------------------------------------

# Make sure these are relative to modal Ω(x) coords
cys_indices = collect(1:length(Omega_coords["000"]))  # = [1, 2, 3]

function dirichlet_shape_coupling_bitwise(
    coords::Vector{SVector{3, Float64}},
    cys_indices::Vector{Int},
    E_D_ref::Float64
)
    E_bitwise = zeros(length(cys_indices))
    for (i, idx) in enumerate(cys_indices)
        E_cys = local_dirichlet_energy(coords, [idx])[1]
        E_bitwise[i] = abs(E_cys - E_D_ref)
    end
    return E_bitwise
end

bitwise_shape_coupling = Dict{String, Vector{Float64}}()
for state in pf_states
    coords = Omega_coords[state]
    bitwise_shape_coupling[state] = dirichlet_shape_coupling_bitwise(coords, cys_indices, E_D_h2o2)
end

println("\nBitwise Dirichlet Shape Coupling to Reactant (Ω_STATE ↔ Ω_REACTANT):")
for state in pf_states
    println("  State $state:")
    bits = collect(state)
    for (i, ΔE) in enumerate(bitwise_shape_coupling[state])
        if bits[i] == '0'  # Only reduced sites can interact with H₂O₂
            println("    Cys $(i) → ΔE_shape = $(round(ΔE, digits=4))")
        else
            println("    Cys $(i) → (oxidized)")
        end
    end
end

# -----------------------------------------------------------------------------
# Discrete Morse Field over Modal Manifold
# -----------------------------------------------------------------------------

# Step 1: Use Dirichlet energy as the Morse scalar field
f_morse = Float64.(G.E_D_vals)

# Step 2: Compute gradient on edges
function morse_gradient(G::GeoGraphReal1, f::Vector{Float64})
    grad_edges = Dict{Tuple{Int, Int}, Float64}()
    for (i, neighbors) in enumerate(G.neighbors)
        for j in neighbors
            if i < j
                Δf = f[j] - f[i]
                grad_edges[(i, j)] = Δf
            end
        end
    end
    return grad_edges
end

# Step 3: Identify Morse critical points
function find_morse_critical_points(G::GeoGraphReal1, f::Vector{Float64})
    minima, maxima, saddles = Int[], Int[], Int[]
    for (i, neighbors) in enumerate(G.neighbors)
        f_i = f[i]
        f_neighbors = f[collect(neighbors)]
        if all(f_i < f_j for f_j in f_neighbors)
            push!(minima, i)
        elseif all(f_i > f_j for f_j in f_neighbors)
            push!(maxima, i)
        else
            push!(saddles, i)
        end
    end
    return minima, maxima, saddles
end

# Step 4: Label and print Morse critical points
function label_morse_states(pf_states, minima, maxima, saddles)
    println("\nMorse Critical Points:")
    for i in minima
        println("  Minimum: ", pf_states[i])
    end
    for i in maxima
        println("  Maximum: ", pf_states[i])
    end
    for i in saddles
        println("  Saddle:  ", pf_states[i])
    end
end

# Run the Morse computation
grad = morse_gradient(G, f_morse)
minima, maxima, saddles = find_morse_critical_points(G, f_morse)
label_morse_states(pf_states, minima, maxima, saddles)

# -----------------------------------------------------------------------------
# Set physical constants
# -----------------------------------------------------------------------------
α = 1.0  # quantum decay constant (unitless)
ε = 78.5         # dielectric constant of water
T = 298.15       # temperature in Kelvin

function compute_Φ_PHYSRT(ε::Float64, T::Float64; q::Float64=1.6e-19, r::Float64=5e-10)
    ε₀ = 8.854e-12  # vacuum permittivity (F/m)
    k_B = 1.38e-23  # Boltzmann constant (J/K)
    solvation_energy = (q^2) / (4π * ε₀ * ε * r)
    return solvation_energy / (k_B * T)
end

Φ_PHYSRT = compute_Φ_PHYSRT(ε, T)

println("Φ_PHYSRT (solvation penalty in kT) = $(round(Φ_PHYSRT, digits=2))")


# -----------------------------------------------------------------------------
# Compute Δ_chem (from SASA and pKa)
# -----------------------------------------------------------------------------
sasa = estimate_sasa(coords, cys_indices)
delta_chem = compute_delta_chem(sasa, pKa_values)

# -----------------------------------------------------------------------------
# Bitflip neighbor function
# -----------------------------------------------------------------------------
function get_bitflip_neighbors(state::String, pf_states::Vector{String})
    neighbors = String[]
    for i in 1:length(state)
        bits = collect(state)
        bits[i] = bits[i] == '0' ? '1' : '0'
        flipped = join(bits)
        if flipped in pf_states
            push!(neighbors, flipped)
        end
    end
    return neighbors
end

# -----------------------------------------------------------------------------
# Δ_chem for a given bitflip transition
# -----------------------------------------------------------------------------
function delta_chem_transition(xi::String, xj::String, Δ_chem::Vector{Float64})
    bits_i = collect(xi)
    bits_j = collect(xj)
    Δ_sum = 0.0
    for i in 1:length(bits_i)
        if bits_i[i] != bits_j[i]
            Δ_sum += Δ_chem[i]
        end
    end
    return Δ_sum
end

# -----------------------------------------------------------------------------
# Transition matrix k[xi][xj]
# -----------------------------------------------------------------------------
transition_matrix = Dict{String, Dict{String, Float64}}()

for xi in pf_states
    neighbors = get_bitflip_neighbors(xi, pf_states)
    transition_matrix[xi] = Dict{String, Float64}()

    for xj in neighbors
        idx_xj = G.pf_states_map[xj]

        # Geometry-based penalties
        Rj = G.R_vals[idx_xj]
        Aj = norm(G.anisotropy[idx_xj])

        # Shape deformation penalty
        coords_xi = Omega_coords[xi]
        local_cys = collect(1:length(coords_xi))  # = [1, 2, 3]
        C_real = sum(dirichlet_shape_coupling_bitwise(coords_xi, local_cys, dirichlet_energy(h2o2.coords)))

        # Chemically grounded Δ_chem term for this bitflip
        Δ_chem_term = delta_chem_transition(xi, xj, delta_chem)

        # Total coupling cost
        C = Rj + Aj + C_real

        # Final transition rate
        k = exp(-2α * C + Δ_chem_term - Φ_PHYSRT)
        transition_matrix[xi][xj] = k
    end
end

# -----------------------------------------------------------------------------
# Print results
# -----------------------------------------------------------------------------
println("\nTransition Rate Matrix (physical + chemical):")
for (xi, neighbors) in transition_matrix
    println("From $xi:")
    for (xj, k) in neighbors
        println("  → $xj : k = $(round(k, digits=6))")
    end
end
