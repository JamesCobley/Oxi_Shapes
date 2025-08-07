# =============================================================================
# Modal Transition Model with Real-to-Abstract Projection
# =============================================================================
using StaticArrays, LinearAlgebra, Statistics

# -----------------------------------------------------------------------------
# 0. Define bitwise directions for abstract states
# -----------------------------------------------------------------------------
pf_states = ["000", "001", "010", "011", "100", "101", "110", "111"]
bit_dirs = [
    SVector(1.0, 0.0, 0.0),  # bit 1 (x)
    SVector(0.0, 1.0, 0.0),  # bit 2 (y)
    SVector(0.0, 0.0, 1.0)   # bit 3 (z)
]
bitwise_dirs = [
    sum((s[i] == '1' ? bit_dirs[i] : SVector(0.0, 0.0, 0.0)) for i in 1:3)
    for s in pf_states
]

# -----------------------------------------------------------------------------
# 1. Real-space geometry helpers
# -----------------------------------------------------------------------------
function load_ca_and_cys(pdb_path::String)
    coords = SVector{3,Float64}[]
    cys_indices = Int[]
    open(pdb_path, "r") do io
        for ln in eachline(io)
            if startswith(ln, "ATOM")
                atom = strip(ln[13:16]); res = strip(ln[18:20])
                if atom == "CA"
                    x = parse(Float64, ln[31:38])
                    y = parse(Float64, ln[39:46])
                    z = parse(Float64, ln[47:54])
                    push!(coords, SVector(x,y,z))
                    if res == "CYS"
                        push!(cys_indices, length(coords))
                    end
                end
            end
        end
    end
    return coords, cys_indices
end

function build_proximity_graph(coords::Vector{SVector{3,Float64}}; cutoff=5.0)
    n = length(coords)
    neighbors = [Int[] for _ in 1:n]
    for i in 1:n-1, j in i+1:n
        if norm(coords[i] - coords[j]) ≤ cutoff
            push!(neighbors[i], j)
            push!(neighbors[j], i)
        end
    end
    return neighbors
end

function real_laplacian_curvature(coords, neighbors, cys_idx)
    curvature = Vector{SVector{3,Float64}}(undef, length(cys_idx))
    for (k, idx) in enumerate(cys_idx)
        nbrs = neighbors[idx]
        Δ = sum(coords[j] - coords[idx] for j in nbrs)
        curvature[k] = Δ / max(length(nbrs), 1)
    end
    return curvature
end

function construct_A_real(G, coords, cys_idx, neighbors)
    cys_curv = real_laplacian_curvature(coords, neighbors, cys_idx)
    avg_norm = sum(norm.(cys_curv)) / length(cys_curv)
    return fill(avg_norm, length(G.R_vals))
end

# -----------------------------------------------------------------------------
# 2. Abstract-graph definition and modal decomposition
# -----------------------------------------------------------------------------
edges = [
    ("000","001"),("000","010"),("000","100"),
    ("001","011"),("001","101"),("010","011"),
    ("010","110"),("100","110"),("100","101"),
    ("011","111"),("101","111"),("110","111")
]
idx = Dict(s => i for (i,s) in enumerate(pf_states))
n = length(pf_states)
A = zeros(n,n)
neighbors = [Int[] for _ in 1:n]
for (u,v) in edges
    i,j = idx[u], idx[v]
    A[i,j] = 1.0; A[j,i] = 1.0
    push!(neighbors[i], j); push!(neighbors[j], i)
end

deg = sum(A, dims=2)[:]
L = Diagonal(deg) - A
λ, V = eigen(Symmetric(L))

# -----------------------------------------------------------------------------
# 3. GeoGraphReal type and constructor
# -----------------------------------------------------------------------------
struct GeoGraphReal
    R_vals::Vector{Float64}
    anisotropy::Vector{SVector{3,Float64}}
    neighbors::Vector{Vector{Int}}
    coords::Vector{SVector{3,Float64}}
    modal_modes::Matrix{Float64}
    adjacency::Matrix{Float64}
end

function GeoGraphReal(states, modal_modes)
    n = length(states)
    return GeoGraphReal(
        zeros(n),                                # R_vals
        [SVector(0.,0.,0.) for _ in 1:n],        # anisotropy
        neighbors,                               # neighbors
        [SVector(0.,0.,0.) for _ in 1:n],        # coords (unused)
        modal_modes,                             # modal_modes
        A                                        # adjacency
    )
end

G = GeoGraphReal(pf_states, V)

# -----------------------------------------------------------------------------
# 4. Build Omega: mapping each PF state → deformed CYS coords
# -----------------------------------------------------------------------------
function deform_sulfenic(coords::Vector{SVector{3,Float64}}, bits::BitVector)
    d = copy(coords)
    for idx in findall(bits)
        n = normalize(d[idx])
        d[idx] += 0.5 * n
    end
    return d
end

function build_Omega_coords(states, coords, cys_idx)
    Omega = Dict{String,Vector{SVector{3,Float64}}}()
    for s in states
        bits = BitVector([c == '1' for c in s])
        Omega[s] = deform_sulfenic(coords, bits)
    end
    return Omega
end

# -----------------------------------------------------------------------------
# 5. Projection and scoring helpers
# -----------------------------------------------------------------------------
real_deformation_energy(c1, c2) = sqrt(sum(norm(c2[i]-c1[i])^2 for i in 1:length(c1)) / length(c1))
real_deformation_vector(c1, c2) = normalize(sum(c2[i] - c1[i] for i in 1:length(c1)))

struct Reactant
    name::String
    coords::Vector{SVector{3,Float64}}
    interaction_radius::Float64
end

oxidant = Reactant("oxidant",
    [SVector(0.0,0.0,0.0), SVector(-1.0,0.0,0.0), SVector(0.0,-1.0,0.0)],
    5.0
)

reactant_orbital_vector(rxn::Reactant) = normalize(sum(rxn.coords[i+1] - rxn.coords[i] for i in 1:length(rxn.coords)-1))
cos_theta(c1, c2, rxn::Reactant) = dot(real_deformation_vector(c1, c2), reactant_orbital_vector(rxn))

function projection_real_to_abstract(def_en, cosθ, i, j, V)
    v_i = V[i, 2:end]; v_j = V[j, 2:end]
    overlap = sum(v_i .* v_j)^2
    return def_en * cosθ * overlap
end

function compute_saddle_points(G::GeoGraphReal, f::Vector{Float64}; barrier=0.1)
    saddles = Dict{Tuple{Int,Int},Float64}()
    for i in 1:length(G.R_vals), j in G.neighbors[i]
        key = (min(i,j), max(i,j))
        saddles[key] = max(f[i], f[j]) + barrier
    end
    return saddles
end

function vibrational_overlap(i, j, V)
    v_i = V[i, 2:end]; v_j = V[j, 2:end]
    return sum(v_i .* v_j)^2
end

function modal_transition_score(i, j, G::GeoGraphReal, V, Omega, saddles; α=1.0)
    R_j    = G.R_vals[j]
    A_j    = norm(G.anisotropy[j])
    barb   = saddles[(min(i,j), max(i,j))]
    c1     = Omega[pf_states[i]]
    c2     = Omega[pf_states[j]]
    def_en = real_deformation_energy(c1, c2)
    cosθ   = cos_theta(c1, c2, oxidant)
    proj   = projection_real_to_abstract(def_en, cosθ, i, j, V)
    return R_j + A_j + proj + α * barb
end

# -----------------------------------------------------------------------------
# 6. Main execution block
# -----------------------------------------------------------------------------
function main(pdb_path::String)
    coords, cys_idx = load_ca_and_cys(pdb_path)
    prox_neighbors = build_proximity_graph(coords; cutoff=5.0)

    Omega   = build_Omega_coords(pf_states, coords, cys_idx)
    A_real  = construct_A_real(G, coords, cys_idx, prox_neighbors)
    G.R_vals .= A_real

    f       = zeros(length(pf_states))
    saddles = compute_saddle_points(G, f; barrier=0.1)

    i, j    = 1, 2  # 000 → 001
    score   = modal_transition_score(i, j, G, V, Omega, saddles)
    println("Modal transition score (000 → 001): ", round(score, digits=6))
end

# run on your PDB
main("/content/AF-P04406-F1-model_v4.pdb")
