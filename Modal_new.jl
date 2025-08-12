# =============================================================================
# Modal Geometric Field (MGF) — single-molecule demo (with anisotropy & tunneling)
# =============================================================================

using LinearAlgebra, Statistics
using StaticArrays
using Graphs
using SparseArrays
using Printf

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
safe_normalize(v::SVector{3,T}) where {T<:Real} = (n = norm(v); n == 0 ? v*zero(T) : v/n)

# Map the R bits to actual cysteine (or chosen) residue indices
function bit_to_cys_map(cys_idx::Vector{Int}, R::Int; mapping::Union{Nothing,Vector{Int}}=nothing)
    if mapping === nothing
        @assert length(cys_idx) ≥ R "Need at least R cysteines to map R bits."
        return cys_idx[1:R]
    else
        @assert length(mapping) == R
        return mapping
    end
end

# -----------------------------------------------------------------------------
# 0. Proteoform lattice (3-bit hypercube) + neighbors
# -----------------------------------------------------------------------------
const pf_states = ["000","001","010","011","100","101","110","111"]
const idx = Dict(s=>i for (i,s) in enumerate(pf_states))
const edges = [
  ("000","001"),("000","010"),("000","100"),
  ("001","011"),("001","101"),("010","011"),
  ("010","110"),("100","110"),("100","101"),
  ("011","111"),("101","111"),("110","111")
]

struct GeoGraphReal7
  pf_states::Vector{String}
  neighbors::Vector{Vector{Int}}
  adjacency::Matrix{Float32}
end

function GeoGraphReal7(pf_states, edges)
  nbrs = [Int[] for _ in 1:length(pf_states)]
  g = SimpleGraph(length(pf_states))
  for (u,v) in edges
    i,j = idx[u], idx[v]
    push!(nbrs[i], j); push!(nbrs[j], i)
    add_edge!(g, i, j)
  end
  A = Float32.(Matrix(adjacency_matrix(g)))
  return GeoGraphReal7(pf_states, nbrs, A)
end

# -----------------------------------------------------------------------------
# 1. PDB parsing (CA only) + CYS index list
# -----------------------------------------------------------------------------
function load_ca_and_cys(pdb_path::String)
    coords = SVector{3,Float64}[]
    cys_idx = Int[]
    open(pdb_path,"r") do io
        for ln in eachline(io)
            if startswith(ln,"ATOM") && strip(ln[13:16])=="CA"
                x,y,z = parse.(Float64, (ln[31:38],ln[39:46],ln[47:54]))
                push!(coords, SVector(x,y,z))
                if strip(ln[18:20])=="CYS"
                    push!(cys_idx, length(coords))
                end
            end
        end
    end
    @assert !isempty(coords) "No CA coordinates parsed."
    return coords, cys_idx
end

# Fallback: generate a mock α-helix-like backbone and mark some residues "CYS"
function mock_coords(n::Int=60; cys_every::Int=10)
    coords = SVector{3,Float64}[]
    for i in 1:n
        θ = 2π * i / 3.6
        x = 2.3*cos(θ); y = 2.3*sin(θ); z = 1.5*i
        push!(coords, SVector(x,y,z))
    end
    cys_idx = [i for i in 1:n if i % cys_every == 0]
    return coords, cys_idx
end

# -----------------------------------------------------------------------------
# 2. Real-space proximity graph (optional)
# -----------------------------------------------------------------------------
function build_proximity_graph(coords; cutoff=5.0)
  n = length(coords)
  neigh = [Int[] for _ in 1:n]
  for i in 1:n-1, j in i+1:n
    if norm(coords[i]-coords[j]) ≤ cutoff
      push!(neigh[i], j); push!(neigh[j], i)
    end
  end
  return neigh
end

# -----------------------------------------------------------------------------
# 3. Modal deformation: sulfenic-like shift along COM-relative direction
# -----------------------------------------------------------------------------
function deform_sulfenic_COM(coords::Vector{SVector{3,Float64}}, on_indices::Vector{Int}; delta=0.5)
    com = sum(coords) / length(coords)
    d = copy(coords)
    for k in on_indices
        dir = safe_normalize(SVector{3,Float64}(coords[k] - com))
        d[k] = d[k] + delta * dir
    end
    return d
end

# Build Ω(state) = deformed coordinates for each bitstring state (on full coords)
function build_Omega_coords(states::Vector{String},
                            all_coords::Vector{SVector{3,Float64}},
                            cys_map::Vector{Int})
    Omega = Dict{String,Vector{SVector{3,Float64}}}()
    for s in states
        bits = [c == '1' for c in s]   # length R
        on_idx = [cys_map[r] for r in 1:length(bits) if bits[r]]
        Omega[s] = deform_sulfenic_COM(all_coords, on_idx)
    end
    return Omega
end

# -----------------------------------------------------------------------------
# 4. Modal curvature vectors from occupancy Laplacian (R = Δρ)
# -----------------------------------------------------------------------------
function modal_curvature_vectors(G::GeoGraphReal7, Omega, rho::Vector{Float32}, cys_map::Vector{Int})
  n = length(G.pf_states)
  m = length(cys_map)
  M = [SVector{3,Float32}(0,0,0) for _ in 1:(n*m)]
  for i in 1:n
    for j in G.neighbors[i]
      w = Float32(rho[j] - rho[i]) # Laplacian action on ρ
      for (kk, resi) in enumerate(cys_map)
        Δ3 = Omega[G.pf_states[j]][resi] - Omega[G.pf_states[i]][resi]
        Δ  = SVector{3,Float32}(Float32(Δ3[1]), Float32(Δ3[2]), Float32(Δ3[3]))
        M[(i-1)*m + kk] += w * Δ
      end
    end
  end
  return M
end

function extract_curv_and_aniso(M::Vector{SVector{3,Float32}}, n::Int, m::Int)
  H = zeros(Float32, n, m)                               # per-(state,bit) curvature magnitude
  A = Array{SVector{3,Float32}}(undef, n, m)             # per-(state,bit) anisotropy direction
  zero3 = SVector{3,Float32}(0,0,0)
  for i in 1:n, k in 1:m
    v = M[(i-1)*m + k]
    H[i,k] = norm(v)
    A[i,k] = H[i,k] == 0f0 ? zero3 : v / H[i,k]
  end
  return H, A
end

# -----------------------------------------------------------------------------
# 5. Real deformation energy, alignment, projection, tunneling weight
# -----------------------------------------------------------------------------
morse_energy(r) = (D_e=0.5; a=1.5; r_e=2.0; D_e*(1 - exp(-a*(r - r_e)))^2)

function δC_real(c1::Vector{SVector{3,Float64}}, c2::Vector{SVector{3,Float64}})
  m = length(c1)
  return sum(morse_energy(norm(c2[i]-c1[i])) for i in 1:m) / m
end

struct Reactant; name::String; coords::Vector{SVector{3,Float64}}; end
reactant_orbital_vector(r::Reactant) = begin
    @assert length(r.coords) ≥ 2
    safe_normalize(sum(r.coords[i+1] - r.coords[i] for i in 1:length(r.coords)-1))
end

# H-weighted cosine between modal anisotropy directions and reactant vector
function alignment_cosine(j::Int, H::AbstractMatrix{<:Real},
                          A::AbstractMatrix{<:SVector{3,Float32}},
                          rxn::Reactant)
    rv = SVector{3,Float32}(reactant_orbital_vector(rxn))
    m  = size(H,2)
    num = 0f0; den = 0f0
    for k in 1:m
        hk = Float32(H[j,k])
        ak = A[j,k]
        c  = (norm(ak) == 0f0) ? 0f0 : dot(ak, rv)   # cosine in [-1,1]
        num += hk * c
        den += hk
    end
    return den == 0f0 ? 0f0 : clamp(num/den, -1f0, 1f0)
end

# Directional anisotropy term (separate from Rj): favors alignment with reactant
function directional_anisotropy(j::Int, H, A, rxn::Reactant)
    rv = SVector{3,Float32}(reactant_orbital_vector(rxn))
    m  = size(H,2)
    s = 0f0
    for k in 1:m
        hk = Float32(H[j,k])
        ak = A[j,k]
        s += hk * abs(norm(ak) == 0f0 ? 0f0 : dot(ak, rv))   # in [0, hk]
    end
    return s
end

# Modal overlap between states i and j (skip DC eigenvector)
function projection_real_to_abstract(δC::Real, cosθ::Real, i::Int, j::Int, V::AbstractMatrix)
    overlap2 = sum(@view(V[i,2:end]) .* @view(V[j,2:end]))^2
    return Float32(δC) * Float32(max(cosθ, 0)) * Float32(overlap2)
end

# Selection/tunneling weight
calign_from_cos(cosθ::Real) = 0.5f0 * (Float32(cosθ) + 1f0)            # [0,1]
function transition_weight(Rj, Aj, proj, cosθ; α=3.0f0)
    Calign = calign_from_cos(cosθ)                                     # [0,1]
    tunneling = exp(-α * (1f0 - Calign))                               # (0,1]
    base = max(Rj + Aj + proj, 0f0)
    return tunneling * base
end

# -----------------------------------------------------------------------------
# 6. Transition gate (patched)
# -----------------------------------------------------------------------------
function is_transition_allowed(i,j,G,V,Omega,rho,rxn,cys_map; θ_thresh=0.0, α=3.0f0)
  M   = modal_curvature_vectors(G, Omega, rho, cys_map)
  H,A = extract_curv_and_aniso(M, length(G.pf_states), length(cys_map))
  Rj  = sum(H[j, :])                                     # modal curvature magnitude
  Aj  = directional_anisotropy(j, H, A, rxn)             # directional anisotropy (patched)

  c1   = Omega[pf_states[i]]
  c2   = Omega[pf_states[j]]
  δC   = δC_real(c1, c2)

  cosr = alignment_cosine(j, H, A, rxn)
  proj = projection_real_to_abstract(δC, cosr, i, j, V)
  w    = transition_weight(Rj, Aj, proj, cosr; α=α)

  allowed = (Rj + Aj + proj > 0f0) && (!isnan(cosr)) && (cosr ≥ Float32(θ_thresh))
  return allowed, Rj, Aj, proj, cosr, w, H, A
end

# -----------------------------------------------------------------------------
# 7. Main demo
# -----------------------------------------------------------------------------
function main(pdb_path::Union{Nothing,String}=nothing; R_bits::Int=3, θ_thresh=0.0, α=3.0f0, delta=0.5)
  # Load structure (PDB if provided and readable, else mock)
  coords = SVector{3,Float64}[]; cys_idx = Int[]
  if pdb_path !== nothing && isfile(pdb_path)
      println("Reading PDB: $pdb_path")
      coords, cys_idx = load_ca_and_cys(pdb_path)
  else
      println("No/invalid PDB path. Using mock structure.")
      coords, cys_idx = mock_coords(60; cys_every=10)   # 60 residues, Cys at 10,20,30,40,50,60
  end
  @assert length(cys_idx) ≥ R_bits "Not enough cysteines to map $R_bits bits."
  cys_map = bit_to_cys_map(cys_idx, R_bits)

  # Build Ω for all states (respect chosen deformation magnitude)
  local function build_Omega_with_delta(states, coords, cys_map)
      Omega = Dict{String,Vector{SVector{3,Float64}}}()
      for s in states
          bits = [c == '1' for c in s]
          on_idx = [cys_map[r] for r in 1:length(bits) if bits[r]]
          Omega[s] = deform_sulfenic_COM(coords, on_idx; delta=delta)
      end
      return Omega
  end
  Omega = build_Omega_with_delta(pf_states, coords, cys_map)

  # Modal Laplacian eigenvectors on the hypercube
  g_bool = SimpleGraph(length(pf_states))
  for (u,v) in edges; add_edge!(g_bool, idx[u], idx[v]); end
  A_bool = Float32.(Matrix(adjacency_matrix(g_bool)))
  L_bool = Diagonal(sum(A_bool, dims=2)[:]) .- A_bool
  V      = eigen(Symmetric(Matrix(L_bool))).vectors

  # Initialize graph & occupancy (ρ=1 at '000')
  G   = GeoGraphReal7(pf_states, edges)
  rho = zeros(Float32, length(pf_states)); rho[1] = 1.0f0

  # Simple reactant vector (replace with realistic trajectory if desired)
  rxn = Reactant("oxidant",
         [SVector(0.0,0.0,0.0), SVector(-1.0,0.0,0.0), SVector(0.0,-1.0,0.0)])

  println("\nTransitions from '000':")
  for j in G.neighbors[1]
    ok,Rj,Aj,proj,cosr,w,H,A = is_transition_allowed(1, j, G, V, Omega, rho, rxn, cys_map; θ_thresh=θ_thresh, α=α)
    @printf("  000 → %s : allowed=%s | Rj=%.4f  Aj=%.4f  proj=%.4f  cosθ=%.4f  weight=%.4f\n",
            pf_states[j], string(ok), Rj, Aj, proj, cosr, w)
    for k in 1:length(cys_map)
        @printf("     bit%-2d  H=%.4f  A=%s\n", k, H[j,k], string(A[j,k]))
    end
  end
end

# ---- Run: provide your PDB path or leave nothing to use mock data ----------
# With your file:
 main("/content/AF-P04406-F1-model_v4.pdb"; R_bits=3, θ_thresh=0.0, α=3.0f0, delta=0.5)

# Mock demo:
 #main(nothing; R_bits=3, θ_thresh=0.0, α=3.0f0, delta=0.5)
