# =============================================================================
# Modal Geometric Field (MGF) — FULL SCRIPT with Dirichlet + Gauss-like check
# =============================================================================

using LinearAlgebra, Statistics
using StaticArrays
using Graphs
using SparseArrays
using Printf
using DelimitedFiles

# ------------------------------ Global knobs ---------------------------------
const LAMBDA_DIRICHLET = 1.0f0  # weight on Dirichlet energy (real→abstract coupling)
const LAMBDA_MORSE     = 0.0f0  # optional Morse contribution (set >0 to mix)
const SIGMA_DIRICHLET  = 8.0    # Å, Gaussian width for real-space edge weights
const DEFAULT_ALPHA    = 3.0f0  # tunneling sharpness
const DEFAULT_DELTA    = 0.5    # deformation magnitude (sulfenic-like)
const W_RADIUS         = 0.7    # exposure radius weight
const W_CROWD          = 0.3    # exposure crowding weight

# ------------------------------ Utilities ------------------------------------
safe_normalize(v::SVector{3,T}) where {T<:Real} = (n = norm(v); n == 0 ? v*zero(T) : v/n)
renormalize_rho!(ρ::AbstractVector{<:Real}) = (s = sum(ρ); s ≈ 1 ? ρ : (ρ ./= s))
function bit_to_cys_map(cys_idx::Vector{Int}, R::Int; mapping::Union{Nothing,Vector{Int}}=nothing)
    if mapping === nothing
        @assert length(cys_idx) ≥ R "Need at least R cysteines to map R bits."
        return cys_idx[1:R]
    else
        @assert length(mapping) == R
        return mapping
    end
end

# --------------------------- Modal hypercube graph ----------------------------
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

# --------------------------- PDB parsing & mock -------------------------------
function load_ca_and_cys(pdb_path::String)
    coords = SVector{3,Float64}[]; cys_idx = Int[]
    open(pdb_path,"r") do io
        for ln in eachline(io)
            if startswith(ln,"ATOM") && strip(ln[13:16])=="CA"
                x,y,z = parse.(Float64, (ln[31:38],ln[39:46],ln[47:54]))
                push!(coords, SVector(x,y,z))
                if strip(ln[18:20])=="CYS"; push!(cys_idx, length(coords)); end
            end
        end
    end
    @assert !isempty(coords) "No CA coordinates parsed."
    return coords, cys_idx
end
function mock_coords(n::Int=60; cys_every::Int=10)
    coords = SVector{3,Float64}[]
    for i in 1:n
        θ = 2π * i / 3.6; x = 2.3*cos(θ); y = 2.3*sin(θ); z = 1.5*i
        push!(coords, SVector(x,y,z))
    end
    cys_idx = [i for i in 1:n if i % cys_every == 0]
    return coords, cys_idx
end

# -------------------- Real-space proximity & exposure -------------------------
function build_proximity_graph(coords; cutoff=5.0)
  n = length(coords); neigh = [Int[] for _ in 1:n]
  for i in 1:n-1, j in i+1:n
    if norm(coords[i]-coords[j]) ≤ cutoff
      push!(neigh[i], j); push!(neigh[j], i)
    end
  end
  return neigh
end

function exposure_scores(coords::Vector{SVector{3,Float64}}, prox_neigh; w_radius=0.7, w_crowd=0.3)
    n = length(coords); com = sum(coords) / n
    r = [norm(coords[i] - com) for i in 1:n]; r ./= (mean(r) + eps())
    deg = [length(prox_neigh[i]) for i in 1:n]
    crowd = [1.0 / (d + 1) for d in deg]; crowd ./= (mean(crowd) + eps())
    e = [w_radius*r[i] + w_crowd*crowd[i] for i in 1:n]
    return e ./ (mean(e) + eps())
end

# -------------------- Modal deformation (exposure-weighted) -------------------
function deform_sulfenic_COM(coords::Vector{SVector{3,Float64}},
                             on_indices::Vector{Int};
                             delta=DEFAULT_DELTA,
                             exposure::Union{Nothing,Vector{Float64}}=nothing)
    com = sum(coords) / length(coords)
    d = copy(coords)
    for k in on_indices
        dir = coords[k] - com; dir = norm(dir) == 0 ? dir : dir / norm(dir)
        scale = exposure === nothing ? 1.0 : exposure[k]
        d[k] = d[k] + delta * scale * dir
    end
    return d
end

function build_Omega_coords(states::Vector{String},
                            all_coords::Vector{SVector{3,Float64}},
                            cys_map::Vector{Int},
                            exposure::Union{Nothing,Vector{Float64}}=nothing;
                            delta=DEFAULT_DELTA)
    Omega = Dict{String,Vector{SVector{3,Float64}}}()
    for s in states
        bits = [c == '1' for c in s]
        on_idx = [cys_map[r] for r in 1:length(bits) if bits[r]]
        Omega[s] = deform_sulfenic_COM(all_coords, on_idx; delta=delta, exposure=exposure)
    end
    return Omega
end

# -------------------- Modal curvature vectors (R = Δρ driver) -----------------
function modal_curvature_vectors(G::GeoGraphReal7, Omega, rho::Vector{Float32}, cys_map::Vector{Int})
  n = length(G.pf_states); m = length(cys_map)
  M = [SVector{3,Float32}(0,0,0) for _ in 1:(n*m)]
  for i in 1:n
    for j in G.neighbors[i]
      w = Float32(rho[j] - rho[i])
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
  H = zeros(Float32, n, m); A = Array{SVector{3,Float32}}(undef, n, m)
  zero3 = SVector{3,Float32}(0,0,0)
  for i in 1:n, k in 1:m
    v = M[(i-1)*m + k]; H[i,k] = norm(v); A[i,k] = H[i,k] == 0f0 ? zero3 : v / H[i,k]
  end
  return H, A
end

# -------------------- Real-space energies: Dirichlet + optional Morse --------
morse_energy(r) = (D_e=0.5; a=1.5; r_e=2.0; D_e*(1 - exp(-a*(r - r_e)))^2)
function δC_real_morse(c1::Vector{SVector{3,Float64}}, c2::Vector{SVector{3,Float64}})
  m = length(c1); return sum(morse_energy(norm(c2[i]-c1[i])) for i in 1:m) / m
end

# Dirichlet energy helpers
function realspace_weights(coords::Vector{SVector{3,Float64}}, prox_neigh; σ::Float64=SIGMA_DIRICHLET)
    n = length(coords)
    W = zeros(Float64, n, n)
    inv2σ2 = 1.0 / (2σ^2)
    for i in 1:n
        for j in prox_neigh[i]
            if W[i,j] == 0.0
                d2 = sum((coords[i] .- coords[j]).^2)
                w = exp(-d2 * inv2σ2)
                W[i,j] = w; W[j,i] = w
            end
        end
    end
    return W
end
deformation_field(c1, c2) = [c2[i] - c1[i] for i in eachindex(c1)]
function dirichlet_energy(coords::Vector{SVector{3,Float64}},
                          prox_neigh,
                          c1::Vector{SVector{3,Float64}},
                          c2::Vector{SVector{3,Float64}};
                          σ::Float64=SIGMA_DIRICHLET)
    d = deformation_field(c1, c2)
    W = realspace_weights(coords, prox_neigh; σ=σ)
    num = 0.0; denom = 0.0; n = length(coords)
    for i in 1:n-1, j in i+1:n
        wij = W[i,j]
        if wij > 0
            num += wij * sum((d[i] .- d[j]).^2)
            denom += wij
        end
    end
    return 0.5 * (denom > 0 ? num / denom : 0.0)
end

# -------------------- Alignment / projection / weights ------------------------
struct Reactant; name::String; coords::Vector{SVector{3,Float64}}; end
reactant_orbital_vector(r::Reactant) = begin
    @assert length(r.coords) ≥ 2
    safe_normalize(sum(r.coords[i+1] - r.coords[i] for i in 1:length(r.coords)-1))
end

function alignment_cosine(j::Int, H::AbstractMatrix{<:Real},
                          A::AbstractMatrix{<:SVector{3,Float32}},
                          rxn::Reactant)
    rv = SVector{3,Float32}(reactant_orbital_vector(rxn))
    m  = size(H,2); num = 0f0; den = 0f0
    for k in 1:m
        hk = Float32(H[j,k]); ak = A[j,k]
        c  = (norm(ak) == 0f0) ? 0f0 : dot(ak, rv)
        num += hk * c; den += hk
    end
    return den == 0f0 ? 0f0 : clamp(num/den, -1f0, 1f0)
end

function directional_anisotropy(j::Int, H, A, rxn::Reactant)
    rv = SVector{3,Float32}(reactant_orbital_vector(rxn))
    m  = size(H,2); s = 0f0
    for k in 1:m
        hk = Float32(H[j,k]); ak = A[j,k]
        s += hk * abs(norm(ak) == 0f0 ? 0f0 : dot(ak, rv))
    end
    return s
end

function projection_real_to_abstract(δC::Real, cosθ::Real, i::Int, j::Int, V::AbstractMatrix)
    overlap2 = sum(@view(V[i,2:end]) .* @view(V[j,2:end]))^2
    return Float32(δC) * Float32(max(cosθ, 0)) * Float32(overlap2)
end

calign_from_cos(cosθ::Real) = 0.5f0 * (Float32(cosθ) + 1f0)
function transition_weight(Rj, Aj, proj, cosθ; α=DEFAULT_ALPHA)
    Calign = calign_from_cos(cosθ); tunneling = exp(-α * (1f0 - Calign))
    base = max(Rj + Aj + proj, 0f0); return tunneling * base
end

# -------------------- Axiom / Laplacian checks -------------------------------
function laplacian_rho(GA::AbstractMatrix{<:Real}, ρ::AbstractVector{<:Real})
    d = vec(sum(GA, dims=2)); L = Diagonal(d) .- GA; return L * ρ
end
function check_axioms(GA::AbstractMatrix{<:Real}, ρ::AbstractVector{<:Real}; tol=1e-6)
    vol_ok = abs(sum(ρ) - 1) ≤ tol
    Δρ = laplacian_rho(GA, ρ)
    lap_ok = abs(sum(Δρ)) ≤ tol
    return vol_ok, lap_ok, sum(ρ), sum(Δρ)
end

# -------------------- Discrete Gauss-like law on hypercube -------------------
function oriented_edges(A_bool::AbstractMatrix{<:Real})
    n = size(A_bool,1); edges = Tuple{Int,Int}[]
    for i in 1:n-1, j in i+1:n
        if A_bool[i,j] != 0; push!(edges, (i,j)); end
    end
    return edges
end
function grad_node_to_edge(φ::AbstractVector{<:Real}, edges::Vector{Tuple{Int,Int}})
    g = zeros(Float64, length(edges))
    for (k,(i,j)) in enumerate(edges); g[k] = φ[j] - φ[i]; end
    return g
end
function div_edge_to_node(F::AbstractVector{<:Real}, edges::Vector{Tuple{Int,Int}}, n_nodes::Int)
    div = zeros(Float64, n_nodes)
    for (k,(i,j)) in enumerate(edges)
        div[i] -= F[k]; div[j] += F[k]
    end
    return div
end
function solve_poisson_mean_zero(L::AbstractMatrix{<:Real}, ρ::AbstractVector{<:Real})
    n = length(ρ); ρ0 = ρ .- mean(ρ)
    φ = pinv(Matrix(L)) * ρ0
    φ .-= mean(φ); return φ
end
function test_modal_gauss(A_bool::AbstractMatrix{<:Real}, ρ::AbstractVector{<:Real}; tol=1e-6)
    n = length(ρ)
    L = Diagonal(vec(sum(A_bool, dims=2))) .- A_bool
    e = oriented_edges(A_bool)
    φ = solve_poisson_mean_zero(L, ρ)
    R_edge = grad_node_to_edge(φ, e)
    divR = div_edge_to_node(R_edge, e, n)
    ρ0 = ρ .- mean(ρ)
    residual = norm(divR - ρ0) / (norm(ρ0) + tol)
    return residual
end

# -------------------- Allowed-transition test (with Dirichlet) ---------------
function is_transition_allowed(i,j,G,V,Omega,rho,rxn,cys_map;
                               θ_thresh=0.0, α=DEFAULT_ALPHA,
                               coords_all=nothing, prox_neigh_all=nothing)
  M   = modal_curvature_vectors(G, Omega, rho, cys_map)
  H,A = extract_curv_and_aniso(M, length(G.pf_states), length(cys_map))
  Rj  = sum(H[j, :])
  Aj  = directional_anisotropy(j, H, A, rxn)

  c1   = Omega[pf_states[i]]
  c2   = Omega[pf_states[j]]

  @assert coords_all !== nothing && prox_neigh_all !== nothing
  δC_D = Float32(dirichlet_energy(coords_all, prox_neigh_all, c1, c2; σ=SIGMA_DIRICHLET))
  δC_M = Float32(δC_real_morse(c1, c2))
  δC   = LAMBDA_DIRICHLET*δC_D + LAMBDA_MORSE*δC_M

  cosr = alignment_cosine(j, H, A, rxn)
  proj = projection_real_to_abstract(δC, cosr, i, j, V)
  w    = transition_weight(Rj, Aj, proj, cosr; α=α)

  allowed = (Rj + Aj + proj > 0f0) && (!isnan(cosr)) && (cosr ≥ Float32(θ_thresh))
  return allowed, Rj, Aj, proj, cosr, w, H, A, δC_D, δC_M
end

# -------------------- Full-manifold scan (geometry-only) ---------------------
function run_full_scan(pdb_path::Union{Nothing,String}=nothing;
                       R_bits::Int=3, θ_thresh=0.0, α::Float32=DEFAULT_ALPHA,
                       delta=DEFAULT_DELTA, w_radius=W_RADIUS, w_crowd=W_CROWD)

  # Load structure
  coords = SVector{3,Float64}[]; cys_idx = Int[]
  if pdb_path !== nothing && isfile(pdb_path)
      println("Reading PDB: $pdb_path"); coords, cys_idx = load_ca_and_cys(pdb_path)
  else
      println("No/invalid PDB path. Using mock structure.")
      coords, cys_idx = mock_coords(60; cys_every=10)
  end
  @assert length(cys_idx) ≥ R_bits "Not enough cysteines to map $R_bits bits."
  cys_map = bit_to_cys_map(cys_idx, R_bits)

  # Modal graph & eigens
  G   = GeoGraphReal7(pf_states, edges)
  A_bool = G.adjacency
  L_bool = Diagonal(vec(sum(A_bool, dims=2))) .- A_bool
  V      = eigen(Symmetric(Matrix(L_bool))).vectors

  # Exposure & Ω
  prox_neigh = build_proximity_graph(coords)
  E = exposure_scores(coords, prox_neigh; w_radius=w_radius, w_crowd=w_crowd)
  Omega = build_Omega_coords(pf_states, coords, cys_map, E; delta=delta)

  # Reactant vector
  rxn = Reactant("oxidant",
         [SVector(0.0,0.0,0.0), SVector(-1.0,0.0,0.0), SVector(0.0,-1.0,0.0)])

  # Axiom check at '000'
  ρ0 = zeros(Float32, length(pf_states)); ρ0[1] = 1.0f0; renormalize_rho!(ρ0)
  vol_ok, lap_ok, s_rho, s_lap = check_axioms(A_bool, ρ0; tol=1e-7)
  @printf("\nAxiom checks (reference ρ at '000'): ∑ρ=%.6f (%s), ∑Δρ=%.6e (%s)\n",
          s_rho, vol_ok ? "OK" : "VIOLATION", s_lap, lap_ok ? "OK" : "VIOLATION")

  # Full scan
  n = length(pf_states)
  adj_allow = zeros(Int, n, n)

  println("\n===== Full State Transition Scan (geometry-only) =====")
  for i in 1:n
      src = pf_states[i]
      ρ = zeros(Float32, n); ρ[i] = 1.0f0; renormalize_rho!(ρ)

      # Gauss-like check at this source occupancy
      gauss_resid = test_modal_gauss(A_bool, ρ; tol=1e-9)
      @printf("\nTransitions from '%s'  [∑ρ=%.1f, Gauss residual=%.3e]\n", src, sum(ρ), gauss_resid)

      for j in G.neighbors[i]
          ok,Rj,Aj,proj,cosr,w,H,A,δC_D,δC_M =
              is_transition_allowed(i, j, G, V, Omega, ρ, rxn, cys_map;
                                    θ_thresh=θ_thresh, α=α,
                                    coords_all=coords, prox_neigh_all=prox_neigh)

          @printf("  %s → %s : allowed=%s | Rj=%.4f  Aj=%.4f  proj=%.4f  cosθ=%.4f  weight=%.4f  Dirichlet=%.4f  Morse=%.4f\n",
                  src, pf_states[j], string(ok), Rj, Aj, proj, cosr, w, δC_D, δC_M)
          for k in 1:length(cys_map)
              @printf("     bit%-2d  H=%.4f  A=%s\n", k, H[j,k], string(A[j,k]))
          end
          if ok; adj_allow[i, j] = 1; end
      end
  end

  println("\nAdjacency matrix of ALLOWED transitions (rows=source, cols=target):")
  display(adj_allow)

  writedlm("transition_matrix.csv", adj_allow, ',')
  open("state_index_map.csv","w") do io
      println(io, "index,state"); for i in 1:n; println(io, "$i,$(pf_states[i])"); end
  end
  println("Saved: transition_matrix.csv, state_index_map.csv")

  return adj_allow
end

# ----------------------------- Run one of these -------------------------------
# With your PDB:
 run_full_scan("/content/AF-P04406-F1-model_v4.pdb"; R_bits=3, θ_thresh=0.0, α=3.0f0, delta=0.5)

# Mock:
# run_full_scan(nothing; R_bits=3, θ_thresh=0.0, α=3.0f0, delta=0.5)
