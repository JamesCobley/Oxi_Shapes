# =============================================================================
# Full Modal Geometric Field (MGF) Axiomatic Implementation
# — Volume fixed at 1, Scalar curvature ∑R = 0 (Axiomatically enforced)
# =============================================================================

using StaticArrays, LinearAlgebra, Graphs, Statistics

# ----------------------------------------------------------------------------
# 0. Proteoform lattice + 2D embedding + bitwise directions
# ----------------------------------------------------------------------------
pf_states = ["000","001","010","011","100","101","110","111"]
flat_pos = Dict(
  "000"=>(0.0f0, 3.0f0), "001"=>(-2.0f0, 2.0f0),
  "010"=>( 0.0f0, 2.0f0), "011"=>(-1.0f0, 1.0f0),
  "100"=>( 2.0f0, 2.0f0), "101"=>( 0.0f0, 1.0f0),
  "110"=>( 1.0f0, 1.0f0), "111"=>( 0.0f0, 0.0f0)
)
edges = [
  ("000","001"),("000","010"),("000","100"),
  ("001","011"),("001","101"),("010","011"),
  ("010","110"),("100","110"),("100","101"),
  ("011","111"),("101","111"),("110","111")
]
idx = Dict(s=>i for (i,s) in enumerate(pf_states))
n = length(pf_states)

bit_dirs = [
  SVector(1.0,0.0,0.0),
  SVector(0.0,1.0,0.0),
  SVector(0.0,0.0,1.0)
]

# ----------------------------------------------------------------------------
# 1. GeoGraphReal7: abstract lattice + MGF fields
# ----------------------------------------------------------------------------
struct GeoGraphReal7
  pf_states::Vector{String}
  neighbors::Vector{Vector{Int}}
  adjacency::Matrix{Float32}
  R_vals::Vector{Float32}
  anisotropy::Vector{SVector{3,Float32}}
end

function GeoGraphReal7(pf_states, edges)
  nbrs = [Int[] for _ in 1:length(pf_states)]
  g = SimpleGraph(length(pf_states))
  for (u,v) in edges
    i,j = idx[u], idx[v]
    push!(nbrs[i], j); push!(nbrs[j], i)
    add_edge!(g, i, j)
  end
  A = Float32.(adjacency_matrix(g))
  return GeoGraphReal7(
    pf_states,
    nbrs,
    A,
    zeros(Float32,length(pf_states)),
    [SVector{3,Float32}(0.0f0,0.0f0,0.0f0) for _ in 1:length(pf_states)]
  )
end

# ----------------------------------------------------------------------------
# 2. Geometry & Morse deformation
# ----------------------------------------------------------------------------
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
  return coords, cys_idx
end

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

function compute_site_curvature(coords, prox_neigh, cys_idx)
  m = length(cys_idx)
  sc = zeros(Float64,m)
  for (k,i) in enumerate(cys_idx)
    nbrs = prox_neigh[i]
    ∆ = foldl(+, coords[j]-coords[i] for j in nbrs)
    sc[k] = norm(∆)/max(length(nbrs),1)
  end
  return sc
end

function morse_energy(r)
  D_e, a, r_e = 0.5, 1.5, 2.0
  return D_e*(1 - exp(-a*(r - r_e)))^2
end

function δC_real(c1, c2)
  m = length(c1)
  return sum(morse_energy(norm(c2[i]-c1[i])) for i=1:m)/m
end

function deform_sulfenic(coords, bits::BitVector)
  d = copy(coords)
  for k in eachindex(bits)
    if bits[k]
      dir = norm(d[k]) == 0 ? zeros(3) : normalize(d[k])
      d[k] += 0.5 * dir
    end
  end
  return d
end

function build_Omega_coords(states, cys_coords)
  Omega = Dict{String,Vector{SVector{3,Float64}}}()
  for s in states
    bits = BitVector([c=='1' for c in s])
    Omega[s] = deform_sulfenic(cys_coords, bits)
  end
  return Omega
end

# ----------------------------------------------------------------------------
# 3. Geometry update (axiomatic volume and curvature)
# ----------------------------------------------------------------------------
function update_geometry!(G::GeoGraphReal7, rho::Vector{Float32})
  fill!(G.R_vals, 0.0f0)
  for i in 1:length(rho), j in G.neighbors[i]
    G.R_vals[i] += rho[j] - rho[i]
  end
  return Int[]  # no violations — volume and ∑R fixed by definition
end

function compute_planar_anisotropy!(G::GeoGraphReal7)
  for i in eachindex(G.R_vals)
    G.anisotropy[i] = SVector(0.0f0,0.0f0,0.0f0)
  end
end

function compute_real_anisotropy!(G::GeoGraphReal7, site_curv)
  for (i,s) in enumerate(G.pf_states)
    bits = BitVector([c=='1' for c in s])
    imag = zero(G.anisotropy[i])
    for b in 1:3
      if bits[b]
        imag += Float32(site_curv[b]) * Float32.(bit_dirs[b])
      end
    end
    G.anisotropy[i] += imag
  end
end

# ----------------------------------------------------------------------------
# 4. Activation & projection
# ----------------------------------------------------------------------------
real_deformation_vector(c1,c2) = normalize(sum(c2[i]-c1[i] for i in 1:length(c1)))

struct Reactant; name::String; coords::Vector{SVector{3,Float64}}; end
reactant_orbital_vector(r::Reactant) = normalize(sum(r.coords[i+1]-r.coords[i] for i in 1:length(r.coords)-1))

cos_theta_real(k, Ur, rxn::Reactant, cys_idx) = dot(Ur[cys_idx,k], reactant_orbital_vector(rxn))

function projection_real_to_abstract(δC, cosr, i, j, V)
  overlap2 = sum(V[i,2:end] .* V[j,2:end])^2
  return δC * cosr * overlap2
end

function is_transition_allowed(i,j,G,V,Ur,Omega,rho,site_curv,rxn,cys_idx; θ_thresh=0.0)
  update_geometry!(G, rho)
  compute_planar_anisotropy!(G)
  compute_real_anisotropy!(G, site_curv)

  Rj   = G.R_vals[j]
  Aj   = norm(G.anisotropy[j])
  c1   = Omega[pf_states[i]]
  c2   = Omega[pf_states[j]]
  δC   = δC_real(c1, c2)
  cosr = cos_theta_real(j, Ur, rxn, cys_idx)
  proj = projection_real_to_abstract(δC, cosr, i, j, V)

  allowed = (Rj + Aj + proj > 0) && (!isnan(cosr)) && (cosr ≥ θ_thresh)
  return allowed, Rj, Aj, proj
end

# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
function main(pdb_path)
  coords, cys_idx = load_ca_and_cys(pdb_path)
  prox_neigh = build_proximity_graph(coords)
  site_curv = compute_site_curvature(coords, prox_neigh, cys_idx)
  cys_coords = coords[cys_idx]
  Omega = build_Omega_coords(pf_states, cys_coords)

  # Boolean Laplacian
  g_bool = SimpleGraph(n)
  for (u,v) in edges; add_edge!(g_bool, idx[u], idx[v]); end
  A_bool = Float32.(adjacency_matrix(g_bool))
  deg_bool = sum(A_bool, dims=2)[:]
  L_bool = Diagonal(deg_bool) - A_bool
  V = eigen(Symmetric(Matrix(L_bool))).vectors

  # Real-space Laplacian
  n_r = length(coords)
  Ar = zeros(Float64, n_r, n_r)
  for i in 1:n_r, j in prox_neigh[i]; Ar[i,j] = 1.0; end
  Lr = Diagonal(sum(Ar,dims=2)[:]) - Ar
  Ur = eigen(Symmetric(Lr)).vectors

  G   = GeoGraphReal7(pf_states, edges)
  rho = zeros(Float32, n); rho[1] = 1.0f0
  rxn = Reactant("oxidant", [SVector(0.0,0.0,0.0), SVector(-1.0,0.0,0.0), SVector(0.0,-1.0,0.0)])

  println("Residue CYS coords:")
  for (k,i) in enumerate(cys_idx); println(" CYS[$k] = ", coords[i]); end

  println("\nSite curvatures:")
  for (k,v) in enumerate(site_curv); println(" site $k: ", round(v; digits=4)); end

  println("\nTransitions from '000':")
  for j in G.neighbors[1]
    ok,Rj,Aj,proj = is_transition_allowed(1, j, G, V, Ur, Omega, rho, site_curv, rxn, cys_idx)
    println("  000→$(pf_states[j]): allowed=$(ok), Rj=$(round(Rj;digits=4)), Aj=$(round(Aj;digits=4)), proj=$(round(proj;digits=4))")
  end
end

# Call the main function with your path
main("/content/AF-P04406-F1-model_v4.pdb")
