# ============================================================================
# Real-structure manifold: bitwise curvature r(x) + vibrational eigenmodes
# (graph built from CA proximity; no latent manifold used here)
# ============================================================================

using LinearAlgebra, Statistics, Printf
using StaticArrays
using Graphs

# ---------------- I/O + mapping ----------------
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

function bit_to_cys_map(cys_idx::Vector{Int}, R::Int; mapping::Union{Nothing,Vector{Int}}=nothing)
    if mapping === nothing
        @assert length(cys_idx) >= R "Need at least $R cysteines."
        return cys_idx[1:R]
    else
        @assert length(mapping) == R
        return mapping
    end
end

# ---------------- Real-manifold graph ----------------
# Weighted edges for CA residues: w_ij = exp(-d^2 / (2σ^2)) if d ≤ cutoff
function build_weighted_graph(coords::Vector{SVector{3,Float64}}; cutoff=6.5, σ=4.0)
    n = length(coords)
    W = zeros(Float64, n, n)
    inv2σ2 = 1.0 / (2σ^2)
    for i in 1:n-1, j in i+1:n
        d = norm(coords[i]-coords[j])
        if d ≤ cutoff
            w = exp(- (d*d) * inv2σ2)
            W[i,j] = w; W[j,i] = w
        end
    end
    d = vec(sum(W, dims=2))
    L = Diagonal(d) - W
    return W, L, d
end

# ---------------- Potential / curvature ----------------
# Solve L φ = ρ - mean(ρ); align sign so occupied bits are minima
function solve_poisson_mean_zero(L::AbstractMatrix, ρ::AbstractVector)
    ρ0 = ρ .- mean(ρ)
    φ  = pinv(Matrix(L)) * ρ0
    φ .-= mean(φ)
    return φ
end

function align_phi_to_bits!(φ::Vector{Float64}, occ_res::Vector{Int})
    # If the average φ over occupied residues is not the minimum, flip sign
    μ_occ = mean(φ[occ_res])
    if μ_occ > minimum(φ) + 1e-12
        φ .*= -1
    end
    return φ
end

# ---------------- Modes ----------------
# Standard eigenmodes L v = λ v (drop near-zero)
function vibrational_modes(L::AbstractMatrix; k::Int=12)
    F = eigen(Symmetric(Matrix(L)))
    λ = F.values; V = F.vectors
    ord = sortperm(λ); λ = λ[ord]; V = V[:,ord]
    keep = findall(x -> x > 1e-9, λ)
    if isempty(keep)
        return Float64[], Array{Float64}(undef, size(V,1), 0)
    end
    kidx = keep[1]:min(keep[1]+k-1, lastindex(λ))
    return λ[kidx], V[:,kidx]
end

# ---------------- Driver ----------------
"""
run_real_manifold(
   pdb_path; R_bits=3, real_state=\"000\", mapping=nothing,
   cutoff=6.5, sigma=4.0, k=8)

- Builds residue graph from PDB (CA atoms).
- Maps first R bits to cysteine indices (or custom mapping).
- Occupancy ρ over residues: ones on residues for bits==1 in `real_state`, else 0.
- Solves L φ = ρ - mean(ρ), aligns sign so occupied residues sit at minima.
- Returns curvature R = L φ, per-bit curvature r_bit, and Laplacian modes.

If `real_state` has no 1 bits (e.g., \"000\"), it will not create a source; in that case,
set a test state like \"100\" to see bitwise curvature for that bit.
"""
function run_real_manifold(pdb_path::String;
                           R_bits::Int=3,
                           real_state::String="100",
                           mapping::Union{Nothing,Vector{Int}}=nothing,
                           cutoff=6.5, sigma=4.0, k=8)

    @assert isfile(pdb_path) "PDB not found at $pdb_path"
    println("Reading PDB: $pdb_path")
    coords, cys_idx = load_ca_and_cys(pdb_path)
    @printf("Parsed %d CA residues; %d CYS CA found\n", length(coords), length(cys_idx))

    cys_map = bit_to_cys_map(cys_idx, R_bits; mapping=mapping)
    println("Bit→Cys mapping (1-based CA indices): ", cys_map)

    # Build weighted residue graph
    W, L, d = build_weighted_graph(coords; cutoff=cutoff, σ=sigma)
    @printf("Graph: %d nodes, %d weighted edges (cutoff=%.1fÅ, σ=%.1fÅ)\n",
            length(coords), count(!iszero, triu(W)) - length(coords), cutoff, sigma)

    # Occupancy over residues: ones at occupied bit residues
    bits = [c=='1' for c in real_state[1:R_bits]]
    occ_res = [cys_map[b] for b in 1:R_bits if bits[b]]
    if isempty(occ_res)
        error("real_state=\"$real_state\" has no 1-bits; pick a state like \"100\" to probe a bit.")
    end
    ρ = zeros(Float64, length(coords))
    ρ[occ_res] .= 1.0
    ρ ./= sum(ρ)   # normalize total mass to 1

    # Potential / curvature on real manifold
    φ = solve_poisson_mean_zero(L, ρ)
    align_phi_to_bits!(φ, occ_res)
    R = L * φ
    E_dir = 0.5 * dot(φ, L*φ)

    @printf("\nVolume invariance (residue domain): sum(ρ)=%.6f | sum(R)=%.3e\n", sum(ρ), sum(R))
    @printf("Dirichlet energy on real manifold:  E[φ]=%.6f\n", E_dir)

    # Bitwise curvature r(x): take curvature at the mapped residue(s)
    r_bits = Dict{Int,Float64}()
    for b in 1:R_bits
        r_bits[b] = R[cys_map[b]]
    end
    println("\nBitwise curvature r_bit at mapped residues:")
    for b in 1:R_bits
        @printf("  bit %d @ residue %d : r = %+ .6e\n", b, cys_map[b], r_bits[b])
    end

    # Modes on residue graph
    λ, V = vibrational_modes(L; k=k)
    @printf("\nLowest %d vibrational eigenmodes of real manifold (L v = λ v):\n", length(λ))
    for m in 1:length(λ)
        # per-bit participation: squared amplitude at that residue normalized by column norm
        colnorm2 = sum(abs2, V[:,m]) + eps()
        @printf("  mode %d : λ = %.6f | ", m, λ[m])
        for b in 1:R_bits
            p = V[cys_map[b], m]^2 / colnorm2
            @printf("bit%d-part=%.4f  ", b, p)
        end
        println()
    end

    # Identify candidate geometric primitive:
    #   - bit with largest |r_bit|
    #   - bit with highest participation in the first nonzero mode
    max_r_bit = argmax([abs(r_bits[b]) for b in 1:R_bits])
    best_mode = 1
    best_part = argmax([V[cys_map[b], best_mode]^2 for b in 1:R_bits])

    println("\nHeuristics for geometric primitive:")
    @printf("  largest |curvature| bit : bit %d (r=%+.3e)\n", max_r_bit, r_bits[max_r_bit])
    @printf("  highest participation in mode %d : bit %d (%.4f)\n",
            best_mode, best_part, V[cys_map[best_part], best_mode]^2 / (sum(abs2,V[:,best_mode])+eps()))

    return (coords=coords, cys_map=cys_map, W=W, L=L, φ=φ, R=R, r_bits=r_bits, λ=λ, V=V)
end

# ---------------- Run example ----------------
# Pick a state that excites a bit; e.g., bit1 on => "100"
res = run_real_manifold("/content/AF-P04406-F1-model_v4.pdb";
                        R_bits=3, real_state="100", cutoff=6.5, sigma=4.0, k=8)
