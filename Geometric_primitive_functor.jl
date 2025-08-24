#!/usr/bin/env julia
# ===============================================================
# Oxi-Shapes MGF — PDE → Modal Projection with Surface Energy & Morse
# (no SASA; structural priors from PDE only)
# ===============================================================

using LinearAlgebra, StaticArrays, Graphs, Printf, Statistics, Random

# ---------------- Params (tune freely) ----------------
const DIST_CUTOFF  = 6.5       # Å, real-graph contact cutoff
const SIGMA_KERNEL = 4.0       # Å, Gaussian width for W
const BETA         = 1.0       # β in exp(-β[…])
const KAPPA_CURV   = 1.0       # κ for ΔU_curv = κ (R[y]-R[x])
const GAMMA_E      = 1.0       # penalty on Dirichlet energy E_b
const GAMMA_R      = 1.0       # penalty on |R_b|
const GAMMA_S      = 1.0       # positive bias for surface energy (normalized)
const LAMBDA_MORSE = 0.7       # min→exp(-λ), saddle→1, max→exp(+λ)
const STEPS        = 60        # Markov steps for demo
const RNG_SEED     = 7

# ---------------- PDB parsing (CA only) ----------------
function load_ca_and_cys(pdb_path::String)
    coords = SVector{3,Float64}[]
    cys_idx = Int[]
    open(pdb_path,"r") do io
        for ln in eachline(io)
            if startswith(ln,"ATOM") && strip(ln[13:16]) == "CA"
                x = parse(Float64, ln[31:38]); y = parse(Float64, ln[39:46]); z = parse(Float64, ln[47:54])
                push!(coords, SVector(x,y,z))
                if strip(ln[18:20]) == "CYS"
                    push!(cys_idx, length(coords))
                end
            end
        end
    end
    @assert !isempty(coords) "No CA atoms parsed."
    return coords, cys_idx
end

# ------------- Real manifold: W, L, graph G ----------------
function build_weighted_graph(coords::Vector{SVector{3,Float64}}; cutoff=DIST_CUTOFF, σ=SIGMA_KERNEL)
    n = length(coords)
    W = zeros(Float64, n, n)
    inv2σ2 = 1.0 / (2σ^2)
    for i in 1:n-1, j in i+1:n
        d = norm(coords[i] - coords[j])
        if d <= cutoff
            w = exp(-(d*d)*inv2σ2)
            W[i,j] = w; W[j,i] = w
        end
    end
    D = Diagonal(vec(sum(W, dims=2)))
    L = D - W
    G = SimpleGraph(n)
    for i in 1:n-1, j in i+1:n
        if W[i,j] > 0
            add_edge!(G, i, j)
        end
    end
    return W, L, G
end

# ------------- Laplacian pseudo-inverse & Poisson -------------
function laplacian_pinv(L::AbstractMatrix; tol=1e-12)
    F = eigen(Symmetric(Matrix(L)))
    λ, U = F.values, F.vectors
    λ⁺ = similar(λ)
    @inbounds for i in eachindex(λ)
        λ⁺[i] = λ[i] > tol ? inv(λ[i]) : 0.0
    end
    U * Diagonal(λ⁺) * U'
end

function solve_poisson_mean_zero(L::AbstractMatrix, ρ::AbstractVector)
    ρ0 = ρ .- mean(ρ)
    φ  = laplacian_pinv(L) * ρ0
    φ .-= mean(φ)
    φ
end

dirichlet_energy(L::AbstractMatrix, φ::AbstractVector) = 0.5 * dot(φ, L*φ)

# ----- Real-graph surface energy density at node i -----
@inline function local_surface_energy(W::AbstractMatrix, φ::AbstractVector, i::Int)
    acc = 0.0
    @inbounds for j in 1:length(φ)
        w = W[i,j]
        w == 0 && continue
        acc += 0.5 * w * (φ[i] - φ[j])^2
    end
    acc
end

# ----------------- Morse classification on G -----------------
# Given scalar field ψ on nodes of G: min if ψ[i] < all neighbors; max if > all; else saddle
function morse_type(G::SimpleGraph, ψ::AbstractVector, i::Int)
    N = neighbors(G, i)
    isempty(N) && return :saddle
    ψi = ψ[i]; ψN = ψ[N]
    if all(ψi .< ψN)
        return :min
    elseif all(ψi .> ψN)
        return :max
    else
        return :saddle
    end
end

# ------------- Modal manifold (R=3 diamond) ----------------
const STATES = ["000","100","010","001","110","101","011","111"]
const IDX = Dict(s=>i for (i,s) in enumerate(STATES))
bitvec(s::String) = [c=='1' for c in collect(s)]
hamming1_neighbors(s::String) = [begin
    tmp = collect(s); tmp[k] = (tmp[k]=='0' ? '1' : '0'); String(tmp)
end for k in 1:length(s)]

# -------- Per-bit PDE geometry + surface + Morse -----------
mutable struct BitGeom
    Rb::Float64         # curvature at source residue (real graph)
    Eb::Float64         # Dirichlet energy of φ_b
    Surf::Float64       # local surface energy at source residue
    Morse::Symbol       # :min, :saddle, :max at source residue
end

function per_bit_geometry_with_surface(W::AbstractMatrix, L::AbstractMatrix, G::SimpleGraph,
                                       cys_map::Vector{Int}, nres::Int)
    B = length(cys_map)
    geom = Vector{BitGeom}(undef, B)
    for (b, resi) in enumerate(cys_map)
        ρ = zeros(Float64, nres); ρ[resi] = 1.0
        φ = solve_poisson_mean_zero(L, ρ)
        R = L * φ
        Rb = R[resi]
        Eb = dirichlet_energy(L, φ)
        Surf = local_surface_energy(W, φ, resi)
        Mt = morse_type(G, φ, resi)
        geom[b] = BitGeom(Rb, Eb, Surf, Mt)
    end
    return geom
end

# ---------- Map Morse type to multiplicative factor ----------
@inline morse_factor(m::Symbol; λ=LAMBDA_MORSE) = m === :min ? exp(-λ) : (m === :max ? exp(+λ) : 1.0)

# ---- Build modal edge priors from real-graph invariants ----
# For any modal edge flipping bit b:
#   w_struct(b) = exp(-γE*E_b - γR*|R_b|) * exp(+γS*Surf̄_b) * MorseFactor_b
# where Surf̄_b is normalized to [0,1] across bits.
function build_modal_priors(geom::Vector{BitGeom};
                            γE::Float64=GAMMA_E, γR::Float64=GAMMA_R, γS::Float64=GAMMA_S)
    B = length(geom)
    Eb = [g.Eb for g in geom]
    Rb = [g.Rb for g in geom]
    Surf = [g.Surf for g in geom]
    # normalize surface energy across bits for comparability
    mn, mx = minimum(Surf), maximum(Surf)
    Surf̄ = (mx > mn) ? (Surf .- mn) ./ (mx - mn) : fill(0.0, B)
    priors = zeros(Float64, B)
    for b in 1:B
        mf = morse_factor(geom[b].Morse)
        priors[b] = exp(-γE*Eb[b] - γR*abs(Rb[b])) * exp(+γS*Surf̄[b]) * mf
        priors[b] = max(priors[b], 1e-12)
    end
    return priors, Eb, Rb, Surf̄
end

# ---- Assemble diamond adjacency & Laplacian from priors ----
function diamond_from_priors(priors::Vector{Float64})
    n = length(STATES)
    Wd = zeros(Float64, n, n)
    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            b = findfirst(k -> bitvec(s)[k] != bitvec(t)[k], 1:3)
            @assert b !== nothing
            w = priors[b]
            Wd[i,j] = w; Wd[j,i] = w
        end
    end
    D = Diagonal(vec(sum(Wd, dims=2)))
    Ld = D - Wd
    return Wd, Ld
end

# -------- Modal curvature from occupancy (mutable) ---------
function modal_curvics(Ld::AbstractMatrix, ρ::AbstractVector; tol=1e-12)
    @assert abs(sum(ρ) - 1.0) < 1e-9 "ρ must sum to 1."
    ρ0 = ρ .- mean(ρ)
    L⁺ = laplacian_pinv(Ld; tol=tol)
    φ = L⁺ * ρ0; φ .-= mean(φ)
    R = Ld * φ
    return φ, R, L⁺
end

# -------- Field-equation kernel on modal manifold ----------
# P(x→y) ∝ w_struct(b) * exp( -β [ΔG_b + κ (R[y]-R[x]) - TΔS] ), normalized over N(x).
function P_outgoing(x::String, ρ::Vector{Float64}, Wd::Matrix{Float64}, Ld::Matrix{Float64};
                    ΔG_b::Vector{Float64}=zeros(3), ΔS::Float64=0.0,
                    β::Float64=BETA, κ::Float64=KAPPA_CURV)
    φ, R, _ = modal_curvics(Ld, ρ)
    i = IDX[x]
    nums = Float64[]; ys = String[]
    for y in hamming1_neighbors(x)
        j = IDX[y]; Wd[i,j] <= 0 && continue
        b = findfirst(k -> bitvec(x)[k] != bitvec(y)[k], 1:3)
        ΔU = κ * (R[j] - R[i])
        num = Wd[i,j] * exp(-β * (ΔG_b[b] + ΔU - ΔS))
        push!(nums, num); push!(ys, y)
    end
    Z = sum(nums)
    Z > 0 ? Dict(ys[k] => nums[k]/Z for k in eachindex(ys)) :
            Dict(y => 1.0/length(ys) for y in ys)
end

# -------- One Markov step (probability-conserving) --------
function step_markov(ρ::Vector{Float64}, Wd::Matrix{Float64}, Ld::Matrix{Float64};
                     ΔG_b::Vector{Float64}=zeros(3), ΔS::Float64=0.0,
                     β::Float64=BETA, κ::Float64=KAPPA_CURV)
    n = length(STATES)
    ρnext = zeros(Float64, n)
    for x in STATES
        i = IDX[x]
        Px = P_outgoing(x, ρ, Wd, Ld; ΔG_b=ΔG_b, ΔS=ΔS, β=β, κ=κ)
        for (y,p) in Px
            j = IDX[y]; ρnext[j] += ρ[i]*p
        end
    end
    s = sum(ρnext); s>0 && (ρnext ./= s)
    ρnext
end

# ------------------------- RUN -------------------------
# Path to your PDB
pdb = "/content/AF-P04406-F1-model_v4.pdb"

coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA; CYS indices: ", cys_idx_all)

# Pick 3 cysteines → 3 bits (edit as needed)
@assert length(cys_idx_all) ≥ 3 "Need ≥3 CYS to map 3 bits."
cys_map = cys_idx_all[1:3]

# Real manifold geometry
Wreal, Lreal, Greal = build_weighted_graph(coords; cutoff=DIST_CUTOFF, σ=SIGMA_KERNEL)

# Bitwise PDE invariants + surface + Morse on the real graph
geom = per_bit_geometry_with_surface(Wreal, Lreal, Greal, cys_map, length(coords))

# Build modal edge priors from real-graph invariants
priors, Eb, Rb, Surf̄ = build_modal_priors(geom; γE=GAMMA_E, γR=GAMMA_R, γS=GAMMA_S)
Wd, Ld = diamond_from_priors(priors)

# Report bitwise diagnostics
println("\n=== Bitwise PDE projection (real → modal) ===")
println("Bit |    R_b(real)     |     E_b(Dir)     |  Surf̄_b(0-1) |  Morse ")
for b in 1:3
    @printf(" %d  |  % .6e  |  % .6e  |    %.3f      |  %s\n",
            b, Rb[b], Eb[b], Surf̄[b], string(geom[b].Morse))
end

println("\nModal edge priors (per bit):  ", round.(priors, digits=6))

# Sanity: curvature conservation on modal lattice for δ at 000
ρ = zeros(Float64, length(STATES)); ρ[IDX["000"]] = 1.0
φd, Rd, _ = modal_curvics(Ld, ρ)
@printf("\nModal curvature conservation: ΣR = %.3e\n", sum(Rd))

# Show outgoing P from 000
function print_outgoing(x, ρ)
    Px = P_outgoing(x, ρ, Wd, Ld)
    println("\nOutgoing P($x → ·):")
    for y in sort(collect(keys(Px))) @printf("  %s → %s : %.6f\n", x, y, Px[y]) end
end
print_outgoing("000", ρ)

# Minimal Markov evolution demo
Random.seed!(RNG_SEED)
println("\n=== Markov evolution on modal manifold (", STEPS, " steps) ===")
for t in 1:STEPS
    ρ = step_markov(ρ, Wd, Ld)
    if t % 10 == 0 || t == 1
        @printf("t=%3d  mass@111=%.6f  max=%s(%.6f)\n",
                t, ρ[IDX["111"]], STATES[argmax(ρ)], maximum(ρ))
    end
end
println("\nFinal ρ (sum=", @sprintf("%.6f", sum(ρ)), "):")
for s in STATES
    @printf("  %s : %.6f\n", s, ρ[IDX[s]])
end
