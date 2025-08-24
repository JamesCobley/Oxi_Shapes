#!/usr/bin/env julia
# ===============================================================
# Oxi-Shapes MGF — Einstein-level closure
# PDE → pushforward measure → modal Laplacian → master equation
# (parameter-free projection; no SASA, no γ/κ/λ knobs)
# ===============================================================

using LinearAlgebra, StaticArrays, Graphs, Printf, Statistics

# ---------------- PDB parsing (CA only) ----------------
function load_ca_and_cys(pdb_path::String)
    coords = SVector{3,Float64}[]
    cys_idx = Int[]
    open(pdb_path,"r") do io
        for ln in eachline(io)
            if startswith(ln, "ATOM") && strip(ln[13:16]) == "CA"
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

# ---------- Real manifold: residue graph W, Laplacian L ----------
# Only 2 physical knobs here: contact cutoff & kernel width.
function build_weighted_graph(coords::Vector{SVector{3,Float64}}; cutoff=6.5, σ=4.0)
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
        W[i,j] > 0 && add_edge!(G, i, j)
    end
    return W, L, G
end

# ---------------- Laplacian pseudoinverse / Poisson --------------
function laplacian_pinv(L::AbstractMatrix; tol=1e-12)
    F = eigen(Symmetric(Matrix(L)))
    λ, U = F.values, F.vectors
    λ⁺ = similar(λ)
    @inbounds for i in eachindex(λ)
        λ⁺[i] = λ[i] > tol ? inv(λ[i]) : 0.0
    end
    return U * Diagonal(λ⁺) * U'
end

# Solve L φ = ρ - mean(ρ) (volume-invariant source), center φ
function solve_poisson_mean_zero(L::AbstractMatrix, ρ::AbstractVector)
    ρ0 = ρ .- mean(ρ)
    φ  = laplacian_pinv(L) * ρ0
    φ .-= mean(φ)
    return φ
end

# Dirichlet energy (total dissipated power): E = 1/2 * φᵀ L φ
dirichlet_energy(L::AbstractMatrix, φ::AbstractVector) = 0.5 * dot(φ, L*φ)

# Local edge energy (power) at node i: S_i = 1/2 ∑_j W_ij (φ_i - φ_j)^2
@inline function local_edge_energy_at(W::AbstractMatrix, φ::AbstractVector, i::Int)
    acc = 0.0
    @inbounds for j in 1:length(φ)
        w = W[i,j]; w == 0 && continue
        acc += 0.5 * w * (φ[i] - φ[j])^2
    end
    return acc
end

# -------- Modal manifold (R=3 Boolean “diamond”) --------
const STATES = ["000","100","010","001","110","101","011","111"]
const IDX    = Dict(s=>i for (i,s) in enumerate(STATES))
bitvec(s::String) = [c=='1' for c in collect(s)]
hamming1_neighbors(s::String) = [begin
    tmp = collect(s); tmp[k] = (tmp[k]=='0' ? '1' : '0'); String(tmp)
end for k in 1:length(s)]

# ------------- Pushforward measure (parameter-free) --------------
# For each bit b (cysteine at resi):
#   Solve Poisson with δ source at resi → φ_b.
#   E_b = 1/2 φᵀ L φ  (global energy; “denominator”)
#   S_b = local_edge_energy_at(W, φ, resi) (power out of source; “numerator”)
#   f_b = S_b / E_b  ∈ (0, 1]  (fraction of energy dissipated at the source)
#   We take the pushforward weight for bit b as  w_b = f_b / Σ_b f_b  (normalized).
#
# Rationale: this is the measure-preserving pushforward of the PDE energy density
# localized to the source node into a bitwise capacity on modal edges. No γ/κ/λ.
function bitwise_pushforward_weights(W::AbstractMatrix, L::AbstractMatrix,
                                     cys_map::Vector{Int}, nres::Int)
    B = length(cys_map)
    φs   = Vector{Vector{Float64}}(undef, B)
    Eb   = zeros(Float64, B)
    Sb   = zeros(Float64, B)
    for (b, resi) in enumerate(cys_map)
        ρ = zeros(Float64, nres); ρ[resi] = 1.0
        φ = solve_poisson_mean_zero(L, ρ)
        φs[b] = φ
        Eb[b] = dirichlet_energy(L, φ)
        Sb[b] = local_edge_energy_at(W, φ, resi)
    end
    fb = Sb ./ Eb                     # fraction of dissipated power localized at the source
    fb .+= eps()                      # guard
    w_raw = fb ./ sum(fb)             # normalized pushforward weights (Σ_b w_b = 1)
    return w_raw, Eb, Sb
end

# ------- Build modal adjacency/Laplacian from w_b (no knobs) -----
function diamond_from_bitweights(w_b::Vector{Float64})
    n = length(STATES)
    Wd = zeros(Float64, n, n)
    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            b = findfirst(k -> bitvec(s)[k] != bitvec(t)[k], 1:3)
            @assert b !== nothing
            w = w_b[b]                # the same bit weight for all flips of bit b
            Wd[i,j] = w; Wd[j,i] = w
        end
    end
    D  = Diagonal(vec(sum(Wd, dims=2)))
    Ld = D - Wd
    return Wd, Ld
end

# --------------- Exact master equation evolution -----------------
# ρ(t) = exp(-t * Ld) ρ(0)
function propagate(Ld::AbstractMatrix, ρ0::AbstractVector, t::Float64)
    @assert abs(sum(ρ0) - 1.0) < 1e-9 "ρ0 must sum to 1."
    K = exp(-t .* Matrix(Ld))      # matrix exponential (heat kernel)
    ρt = K * ρ0
    s = sum(ρt); s > 0 && (ρt ./= s)  # numerical hygiene
    return ρt
end

# -------------------- Diagnostics / printing ---------------------
function initial_outgoing_rates_from(x::String, ρ0::Vector{Float64}, Wd::Matrix{Float64})
    i = IDX[x]
    # For ρ0=δ_x, instantaneous outflow rates along edges ∝ Wd[i,j]
    nbrs = hamming1_neighbors(x)
    ws = [Wd[i, IDX[y]] for y in nbrs]
    Z = sum(ws)
    return Dict(nbrs[k] => ws[k]/Z for k in eachindex(nbrs))
end

function check_modal_conservation(Ld::AbstractMatrix, ρ::AbstractVector; label="modal")
    φ = laplacian_pinv(Ld) * (ρ .- mean(ρ))
    φ .-= mean(φ)
    R = Ld * φ
    @printf("Curvature conservation (%s): ΣR = %.3e\n", label, sum(R))
end

# =============================== RUN =============================
# Set your PDB path:
pdb = "/content/AF-P04406-F1-model_v4.pdb"

coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA; CYS indices: ", cys_idx_all)

# Choose 3 cysteines → 3 bits (edit as needed)
@assert length(cys_idx_all) ≥ 3 "Need ≥3 CYS to map 3 bits."
cys_map = cys_idx_all[1:3]

# Real manifold geometry
Wreal, Lreal, _ = build_weighted_graph(coords; cutoff=6.5, σ=4.0)

# Pushforward (parameter-free) bit weights
w_b, E_b, S_b = bitwise_pushforward_weights(Wreal, Lreal, cys_map, length(coords))

println("\n=== PDE → Modal pushforward (parameter-free) ===")
println("Bit |   E_b (Dirichlet)   |   S_b (local power@source)   |   w_b (Σ=1)")
for b in 1:3
    @printf(" %d  |   % .6e        |   % .6e                   |   %.6f\n",
            b, E_b[b], S_b[b], w_b[b])
end

# Modal Laplacian from pushforward weights
Wd, Ld = diamond_from_bitweights(w_b)

# Check discrete curvature conservation on modal manifold
ρ0 = zeros(Float64, length(STATES)); ρ0[IDX["000"]] = 1.0
check_modal_conservation(Ld, ρ0; label="modal (δ at 000)")

# Instantaneous (t→0⁺) outgoing probabilities from 000 (rates ∝ Wd[000,·])
P0 = initial_outgoing_rates_from("000", ρ0, Wd)
println("\nInstantaneous outgoing P(000 → ·) from pushforward weights:")
for y in sort(collect(keys(P0)))
    @printf("  000 → %s : %.6f\n", y, P0[y])
end

# Exact master-equation evolution (choose physical time points)
times = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
println("\n=== Master-equation evolution ρ(t) ===")
for t in times
    ρt = propagate(Ld, ρ0, t)
    @printf("t = %4.1f   max=%s(%.6f)   mass@111=%.6f\n",
            t, STATES[argmax(ρt)], maximum(ρt), ρt[IDX["111"]])
end

println("\nFinal ρ at t = $(times[end]) (sum=", @sprintf("%.6f", sum(propagate(Ld, ρ0, times[end]))), "):")
ρT = propagate(Ld, ρ0, times[end])
for s in STATES
    @printf("  %s : %.6f\n", s, ρT[IDX[s]])
end
