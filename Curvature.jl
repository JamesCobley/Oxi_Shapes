#!/usr/bin/env julia
# ===============================================================
# Oxi-Shapes MGF — Modal curvature → activation energy (Einstein-style)
# Uses absolute edge curvature magnitude for ΔG‡, keeps signed curvature for diagnostics
# ===============================================================

using LinearAlgebra, StaticArrays, Graphs, Printf, Statistics
using Ripserer

# ---------------- PDB parsing (CA only) ----------------
function load_ca_and_cys(pdb_path::String)
    coords = SVector{3,Float64}[]
    cys_idx = Int[]
    open(pdb_path,"r") do io
        for ln in eachline(io)
            if startswith(ln, "ATOM") && strip(ln[13:16]) == "CA"
                x = parse(Float64, ln[31:38])
                y = parse(Float64, ln[39:46])
                z = parse(Float64, ln[47:54])
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

# ---------- Real-manifold graph & Laplacian (kept for diagnostics only) ----------
function build_weighted_graph(coords::Vector{SVector{3,Float64}}; cutoff=6.5, sigma=4.0)
    n = length(coords)
    W = zeros(Float64, n, n)
    inv2sig2 = 1.0 / (2.0 * sigma^2)
    for i in 1:(n-1)
        for j in (i+1):n
            d = norm(coords[i] - coords[j])
            if d <= cutoff
                w = exp(-(d*d)*inv2sig2)
                W[i,j] = w
                W[j,i] = w
            end
        end
    end
    D = Diagonal(vec(sum(W, dims=2)))
    L = D - W
    return W, L
end

# ---------- Laplacian pseudoinverse / Poisson ----------
function laplacian_pinv(L::AbstractMatrix; tol=1e-12)
    F = eigen(Symmetric(Matrix(L)))
    lam, U = F.values, F.vectors
    lam_inv = similar(lam)
    @inbounds for i in eachindex(lam)
        lam_inv[i] = lam[i] > tol ? inv(lam[i]) : 0.0
    end
    return U * Diagonal(lam_inv) * U'
end

function solve_poisson_mean_zero(L::AbstractMatrix, rho::AbstractVector)
    rho0 = rho .- mean(rho)
    phi  = laplacian_pinv(L) * rho0
    phi .-= mean(phi)
    return phi
end

# ---------- Modal manifold (R=3 Boolean “diamond”) ----------
const STATES = ["000","100","010","001","110","101","011","111"]
const IDX    = Dict(s=>i for (i,s) in enumerate(STATES))

bitvec(s::String) = (c = collect(s); [c[1]=='1', c[2]=='1', c[3]=='1'])

function hamming1_neighbors(s::String)
    c = collect(s)
    nbrs = String[]
    for k in 1:3
        tmp = copy(c)
        tmp[k] = (tmp[k] == '0' ? '1' : '0')
        push!(nbrs, String(tmp))
    end
    return nbrs
end

function flipped_bit_index(s::String, t::String)::Int
    bs = bitvec(s); bt = bitvec(t)
    for k in 1:3
        if bs[k] != bt[k]
            return k
        end
    end
    error("States do not differ by Hamming distance 1: $s → $t")
end

# ---------- Axiom 2: Volume conservation ----------
function normalize_modal!(ρ::Vector{Float64})
    total = sum(ρ)
    if total > 0
        ρ ./= total
    end
    return ρ
end

# ---------- Modal Laplacian from occupancy ONLY ----------
function modal_laplacian_from_occupancy(ρ::Vector{Float64}; α::Float64=1.0)
    n = length(STATES)
    W = zeros(Float64, n, n)
    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            w_curv = exp(-α * abs(ρ[i] - ρ[j]))
            W[i,j] = w_curv
            W[j,i] = w_curv
        end
    end
    D  = Diagonal(vec(sum(W, dims=2)))
    Ld = D - W
    return Ld, W
end

# ---------- Forman-Ricci curvature on edges (weighted) ----------
function forman_edge_curvature(W::AbstractMatrix, i::Int, j::Int; node_weight::Float64=1.0)
    w_e = W[i,j]
    w_e == 0 && return 0.0
    term = (node_weight / w_e) + (node_weight / w_e)
    # neighbors of i excluding j
    for k in 1:size(W,1)
        k == j && continue
        w_ik = W[i,k]
        w_ik == 0 && continue
        term -= node_weight / sqrt(w_e * w_ik)
    end
    # neighbors of j excluding i
    for k in 1:size(W,1)
        k == i && continue
        w_jk = W[j,k]
        w_jk == 0 && continue
        term -= node_weight / sqrt(w_e * w_jk)
    end
    return term
end

# ---------- Einstein-style mapping: |curvature| → ΔG‡ (J/mol) ----------
"""
    curvature_to_activation(K; κ_modal)

Map absolute edge curvature |K| to activation energy ΔG‡ (J/mol):
ΔG = κ_modal * |K|
"""
function curvature_to_activation(K::Float64; κ_modal::Float64)
    return κ_modal * abs(K)
end

# =============================== RUN =============================
pdb = "/content/AF-P04406-F1-model_v4.pdb"
T   = 293.15

coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA; CYS indices: ", cys_idx_all)
@assert length(cys_idx_all) >= 3 "Need ≥3 CYS"
cys_map = cys_idx_all[1:3]

ρ = zeros(Float64, length(STATES))
ρ[IDX["000"]] = 1.0
normalize_modal!(ρ)

Ldρ, W_occ = modal_laplacian_from_occupancy(ρ; α=1.0)

K_edge = fill(0.0, length(STATES), length(STATES))
for s in STATES
    i = IDX[s]
    for t in hamming1_neighbors(s)
        j = IDX[t]
        K_edge[i,j] = forman_edge_curvature(W_occ, i, j)
        K_edge[j,i] = K_edge[i,j]
    end
end

println("\n=== Modal edge Forman curvatures (signed) ===")
for s in STATES
    i = IDX[s]
    for t in hamming1_neighbors(s)
        j = IDX[t]
        b = flipped_bit_index(s,t)
        @printf("%s ↔ %s (bit %d):  K_forman=%.4f\n", s, t, b, K_edge[i,j])
    end
end

κ_modal = 1.0e4   # J/mol
ΔG_curv_bits = zeros(Float64, 3)
for b in 1:3
    s0 = "000"; t = hamming1_neighbors(s0)[b]
    i, j = IDX[s0], IDX[t]
    ΔG_curv_bits[b] = curvature_to_activation(abs(K_edge[i,j]); κ_modal=κ_modal)
end

@printf("\nΔG‡ from modal curvature (per bit, using |K|) [kJ/mol]: [%.2f, %.2f, %.2f]\n",
        ΔG_curv_bits[1]/1e3, ΔG_curv_bits[2]/1e3, ΔG_curv_bits[3]/1e3)
