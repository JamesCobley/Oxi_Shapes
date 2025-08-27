#!/usr/bin/env julia
# ===============================================================
# Oxi-Shapes MGF — Modal curvature → activation energy
# Clean formulation: curvature baseline × (1 - ln w_E) functor
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

# ---------- Real-manifold PDE weights (Eb, Sb, wE) ----------
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

function dirichlet_energy(L::AbstractMatrix, phi::AbstractVector)
    return 0.5 * dot(phi, L*phi)
end

function local_edge_energy_at(W::AbstractMatrix, phi::AbstractVector, i::Int)
    acc = 0.0
    @inbounds for j in 1:length(phi)
        w = W[i,j]; w == 0 && continue
        acc += 0.5 * w * (phi[i] - phi[j])^2
    end
    return acc
end

function build_weighted_graph(coords::Vector{SVector{3,Float64}}; cutoff=6.5, sigma=4.0)
    n = length(coords)
    W = zeros(Float64, n, n)
    inv2sig2 = 1.0 / (2.0 * sigma^2)
    for i in 1:(n-1), j in (i+1):n
        d = norm(coords[i] - coords[j])
        if d <= cutoff
            w = exp(-(d*d)*inv2sig2)
            W[i,j] = w; W[j,i] = w
        end
    end
    D = Diagonal(vec(sum(W, dims=2)))
    L = D - W
    return W, L
end

function bitwise_pushforward(W::AbstractMatrix, L::AbstractMatrix,
                             cys_map::Vector{Int}, nres::Int)
    B = length(cys_map)
    Eb = zeros(Float64, B)
    Sb = zeros(Float64, B)
    for (b, resi) in enumerate(cys_map)
        rho = zeros(Float64, nres); rho[resi] = 1.0
        phi = solve_poisson_mean_zero(L, rho)
        Eb[b] = dirichlet_energy(L, phi)
        Sb[b] = local_edge_energy_at(W, phi, resi)
    end
    fb = Sb ./ Eb
    fb .+= eps()
    w  = fb ./ sum(fb)
    return w, Eb, Sb
end

# ---------- Modal manifold (R=3 Boolean “diamond”) ----------
const STATES = ["000","100","010","001","110","101","011","111"]
const IDX    = Dict(s=>i for (i,s) in enumerate(STATES))

function hamming1_neighbors(s::String)
    c = collect(s)
    nbrs = String[]
    for k in 1:3
        tmp = copy(c); tmp[k] = (tmp[k] == '0' ? '1' : '0')
        push!(nbrs, String(tmp))
    end
    return nbrs
end

function flipped_bit_index(s::String, t::String)::Int
    bs = collect(s); bt = collect(t)
    for k in 1:3
        if bs[k] != bt[k]; return k; end
    end
    error("States not Hamming-1 neighbors: $s → $t")
end

function normalize_modal!(ρ::Vector{Float64})
    total = sum(ρ); total > 0 && (ρ ./= total); return ρ
end

# ---------- Modal Laplacian from occupancy ----------
function modal_laplacian_from_occupancy(ρ::Vector{Float64}; α::Float64=1.0)
    n = length(STATES); W = zeros(Float64, n, n)
    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            w_curv = exp(-α * abs(ρ[i] - ρ[j]))
            W[i,j] = w_curv; W[j,i] = w_curv
        end
    end
    D  = Diagonal(vec(sum(W, dims=2)))
    Ld = D - W
    return Ld, W
end

# ---------- Forman-Ricci curvature ----------
function forman_edge_curvature(W::AbstractMatrix, i::Int, j::Int; node_weight::Float64=1.0)
    w_e = W[i,j]; w_e == 0 && return 0.0
    term = (node_weight / w_e) + (node_weight / w_e)
    for k in 1:size(W,1)
        k == j && continue
        w_ik = W[i,k]; w_ik == 0 && continue
        term -= node_weight / sqrt(w_e * w_ik)
    end
    for k in 1:size(W,1)
        k == i && continue
        w_jk = W[j,k]; w_jk == 0 && continue
        term -= node_weight / sqrt(w_e * w_jk)
    end
    return term
end

# ---------- Einstein mapping: |K| → ΔG‡ ----------
function curvature_to_activation(K::Float64; κ_modal::Float64)
    return κ_modal * abs(K)
end

# =============================== RUN =============================
pdb = "/content/AF-P04406-F1-model_v4.pdb"
T   = 293.15

# --- Parse structure & select cysteines ---
coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA; CYS indices: ", cys_idx_all)
@assert length(cys_idx_all) >= 3 "Need ≥3 CYS"
cys_map = cys_idx_all[1:3]

# --- PDE weights (Eb, Sb, wE) ---
Wreal, Lreal = build_weighted_graph(coords; cutoff=6.5, sigma=4.0)
w_E, Eb, Sb = bitwise_pushforward(Wreal, Lreal, cys_map, length(coords))
println("\n=== PDE weights ===")
for b in 1:3
    @printf("bit %d : Eb=%.3e, Sb=%.3e, w_E=%.3f\n", b, Eb[b], Sb[b], w_E[b])
end

# --- Modal Laplacian & curvature spectrum ---
ρ = zeros(Float64, length(STATES)); ρ[IDX["000"]] = 1.0; normalize_modal!(ρ)
Ldρ, W_occ = modal_laplacian_from_occupancy(ρ; α=1.0)

K_edge = fill(0.0, length(STATES), length(STATES))
for s in STATES, t in hamming1_neighbors(s)
    i, j = IDX[s], IDX[t]
    K_edge[i,j] = forman_edge_curvature(W_occ, i, j)
    K_edge[j,i] = K_edge[i,j]
end

println("\n=== Modal edge curvatures (signed) ===")
for s in STATES, t in hamming1_neighbors(s)
    i, j = IDX[s], IDX[t]; b = flipped_bit_index(s,t)
    @printf("%s ↔ %s (bit %d): K_forman=%.4f\n", s, t, b, K_edge[i,j])
end

# --- Curvature baseline and log-scaled effective ΔG‡ ---
κ_modal = 1.0e4   # J/mol/m² effective coupling
ΔG_curv_bits = zeros(Float64, 3)
for b in 1:3
    s0 = "000"; t = hamming1_neighbors(s0)[b]
    i, j = IDX[s0], IDX[t]
    ΔG_curv_bits[b] = curvature_to_activation(K_edge[i,j]; κ_modal=κ_modal)
end

ΔG_eff_bits = ΔG_curv_bits .* (1 .- log.(w_E))   # clean log-scaling

@printf("\nΔG‡ curvature baseline [kJ/mol]: [%.2f, %.2f, %.2f]\n",
        ΔG_curv_bits[1]/1e3, ΔG_curv_bits[2]/1e3, ΔG_curv_bits[3]/1e3)
@printf("ΔG‡ effective (log scaled) [kJ/mol]: [%.2f, %.2f, %.2f]\n",
        ΔG_eff_bits[1]/1e3, ΔG_eff_bits[2]/1e3, ΔG_eff_bits[3]/1e3)
