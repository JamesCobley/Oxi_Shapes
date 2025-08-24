#!/usr/bin/env julia
# ===============================================================
# Oxi-Shapes MGF — Einstein-level closure + Persistent Topology
# PDE → pushforward measure → modal Laplacian → master equation
# + persistent homology of PDE potential (Ripserer)
# ===============================================================

using LinearAlgebra, StaticArrays, Graphs, Printf, Statistics
using Ripserer   # persistent homology

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

function solve_poisson_mean_zero(L::AbstractMatrix, ρ::AbstractVector)
    ρ0 = ρ .- mean(ρ)
    φ  = laplacian_pinv(L) * ρ0
    φ .-= mean(φ)
    return φ
end

dirichlet_energy(L::AbstractMatrix, φ::AbstractVector) = 0.5 * dot(φ, L*φ)

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
function bitwise_pushforward_weights(W::AbstractMatrix, L::AbstractMatrix,
                                     cys_map::Vector{Int}, nres::Int)
    B = length(cys_map)
    Eb   = zeros(Float64, B)
    Sb   = zeros(Float64, B)
    for (b, resi) in enumerate(cys_map)
        ρ = zeros(Float64, nres); ρ[resi] = 1.0
        φ = solve_poisson_mean_zero(L, ρ)
        Eb[b] = dirichlet_energy(L, φ)
        Sb[b] = local_edge_energy_at(W, φ, resi)

        # Persistent topology for this source
        diagram = persistent_topology_analysis(φ, coords)
        println("\n--- Persistent topology for cysteine bit $b (resi=$resi) ---")
        print_persistence(diagram)
    end
    fb = Sb ./ Eb
    fb .+= eps()
    w_raw = fb ./ sum(fb)
    return w_raw, Eb, Sb
end

# ------- Build modal adjacency/Laplacian from w_b ---------------
function diamond_from_bitweights(w_b::Vector{Float64})
    n = length(STATES)
    Wd = zeros(Float64, n, n)
    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            b = findfirst(k -> bitvec(s)[k] != bitvec(t)[k], 1:3)
            @assert b !== nothing
            w = w_b[b]
            Wd[i,j] = w; Wd[j,i] = w
        end
    end
    D  = Diagonal(vec(sum(Wd, dims=2)))
    Ld = D - Wd
    return Wd, Ld
end

# ---------------- Master equation propagation -------------------
function propagate(Ld::AbstractMatrix, ρ0::AbstractVector, t::Float64)
    K = exp(-t .* Matrix(Ld))
    ρt = K * ρ0
    ρt ./= sum(ρt)
    return ρt
end

# ---------------- Persistent topology utilities -----------------
function persistent_topology_analysis(φ::Vector{Float64}, coords::Vector{SVector{3,Float64}})
    n = length(φ)
    D = zeros(Float64, n, n)
    for i in 1:n-1, j in i+1:n
        d = norm(coords[i] - coords[j])
        D[i,j] = d * (abs(φ[i]-φ[j]) + 1e-6)
        D[j,i] = D[i,j]
    end
    diagram = ripserer(D, dim_max=1)
    return diagram
end

function print_persistence(diagram)
    for (dim, bars) in enumerate(diagram)
        for (i, bar) in enumerate(bars)
            birth = round(bar.birth, digits=3)
            death = bar.death === Inf ? "∞" : string(round(bar.death, digits=3))
            println("Dim $dim | Feature $i : birth=$birth, death=$death")
        end
    end
end

# =============================== RUN =============================
pdb = "/content/AF-P04406-F1-model_v4.pdb"

coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA; CYS indices: ", cys_idx_all)

@assert length(cys_idx_all) ≥ 3 "Need ≥3 CYS to map 3 bits."
cys_map = cys_idx_all[1:3]

Wreal, Lreal, _ = build_weighted_graph(coords; cutoff=6.5, σ=4.0)

w_b, E_b, S_b = bitwise_pushforward_weights(Wreal, Lreal, cys_map, length(coords))

println("\n=== PDE → Modal pushforward (parameter-free) ===")
println("Bit |   E_b (Dirichlet)   |   S_b (local power@source)   |   w_b (Σ=1)")
for b in 1:3
    @printf(" %d  |   % .6e        |   % .6e                   |   %.6f\n",
            b, E_b[b], S_b[b], w_b[b])
end

Wd, Ld = diamond_from_bitweights(w_b)

ρ0 = zeros(Float64, length(STATES)); ρ0[IDX["000"]] = 1.0

times = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
println("\n=== Master-equation evolution ρ(t) ===")
for t in times
    ρt = propagate(Ld, ρ0, t)
    @printf("t = %4.1f   max=%s(%.6f)   mass@111=%.6f\n",
            t, STATES[argmax(ρt)], maximum(ρt), ρt[IDX["111"]])
end

ρT = propagate(Ld, ρ0, times[end])
println("\nFinal ρ at t = $(times[end]) (sum=$(round(sum(ρT), digits=6))):")
for s in STATES
    @printf("  %s : %.6f\n", s, ρT[IDX[s]])
end
