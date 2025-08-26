#!/usr/bin/env julia
# ===============================================================
# Oxi-Shapes MGF — PDE + Persistent Topology (Morse) + Modal Energy
# SINGLE-MOLECULE ΔG‡ (no fitting) + curvature-correct edge weights
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

# ---------- Real-manifold graph & Laplacian ----------
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

dirichlet_energy(L::AbstractMatrix, phi::AbstractVector) = 0.5 * dot(phi, L*phi)

function local_edge_energy_at(W::AbstractMatrix, phi::AbstractVector, i::Int)
    acc = 0.0
    @inbounds for j in 1:length(phi)
        w = W[i,j]
        w == 0 && continue
        acc += 0.5 * w * (phi[i] - phi[j])^2
    end
    return acc
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

# ---------- PDE pushforward weights ----------
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

# --- Graph-radius based topology weights ---
function morse_topology_weights(coords::Vector{SVector{3,Float64}},
                                cys_map::Vector{Int},
                                W::Matrix{Float64}; hop_radius::Int=2)
    n = length(coords)
    depth_local  = zeros(Float64, length(cys_map))  # H0 burial proxy
    loop_local   = zeros(Float64, length(cys_map))  # H1 loopiness proxy

    # Build unweighted connectivity graph from W>0
    G = SimpleGraph(n)
    for i in 1:(n-1), j in (i+1):n
        W[i,j] > 0 && add_edge!(G,i,j)
    end

    for (b,resi) in enumerate(cys_map)
        # --- collect nodes within hop_radius ---
        nodes = Set([resi])
        frontier = Set([resi])
        for _ in 1:hop_radius
            newfrontier = Set{Int}()
            for v in frontier, nb in neighbors(G,v)
                push!(newfrontier, nb)
            end
            union!(nodes, newfrontier)
            frontier = newfrontier
        end
        idxs = collect(nodes)
        m = length(idxs)
        if m < 4
            depth_local[b] = 1e-3
            loop_local[b]  = 1e-3
            continue
        end

        # --- local pairwise distances ---
        Dloc = [norm(coords[idxs[i]] - coords[idxs[j]]) for i in 1:m, j in 1:m]

        # --- persistent homology up to H1 ---
        dgms = ripserer(Dloc, dim_max=1)

        # burial proxy: median of distances (robust depth measure)
        depth_local[b] = median(Dloc)

        # loopiness proxy: total finite H1 persistence
        if length(dgms) >= 2 && !isempty(dgms[2])
            loop_local[b] = sum(bar -> (isfinite(bar.death) ?
                                        (bar.death - bar.birth) : 0.0), dgms[2])
        else
            loop_local[b] = 1e-6
        end
    end

    # --- normalize and combine ---
    depth_local ./= sum(depth_local) > 0 ? sum(depth_local) : 1.0
    loop_local  ./= sum(loop_local)  > 0 ? sum(loop_local)  : 1.0
    w_morse = (depth_local .+ loop_local) ./ sum(depth_local .+ loop_local)

    return w_morse, depth_local, loop_local
end

# ---------- Internal activation from geometry ----------
const Rgas = 8.314462618

function activation_from_internal(T::Float64,
                                  w_morse::Vector{Float64},
                                  Eb::Vector{Float64},
                                  Sb::Vector{Float64}; enthalpy::Symbol=:Eb)
    wbar = mean(w_morse)
    dS_act  = Rgas .* log.((w_morse ./ wbar) .+ eps())
    dH_act = if enthalpy === :Eb
        (Rgas*T) .* Eb
    elseif enthalpy === :Sb
        (Rgas*T) .* Sb
    else
        error("enthalpy must be :Eb or :Sb")
    end
    dG_act = dH_act .- T .* dS_act
    return dH_act, dS_act, dG_act
end

# ---------- Build modal Laplacian from dG_act ----------
function edge_weights_from_activation(dG_act::Vector{Float64}, T::Float64)
    β = 1.0 / (Rgas*T)
    w_raw = exp.(-β .* dG_act) .+ eps()
    return w_raw ./ sum(w_raw)
end

function diamond_from_weights(wbits::Vector{Float64})
    n = length(STATES)
    Wd = zeros(Float64, n, n)
    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            b = flipped_bit_index(s, t)
            w = wbits[b]
            Wd[i,j] = w
            Wd[j,i] = w
        end
    end
    D  = Diagonal(vec(sum(Wd, dims=2)))
    Ld = D - Wd
    return Wd, Ld
end

function diamond_from_activation(dG_act::Vector{Float64}, T::Float64)
    wbits = edge_weights_from_activation(dG_act, T)
    return diamond_from_weights(wbits), wbits
end

# ---------- Modal diagnostics ----------
function modal_dirichlet_energy(Ld::AbstractMatrix; src_state::String="000")
    n = size(Ld,1)
    rho = zeros(Float64, n); rho[IDX[src_state]] = 1.0
    rho .-= mean(rho)
    phi = laplacian_pinv(Ld) * rho
    phi .-= mean(phi)
    return 0.5 * dot(phi, Ld*phi)
end

function initial_outgoing_probs(x::String, Wd::AbstractMatrix)
    i = IDX[x]
    nbrs = hamming1_neighbors(x)
    ws = [Wd[i, IDX[y]] for y in nbrs]
    Z = sum(ws)
    return Dict(nbrs[k] => (Z>0 ? ws[k]/Z : 1.0/length(nbrs)) for k in eachindex(nbrs))
end

function propagate(Ld::AbstractMatrix, rho0::AbstractVector, t::Float64)
    K = exp(-t .* Matrix(Ld))  # matrix exponential
    rho_t = K * rho0
    s = sum(rho_t)
    s > 0 && (rho_t ./= s)
    return rho_t
end

function modal_curvature(Ld::AbstractMatrix; src_state::String="000")
    n = size(Ld,1)
    ρ = zeros(Float64, n); ρ[IDX[src_state]] = 1.0
    ρ .-= mean(ρ)
    φ = laplacian_pinv(Ld) * ρ
    φ .-= mean(φ)
    R = Ld * φ
    return R, φ
end

# ---------- Vibrational eigenmodes of modal Laplacian ----------
"""
    vibrational_modes(Ld; k=5)

Compute the lowest-k nonzero eigenvalues/eigenvectors of the modal Laplacian.
Eigenvalues λ correspond to squared vibrational frequencies ω²,
and eigenvectors are the vibrational eigenmodes.
"""
function vibrational_modes(Ld::AbstractMatrix; k::Int=5)
    F = eigen(Symmetric(Matrix(Ld)))
    vals, vecs = F.values, F.vectors
    # Discard near-zero eigenvalues (numerical nullspace)
    idxs = findall(>(1e-8), vals)
    keep = first(idxs, min(k, length(idxs)))
    return vals[keep], vecs[:, keep]
end

# =============================== RUN =============================
pdb = "/content/AF-P04406-F1-model_v4.pdb"   # <-- your PDB
T   = 293.15                                  # 20 °C test

coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA; CYS indices: ", cys_idx_all)
@assert length(cys_idx_all) >= 3 "Need ≥3 CYS"
cys_map = cys_idx_all[1:3]

Wreal, Lreal = build_weighted_graph(coords; cutoff=6.5, sigma=4.0)
w_energy, Eb, Sb = bitwise_pushforward(Wreal, Lreal, cys_map, length(coords))
println("\n=== PDE (real) per-bit ===")
for b in 1:3
    @printf("bit %d : Eb=% .6e  Sb=% .6e\n", b, Eb[b], Sb[b])
end

w_morse, tau0, tau1 = morse_topology_weights(coords, cys_map, Wreal)
println("\n=== Topology (Morse) weights ===")
for b in 1:3
    @printf("bit %d : depth(H0)=%.6f  loops(H1)=%.6f  wM=%.6f\n",
            b, tau0[b], tau1[b], w_morse[b])
end

dH_act, dS_act, dG_act = activation_from_internal(T, w_morse, Eb, Sb; enthalpy=:Eb)
println("\n=== Internal geometric activation (no fit) at T = $(T) K ===")
for b in 1:length(dG_act)
    @printf("bit %d : ΔH‡=%.2f kJ/mol   TΔS‡=%.2f kJ/mol   ΔG‡=%.2f kJ/mol\n",
            b, dH_act[b]/1e3, (T*dS_act[b])/1e3, dG_act[b]/1e3)
end
@printf("\nSingle-molecule test (000 → 100): ΔG‡(bit1) = %.2f kJ/mol\n", dG_act[1]/1e3)

((WdG, LdG), w_fromG) = diamond_from_activation(dG_act, T)
println("\n=== Edge weights from activation ===")
@printf("w(bits) = [%.3f, %.3f, %.3f]\n", w_fromG[1], w_fromG[2], w_fromG[3])

println("\n=== Instantaneous outflow P(000→·) ===")
P0 = initial_outgoing_probs("000", WdG)
@printf("000→001=%.3f  000→010=%.3f  000→100=%.3f\n", P0["001"], P0["010"], P0["100"])

E_G = modal_dirichlet_energy(LdG)
R_G, φ_G = modal_curvature(LdG)
println("\n--- ΔG‡-weighted channel diagnostics ---")
@printf("Dirichlet energy: %.6e\n", E_G)
@printf("Curvature sum: %.3e\n", sum(R_G))
for s in STATES[1:5]
    @printf("  %s : R=%.3e\n", s, R_G[IDX[s]])
end

println("\n=== Vibrational eigenmodes of modal Laplacian (ΔG‡ channel) ===")
eigvals, eigvecs = vibrational_modes(LdG; k=5)
for i in 1:length(eigvals)
    @printf("mode %d : ω²=%.6f   ω=%.6f\n", i, eigvals[i], sqrt(eigvals[i]))
end
