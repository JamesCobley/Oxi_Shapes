#!/usr/bin/env julia
# ===============================================================
# Oxi-Shapes MGF — PDE + Persistent Topology (Morse) + Modal Energy
# Parameter-free PDE pushforward + substrate topology → dual channel
# ===============================================================

using LinearAlgebra, StaticArrays, Graphs, Printf, Statistics
using Ripserer  # pkg> add Ripserer

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
    for i in 1:n-1
        for j in i+1:n
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

# safer than findfirst(...) for older parsers
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
    fb .+= eps()          # guard
    w  = fb ./ sum(fb)    # normalized
    return w, Eb, Sb
end

# --- Graph-radius based topology weights (parameter-free) ---
function morse_topology_weights(coords::Vector{SVector{3,Float64}},
                                cys_map::Vector{Int},
                                W::Matrix{Float64}; hop_radius::Int=2)
    n = length(coords)

    depth_local  = zeros(Float64, length(cys_map))  # H0 burial proxy
    loop_local   = zeros(Float64, length(cys_map))  # H1 loopiness proxy

    # build unweighted graph from W
    G = SimpleGraph(n)
    for i in 1:n-1, j in i+1:n
        W[i,j] > 0 && add_edge!(G,i,j)
    end

    for (b,resi) in enumerate(cys_map)
        # --- collect nodes within hop_radius ---
        nodes = Set([resi])
        frontier = Set([resi])
        for _ in 1:hop_radius
            newfrontier = Set{Int}()
            for v in frontier
                for nb in neighbors(G,v)
                    push!(newfrontier, nb)
                end
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

        # local pairwise distances
        Dloc = [norm(coords[idxs[i]] - coords[idxs[j]]) for i in 1:m, j in 1:m]

        # persistent homology up to H1
        dgms = ripserer(Dloc, dim_max=1)

        # burial proxy: death time of resi’s local H0 bar (or 2nd nearest neighbor)
        pos = findfirst(==(resi), idxs)
        if pos !== nothing && !isempty(dgms[1]) && pos <= length(dgms[1])
            d = dgms[1][pos].death
            depth_local[b] = isfinite(d) ? d : maximum(Dloc)
        else
            dd = sort(Dloc[pos,:])
            depth_local[b] = dd[min(3,length(dd))]
        end

        # loopiness proxy: total finite H1 persistence
        if length(dgms) >= 2 && !isempty(dgms[2])
            loop_local[b] = sum(bar -> (isfinite(bar.death) ? (bar.death - bar.birth) : 0.0), dgms[2])
        else
            loop_local[b] = 1e-6
        end
    end

    # normalize and combine
    depth_local ./= sum(depth_local) > 0 ? sum(depth_local) : 1.0
    loop_local  ./= sum(loop_local)  > 0 ? sum(loop_local)  : 1.0
    w_morse = (depth_local .+ loop_local) ./ sum(depth_local .+ loop_local)
    return w_morse, depth_local, loop_local
end


# ---------- Build modal Laplacians ----------
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

function diamond_from_dual(w_energy::Vector{Float64}, w_morse::Vector{Float64};
                           mode::Symbol = :average)
    # modes: :average → (wE + wM)/2
    #        :tropical → exp.( - min( -log(wE), -log(wM) ) )
    wbits = similar(w_energy)
    if mode == :average
        @inbounds for i in eachindex(wbits)
            wbits[i] = 0.5*(w_energy[i] + w_morse[i])
        end
    elseif mode == :tropical
        @inbounds for i in eachindex(wbits)
            cE = -log(w_energy[i] + eps())
            cM = -log(w_morse[i] + eps())
            wbits[i] = exp( -min(cE, cM) )
        end
        wbits ./= sum(wbits)
    else
        error("Unknown fusion mode: $mode")
    end
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

# ---------- Propagation ----------
function propagate(Ld::AbstractMatrix, rho0::AbstractVector, t::Float64)
    K = exp(-t .* Matrix(Ld))
    rho_t = K * rho0
    s = sum(rho_t)
    s > 0 && (rho_t ./= s)
    return rho_t
end

# ---------- Modal curvature diagnostics ----------
function modal_curvature(Ld::AbstractMatrix; src_state::String="000")
    n = size(Ld,1)
    ρ = zeros(Float64, n); ρ[IDX[src_state]] = 1.0
    ρ .-= mean(ρ)
    φ = laplacian_pinv(Ld) * ρ
    φ .-= mean(φ)

    R = Ld * φ  # curvature vector
    return R, φ
end

# ---------- Simple Forman curvature on weighted graph ----------
# Ref: Forman-Ricci curvature for weighted graphs
# κ(i,j) = w(i,j) * ( (1/w(i,j))*(w(i)/deg(i) + w(j)/deg(j)) - sum over 2-faces ... )
# For CA graph we ignore higher cells, use edge-based approx

function forman_curvature(W::Matrix{Float64})
    n = size(W,1)
    κ = zeros(Float64, n, n)
    degs = vec(sum(W,dims=2))
    for i in 1:n-1, j in i+1:n
        wij = W[i,j]; wij == 0 && continue
        κ[i,j] = (degs[i] + degs[j]) / wij - sum(W[i,:]./wij) - sum(W[j,:]./wij)
        κ[j,i] = κ[i,j]
    end
    return κ
end

function cysteine_curvatures(W::Matrix{Float64}, cys_map::Vector{Int})
    κ = forman_curvature(W)
    κ_nodes = zeros(Float64, length(cys_map))
    for (b,resi) in enumerate(cys_map)
        # average curvature of edges attached to cysteine
        neighs = findall(x->x>0, W[resi,:])
        κ_nodes[b] = mean(κ[resi, neighs])
    end
    # normalize to positive scale
    κ_nodes .-= minimum(κ_nodes)
    κ_nodes ./= maximum(κ_nodes) > 0 ? maximum(κ_nodes) : 1
    return κ_nodes
end

# ---------- Build modal Laplacian with curvature-as-harder-to-move ----------
"""
Curvature penalty via sign flip: higher κ → smaller edge weight.

wbits = normalize( w_energy .* clamp.(1 .- κ_nodes, eps(), Inf) )

- No new parameters introduced.
- Assumes κ_nodes has been normalized to [0,1] upstream.
"""
function diamond_from_curved(w_energy::Vector{Float64}, κ_nodes::Vector{Float64};
                             mode::Symbol = :geometric)  # kept for API compatibility
    @assert length(w_energy) == length(κ_nodes)
    # guard against negatives or >1 due to numerical drift
    κ = clamp.(κ_nodes, 0.0, 1.0)
    w_e = max.(w_energy, eps())

    # SIGN FLIP: more curvature → less weight
    wbits = w_e .* clamp.(1 .- κ, eps(), Inf)

    # optional tropical variant still supported (sign-flipped logic):
    if mode == :tropical
        # equivalently: prefer the harsher cost; here curvature contributes as (1-κ)
        cE = .-log.(w_e .+ eps())
        cK = .-log.(clamp.(1 .- κ, eps(), 1.0))  # larger κ → larger cost
        c  = max.(cE, cK)
        wbits = exp.(-c)
    end

    # normalize safely
    s = sum(wbits)
    wbits = s > 0 ? (wbits ./ s) : fill(1.0/length(wbits), length(wbits))

    return diamond_from_weights(wbits), wbits
end


# =============================== RUN =============================
pdb = "/content/AF-P04406-F1-model_v4.pdb"   # <-- set your path

coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA; CYS indices: ", cys_idx_all)
@assert length(cys_idx_all) ≥ 3 "Need ≥3 CYS to map 3 bits."
cys_map = cys_idx_all[1:3]

# PDE on the PDB substrate (graph Laplacian on CA point cloud)
Wreal, Lreal = build_weighted_graph(coords; cutoff=6.5, sigma=4.0)
w_energy, Eb, Sb = bitwise_pushforward(Wreal, Lreal, cys_map, length(coords))
println("\n=== PDE → bit weights (energy channel) ===")
for b in 1:3
    @printf("bit %d : Eb=% .6e  Sb=% .6e  wE=%.6f\n", b, Eb[b], Sb[b], w_energy[b])
end

# Persistent topology on the substrate → Morse landscape
w_morse, tau0, tau1 = morse_topology_weights(coords, cys_map, Wreal)
println("\n=== Substrate topology → Morse weights ===")
for b in 1:3
    @printf("bit %d : depth(H0)=%.6f  loops(H1)=%.6f  wM=%.6f\n", 
            b, tau0[b], tau1[b], w_morse[b])
end

# Build three modal Laplacians: energy-only, morse-only, and fused (average & tropical)
(WdE, LdE) = diamond_from_weights(w_energy)
(WdM, LdM) = diamond_from_weights(w_morse)
((WdA, LdA), w_avg) = diamond_from_dual(w_energy, w_morse; mode=:average)
((WdT, LdT), w_trop) = diamond_from_dual(w_energy, w_morse; mode=:tropical)

# Modal Dirichlet energies (source at 000) as scalar diagnostics
E_E = modal_dirichlet_energy(LdE)
E_M = modal_dirichlet_energy(LdM)
E_A = modal_dirichlet_energy(LdA)
E_T = modal_dirichlet_energy(LdT)
println("\n=== Modal Dirichlet energy (source 000) ===")
@printf("energy-only : %.6e\n", E_E)
@printf("morse-only  : %.6e\n", E_M)
@printf("average     : %.6e\n", E_A)
@printf("tropical    : %.6e\n", E_T)

# Instantaneous outflow from 000
println("\n=== Instantaneous outflow P(000→·) ===")
for (label, Wd) in [("energy",WdE), ("morse",WdM), ("avg",WdA), ("trop",WdT)]
    P0 = initial_outgoing_probs("000", Wd)
    @printf("%-6s : 000→001=%.3f  000→010=%.3f  000→100=%.3f\n",
            label, P0["001"], P0["010"], P0["100"])
end

# Master-equation propagation on fused Laplacians
times = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
rho0 = zeros(Float64, length(STATES)); rho0[IDX["000"]] = 1.0

function report_evolution(label::String, Ld::AbstractMatrix)
    println("\n=== Evolution on ", label, " Laplacian ===")
    for t in times
        rho_t = propagate(Ld, rho0, t)
        @printf("t=%4.1f   max=%s(%.6f)   mass@111=%.6f\n",
                t, STATES[argmax(rho_t)], maximum(rho_t), rho_t[IDX["111"]])
    end
end

report_evolution("energy-only", LdE)
report_evolution("morse-only",  LdM)
report_evolution("average",     LdA)
report_evolution("tropical",    LdT)

# Modal curvature + Dirichlet energies
for (label, Ld) in [("energy",LdE), ("morse",LdM), ("avg",LdA), ("trop",LdT)]
    E = modal_dirichlet_energy(Ld)
    R, φ = modal_curvature(Ld)
    println("\n--- $label channel ---")
    @printf("Dirichlet energy: %.6e\n", E)
    @printf("Curvature sum: %.3e (should ≈0)\n", sum(R))
    println("Local curvature R[i] for first 5 states:")
    for s in STATES[1:5]
        @printf("  %s : %.4e\n", s, R[IDX[s]])
    end
end

# PDE weights
Wreal, Lreal = build_weighted_graph(coords; cutoff=6.5, sigma=4.0)
w_energy, Eb, Sb = bitwise_pushforward(Wreal, Lreal, cys_map, length(coords))

# Curvature channel
κ_nodes = cysteine_curvatures(Wreal, cys_map)
println("\n=== Substrate curvature per cysteine ===")
for (b,κ) in enumerate(κ_nodes)
    @printf("bit %d : κ_node=%.6f\n", b, κ)
end

# Curvature-infused modal Laplacian
((WdC, LdC), w_curved) = diamond_from_curved(w_energy, κ_nodes)  # sign-flipped fusion

println("\n=== Curvature-fused modal weights ===")
println(w_curved)

report_evolution("curvature-fused", LdC)
