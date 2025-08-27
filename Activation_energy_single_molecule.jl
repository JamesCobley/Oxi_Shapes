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

# --- Graph-radius based topology weights (optional diagnostics) ---
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
        # collect nodes within hop_radius
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

        # local pairwise distances
        Dloc = [norm(coords[idxs[i]] - coords[idxs[j]]) for i in 1:m, j in 1:m]

        # persistent homology up to H1
        dgms = ripserer(Dloc, dim_max=1)

        # burial proxy: median distance
        depth_local[b] = median(Dloc)

        # loopiness proxy: total finite H1 persistence
        if length(dgms) >= 2 && !isempty(dgms[2])
            loop_local[b] = sum(bar -> (isfinite(bar.death) ?
                                        (bar.death - bar.birth) : 0.0), dgms[2])
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

# ---------- Internal activation from geometry ----------
const Rgas = 8.314462618  # J/mol/K

"""
    activation_from_dirichlet(T, Eb)

Convert Dirichlet energies per cysteine bit into activation energies ΔG‡ (J/mol).
"""
function activation_from_dirichlet(T::Float64, Eb::Vector{Float64})
    return (Rgas * T) .* Eb
end

# ---------- Axiom 2: Volume conservation ----------
"""
    normalize_modal!(ρ)  # sum(ρ) = 1
"""
function normalize_modal!(ρ::Vector{Float64})
    total = sum(ρ)
    if total > 0
        ρ ./= total
    end
    return ρ
end

# ---------- Modal curvature (Axiom 4) ----------
"""
    modal_curvature(Ld, ρ) = Ld * ρ
"""
function modal_curvature(Ld::AbstractMatrix, ρ::Vector{Float64})
    return Ld * ρ
end

# ---------- Modal Dirichlet energy ----------
"""
    modal_dirichlet_energy(Ld, ρ) = 0.5 * ρ' * Ld * ρ
"""
function modal_dirichlet_energy(Ld::AbstractMatrix, ρ::Vector{Float64})
    return 0.5 * dot(ρ, Ld * ρ)
end

# ---------- Curvature + Real-activation weighted modal Laplacian ----------
"""
    modal_laplacian_curved_real(ρ, ΔG_real_bits, T; α=1.0)

Build occupancy- and real-activation–dependent Laplacian on the modal manifold.

- ρ            : occupancy over STATES (sum = 1)
- ΔG_real_bits : ΔG‡ per bit (J/mol) from real geometry (length = number of bits)
- T            : temperature (K)
- α            : curvature sensitivity

Edge weights:
W_ij = exp(-α * |ρ_i - ρ_j|) * exp(- (ΔG_real_bits[b(i↔j)] / (Rgas*T)) )

Returns: (Ld, W_total, W_real_only)
"""
function modal_laplacian_curved_real(
    ρ::Vector{Float64},
    ΔG_real_bits::Vector{Float64},
    T::Float64; α::Float64=1.0
)
    n = length(STATES)
    β = 1.0 / (Rgas * T)

    W           = zeros(Float64, n, n)
    W_real_only = zeros(Float64, n, n)

    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            b = flipped_bit_index(s, t)
            w_curv = exp(-α * abs(ρ[i] - ρ[j]))
            w_real = exp(-β * ΔG_real_bits[b])
            w = w_curv * w_real
            W[i,j] = w;            W[j,i] = w
            W_real_only[i,j] = w_real; W_real_only[j,i] = w_real
        end
    end
    D  = Diagonal(vec(sum(W, dims=2)))
    Ld = D - W
    return Ld, W, W_real_only
end

# ---------- Modal Morse potential (uses curvature-modified Laplacian) ----------
"""
    modal_morse_potential(Ld, ρ)

Solve Ld φ = ρ (mean-zero) to get the modal Morse potential φ.
ρ is normalized to sum=1 (Axiom 2) before solving.
"""
function modal_morse_potential(Ld::AbstractMatrix, ρ::Vector{Float64})
    ρ = copy(ρ)
    s = sum(ρ); s > 0 && (ρ ./= s)
    ρ .-= mean(ρ)
    φ = laplacian_pinv(Ld) * ρ
    φ .-= mean(φ)
    return φ
end

# ---------- Modal Morse edge saddles (per allowed edge) ----------
"""
    modal_morse_edgebarriers(φ) → S (S_ij = 0.5*(φ_i - φ_j)^2 on Hamming-1 edges)
"""
function modal_morse_edgebarriers(φ::Vector{Float64})
    n = length(STATES)
    S = fill(NaN, n, n)
    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            sij = 0.5 * (φ[i] - φ[j])^2
            S[i,j] = sij
            S[j,i] = sij
        end
    end
    return S
end

# ---------- Curvature-modified spectrum (optional) ----------
function vibrational_modes_curved(ρ::Vector{Float64}; α::Float64=1.0, T::Float64=298.15)
    R = length(STATES[1])                     # number of bits from the state strings
    zeros_bits = zeros(Float64, R)            # disable real term cleanly
    Ldρ, _, _ = modal_laplacian_curved_real(ρ, zeros_bits, T; α=α)
    F = eigen(Symmetric(Matrix(Ldρ)))
    return F.values, F.vectors
end

# ---------- Spectroscopic vibrational gating (real + modal, spectrum-tied) ----------
"""
    vibrational_gating_spectrum(T, Eb, S_modal, λ; ν0=700.0, γ_modal=1.0)

Combine real (PDE Dirichlet) and modal (Morse curvature) penalties,
then rescale to chemically realistic ΔG‡ using vibrational quanta.
Number of quanta N_b is derived from the curvature spectrum λ.

Arguments:
- `T`      : temperature (K)
- `Eb`     : vector of per-bit Dirichlet energies (dimensionless)
- `S_modal`: modal saddle penalties (matrix, dimensionless)
- `λ`      : vector of modal Laplacian eigenvalues (from vibrational_modes_curved)
- `ν0`     : reference vibrational frequency (cm^-1)
- `γ_modal`: scaling factor for modal contribution

Returns:
- ΔG_gated : vector of gated ΔG‡ (J/mol)
- weights  : normalized Boltzmann weights
- report   : summary string
"""
# ---------- Spectroscopic vibrational gating ----------
function vibrational_gating_spectrum(
    T::Float64,
    Eb::Vector{Float64},
    S_modal::Matrix{Float64},
    λ::Vector{Float64};
    ν0::Float64 = 700.0,
    γ_modal::Float64 = 1.0,
    target_mean_quanta::Float64 = 3.5  # sets scale of N; tweak if you want 2–5, etc.
)
    Rgas = 8.314462618
    RT   = Rgas * T
    hc_per_cm = 0.01196e3          # J/mol per cm^-1
    one_quantum = hc_per_cm * ν0

    nbits = length(Eb)
    @assert nbits <= length(λ)-1 "Need at least nbits nonzero λ entries (λ1=0)."

    # Map bit b ↔ smallest nonzero modes λ[2], λ[3], λ[4], ...
    ω = sqrt.(max.(λ[2:1+nbits], 1e-15))  # dimensionless “frequencies”

    # Choose a scale κ so that mean(N_raw) ≈ target_mean_quanta
    # N_raw = κ * ω / ν0  ⇒  κ = target_mean_quanta * ν0 / mean(ω)
    κ = target_mean_quanta * ν0 / mean(ω)

    ΔG_gated = zeros(Float64, nbits)
    weights  = zeros(Float64, nbits)
    io = IOBuffer()
    println(io, "\n=== Spectroscopic vibrational gating (spectrum-tied) ===")
    @printf(io, "ν0 = %.1f cm^-1  →  one quantum = %.2f kJ/mol\n", ν0, one_quantum/1e3)

    for b in 1:nbits
        # Real term (protein PDE)
        ΔG_real  = RT * Eb[b]

        # Modal Morse saddle: take the edge 000 → flip(bit b)
        s = "000"
        t = hamming1_neighbors(s)[b]
        i, j = IDX[s], IDX[t]
        ΔG_modal = γ_modal * RT * S_modal[i,j]

        # Geometric barrier: real + modal (in J/mol)
        ΔG_geom = ΔG_real + ΔG_modal

        # Quanta from spectrum: N_b = round( κ * ω_b / ν0 ), at least 1
        N = max(1, round(Int, κ * ω[b] / ν0))

        add_vib = N * one_quantum
        ΔG_g = ΔG_geom + add_vib              # gated barrier
        ΔG_gated[b] = ΔG_g
        weights[b]  = exp(-ΔG_g / RT)

        @printf(io,
            "bit %d : ΔG_real=%.2f  ΔG_modal=%.2f  N=%d  add_vib=%.2f  ⇒  ΔG_gated=%.2f kJ/mol\n",
            b, ΔG_real/1e3, ΔG_modal/1e3, N, add_vib/1e3, ΔG_g/1e3)
    end

    # Normalize weights
    weights ./= sum(weights)
    @printf(io, "\nGated weights [bits] = %s\n", string(round.(weights, digits=3)))

    return ΔG_gated, weights, String(take!(io))
end

# =============================== RUN =============================
pdb = "/content/AF-P04406-F1-model_v4.pdb"   # <-- your PDB path
T   = 293.15                                  # 20 °C

# --- Parse PDB & pick first 3 cysteines as bits ---
coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA; CYS indices: ", cys_idx_all)
@assert length(cys_idx_all) >= 3 "Need ≥3 CYS"
cys_map = cys_idx_all[1:3]

# --- Real manifold (protein) Laplacian ---
Wreal, Lreal = build_weighted_graph(coords; cutoff=6.5, sigma=4.0)

# --- Per-bit PDE energies on the real manifold ---
_, Eb, Sb = bitwise_pushforward(Wreal, Lreal, cys_map, length(coords))
println("\n=== PDE (real) per-bit energies ===")
for b in 1:3
    @printf("bit %d : Eb=% .6e  Sb=% .6e\n", b, Eb[b], Sb[b])
end

# --- Convert real Dirichlet to activation energies (ΔG‡_real per bit, J/mol) ---
ΔG_real_bits = activation_from_dirichlet(T, Eb)
@printf("\nΔG‡_real (per bit) [kJ/mol]: [%.2f, %.2f, %.2f]\n",
        ΔG_real_bits[1]/1e3, ΔG_real_bits[2]/1e3, ΔG_real_bits[3]/1e3)

# --- Choose an initial occupancy ρ over STATES (Axiom 2 enforced) ---
ρ = zeros(Float64, length(STATES))
ρ[IDX["000"]] = 1.0             # all mass in "000" for this test
normalize_modal!(ρ)

# --- Build curvature + real-weighted modal Laplacian ---
Ldρ, W_total, W_real = modal_laplacian_curved_real(ρ, ΔG_real_bits, T; α=1.0)

# --- Modal Morse potential (on the curvature-modified Laplacian) ---
φ = modal_morse_potential(Ldρ, ρ)
S_modal = modal_morse_edgebarriers(φ)   # dimensionless Morse saddle height per allowed edge

println("\n=== Edge diagnostics (curved Laplacian + real edge weights) ===")
for s in STATES
    i = IDX[s]
    for t in hamming1_neighbors(s)
        j = IDX[t]
        b = flipped_bit_index(s,t)
        @printf("%s ↔ %s  (bit %d):  W_real=%.3e   W_total=%.3e   S_modal=%.4e\n",
                s, t, b, W_real[i,j], W_total[i,j], S_modal[i,j])
    end
end

# --- Axiom 4 curvature and modal Dirichlet energy for current ρ ---
R_modal = modal_curvature(Ldρ, ρ)
E_modal = modal_dirichlet_energy(Ldρ, ρ)
@printf("\nModal Dirichlet energy E_modal = %.6e   (sum R = %.3e)\n", E_modal, sum(R_modal))

# --- (Optional) spectral fingerprint with curvature only (no real weights) ---
vals_curved, _ = vibrational_modes_curved(ρ; α=1.0)
println("\n=== Curvature-only modal spectrum (all 8 eigenvalues) ===")
for k in 1:length(vals_curved)
    @printf("λ[%d] = %.6f  (ω = %.6f)\n", k, vals_curved[k], sqrt(max(vals_curved[k],0.0)))
end


# after you've computed φ and S_modal, and have Ldρ already:
F = eigen(Symmetric(Matrix(Ldρ)))
λ = F.values

F = eigen(Symmetric(Matrix(Ldρ)))
λ = F.values

ΔG_gated, w_gated, report = vibrational_gating_spectrum(T, Eb, S_modal, λ; ν0=700.0, γ_modal=1.0)
println(report)
