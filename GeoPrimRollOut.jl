# ===============================================================
# ONE-PROTEIN RUNNER — MGF state-conditional activation energies
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

# ---------------- Real graph + PDE weights ----------------
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

"Compute bitwise PDE pushforward weights w_E over cysteines"
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

# ---------------- Boolean manifold utils (arbitrary R) ----------------
function bitstrings(R::Int)
    out = Vector{String}(undef, 2^R)
    for i in 0:(2^R-1)
        s = bitstring(i)
        out[i+1] = s[end-R+1:end]
    end
    return out
end

function hamming1_neighbors(s::String)
    c = collect(s)
    nbrs = String[]
    for k in 1:length(c)
        c2 = copy(c); c2[k] = (c2[k] == '0' ? '1' : '0')
        push!(nbrs, String(c2))
    end
    return nbrs
end

function flipped_bit_index(s::String, t::String)::Int
    bs, bt = collect(s), collect(t)
    for k in 1:length(bs)
        if bs[k] != bt[k]; return k; end
    end
    error("States not Hamming-1 neighbors: $s → $t")
end

function build_state_index(R::Int)
    S = bitstrings(R)
    IDX = Dict(s=>i for (i,s) in enumerate(S))
    return S, IDX
end

# ---------------- Modal weights & Forman curvature ----------------
function modal_weights_from_occupancy(ρ::AbstractVector{<:Real}, α::Float64,
                                      STATES::Vector{String}, IDX::Dict{String,Int})
    N = length(STATES)
    W = zeros(Float64, N, N)
    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            wcurv = exp(-α * abs(ρ[i] - ρ[j]))  # kernel; α tunes sensitivity
            W[i,j] = wcurv; W[j,i] = wcurv
        end
    end
    return W
end

function forman_edge_curvature(W::AbstractMatrix, i::Int, j::Int; node_weight::Float64=1.0)
    w_e = W[i,j]; w_e == 0 && return 0.0
    term = (2*node_weight / w_e)
    for k in 1:size(W,1)
        k == j && continue
        w_ik = W[i,k]; w_ik == 0 && continue
        term -= node_weight / sqrt(max(w_e * w_ik, eps()))
    end
    for k in 1:size(W,1)
        k == i && continue
        w_jk = W[j,k]; w_jk == 0 && continue
        term -= node_weight / sqrt(max(w_e * w_jk, eps()))
    end
    return term
end

# ---------------- Activation-energy field (R-invariant) ----------------
function activation_energy_field(ρ::Vector{Float64}, wE::Vector{Float64};
                                 κ_modal::Float64=1.0e4, α::Float64=1.0,
                                 STATES::Vector{String}, IDX::Dict{String,Int})
    Wm = modal_weights_from_occupancy(ρ, α, STATES, IDX)
    ΔG_eff = Dict{Tuple{Int,Int},Float64}()
    K_edge = Dict{Tuple{Int,Int},Float64}()
    R = length(wE)                              # number of cysteines (bits)
    log_uniform = log(inv(R))                   # ln(1/R) baseline

    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            K = forman_edge_curvature(Wm, i, j)
            b = flipped_bit_index(s,t)          # which bit flips on s->t
            ΔG_curv = κ_modal * abs(K)

            # --- R-invariant primitive correction ---
            # Use deviation from uniform: log(wE[b]) - log(1/R)
            # So when wE[b] = 1/R (all equal), factor = 1 exactly.
            log_corr = log(max(wE[b], eps())) - log_uniform
            ΔG_eff[(i,j)] = ΔG_curv * (1 - log_corr)

            K_edge[(i,j)] = K
        end
    end
    return ΔG_eff, K_edge
end

# ---------------- Helpers ----------------
normalize!(v) = (s = sum(v); s>0 && (v ./= s); v)
rho_delta(STATES, IDX, s::String) = (ρ = zeros(Float64, length(STATES)); ρ[IDX[s]] = 1.0; ρ)

function top_outgoing(ΔG_eff::Dict{Tuple{Int,Int},Float64}, STATES, IDX, s::String; k::Int=3, only_oxidation::Bool=false)
    i = IDX[s]
    pairs = Tuple{String,Int,Float64}[]
    for t in hamming1_neighbors(s)
        j = IDX[t]
        if only_oxidation
            b = flipped_bit_index(s,t)
            if s[b] == '1'
                continue
            end
        end
        push!(pairs, (t, flipped_bit_index(s,t), ΔG_eff[(i,j)]))
    end
    sort!(pairs, by = x->x[3])  # ascending energy
    return pairs[1:min(k, length(pairs))]
end

# ---------------- Pretty labels & helpers ----------------
"Human-readable bit labels like: bit 3 (Cys residue CA index 247)"
function bit_label(bit::Int, cys_map::Vector{Int})
    ridx = (bit <= length(cys_map)) ? cys_map[bit] : missing
    isnothing(ridx) ? "bit $bit" : "bit $bit (Cys_CA=$ridx)"
end

"Return true if the move is oxidation (0→1) for the flipped bit."
is_oxidation(s::String, t::String) = begin
    b = flipped_bit_index(s,t); s[b] == '0' && t[b] == '1'
end

# ---------------- Tabulate field (for printing / export) ----------------
"""
Collect a tidy table of all Hamming-1 edges:
Returns a Vector of NamedTuples with fields:
:from, :to, :bit, :bit_label, :direction, :dG_kJ, :K_forman
You can restrict to oxidation-only and/or a subset of states.
"""
function collect_edge_table(ΔG_eff::Dict{Tuple{Int,Int},Float64},
                            K_edge::Dict{Tuple{Int,Int},Float64},
                            STATES::Vector{String}, IDX::Dict{String,Int},
                            cys_map::Vector{Int};
                            only_oxidation::Bool=false,
                            states_subset::Union{Nothing,Vector{String}}=nothing)

    states = isnothing(states_subset) ? STATES : states_subset
    rows = NamedTuple[]
    for s in states
        i = IDX[s]
        nbrs = hamming1_neighbors(s)
        for t in nbrs
            j = IDX[t]
            if only_oxidation && !is_oxidation(s,t)
                continue
            end
            b = flipped_bit_index(s,t)
            dir = is_oxidation(s,t) ? "oxidation" : "reduction"
            push!(rows, (
                from = s,
                to = t,
                bit = b,
                bit_label = bit_label(b, cys_map),
                direction = dir,
                dG_kJ = ΔG_eff[(i,j)]/1e3,
                K_forman = K_edge[(i,j)]
            ))
        end
    end
    return rows
end

# ===============================================================
# RUN FOR ONE PROTEIN
# ===============================================================
# 1) Point to your structure file (AlphaFold or PDB for THIS protein)
pdb_path = "/content/AF-P18031-F1-model_v4.pdb"   # <-- CHANGE THIS

# 2) Parse structure and cysteines
coords, cys_idx_all = load_ca_and_cys(pdb_path)
@printf("Parsed %d CA atoms; %d Cys CA indices found.\n", length(coords), length(cys_idx_all))
@assert length(cys_idx_all) >= 2 "Need at least 2 cysteines."

# (Optional) choose a subset for an R-bit manifold; otherwise use all Cys:
cys_map = cys_idx_all                     # use all cysteines
R = length(cys_map)

# 3) Build real graph & PDE weights
Wreal, Lreal = build_weighted_graph(coords; cutoff=6.5, sigma=4.0)
wE, Eb, Sb   = bitwise_pushforward(Wreal, Lreal, cys_map, length(coords))
@printf("\nBitwise PDE weights (w_E, sum=1): %s\n", join(round.(wE; digits=3), ", "))

R = length(wE)
@printf("R = %d, mean wE = %.4f, baseline ln(1/R)=%.4f\n", R, mean(wE), log(1/R))

# 4) Build modal space
STATES, IDX = build_state_index(R)
s0 = "0"^R   # fully reduced canonical start

# 5) Choose occupancy ρ (single-molecule at s0 for primitive)
ρ = rho_delta(STATES, IDX, s0)

# 6) Compute ΔG‡_eff field
κ_modal = 1.0e4  # global coupling (J/mol), calibrate once if you want kJ/mol to align
α        = 1.0   # curvature sensitivity hyperparam
ΔG_eff, K_edge = activation_energy_field(ρ, wE; κ_modal=κ_modal, α=α, STATES=STATES, IDX=IDX)

# 7) Report the favoured next oxidation moves from s0
println("\nTop outgoing oxidation moves from $s0 (lowest ΔG‡ first):")
for (t, bit, dge) in top_outgoing(ΔG_eff, STATES, IDX, s0; k=min(R,3), only_oxidation=true)
    @printf("  %s  --flip bit %d-->  %s    ΔG‡_eff = %.2f kJ/mol\n", s0, bit, t, dge/1e3)
end

# 8) Take a step (simulate oxidation of the best bit), then recompute for new state
best = first(top_outgoing(ΔG_eff, STATES, IDX, s0; k=1, only_oxidation=true))
t1 = best[1]   # new state after first oxidation
println("\nEvolving one oxidation step to state: ", t1)

# For single molecule: move mass from s0 to t1
ρ_next = zeros(Float64, length(STATES)); ρ_next[IDX[t1]] = 1.0
ΔG_eff2, _ = activation_energy_field(ρ_next, wE; κ_modal=κ_modal, α=α, STATES=STATES, IDX=IDX)

println("\nTop outgoing oxidation moves from $(t1):")
for (t, bit, dge) in top_outgoing(ΔG_eff2, STATES, IDX, t1; k=min(R-1,3), only_oxidation=true)
    @printf("  %s  --flip bit %d-->  %s    ΔG‡_eff = %.2f kJ/mol\n", t1, bit, t, dge/1e3)
end
