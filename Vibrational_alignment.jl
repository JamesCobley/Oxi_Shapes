############################
# Oxi-Shapes MGF – Full Pipeline (axiomatic + tropical + alignment)
############################

using LinearAlgebra, Statistics, Printf
using StaticArrays, Graphs

# -------------------- PDB parsing (CA only) ---------------------
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

# -------------------- Weighted residue graph (real manifold) ----
function build_weighted_graph(coords::Vector{SVector{3,Float64}}; cutoff=6.5, σ=4.0)
    n = length(coords)
    W = zeros(Float64, n, n)
    inv2σ2 = 1.0 / (2σ^2)
    for i in 1:n-1, j in i+1:n
        d = norm(coords[i] - coords[j])
        if d <= cutoff
            w = exp(-(d*d) * inv2σ2)
            W[i,j] = w; W[j,i] = w
        end
    end
    d = vec(sum(W, dims=2))
    L = Diagonal(d) - W
    return W, L
end

# -------------------- Laplacian pseudoinverse & Poisson ---------
function laplacian_pinv(L::AbstractMatrix; tol=1e-12)
    F = eigen(Symmetric(Matrix(L)))
    λ, U = F.values, F.vectors
    λplus = similar(λ)
    @inbounds for i in eachindex(λ)
        λplus[i] = λ[i] > tol ? inv(λ[i]) : 0.0
    end
    return U * Diagonal(λplus) * U'
end

function solve_poisson_mean_zero(L::AbstractMatrix, ρ::AbstractVector)
    ρ0 = ρ .- mean(ρ)          # volume-invariant RHS
    φ  = laplacian_pinv(L) * ρ0
    φ .-= mean(φ)              # remove numerical drift
    return φ
end

dirichlet_energy(L::AbstractMatrix, φ::AbstractVector) = 0.5 * dot(φ, L * φ)

# -------------------- Diamond topology (modal manifold) ---------
if !isdefined(Main, :STATES)
    const STATES = ["000","100","010","001","110","101","011","111"]
    const IDX = Dict(s=>i for (i,s) in enumerate(STATES))
end

bitvec(s::String) = [c=='1' for c in collect(s)]

function hamming1_neighbors(s::String)
    chars = collect(s)
    res = String[]
    for i in 1:length(chars)
        tmp = copy(chars)
        tmp[i] = (chars[i] == '0') ? '1' : '0'
        push!(res, String(tmp))
    end
    return res
end

# -------------------- Per-bit geometry on real manifold ---------
function per_bit_geometry(L::AbstractMatrix, cys_map::Vector{Int}, nres::Int)
    Rb = zeros(Float64, length(cys_map))
    Eb = zeros(Float64, length(cys_map))
    for (b, resi) in enumerate(cys_map)
        ρ = zeros(Float64, nres); ρ[resi] = 1.0
        φ = solve_poisson_mean_zero(L, ρ)
        R = L * φ
        Rb[b] = R[resi]
        Eb[b] = dirichlet_energy(L, φ)
    end
    return Rb, Eb
end

# -------------------- Optional anisotropy diagnostics -----------
function diamond_xy()
    xy = Dict{String,SVector{2,Float64}}()
    xy["000"] = SVector(0.0, 0.0)
    xy["100"] = SVector(-1.0, 1.0); xy["010"] = SVector(0.0, 1.0); xy["001"] = SVector(1.0, 1.0)
    xy["110"] = SVector(-1.0, 2.0); xy["101"] = SVector(0.0, 2.0); xy["011"] = SVector(1.0, 2.0)
    xy["111"] = SVector(0.0, 3.0)
    return xy
end

function node_anisotropy_curvature(Rx::Vector{Float64})
    xy = diamond_xy()
    ρ = zeros(Float64, length(STATES)); ρ[IDX["000"]] = 1.0
    coords3 = Dict{String,SVector{3,Float64}}()
    for s in STATES
        p = xy[s]; coords3[s] = SVector(p[1], p[2], -ρ[IDX[s]])
    end
    A_node = zeros(Float64, length(STATES))
    for s in STATES
        i = IDX[s]; N = hamming1_neighbors(s)
        isempty(N) && continue
        acc = 0.0
        for t in N
            j = IDX[t]
            ΔR = Rx[i] - Rx[j]
            Δx = coords3[s] - coords3[t]
            acc += ΔR / (norm(Δx) + eps())
        end
        A_node[i] = acc / length(N)
    end
    mn, mx = minimum(A_node), maximum(A_node)
    return (A_node .- mn) ./ (mx - mn + eps())
end

function bit_anisotropy_curvature(Rx::Vector{Float64})
    xy = diamond_xy()
    ρ = zeros(Float64, length(STATES)); ρ[IDX["000"]] = 1.0
    coords3 = Dict{String,SVector{3,Float64}}()
    for s in STATES
        p = xy[s]; coords3[s] = SVector(p[1], p[2], -ρ[IDX[s]])
    end
    A_bit_raw = zeros(Float64, 3)
    for b in 1:3
        vals = Float64[]
        for s in STATES
            i = IDX[s]; bits_s = bitvec(s)
            for t in hamming1_neighbors(s)
                bits_t = bitvec(t)
                if bits_s[b] != bits_t[b] && sum(bits_s .!= bits_t) == 1
                    j = IDX[t]
                    ΔR = Rx[i] - Rx[j]
                    Δx = coords3[s] - coords3[t]
                    push!(vals, ΔR / (norm(Δx) + eps()))
                end
            end
        end
        A_bit_raw[b] = isempty(vals) ? 0.0 : mean(vals)
    end
    mn, mx = minimum(A_bit_raw), maximum(A_bit_raw)
    return (A_bit_raw .- mn) ./ (mx - mn + eps())
end

# -------------------- Diamond weights & eigenmodes --------------
# Edge weight (allowed flip of bit b): w = A_b_real[b] * exp(-γ * E_b[b])
function diamond_adjacency_weights(A_b_real::Vector{Float64}, E_b::Vector{Float64}; γ::Float64=1.0)
    n = length(STATES)
    Wd = zeros(Float64, n, n)
    for s in STATES
        i = IDX[s]
        for t in hamming1_neighbors(s)
            j = IDX[t]
            bflip = findfirst(k -> bitvec(s)[k] != bitvec(t)[k], 1:3)
            @assert bflip !== nothing
            w = max(1e-12, max(1e-9, A_b_real[bflip]) * exp(-γ * E_b[bflip]))
            Wd[i,j] = w; Wd[j,i] = w
        end
    end
    D = Diagonal(vec(sum(Wd, dims=2)))
    Ld = D - Wd
    return Wd, Ld
end

function diamond_modes(Ld::AbstractMatrix; k::Int=7)
    F = eigen(Symmetric(Matrix(Ld)))
    λ = F.values; V = F.vectors
    ord = sortperm(λ); λ = λ[ord]; V = V[:,ord]
    keep = findall(x -> x > 1e-12, λ)
    isempty(keep) && return Float64[], Array{Float64}(undef, size(V,1), 0)
    kidx = keep[1]:min(keep[1]+k-1, lastindex(λ))
    return λ[kidx], V[:,kidx]
end

# -------------------- Morse classification ----------------------
function classify_morse_on_graph(g::SimpleGraph, φ::Vector{Float64})
    minima = Int[]; saddles = Int[]; maxima = Int[]
    for v in vertices(g)
        N = neighbors(g, v)
        ϕv = φ[v]; ϕN = φ[N]
        if all(ϕv .< ϕN)
            push!(minima, v)
        elseif all(ϕv .> ϕN)
            push!(maxima, v)
        else
            push!(saddles, v)
        end
    end
    return minima, saddles, maxima
end

# -------------------- Tropical eigen (min-plus) -----------------
function tropical_eigen(W::Matrix{Float64}; max_iter=500, tol=1e-9)
    n = size(W,1)
    C = -log.(W .+ 1e-12)  # tropical arc costs
    v = zeros(Float64, n)
    for _ in 1:max_iter
        new_v = fill(Inf, n)
        for i in 1:n, j in 1:n
            new_v[i] = min(new_v[i], C[i,j] + v[j])
        end
        new_v .-= minimum(new_v)
        if maximum(abs.(new_v .- v)) < tol
            return 0.0, new_v
        end
        v .= new_v
    end
    return 0.0, v
end

# -------------------- Motif algebra (discrete) ------------------
struct Motif
    word::Vector{Int}
    cost::Float64
end
motif_cost(word::Vector{Int}, gen_costs::Vector{Float64}) = sum(gen_costs[g] for g in word)
function generate_motifs(gen_costs::Vector{Float64}; kmax=2)
    motifs = Motif[]
    G = length(gen_costs)
    for L in 1:kmax
        for word in Iterators.product(fill(1:G,L)...)
            w = collect(word)
            push!(motifs, Motif(w, motif_cost(w, gen_costs)))
        end
    end
    return motifs
end
motif_string(m::Motif) = isempty(m.word) ? "e" : join(["g$(g)" for g in m.word], "*")

# -------------------- Curvature & transport on diamond ----------
function modal_curvics(L::AbstractMatrix, ρ::AbstractVector; tol=1e-12)
    @assert abs(sum(ρ) - 1.0) < 1e-12 "ρ must sum to 1."
    ρ0 = ρ .- mean(ρ)
    Lplus = laplacian_pinv(L; tol=tol)
    φ = Lplus * ρ0; φ .-= mean(φ)
    R = L * φ
    return φ, R, Lplus
end

function edge_flows_from(φ::AbstractVector, W::AbstractMatrix, s::String)
    i = IDX[s]
    flows = Dict{String,Float64}()
    for t in hamming1_neighbors(s)
        j = IDX[t]
        if W[i,j] > 0
            flows[t] = W[i,j] * (φ[j] - φ[i])
        end
    end
    flows
end

reff(i::Int, j::Int, Lplus::AbstractMatrix) = begin
    e = zeros(size(Lplus,1)); e[i]=1; e[j]-=1
    dot(e, Lplus*e)
end

function edge_score(i::Int, j::Int, W::AbstractMatrix, Lplus::AbstractMatrix; cosθ::Float64=1.0, eps::Float64=1e-12)
    W[i,j] <= 0 && return 0.0
    (W[i,j] * cosθ) / (reff(i,j,Lplus) + eps)
end

function recommend_next_move(s::String, ρ::AbstractVector, W::AbstractMatrix, L::AbstractMatrix;
                             cosθ_map::Dict{Tuple{String,String},Float64}=Dict())
    φ, R, Lplus = modal_curvics(L, ρ)
    i = IDX[s]
    best_t, best_score = "", -Inf
    println("\nFlows and action-like scores from $s (allowed = Hamming-1):")
    for t in hamming1_neighbors(s)
        j = IDX[t]; W[i,j] <= 0 && continue
        cosθ = get(cosθ_map, (s,t), 1.0)
        jflow = W[i,j]*(φ[j]-φ[i])
        score = edge_score(i,j,W,Lplus; cosθ=cosθ)
        Reff  = reff(i,j,Lplus)
        @printf("  %s → %s :  flow=% .4e,  Reff=%.4e,  cosθ=%.3f,  Score=%.4e\n",
                s, t, jflow, Reff, cosθ, score)
        if score > best_score
            best_score, best_t = score, t
        end
    end
    @printf("\n→ Recommended next move:  %s → %s  (max Score=%.4e)\n", s, best_t, best_score)
    return best_t, best_score, φ, R, Lplus
end

function edge_cost(i::Int, j::Int, W::AbstractMatrix, Lplus::AbstractMatrix; cosθ::Float64=1.0, eps::Float64=1e-12)
    W[i,j] <= 0 && return Inf
    reff(i,j,Lplus) / (W[i,j]*cosθ + eps)
end

function min_action_path(s::String, t::String, W::AbstractMatrix, L::AbstractMatrix;
                         cosθ_map::Dict{Tuple{String,String},Float64}=Dict())
    ρ = zeros(Float64, length(STATES)); ρ[IDX[s]] = 1.0
    _, _, Lplus = modal_curvics(L, ρ)
    n = length(STATES); sidx = IDX[s]; tidx = IDX[t]
    dist = fill(Inf, n); prev = fill(0, n); visited = falses(n)
    dist[sidx] = 0.0
    for _ in 1:n
        u = 0; du = Inf
        for v in 1:n
            if !visited[v] && dist[v] < du
                du = dist[v]; u = v
            end
        end
        u == 0 && break
        visited[u] = true
        u == tidx && break
        su = STATES[u]
        for tv in hamming1_neighbors(su)
            v = IDX[tv]; W[u,v] <= 0 && continue
            cosθ = get(cosθ_map, (su,tv), 1.0)
            c = edge_cost(u,v,W,Lplus; cosθ=cosθ)
            if dist[u] + c < dist[v]
                dist[v] = dist[u] + c; prev[v] = u
            end
        end
    end
    path = String[]; cur = tidx
    if dist[tidx] < Inf
        while cur != 0
            pushfirst!(path, STATES[cur]); cur = prev[cur]
        end
    end
    return path, dist[tidx]
end

# -------------------- Axiomatic conservation & functional drift --
function check_conservation_real(Lreal::AbstractMatrix, ρ_real::AbstractVector; label::String="real", tol=1e-6)
    @assert abs(sum(ρ_real) - 1.0) < 1e-12 "ρ_real must sum to 1."
    φ = solve_poisson_mean_zero(Lreal, ρ_real)
    R = Lreal * φ
    s = sum(R); ok = abs(s) ≤ tol
    @printf("Curvature conservation (%s): sum R = %.3e  ->  %s\n", label, s, ok ? "OK (≈0)" : "VIOLATION")
    return R, ok
end

function check_conservation_modal(Ld::AbstractMatrix, ρ_d::AbstractVector; label::String="diamond", tol=1e-6)
    @assert abs(sum(ρ_d) - 1.0) < 1e-12 "ρ_d must sum to 1."
    φ = solve_poisson_mean_zero(Ld, ρ_d)
    R = Ld * φ
    s = sum(R); ok = abs(s) ≤ tol
    @printf("Curvature conservation (%s): sum R = %.3e  ->  %s\n", label, s, ok ? "OK (≈0)" : "VIOLATION")
    return R, ok
end

function functional_drift(R::AbstractVector, weights::AbstractVector; label::String="functional")
    @assert length(R) == length(weights)
    s = sum(R .* weights)
    @printf("Functional drift (%s): weighted sum R = %.3e  ->  DEFORMED (not conserved)\n", label, s)
    return s
end

# Helper: per-node SASA weight = mean SASA of active bits in that node (0 if none)
function node_sasa_weights(A_b_real::Vector{Float64})
    w = zeros(Float64, length(STATES))
    for s in STATES
        bits = bitvec(s); I = findall(identity, bits)
        w[IDX[s]] = isempty(I) ? 0.0 : mean(A_b_real[I])
    end
    return w
end

# =================== Vibrational alignment (no QM) ===================
# External reagent field as localized Gaussian forcing displaced by u
# around residue `resi`. Uses real residue graph (Wreal, Lreal).
function external_forcing_gaussian(coords::Vector{SVector{3,Float64}},
                                   resi::Int; u::SVector{3,Float64}=SVector(0.0,0.0,1.0),
                                   shift::Float64=3.0, σ::Float64=2.0)
    n = length(coords)
    center = coords[resi] + shift * (u / (norm(u)+eps()))
    s = zeros(Float64, n)
    inv2σ2 = 1.0/(2σ^2)
    for j in 1:n
        d2 = sum((coords[j] .- center).^2)
        s[j] = exp(-d2*inv2σ2)
    end
    s ./= sum(s) + eps()
    return s
end

# Alignment factor κ_b from local gradient magnitude of φ_ext at residue b
function alignment_factor_for_bit(Wreal::AbstractMatrix, Lreal::AbstractMatrix,
                                  coords::Vector{SVector{3,Float64}}, resi::Int;
                                  u::SVector{3,Float64}=SVector(0.0,0.0,1.0),
                                  shift::Float64=3.0, σ::Float64=2.0)
    s_ext = external_forcing_gaussian(coords, resi; u=u, shift=shift, σ=σ)
    φ_ext = solve_poisson_mean_zero(Lreal, s_ext)
    # Dirichlet energy of reagent field (for reporting)
    E_ext = dirichlet_energy(Lreal, φ_ext)
    # gradient magnitude about resi using weighted 1-ring in Wreal
    Ng = findall(w -> w>0, Wreal[resi,:])
    if isempty(Ng); return 0.0, E_ext end
    acc = 0.0
    for j in Ng
        acc += sqrt(Wreal[resi,j]) * abs(φ_ext[resi] - φ_ext[j])
    end
    g = acc / length(Ng)
    return g, E_ext
end

# Compute normalized κ_b ∈ [0,1] for all bits, plus per-bit E_ext
function alignment_factors_for_bits(Wreal::AbstractMatrix, Lreal::AbstractMatrix,
                                    coords::Vector{SVector{3,Float64}}, cys_map::Vector{Int};
                                    u::SVector{3,Float64}=SVector(0.0,0.0,1.0),
                                    shift::Float64=3.0, σ::Float64=2.0)
    G = length(cys_map)
    g = zeros(Float64, G)
    Eext = zeros(Float64, G)
    for (k, resi) in enumerate(cys_map)
        g[k], Eext[k] = alignment_factor_for_bit(Wreal, Lreal, coords, resi; u=u, shift=shift, σ=σ)
    end
    mn, mx = minimum(g), maximum(g)
    κ = (g .- mn) ./ (mx - mn + eps())  # normalize to [0,1]
    return κ, g, Eext
end

# Map κ_b onto modal edges as cosθ for gating (Hamming-1 flips)
function build_cosθ_map_from_kappa(κ::Vector{Float64})
    m = Dict{Tuple{String,String},Float64}()
    for s in STATES
        bs = bitvec(s)
        for t in hamming1_neighbors(s)
            bt = bitvec(t)
            bflip = findfirst(k -> bs[k] != bt[k], 1:3)
            @assert bflip !== nothing
            m[(s,t)] = κ[bflip]
        end
    end
    return m
end

# =========================== RUN ================================
# --- Inputs
pdb = "/content/AF-P04406-F1-model_v4.pdb"         # <- update if needed
coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA residues; CYS CA at indices: ", cys_idx_all)

# Bit mapping (Cys indices) & SASA-like exposures (functional weights)
cys_map  = [152,156,247]
A_b_real = [25.0, 0.0, 0.47]

# Real manifold
Wreal, Lreal = build_weighted_graph(coords; cutoff=6.5, σ=4.0)
Rb, Eb = per_bit_geometry(Lreal, cys_map, length(coords))   # bitwise curvature & Dirichlet (real)
# Diagnostic aggregation only
function node_attributes_from_bits(Rb::Vector{Float64}, Eb::Vector{Float64})
    Rx = zeros(Float64, length(STATES))
    Ex = zeros(Float64, length(STATES))
    for s in STATES
        bits = bitvec(s); I = findall(identity, bits)
        if isempty(I); Rx[IDX[s]]=0.0; Ex[IDX[s]]=0.0
        else           Rx[IDX[s]]=mean(Rb[I]); Ex[IDX[s]]=mean(Eb[I]) end
    end
    return Rx, Ex
end
Rx, Ex = node_attributes_from_bits(Rb, Eb)

# Optional anisotropy diagnostics
A_node = node_anisotropy_curvature(Rx)
A_bit_curv = bit_anisotropy_curvature(Rx)

# Modal manifold (diamond)
Wd, Ld = diamond_adjacency_weights(A_b_real, Eb; γ=1.0)
λd, Vd = diamond_modes(Ld; k=7)

# Morse classification (on Ex)
gD = SimpleGraph(length(STATES))
for s in STATES, t in hamming1_neighbors(s); add_edge!(gD, IDX[s], IDX[t]); end
mins, sads, maxs = classify_morse_on_graph(gD, Ex)

# --------------------------- REPORTING ---------------------------
println("\n=== Bitwise geometry (real manifold) ===")
println("Bit |   R_b(curv)    |  A_b(SASA) |   E_b(Dir)   |  Anisotropy_bit(curv, 0-1)")
for b in 1:3
    @printf(" %d  |  % .6e  |  %8.3f  |  % .6f  |    %.4f\n", b, Rb[b], A_b_real[b], Eb[b], A_bit_curv[b])
end

println("\n=== Nodewise geometry on diamond (diagnostic aggregation) ===")
println("Node |   R(x) (avg bits)   |  A_node(curv, 0-1)  |   E_node (avg bits)")
for s in STATES
    i = IDX[s]
    @printf(" %-3s |  % .6e         |       %.4f          |  % .6f\n", s, Rx[i], A_node[i], Ex[i])
end

println("\nMorse minima: ", [STATES[i] for i in mins])
println("Morse saddles: ", [STATES[i] for i in sads])
println("Morse maxima: ", [STATES[i] for i in maxs])

println("\nDiamond Laplacian eigenvalues:")
println(round.(λd, digits=6))
if !isempty(λd)
    println("\nVibrational eigenmodes (normalized):")
    for m in 1:length(λd)
        vnorm = Vd[:,m] ./ (maximum(abs.(Vd[:,m])) + eps())
        @printf("Mode %d (λ=%.6f): %s\n", m, λd[m], string(round.(vnorm, digits=3)))
    end
end

# --------- Axiomatic curvature conservation (REAL & MODAL) -------
println("\n=== Axiomatic curvature conservation checks ===")
for (k, resi) in enumerate(cys_map)
    ρ_real = zeros(Float64, size(Lreal,1)); ρ_real[resi] = 1.0
    _ = check_conservation_real(Lreal, ρ_real; label="real: δ at CYS[$k]")
end
ρ_real_uniform = fill(1/size(Lreal,1), size(Lreal,1))
_ = check_conservation_real(Lreal, ρ_real_uniform; label="real: uniform over residues")

ρd_000 = zeros(Float64, length(STATES)); ρd_000[IDX["000"]] = 1.0
R_d_000, _ = check_conservation_modal(Ld, ρd_000; label="diamond: δ at 000")

ρd_uniform = fill(1/length(STATES), length(STATES))
_ = check_conservation_modal(Ld, ρd_uniform; label="diamond: uniform")

ρd_rand = rand(length(STATES)); ρd_rand ./= sum(ρd_rand)
_ = check_conservation_modal(Ld, ρd_rand; label="diamond: random")

println("\n— Functional (non-axiomatic) drift diagnostics —")
node_w_sasa = node_sasa_weights(A_b_real)
_ = functional_drift(R_d_000, node_w_sasa; label="diamond: δ(000) with per-node mean SASA")

# --------- Curvature transport: best move & min-action path ------
ρ0 = zeros(Float64, length(STATES)); ρ0[IDX["000"]] = 1.0
cosθ_map_static = Dict{Tuple{String,String},Float64}()  # static (no alignment)
best_t, best_score, φd, Rd, Lplus_d = recommend_next_move("000", ρ0, Wd, Ld; cosθ_map=cosθ_map_static)

path, total_cost = min_action_path("000", "111", Wd, Ld; cosθ_map=cosθ_map_static)
@printf("\nMinimum-action path 000 → 111 : %s  (total cost = %.6f)\n", join(path, " → "), total_cost)

# --------------- Tropical eigen + motif algebra ------------------
λ_trop, v_trop = tropical_eigen(Wd)
println("\nTropical eigenmode (min-plus): λ=", λ_trop, ", v=", round.(v_trop, digits=3))

gen_costs = Eb
motifs = generate_motifs(gen_costs; kmax=2)
println("\nMotif algebra (up to length 2):")
for m in motifs
    println("  ", motif_string(m), " : cost=", round(m.cost, digits=3))
end

# ===================== Maxwell-like modal laws (checks) =====================
function build_incidence(W::AbstractMatrix)
    n = size(W,1)
    edges = Vector{Tuple{Int,Int,Float64}}()
    for i in 1:n-1, j in i+1:n
        wij = W[i,j]
        if wij > 0
            push!(edges, (i,j,wij))  # oriented i→j
        end
    end
    m = length(edges)
    B = zeros(Float64, n, m)
    ce = zeros(Float64, m)
    for (e,(i,j,w)) in enumerate(edges)
        B[i,e] =  1.0; B[j,e] = -1.0; ce[e] = w
    end
    return B, Diagonal(ce), edges
end

function gauss_curvature(L::AbstractMatrix, ρ::AbstractVector; label::String="")
    R = L * ρ
    s = sum(R)
    println(@sprintf("Gauss (ΣR=0)%s: sum R = %.3e  ->  %s",
                     isempty(label) ? "" : " ["*label*"]",
                     s, abs(s) ≤ 1e-8 ? "OK" : "FAIL"))
    return R
end

function transport_closure(L::AbstractMatrix, B::AbstractMatrix, Ce::Diagonal,
                           ρ₀::AbstractVector, ρ₁::AbstractVector; label::String="")
    Δρ = ρ₁ - ρ₀
    ΔR = L * Δρ
    S  = - Ce * (B' * Δρ)
    res = ΔR + B * S
    println(@sprintf("Transport closure (ΔR + div S = 0)%s: ‖res‖₂ = %.3e  ->  %s",
                     isempty(label) ? "" : " ["*label*"]",
                     norm(res), norm(res) ≤ 1e-8 ? "OK" : "FAIL"))
    return ΔR, S, res
end

B, Ce, edges = build_incidence(Wd)
n = length(STATES)
ρ_000 = zeros(Float64, n); ρ_000[IDX["000"]] = 1.0
ρ_100 = zeros(Float64, n); ρ_100[IDX["100"]] = 1.0
ρ_010 = zeros(Float64, n); ρ_010[IDX["010"]] = 1.0
ρ_001 = zeros(Float64, n); ρ_001[IDX["001"]] = 1.0

println("\n=== Maxwell-like modal checks (single-molecule curvature transport) ===")
_ = gauss_curvature(Ld, ρ_000; label="ρ = δ(000)")
_ = gauss_curvature(Ld, ρ_100; label="ρ = δ(100)")
_ = gauss_curvature(Ld, ρ_010; label="ρ = δ(010)")
_ = gauss_curvature(Ld, ρ_001; label="ρ = δ(001)")
_ = transport_closure(Ld, B, Ce, ρ_000, ρ_100; label="δ(000) → δ(100)")
_ = transport_closure(Ld, B, Ce, ρ_000, ρ_010; label="δ(000) → δ(010)")
_ = transport_closure(Ld, B, Ce, ρ_000, ρ_001; label="δ(000) → δ(001)")

ρ_110 = zeros(Float64, n); ρ_110[IDX["110"]] = 1.0
ρ_111 = zeros(Float64, n); ρ_111[IDX["111"]] = 1.0
_ = transport_closure(Ld, B, Ce, ρ_000, ρ_010; label="step 1: 000→010")
_ = transport_closure(Ld, B, Ce, ρ_010, ρ_110; label="step 2: 010→110")
_ = transport_closure(Ld, B, Ce, ρ_110, ρ_111; label="step 3: 110→111")

# ===================== Vibrational alignment demo ===========================
# Crystal information for H2O2 (P 41 21 2, No. 92): take principal axis along c
u_H2O2 = SVector(0.0, 0.0, 1.0)   # tetragonal c-axis direction
shift   = 3.0                      # Å, displacement of forcing toward approach
σfield  = 2.0                      # Å, spread of forcing

κ, g_raw, Eext = alignment_factors_for_bits(Wreal, Lreal, coords, cys_map;
                                            u=u_H2O2, shift=shift, σ=σfield)
println("\n=== Vibrational alignment (curvature→Dirichlet, no QM) ===")
for b in 1:length(cys_map)
    @printf(" Bit %d (Cys %d): κ=%.3f,  g_raw=%.4e,  E_ext=%.6f\n", b, cys_map[b], κ[b], g_raw[b], Eext[b])
end

# Gate transitions tropically by using κ_b as cosθ for each bit flip
cosθ_map_dyn = build_cosθ_map_from_kappa(κ)

println("\n— Dynamic (alignment-gated) recommendation from 000 —")
_ = recommend_next_move("000", ρ0, Wd, Ld; cosθ_map=cosθ_map_dyn)

println("\n— Dynamic min-action path 000→111 (gated) —")
path_dyn, total_cost_dyn = min_action_path("000", "111", Wd, Ld; cosθ_map=cosθ_map_dyn)
@printf("  Path: %s  (total cost = %.6f)\n", join(path_dyn, " → "), total_cost_dyn)
