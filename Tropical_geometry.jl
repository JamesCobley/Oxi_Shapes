############################
# Oxi-Shapes MGF – Full Pipeline (axiomatic + tropical + algebra)
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
    ρ0 = ρ .- mean(ρ)          # volume invariance
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

# -------------------- Aggregate bit → node (diagnostic only) ----
function node_attributes_from_bits(Rb::Vector{Float64}, Eb::Vector{Float64})
    Rx = zeros(Float64, length(STATES))
    Ex = zeros(Float64, length(STATES))
    for s in STATES
        bits = bitvec(s)
        I = findall(identity, bits)
        if isempty(I)
            Rx[IDX[s]] = 0.0; Ex[IDX[s]] = 0.0
        else
            Rx[IDX[s]] = mean(Rb[I]); Ex[IDX[s]] = mean(Eb[I])
        end
    end
    return Rx, Ex
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

# -------------------- Curvature, flows & paths (modal) ----------
# Curvature from ρ on diamond; returns φ, R, L^+
function modal_curvics(L::AbstractMatrix, ρ::AbstractVector; tol=1e-12)
    @assert abs(sum(ρ) - 1.0) < 1e-12 "ρ must sum to 1."
    ρ0 = ρ .- mean(ρ)
    Lplus = laplacian_pinv(L; tol=tol)
    φ = Lplus * ρ0; φ .-= mean(φ)
    R = L * φ
    return φ, R, Lplus
end

# Edge flow j_{x->y} = w_xy * (φ[y] - φ[x])
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

# ------------------------------ RUN -----------------------------
# Adjust this path as needed
pdb = "/content/AF-P04406-F1-model_v4.pdb"
coords, cys_idx_all = load_ca_and_cys(pdb)
println("Parsed ", length(coords), " CA residues; CYS CA at indices: ", cys_idx_all)

# Bit mapping & SASA-like exposures (functional weights)
cys_map  = [152,156,247]
A_b_real = [25.0, 0.0, 0.47]

# Real manifold
_, Lreal = build_weighted_graph(coords; cutoff=6.5, σ=4.0)
Rb, Eb = per_bit_geometry(Lreal, cys_map, length(coords))   # bitwise curvature & Dirichlet on real graph
Rx, Ex = node_attributes_from_bits(Rb, Eb)                  # diagnostic aggregation

# Optional anisotropy from diagnostic Rx
A_node = node_anisotropy_curvature(Rx)
A_bit_curv = bit_anisotropy_curvature(Rx)

# Modal manifold (diamond)
Wd, Ld = diamond_adjacency_weights(A_b_real, Eb; γ=1.0)
λd, Vd = diamond_modes(Ld; k=7)

# Morse classification on energy landscape Ex
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
# Real manifold: δ at each CYS & uniform
for (k, resi) in enumerate(cys_map)
    ρ_real = zeros(Float64, size(Lreal,1)); ρ_real[resi] = 1.0
    _ = check_conservation_real(Lreal, ρ_real; label="real: δ at CYS[$k]")
end
ρ_real_uniform = fill(1/size(Lreal,1), size(Lreal,1))
_ = check_conservation_real(Lreal, ρ_real_uniform; label="real: uniform over residues")

# Modal manifold: δ at nodes, uniform, random
ρd_000 = zeros(Float64, length(STATES)); ρd_000[IDX["000"]] = 1.0
R_d_000, _ = check_conservation_modal(Ld, ρd_000; label="diamond: δ at 000")

ρd_uniform = fill(1/length(STATES), length(STATES))
_ = check_conservation_modal(Ld, ρd_uniform; label="diamond: uniform")

ρd_rand = rand(length(STATES)); ρd_rand ./= sum(ρd_rand)
_ = check_conservation_modal(Ld, ρd_rand; label="diamond: random")

# Functional drift (clearly non-conserved): SASA weights per node (mean SASA of active bits)
println("\n— Functional (non-axiomatic) drift diagnostics —")
node_w_sasa = node_sasa_weights(A_b_real)
_ = functional_drift(R_d_000, node_w_sasa; label="diamond: δ(000) with per-node mean SASA")

# --------- Curvature transport: best move & min-action path ------
ρ0 = zeros(Float64, length(STATES)); ρ0[IDX["000"]] = 1.0
cosθ_map = Dict{Tuple{String,String},Float64}()  # optional overlaps in [0,1]
best_t, best_score, φd, Rd, Lplus_d = recommend_next_move("000", ρ0, Wd, Ld; cosθ_map=cosθ_map)

path, total_cost = min_action_path("000", "111", Wd, Ld; cosθ_map=cosθ_map)
@printf("\nMinimum-action path 000 → 111 : %s  (total cost = %.6f)\n", join(path, " → "), total_cost)

# --------------- Tropical eigen + motif algebra ------------------
λ_trop, v_trop = tropical_eigen(Wd)
println("\nTropical eigenmode (min-plus): λ=", λ_trop, ", v=", round.(v_trop, digits=3))

gen_costs = Eb                      # generator costs from per-bit Dirichlet energies
motifs = generate_motifs(gen_costs; kmax=2)
println("\nMotif algebra (up to length 2):")
for m in motifs
    println("  ", motif_string(m), " : cost=", round(m.cost, digits=3))
end
