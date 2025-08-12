# ============================================================================
# Extensions: Morse barriers + Ascending/Descending manifolds (graph-only)
# ============================================================================

using LinearAlgebra, Statistics
using Graphs
using Printf

# ---------- Hypercube setup (same as before) ----------
const pf_states = ["000","001","010","011","100","101","110","111"]
const idx = Dict(s=>i for (i,s) in enumerate(pf_states))
const pf_edges = [
  ("000","001"),("000","010"),("000","100"),
  ("001","011"),("001","101"),
  ("010","011"),("010","110"),
  ("100","110"),("100","101"),
  ("011","111"),("101","111"),("110","111")
]

function build_boolean_graph(pf_states, pf_edges)
    g = SimpleGraph(length(pf_states))
    for (u,v) in pf_edges
        add_edge!(g, idx[u], idx[v])
    end
    A = Matrix(adjacency_matrix(g))
    d = vec(sum(A, dims=2))
    L = Diagonal(d) - A
    return g, A, L, d
end

# ---------- Poisson, Dirichlet, Morse classification (as before) ----------
function solve_poisson_mean_zero(L::AbstractMatrix, ρ::AbstractVector)
    ρ0 = ρ .- mean(ρ)
    φ  = pinv(Matrix(L)) * ρ0
    φ .-= mean(φ)
    return φ
end
dirichlet_energy(L::AbstractMatrix, φ::AbstractVector) = 0.5 * (φ' * L * φ)

function classify_morse_nodes(g::SimpleGraph, φ::AbstractVector)
    minima = Int[]; maxima = Int[]; saddles = Int[]
    for i in 1:nv(g)
        vals = φ[neighbors(g, i)]
        if all(φ[i] .< vals)
            push!(minima, i)
        elseif all(φ[i] .> vals)
            push!(maxima, i)
        else
            push!(saddles, i)
        end
    end
    return minima, saddles, maxima
end

function edge_gradients(g::SimpleGraph, φ::AbstractVector)
    grads = Dict{Tuple{Int,Int},Float64}()
    for e in Graphs.edges(g)
        i, j = src(e), dst(e)
        grads[(i,j)] = φ[j] - φ[i]
        grads[(j,i)] = φ[i] - φ[j]
    end
    return grads
end

function vibrational_modes(L::AbstractMatrix; D::Union{Nothing,AbstractVector}=nothing, k::Int=7)
    if D === nothing
        F = eigen(Symmetric(Matrix(L))); λ = F.values; V = F.vectors
    else
        Dinvh = Diagonal(1 ./ sqrt.(D .+ eps()))
        Ltilde = Symmetric(Dinvh * Matrix(L) * Dinvh)
        F = eigen(Ltilde)
        λ = F.values
        V = Dinvh * F.vectors
    end
    ord = sortperm(λ)
    λs = λ[ord]; Vs = V[:, ord]
    keep = findall(x -> x > 1e-9, λs)
    if isempty(keep)
        return Float64[], Array{Float64}(undef, size(Vs,1), 0)
    end
    kkeep = first(keep):(min(lastindex(λs), first(keep)+k-1))
    return λs[kkeep], Vs[:, kkeep]
end

# ---------- NEW: Barrier heights (minimal maximum φ along any path) ----------
# For each sink t, run a Dijkstra-like search with node costs = φ,
# edge cost = max(φ(u), φ(v)); path cost = max(node φ on the path).
# This computes minimal-max potential (also called minimax path / bottleneck path).
function minimax_barrier_table(g::SimpleGraph, φ::AbstractVector)
    n = nv(g)
    # Precompute adjacency
    N = [neighbors(g, i) for i in 1:n]

    # For each source s, run minimax to all t
    barrier = fill(0.0, n, n)
    for s in 1:n
        # dist[i] = minimal achievable max φ along any s→i path
        dist = fill(Inf, n)
        dist[s] = φ[s]
        visited = falses(n)
        # simple O(n^2) label-setting since graph is small
        for _ in 1:n
            # pick unvisited with smallest dist
            u = argmin((visited[i] ? Inf : dist[i]) for i in 1:n)
            visited[u] = true
            for v in N[u]
                # bottleneck cost on edge
                cand = max(dist[u], φ[v])
                if cand < dist[v]
                    dist[v] = cand
                end
            end
        end
        # Fill row s
        barrier[s, :] = dist
    end
    return barrier
end

# ---------- NEW: Ascending / Descending manifolds ----------
# Discrete gradient flow: from a node, step to any neighbor with strictly lower φ (descending)
# or strictly higher φ (ascending). We collect basins by repeatedly descending to minima.
function descending_basins(g::SimpleGraph, φ::AbstractVector)
    n = nv(g)
    N = [neighbors(g, i) for i in 1:n]
    minima = Int[]
    for i in 1:n
        if all(φ[i] .< φ[N[i]])
            push!(minima, i)
        end
    end
    # map each node to a chosen minimum via steepest descent (break ties by index for determinism)
    basin_of = fill(0, n)
    for i in 1:n
        cur = i
        seen = Set{Int}()
        while true
            push!(seen, cur)
            # neighbors strictly lower
            lower = [v for v in N[cur] if φ[v] < φ[cur] - eps()]
            if isempty(lower)
                # at a local minimum (or flat sink)
                basin_of[i] = cur
                break
            else
                # choose the lowest φ (break ties by smallest index)
                bestφ = minimum(φ[lower])
                candidates = [v for v in lower if abs(φ[v]-bestφ) ≤ 1e-12]
                cur = minimum(candidates)
                if cur in seen; basin_of[i] = cur; break; end
            end
        end
    end
    # group nodes by basin
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        b = basin_of[i]
        groups[b] = get(groups, b, Int[]) ; push!(groups[b], i)
    end
    return groups
end

function ascending_basins(g::SimpleGraph, φ::AbstractVector)
    return descending_basins(g, -φ)  # reuse by flipping sign
end

# ---------- Driver with additions ----------
function graph_only_scan_with_morse(real_state::String="000"; generalized=false, k=7)
    @assert haskey(idx, real_state) "Unknown state: $real_state"
    g, A, L, d = build_boolean_graph(pf_states, pf_edges)

    ρ = zeros(length(pf_states)); ρ[idx[real_state]] = 1.0
    φ = solve_poisson_mean_zero(L, ρ)
    R = L * φ
    E_dir = dirichlet_energy(L, φ)

    @printf("Volume invariance: sum(ρ)=%.1f  |  sum(R)=%.3e  (OK if ~0)\n", sum(ρ), sum(R))
    @printf("Dirichlet energy:  E[φ] = %.6f\n", E_dir)

    mins, sads, maxs = classify_morse_nodes(g, φ)
    println("\nDiscrete Morse classification on φ:")
    println("  minima:  ", [pf_states[i] for i in mins])
    println("  saddles: ", [pf_states[i] for i in sads])
    println("  maxima:  ", [pf_states[i] for i in maxs])

    grads = edge_gradients(g, φ)
    println("\nEdge gradients g(i→j)=φ(j)-φ(i) (only first-order edges):")
    for (u,v) in pf_edges
        i, j = idx[u], idx[v]
        @printf("  %s ↔ %s : g(i→j)= %+ .6f | g(j→i)= %+ .6f\n", u, v, grads[(i,j)], grads[(j,i)])
    end

    # Vibrational eigenmodes
    if generalized
        λ, V = vibrational_modes(L; D=d, k=k)
        println("\nVibrational modes (generalized, L v = λ D v):")
    else
        λ, V = vibrational_modes(L; D=nothing, k=k)
        println("\nVibrational modes (standard, L v = λ v):")
    end
    for m in 1:length(λ)
        @printf("  mode %d : λ=%.6f  participation(real=%s)=%.4f\n",
                m, λ[m], real_state, V[idx[real_state], m]^2)
    end

    # ---- NEW: barrier table ----
    barrier = minimax_barrier_table(g, φ)
    println("\nMinimal barrier φ* (minimax) between states (row s → col t):")
    print("        "); for t in 1:length(pf_states); @printf("%6s", pf_states[t]); end; println()
    for s in 1:length(pf_states)
        @printf("%6s ", pf_states[s])
        for t in 1:length(pf_states)
            @printf("%6.3f", barrier[s,t])
        end
        println()
    end

    # ---- NEW: basins ----
    desc = descending_basins(g, φ)
    asc  = ascending_basins(g, φ)
    println("\nDescending basins (gradient ↓ to minima):")
    for (b, nodes) in sort(collect(desc); by=x->x[1])
        println("  basin at ", pf_states[b], " ← ", [pf_states[i] for i in sort(nodes)])
    end
    println("\nAscending basins (gradient ↑ to maxima):")
    for (b, nodes) in sort(collect(asc); by=x->x[1])
        println("  basin at ", pf_states[b], " → ", [pf_states[i] for i in sort(nodes)])
    end

    return (φ=φ, R=R, E_dir=E_dir, mins=mins, sads=sads, maxs=maxs, λ=λ, V=V,
            grads=grads, barrier=barrier, desc=desc, asc=asc)
end

# ---- Run ----
results2 = graph_only_scan_with_morse("000"; generalized=false, k=7)
