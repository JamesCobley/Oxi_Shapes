# =============================================================================
# GeoBrain: Real-Flow & Imagination Pipeline
# =============================================================================
using LinearAlgebra
using SparseArrays
using GeometryBasics
using Graphs
using StatsBase
using Statistics: mean
using Random
using UUIDs
using BSON: @save, @load
using Dates

# =============================================================================
# Geomtric object 1: The GeoNode = real and imagined Oxi-shape
# =============================================================================
struct GeoGraphReal
    n::Int
    flat_x::Vector{Float32}
    flat_y::Vector{Float32}
    neighbors::Vector{Vector{Int}}
    d0::Vector{Vector{Float32}}
    edges_idx::Vector{Tuple{Int,Int}}
    adjacency::Matrix{Float32}
    points3D::Vector{Point3{Float32}}
    R_vals::Vector{Float32}
    anisotropy::Vector{Float32}
end

function GeoGraphReal(pf_states::Vector{String},
                      flat_pos::Dict{String,Tuple{Float64,Float64}},
                      edges::Vector{Tuple{String,String}})
    n = length(pf_states)
    idx_map = Dict(s=>i for (i,s) in enumerate(pf_states))
    fx = Float32[flat_pos[s][1] for s in pf_states]
    fy = Float32[flat_pos[s][2] for s in pf_states]
    eidx = [(idx_map[u], idx_map[v]) for (u,v) in edges]
    nbrs = [Int[] for _ in 1:n]
    for (i,j) in eidx
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end
    d0 = [Float32[sqrt((fx[i]-fx[j])^2 + (fy[i]-fy[j])^2) for j in nbrs[i]] for i in 1:n]
    g = SimpleGraph(n)
    for (i,j) in eidx
        add_edge!(g,i,j)
    end
    A = Float32.(adjacency_matrix(g))
    pts3D = Vector{Point3{Float32}}(undef, n)
    Rbuf  = zeros(Float32, n)
    anis  = zeros(Float32, n)
    return GeoGraphReal(n, fx, fy, nbrs, d0, eidx, A, pts3D, Rbuf, anis)
end

function lift_to_z_plane(rho::Vector{Float32}, pf_states, flat_pos)
    return [Point3(Float32(flat_pos[s][1]), Float32(flat_pos[s][2]), -rho[i]) for (i, s) in enumerate(pf_states)]
end

function build_neighbor_indices(pf_states::Vector{String}, edges::Vector{Tuple{String, String}}, idx::Dict{String, Int})
    return Dict(s => [idx[v] for (u, v) in edges if u == s] ∪ [idx[u] for (u, v) in edges if v == s] for s in pf_states)
end

function compute_anisotropy_from_curvature(
    R_vals::Vector{Float32},
    pf_states::Vector{String},
    flat_pos::Dict{String, Tuple{Float64, Float64}},
    edges::Vector{Tuple{String, String}}
)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = build_neighbor_indices(pf_states, edges, idx)

    anisotropy = map(pf_states) do s
        i = idx[s]
        grads = map(neighbor_indices[s]) do j
            dx = Float32(flat_pos[s][1] - flat_pos[pf_states[j]][1])
            dy = Float32(flat_pos[s][2] - flat_pos[pf_states[j]][2])
            dist = sqrt(dx^2 + dy^2)
            dist > 1f-6 ? abs(R_vals[i] - R_vals[j]) / dist : 0.0f0
        end
        grads_nonzero = filter(x -> x > 0, grads)
        isempty(grads_nonzero) ? 0.0f0 : mean(grads_nonzero)
    end

    return collect(anisotropy)
end

function initialize_sheaf_stalks(flat_pos, pf_states)
    Dict(s => [Float32(flat_pos[s][1]), Float32(flat_pos[s][2])] for s in pf_states)
end

function sheaf_consistency(stalks, edges; threshold=2.5f0)
    [(u, v, norm(stalks[u] .- stalks[v])) for (u, v) in edges if norm(stalks[u] .- stalks[v]) > threshold]
end
function update_real_geometry!(G::GeoGraphReal, ρ::Vector{Float32}; ε::Float32=1e-3)
    violated = Int[]

    # Step 1: z-lift from ρ to 3D points
    @inbounds for i in 1:G.n
        G.points3D[i] = Point3(G.flat_x[i], G.flat_y[i], -ρ[i])
        G.R_vals[i] = 0f0
    end

    # Step 2: Compute scalar curvature R(x)
    @inbounds for i in 1:G.n
        pi = G.points3D[i]
        for (k,j) in enumerate(G.neighbors[i])
            d3 = norm(pi - G.points3D[j])
            G.R_vals[i] += d3 - G.d0[i][k]
        end
    end

    # Step 3: Compute anisotropy A(x)
    @inbounds for i in 1:G.n
        acc, cnt = 0f0, 0
        Ri = G.R_vals[i]
        for (k,j) in enumerate(G.neighbors[i])
            dist = G.d0[i][k]
            ΔR = abs(Ri - G.R_vals[j])
            if dist > 1e-6f0
                acc += ΔR / dist
                cnt += 1
            end
        end
        G.anisotropy[i] = cnt > 0 ? acc / cnt : 0f0
    end

    # Step 4: Sheath integrity & volume check
    for i in 1:G.n
        # Local volume (ρ at self + neighbors)
        vol = ρ[i] + sum(ρ[j] for j in G.neighbors[i])
        expected_vol = (1 + length(G.neighbors[i])) / G.n
        vol_ok = abs(vol - expected_vol) ≤ ε

        # Sheath logic (curvature must be supported by anisotropy)
        shape_ok = abs(G.R_vals[i]) ≤ ε * (1.0f0 + G.anisotropy[i])

        if !(vol_ok && shape_ok)
            push!(violated, i)
        end
    end

    return violated
end

function update_imagined_geometry!(G::GeoGraphReal, ρ_imag::Vector{Float32}; ε::Float32=1e-3)
    violated = Int[]

    # Step 1: z-lift imagined ρ to 3D points
    @inbounds for i in 1:G.n
        G.points3D[i] = Point3(G.flat_x[i], G.flat_y[i], -ρ_imag[i])
        G.R_vals[i] = 0f0
    end

    # Step 2: Compute curvature for imagined field
    @inbounds for i in 1:G.n
        pi = G.points3D[i]
        for (k,j) in enumerate(G.neighbors[i])
            d3 = norm(pi - G.points3D[j])
            G.R_vals[i] += d3 - G.d0[i][k]
        end
    end

    # Step 3: Compute anisotropy for imagined field
    @inbounds for i in 1:G.n
        acc, cnt = 0f0, 0
        Ri = G.R_vals[i]
        for (k,j) in enumerate(G.neighbors[i])
            dist = G.d0[i][k]
            ΔR = abs(Ri - G.R_vals[j])
            if dist > 1e-6f0
                acc += ΔR / dist
                cnt += 1
            end
        end
        G.anisotropy[i] = cnt > 0 ? acc / cnt : 0f0
    end

    # Step 4: Sheath stress check (optional during training)
    # (Optional: compute sheath stress on ρ_imag)

    return nothing
end

function update_imagined_geometry!(G::GeoGraphReal, ρ_imag::Vector{Float32}; ε::Float32=1e-3)
    violated = Int[]

    # Step 1: z-lift imagined ρ to 3D points
    @inbounds for i in 1:G.n
        G.points3D[i] = Point3(G.flat_x[i], G.flat_y[i], -ρ_imag[i])
        G.R_vals[i] = 0f0
    end

    # Step 2: Compute curvature for imagined field
    @inbounds for i in 1:G.n
        pi = G.points3D[i]
        for (k,j) in enumerate(G.neighbors[i])
            d3 = norm(pi - G.points3D[j])
            G.R_vals[i] += d3 - G.d0[i][k]
        end
    end

    # Step 3: Compute anisotropy for imagined field
    @inbounds for i in 1:G.n
        acc, cnt = 0f0, 0
        Ri = G.R_vals[i]
        for (k,j) in enumerate(G.neighbors[i])
            dist = G.d0[i][k]
            ΔR = abs(Ri - G.R_vals[j])
            if dist > 1e-6f0
                acc += ΔR / dist
                cnt += 1
            end
        end
        G.anisotropy[i] = cnt > 0 ? acc / cnt : 0f0
    end

    # Step 4: Sheath stress check (optional during training)
    # (Optional: compute sheath stress on ρ_imag)

    return nothing
end

struct GeoNode
    ρ_real::Vector{Float32}
    R_real::Vector{Float32}
    A_real::Vector{Float32}
    ρ_imag::Vector{Float32}
    R_imag::Vector{Float32}
    A_imag::Vector{Float32}
    λ::Float32
    sheath_stress::Vector{Float32}  # from imag field
    flux::Vector{Float32}
    action_cost::Float32
end

# =============================================================================
# Module 2: Simplex Construction and Flow
# =============================================================================

# =============================================================================
# 1. Real-Flow (Oxi-Shapes Alive)
# =============================================================================

"""
    init_alive_buffers!(G, bitcounts)

Pre-allocate counts, inflow, outflow, and degeneracy penalties.
"""
function init_alive_buffers!(G::GeoGraphReal, bitcounts::Vector{Int})
    n = G.n
    counts = Vector{Int}(undef, n)
    inflow_int = zeros(Int, n)
    outflow_int = zeros(Int, n)
    R_total = length(bitcounts)
    binom = Dict(k => binomial(R_total, k) for k in 0:R_total)
    deg_pen = Float32[1f0/binom[bitcounts[i]] for i in 1:n]
    return (counts=counts, inflow_int=inflow_int, outflow_int=outflow_int, deg_pen=deg_pen)
end

"""
    oxi_shapes_alive!(ρ, G, buffers; max_moves)

Discrete-count stochastic flow; updates ρ in-place according to real flow rules.
"""
function oxi_shapes_alive!(ρ::Vector{Float32}, G::GeoGraphReal, buffers; max_moves::Int=10)
    n = G.n; counts = buffers.counts
    @inbounds for i in 1:n counts[i] = round(Int, ρ[i]*100) end
    counts[n] = 100 - sum(counts[1:n-1])

    update_real_geometry!(G, ρ)

    fill!(buffers.inflow_int, 0); fill!(buffers.outflow_int, 0)
    total_moves = rand(0:max_moves)
    nonzero = findall(>(0), counts)

    for _ in 1:total_moves
        isempty(nonzero) && break
        i = rand(nonzero); nbrs = G.neighbors[i]
        isempty(nbrs) && continue

        wsum = 0f0; ws = Float32[]; push!(ws, 0f0)
        for j in nbrs
            ΔS = 0.1f0 + 0.01f0 + abs(G.R_vals[j]) + buffers.deg_pen[j]
            Δf = exp(counts[i]/100) - G.R_vals[i] + ΔS
            w = exp(-Δf) * exp(-G.anisotropy[j]); wsum += w; push!(ws, w)
        end
        wsum < 1e-8f0 && continue

        r = rand() * wsum; cum = 0f0; chosen = nbrs[1]
        for (k, j) in enumerate(nbrs)
            cum += ws[k+1]
            if cum >= r
                chosen = j
                break
            end
        end

        buffers.inflow_int[chosen] += 1
        buffers.outflow_int[i] += 1

        if counts[i] - buffers.outflow_int[i] == 0
            deleteat!(nonzero, findfirst(==(i), nonzero))
        end
        if counts[chosen] + buffers.inflow_int[chosen] == 1
            push!(nonzero, chosen)
        end
    end

    @inbounds for i in 1:n
        counts[i] += buffers.inflow_int[i] - buffers.outflow_int[i]
        ρ[i] = counts[i] / 100f0
    end

    return ρ
end

# =============================================================================
# 2. Imagination Pipeline
# =============================================================================

struct LivingSimplexTensor
    real::Vector{Float32}
    imag::Vector{Float32}
end

init_living_simplex_tensor(ρ0::Vector{Float32}) = 
    LivingSimplexTensor(copy(ρ0), copy(ρ0))

build_imagined_manifold!(field::LivingSimplexTensor, G::GeoGraphReal) = 
    update_real_geometry!(G, field.imag)

function evolve_imagination_single_counts!(field::LivingSimplexTensor, G::GeoGraphReal, buffers; moves::Int=10)
    counts = buffers.counts
    @inbounds for i in 1:G.n counts[i] = round(Int, field.imag[i]*100) end
    counts[G.n] = 100 - sum(counts[1:G.n-1])

    update_real_geometry!(G, field.imag)

    fill!(buffers.inflow_int, 0); fill!(buffers.outflow_int, 0)
    nmoves = rand(0:moves)
    nonzero = findall(>(0), counts)

    for _ in 1:nmoves
        isempty(nonzero) && break
        i = nonzero[argmax(G.R_vals[nonzero])]; nbrs = G.neighbors[i]
        isempty(nbrs) && continue

        best_j, bc = nbrs[1], Inf
        for j in nbrs
            ΔS = 0.1f0 + 0.01f0 + abs(G.R_vals[j]) + buffers.deg_pen[j]
            Δf = exp(counts[i]/100) - G.R_vals[i] + ΔS
            if Δf < bc
                bc, best_j = Δf, j
            end
        end

        buffers.inflow_int[best_j] += 1
        buffers.outflow_int[i] += 1

        if counts[i] - buffers.outflow_int[i] == 0
            deleteat!(nonzero, findfirst(==(i), nonzero))
        end
        if counts[best_j] + buffers.inflow_int[best_j] == 1
            push!(nonzero, best_j)
        end
    end

    @inbounds for i in 1:G.n
        counts[i] += buffers.inflow_int[i] - buffers.outflow_int[i]
        field.imag[i] = counts[i] / 100f0
    end

    return field.imag
end

compute_lambda(ρ_real::Vector{Float32}, ρ_imag::Vector{Float32}) = 
    mean(abs.(ρ_imag .- ρ_real))

function step_imagination!(field::LivingSimplexTensor, G::GeoGraphReal, buffers; max_moves::Int=10)
    build_imagined_manifold!(field, G)
    evolve_imagination_single_counts!(field, G, buffers; moves=max_moves)
    return compute_lambda(field.real, field.imag)
end

# =============================================================================
# 3. Simplex Tensor (λ-Surface) Construction
# =============================================================================

"""
    build_simplex_surface(simplex::Vector{Vector{GeoNode}}) → Matrix{Float32}

Given the simplex tensor (list of GeoNode sequences per run),
builds the λ-surface where each point is -λ.
"""
function build_simplex_surface(simplex::Vector{Vector{GeoNode}})
    n_runs = length(simplex)
    rollout_steps = maximum(length(run) for run in simplex)
    λ_surface = fill(0f0, rollout_steps, n_runs)

    for (r, run) in enumerate(simplex)
        for (t, node) in enumerate(run)
            λ_surface[t, r] = -node.λ
        end
    end

    return λ_surface
end

"""
    compute_simplex_dirichlet_energy(lambda_surface::Matrix{Float32}) → Float32

Computes the total Dirichlet energy of the simplex λ-surface.
"""
function compute_simplex_dirichlet_energy(lambda_surface::Matrix{Float32})
    T, R = size(lambda_surface)
    energy = 0.0f0

    for t in 1:T
        for r in 1:R
            if t < T
                energy += (lambda_surface[t, r] - lambda_surface[t+1, r])^2
            end
            if r < R
                energy += (lambda_surface[t, r] - lambda_surface[t, r+1])^2
            end
        end
    end

    return energy
end

"""
    build_simplex_laplacian(lambda_surface::Matrix{Float32}) → Matrix{Float32}

Builds the graph Laplacian over the Simplex surface grid.
"""
function build_simplex_laplacian(lambda_surface::Matrix{Float32})
    T, R = size(lambda_surface)
    n = T * R
    W = zeros(Float32, n, n)

    for t in 1:T
        for r in 1:R
            idx = (t-1)*R + r
            if t < T
                W[idx, idx+R] = 1.0f0
                W[idx+R, idx] = 1.0f0
            end
            if r < R
                W[idx, idx+1] = 1.0f0
                W[idx+1, idx] = 1.0f0
            end
        end
    end

    D = Diagonal(sum(W, dims=2))
    L = D - W

    return L
end

# =============================================================================
# Flow Helper
# =============================================================================

# --- Geodesic Definitions
const geodesics = [
    ["000", "100", "101", "111"],
    ["000", "100", "110", "111"],
    ["000", "010", "110", "111"],
    ["000", "010", "011", "111"],
    ["000", "001", "101", "111"],
    ["000", "001", "011", "111"]
]

# --- Geodesic Classifier
function dominant_geodesic(trajectory::Vector{String}, geodesics::Vector{Vector{String}})
    best_path, max_score = nothing, 0
    for path in geodesics
        score = count(s -> s in trajectory, path)
        if score > max_score
            max_score = score
            best_path = path
        end
    end
    return best_path
end

# --- K-Distribution Computation
function compute_k_distribution(ρ::Vector{Float32}, pf_states::Vector{String})
    k_counts = Dict{Int, Float32}()
    for (i, s) in enumerate(pf_states)
        k = count(==('1'), s)
        k_counts[k] = get(k_counts, k, 0.0f0) + ρ[i]
    end
    return k_counts
end

# --- Weighted Mean Oxidation
function weighted_mean_oxidation(k_dist::Dict{Int, Float32})
    return sum(Float32(k) * v for (k, v) in k_dist)
end

# --- Shannon Entropy
function shannon_entropy(p::Vector{Float32})
    p_nonzero = filter(x -> x > 0f0, p)
    return -sum(x -> x * log2(x), p_nonzero)
end

# --- Fisher Information
function fisher_information(ρ::Vector{Float32})
    grad = diff(ρ)
    return sum(grad .^ 2)
end

# --- Lyapunov Exponent
function lyapunov_exponent(series::Vector{Vector{Float32}})
    exponents = Float32[]
    for t in 2:length(series)
        norm_prev = norm(series[t-1])
        norm_diff = norm(series[t] .- series[t-1])
        if norm_prev > 0f0 && norm_diff > 0f0
            push!(exponents, log(norm_diff / norm_prev))
        end
    end
    return isempty(exponents) ? 0f0 : mean(exponents)
end

# --- Transition Classification
function classify_transition(prev::Vector{Float32}, curr::Vector{Float32}, pf_states::Vector{String})
    k_prev = sum(Float32(count(==('1'), pf_states[i])) * prev[i] for i in 1:length(prev)) / 3f0
    k_curr = sum(Float32(count(==('1'), pf_states[i])) * curr[i] for i in 1:length(curr)) / 3f0

    if k_curr > k_prev + 1f-4
        return :oxidizing
    elseif k_curr < k_prev - 1f-4
        return :reducing
    else
        return :neutral
    end
end

# --- Trace Analysis
struct Trace
    steps::Vector{String}
end

function transitions(trace::Trace)
    return [(trace.steps[i], trace.steps[i+1]) for i in 1:length(trace.steps)-1]
end

function compress(trace::Trace)
    compressed = String[]
    for step in trace.steps
        if length(compressed) ≥ 2 && step == compressed[end-1]
            pop!(compressed)
        else
            push!(compressed, step)
        end
    end
    return compressed
end

function recurrence_patterns(trace::Trace; window::Int=3)
    patterns = Dict{Vector{String}, Int}()
    for i in 1:(length(trace.steps) - window + 1)
        pat = trace.steps[i:i+window-1]
        patterns[pat] = get(patterns, pat, 0) + 1
    end
    return Dict(k => v for (k, v) in patterns if v > 1)
end

function delta_commutator(trace::Trace)
    ts = transitions(trace)
    return length(unique(ts)) / max(length(ts), 1)
end

function trace_analysis(trace_steps::Vector{String})
    trace = Trace(trace_steps)
    return (
        compressed = compress(trace),
        patterns = recurrence_patterns(trace),
        delta = delta_commutator(trace)
    )
end

# --- FlowTrace Structure
struct FlowTrace
    run_id::String
    ρ_series::Vector{Vector{Float32}}
    flux_series::Vector{Vector{Float32}}
    R_series::Vector{Vector{Float32}}
    k_states::Vector{Dict{Int, Float32}}
    mean_oxidation_series::Vector{Float32}
    shannon_entropy_series::Vector{Float32}
    fisher_info_series::Vector{Float32}
    transition_classes::Vector{Symbol}
    on_geodesic_flags::Vector{Bool}
    geodesic_path::Vector{String}
    lyapunov::Float32
    action_cost::Float32
end

# --- Record Flow Trace
function record_flow_trace!(
    ρ0::Vector{Float32}, T::Int, pf_states::Vector{String}, flat_pos::Dict{String, Tuple{Float64, Float64}}, edges::Vector{Tuple{String, String}};
    max_moves_per_step=10, run_id::String="default_run"
)
    ρ = copy(ρ0)
    ρ_series = [copy(ρ)]
    flux_series = Vector{Vector{Float32}}()
    R_series = Vector{Vector{Float32}}()
    k_states = Vector{Dict{Int, Float32}}()
    mean_oxidation_series = Float32[]
    shannon_entropy_series = Float32[]
    fisher_info_series = Float32[]
    transition_classes = Symbol[]
    on_geodesic_flags = Bool[]
    trajectory = String[]

    ox_moves = 0
    red_moves = 0
    total_moves = 0
    cumulative_entropy = 0.0f0

    # Initialize GeoGraphReal
    G = GeoGraphReal(pf_states, flat_pos, edges)
    bitcounts = [count(==('1'), s) for s in pf_states]
    buffers = init_alive_buffers!(G, bitcounts)

    for t in 1:T
        ρ_old = copy(ρ)
        oxi_shapes_alive!(ρ, G, buffers; max_moves=max_moves_per_step)

        push!(ρ_series, copy(ρ))
        curr_state = pf_states[argmax(ρ)]
        push!(trajectory, curr_state)

        flux = ρ .- ρ_old
        push!(flux_series, copy(flux))

        k_dist = compute_k_distribution(ρ, pf_states)
        push!(k_states, deepcopy(k_dist))

        mean_ox = Float32(weighted_mean_oxidation(k_dist))
        entropy = Float32(shannon_entropy(ρ))
        fisher_info = Float32(fisher_information(ρ))
        cumulative_entropy += entropy

        push!(mean_oxidation_series, mean_ox)
        push!(shannon_entropy_series, entropy)
        push!(fisher_info_series, fisher_info)

        update_real_geometry!(G, ρ)
        push!(R_series, copy(G.R_vals))

        class = classify_transition(ρ_old, ρ, pf_states)
        push!(transition_classes, class)
        class == :oxidizing && (ox_moves += 1)
        class == :reducing && (red_moves += 1)
        total_moves += 1
    end

    geo_path = dominant_geodesic(trajectory, geodesics)
    for state in trajectory
        push!(on_geodesic_flags, state in geo_path)
    end

    lyap = Float32(lyapunov_exponent(ρ_series))
    action_cost = cumulative_entropy / Float32(total_moves)

    trace_meta = trace_analysis(trajectory)

    return FlowTrace(
        run_id,
        ρ_series,
        flux_series,
        R_series,
        k_states,
        mean_oxidation_series,
        shannon_entropy_series,
        fisher_info_series,
        transition_classes,
        on_geodesic_flags,
        geo_path,
        lyap,
        action_cost
    ), trace_meta
end

# =============================================================================
# Geometric object 3: The Hypergraph learning manifold
# =============================================================================

struct HypergraphBrain
    edge_sets::Dict{Symbol, Vector{Vector{Int}}}
    weights::Dict{Symbol, Vector{Float32}}
    φ::Vector{Float32}
    Ψ::Vector{Float32}
    λ::Vector{Float32}
    L::SparseMatrixCSC{Float32, Int}
    modes::Matrix{Float32}
    trace_patterns::Vector{Dict{Vector{String}, Int}}
    delta_commutators::Vector{Float32}
end

function build_hypergraph_from_simplex(simplex::Vector{Vector{GeoNode}},
                                       trace_metadata::Vector{Dict{Symbol, Any}};
                                       curvature_bins=5,
                                       cost_bins=5,
                                       spectral_bins=5,
                                       mode_rank=10,
                                       α=1.0f0,
                                       β=1.0f0)

    # 1. Flatten all nodes
    nodes = reduce(vcat, simplex)
    N = length(nodes)

    # 2. Build edge sets
    edge_sets = Dict{Symbol, Vector{Vector{Int}}}()

    # Action-cost hyperedges
    costs = [n.action_cost for n in nodes]
    bins_cost = StatsBase.cut(costs, cost_bins)
    edge_sets[:action_cost] = group_by_bins(bins_cost)

    # Curvature hyperedges
    curvs = [mean(n.R_real) for n in nodes]
    bins_curv = StatsBase.cut(curvs, curvature_bins)
    edge_sets[:curvature] = group_by_bins(bins_curv)

    # Delta-commutator clustering
    deltas = [m[:delta] for m in trace_metadata]
    bins_delta = StatsBase.cut(deltas, spectral_bins)
    edge_sets[:delta_commutator] = group_by_bins(bins_delta)

    # Recurrence pattern tags (symbolic edge types)
    trace_patterns = [m[:patterns] for m in trace_metadata]

    # 3. Fields
    λ = [n.λ for n in nodes]
    local_energy = [sum(abs.(n.R_real)) for n in nodes]
    local_entropy = [sum(abs.(n.flux)) for n in nodes]
    Ψ = [α * e + β * s for (e, s) in zip(local_energy, local_entropy)]
    φ = zeros(Float32, N)

    # 4. Build Laplacian and spectrum
    λ_surface = build_simplex_surface(simplex)
    L = build_simplex_laplacian(λ_surface)
    evals, evecs = eigen(Matrix(L))
    modes = evecs[:, 1:min(mode_rank, size(evecs, 2))]

    # 5. Spectral edge bins
    add_spectral_binning!(edge_sets, modes, spectral_bins)

    # 6. Weights
    weights = Dict{Symbol, Vector{Float32}}()
    for k in keys(edge_sets)
        weights[k] = ones(Float32, length(edge_sets[k]))
    end

    return HypergraphBrain(
        edge_sets,
        weights,
        φ,
        Ψ,
        λ,
        sparse(L),
        modes,
        trace_patterns,
        deltas
    )
end

function group_by_bins(binning::Vector{Int})
    groups = Dict{Int, Vector{Int}}()
    for (i, b) in enumerate(binning)
        push!(get!(groups, b, Int[]), i)
    end
    return collect(values(groups))
end

function add_spectral_binning!(edge_sets::Dict{Symbol, Vector{Vector{Int}}},
                               modes::Matrix{Float32},
                               spectral_bins::Int)
    projections = eachrow(modes)
    clusters = Dict{Int, Vector{Int}}()
    for (i, vec) in enumerate(projections)
        key = round(Int, sum(vec) * spectral_bins)
        push!(get!(clusters, key, Int[]), i)
    end
    edge_sets[:spectral] = collect(values(clusters))
end



