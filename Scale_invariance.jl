# =============================================================================
# Unified Oxi‑Shapes (R‑agnostic) – Modal topology + Simulation pipeline
# =============================================================================

using LinearAlgebra, SparseArrays, StaticArrays, GeometryBasics, Graphs
using StatsBase, Statistics: mean
using Random, UUIDs, BSON: @save, @load
using Dates

# =========================
# 1) R‑agnostic modal topology
# =========================

"All 2^R proteoform states as fixed-width binary strings."
function generate_states(R::Int)::Vector{String}
    @assert R ≥ 1
    [lpad(string(i, base=2), R, '0') for i in 0:(2^R - 1)]
end

"Edges for the R-hypercube: connect states that differ by exactly one bit."
function generate_edges(pf_states::Vector{String})::Vector{Tuple{String,String}}
    R = length(pf_states[1])
    stset = Set(pf_states)
    edges = Tuple{String,String}[]
    buf = Vector{UInt8}(undef, R)
    for s in pf_states
        @inbounds for i in 1:R
            buf[i] = UInt8(s[i])
        end
        @inbounds for i in 1:R
            old = buf[i]
            buf[i] = (old == UInt8('0')) ? UInt8('1') : UInt8('0')
            t = String(buf)
            if t in stset && s < t
                push!(edges, (s, t))
            end
            buf[i] = old
        end
    end
    return edges
end

"Layered 2D layout by Hamming weight (k levels) with centered x-positions."
function assign_positions(pf_states::Vector{String})::Dict{String,Tuple{Float64,Float64}}
    buckets = Dict{Int, Vector{String}}()
    for s in pf_states
        k = count(==('1'), s)
        push!(get!(buckets, k, String[]), s)
    end
    coords = Dict{String,Tuple{Float64,Float64}}()
    for k in sort!(collect(keys(buckets)))
        layer = sort!(buckets[k])
        xs = range(-0.5*(length(layer)-1), 0.5*(length(layer)-1), length=length(layer))
        for (j,s) in enumerate(layer)
            coords[s] = (float(xs[j]), float(k))
        end
    end
    return coords
end

"Convenience: return states, positions, edges for any R."
function build_modal_topology(R::Int)
    pf_states = generate_states(R)
    flat_pos  = assign_positions(pf_states)
    edges     = generate_edges(pf_states)
    return pf_states, flat_pos, edges
end

"Sample M canonical shortest geodesics from 0…0 to 1…1 by permuting bit-flip order."
function sample_geodesics(R::Int; M::Int=6, seed::Int=0)
    Random.seed!(seed)
    start = lpad("0", R, '0')
    goal  = lpad("1", R, '1')

    orders = Vector{Vector{Int}}()
    push!(orders, collect(1:R))
    push!(orders, collect(R:-1:1))
    while length(orders) < M
        ord = collect(1:R)
        shuffle!(ord)
        any(==(ord), orders) || push!(orders, ord)
    end

    geodesics = Vector{Vector{String}}()
    for ord in orders
        s = start
        path = String[s]
        buf = Vector{UInt8}(s)
        for i in ord
            buf[i] = UInt8('1')  # flip 0→1 along this coordinate
            push!(path, String(buf))
        end
        push!(geodesics, path)
    end
    return geodesics
end

# =========================
# 2) Geometry on the modal graph
# =========================

@kwdef struct GeoGraphReal
    pf_states::Vector{String}
    flat_pos::Dict{String, Tuple{Float64, Float64}}
    edges::Vector{Tuple{String, String}}

    n::Int
    flat_x::Vector{Float32}
    flat_y::Vector{Float32}
    neighbors::Vector{Vector{Int}}
    d0::Vector{Vector{Float32}}
    edges_idx::Vector{Tuple{Int, Int}}
    adjacency::Matrix{Float32}

    points3D::Vector{Point3{Float32}}
    R_vals::Vector{Float32}
    anisotropy::Vector{SVector{3, Float32}}  # a 3D vector per node
end

function GeoGraphReal(pf_states, flat_pos, edges)
    n = length(pf_states)
    idx_map = Dict(s => i for (i, s) in enumerate(pf_states))
    fx = Float32[flat_pos[s][1] for s in pf_states]
    fy = Float32[flat_pos[s][2] for s in pf_states]
    eidx = [(idx_map[u], idx_map[v]) for (u, v) in edges]
    nbrs = [Int[] for _ in 1:n]
    for (i, j) in eidx
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end
    d0 = [Float32[sqrt((fx[i] - fx[j])^2 + (fy[i] - fy[j])^2) for j in nbrs[i]] for i in 1:n]
    g = SimpleGraph(n)
    for (i, j) in eidx
        add_edge!(g, i, j)
    end
    A = Float32.(adjacency_matrix(g))
    pts3D = Vector{Point3{Float32}}(undef, n)
    Rbuf = zeros(Float32, n)
    anis = [SVector{3, Float32}(0f0, 0f0, 0f0) for _ in 1:n]

    return GeoGraphReal(pf_states, flat_pos, edges, n, fx, fy, nbrs, d0, eidx, A, pts3D, Rbuf, anis)
end

"Update 3D embedding (z = −ρ) and compute discrete curvature + anisotropy bundles."
function update_real_geometry!(G::GeoGraphReal, rho::Vector{Float32}; eps::Float32 = 1f-3)
    violated = Int[]

    @inbounds for i in 1:G.n
        G.points3D[i] = Point3(G.flat_x[i], G.flat_y[i], -rho[i])
        G.R_vals[i] = 0.0f0
    end

    @inbounds for i in 1:G.n
        pi = G.points3D[i]
        for (k, j) in enumerate(G.neighbors[i])
            d3 = norm(pi - G.points3D[j])
            G.R_vals[i] += d3 - G.d0[i][k]
        end
    end

    @inbounds for i in 1:G.n
        pi = SVector{3, Float32}(G.points3D[i])
        bundle = SVector{3, Float32}(0f0, 0f0, 0f0)
        for j in G.neighbors[i]
            pj = SVector{3, Float32}(G.points3D[j])
            ΔR = G.R_vals[i] - G.R_vals[j]
            dir = pi - pj
            nd = norm(dir)
            if nd > 1e-6
                bundle += (ΔR / (nd^2)) * dir
            end
        end
        n_neighbors = length(G.neighbors[i])
        G.anisotropy[i] = n_neighbors > 0 ? bundle / n_neighbors : bundle
    end

    for i in 1:G.n
        vol = rho[i] + sum(rho[j] for j in G.neighbors[i])
        expected_vol = (1 + length(G.neighbors[i])) / G.n
        vol_ok = abs(vol - expected_vol) ≤ eps
        shape_ok = abs(G.R_vals[i]) ≤ eps * (1.0f0 + norm(G.anisotropy[i]))
        if !(vol_ok && shape_ok)
            push!(violated, i)
        end
    end

    return violated
end

@kwdef struct GeoNode
    ρ_real::Vector{Float32}
    R_real::Vector{Float32}
    A_real::Vector{SVector{3, Float32}}
    sheath_stress::Vector{Float32}
    flux::Vector{Float32}
    action_cost::Float32
end

# =========================
# 3) Real‑flow (Oxi‑Shapes Alive)
# =========================

"Buffers derived from topology; includes degeneracy penalty by Hamming weight."
function init_alive_buffers!(G::GeoGraphReal, pf_states::Vector{String})
    R = length(pf_states[1])
    n = G.n
    counts = Vector{Int}(undef, n)
    inflow_int = zeros(Int, n)
    outflow_int = zeros(Int, n)
    # Binomial degeneracy penalty by layer k (1 / C(R,k))
    binom = Dict(k => binomial(R, k) for k in 0:R)
    bitcounts = [count(==('1'), s) for s in pf_states]
    deg_pen = Float32[1f0 / binom[bitcounts[i]] for i in 1:n]
    return (counts=counts, inflow_int=inflow_int, outflow_int=outflow_int, deg_pen=deg_pen)
end

"Single step of occupancy transport with geometric penalties."
function oxi_shapes_alive!(rho:Vector{Float32}, G::GeoGraphReal, buffers; max_moves::Int=10)
    n = G.n
    counts = buffers.counts

    @inbounds for i in 1:n
        counts[i] = round(Int, rho[i] * 100)
    end
    counts[n] = 100 - sum(counts[1:n-1])  # keep total mass = 100 (discrete)

    update_real_geometry!(G, rho)

    fill!(buffers.inflow_int, 0)
    fill!(buffers.outflow_int, 0)
    total_moves = rand(0:max_moves)

    nonzero = findall(>(0), counts)

    for _ in 1:total_moves
        isempty(nonzero) && break
        i = rand(nonzero)

        if counts[i] - buffers.outflow_int[i] ≤ 0
            continue
        end

        nbrs = G.neighbors[i]
        isempty(nbrs) && continue

        wsum = 0f0
        ws = Float32[]; push!(ws, 0f0)  # offset

        for j in nbrs
            ΔS = 0.1f0 + 0.01f0 + abs(G.R_vals[j]) + buffers.deg_pen[j]
            Δf = exp(counts[i]/100) - G.R_vals[i] + ΔS

            # local anisotropic penalty around j
            anisotropy_penalty = 0f0
            for k in G.neighbors[j]
                Δfk = exp(counts[j]/100) - G.R_vals[j] + buffers.deg_pen[j]
                anisotropy_penalty += exp(-Δfk - norm(G.anisotropy[k]))
            end

            w = exp(-Δf) * anisotropy_penalty
            wsum += w
            push!(ws, w)
        end

        wsum < 1f-8 && continue

        r = rand() * wsum
        cum = 0f0
        chosen = nbrs[1]
        for (kk, j) in enumerate(nbrs)
            cum += ws[kk+1]
            if cum ≥ r
                chosen = j
                break
            end
        end

        buffers.inflow_int[chosen] += 1
        buffers.outflow_int[i] += 1

        if counts[i] - buffers.outflow_int[i] == 0
            idx = findfirst(==(i), nonzero)
            idx !== nothing && deleteat!(nonzero, idx)
        end
        if counts[chosen] + buffers.inflow_int[chosen] == 1
            push!(nonzero, chosen)
        end
    end

    @inbounds for i in 1:n
        net = counts[i] + buffers.inflow_int[i] - buffers.outflow_int[i]
        counts[i] = max(0, net)
        rho[i] = counts[i] / 100f0
    end

    return rho
end

# =========================
# 4) Simplex + metrics
# =========================

simplex = Vector{Vector{GeoNode}}()  # (kept for compatibility)

"Compute distribution of occupancy by Hamming weight k."
function compute_k_distribution(ρ::Vector{Float32}, pf_states::Vector{String})
    k_counts = Dict{Int, Float32}()
    for (i, s) in enumerate(pf_states)
        k = count(==('1'), s)
        k_counts[k] = get(k_counts, k, 0.0f0) + ρ[i]
    end
    return k_counts
end

weighted_mean_oxidation(k_dist::Dict{Int, Float32}) =
    sum(Float32(k) * v for (k, v) in k_dist)

"Shannon entropy in bits, ignoring zeros."
function shannon_entropy(p::Vector{Float32})
    p_nonzero = filter(x -> x > 0f0, p)
    return -sum(x -> x * log2(x), p_nonzero)
end

"Simple 1D Fisher information across state index ordering."
function fisher_information(ρ::Vector{Float32})
    grad = diff(ρ)
    return sum(grad .^ 2)
end

"Crude Lyapunov estimate from successive norms."
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

"Classify direction of oxidation based on change in mean k/R."
function classify_transition(prev::Vector{Float32}, curr::Vector{Float32}, pf_states::Vector{String})
    R = length(pf_states[1])
    k_prev = sum(Float32(count(==('1'), pf_states[i])) * prev[i] for i in 1:length(prev)) / Float32(R)
    k_curr = sum(Float32(count(==('1'), pf_states[i])) * curr[i] for i in 1:length(curr)) / Float32(R)
    if k_curr > k_prev + 1f-4
        return :oxidizing
    elseif k_curr < k_prev - 1f-4
        return :reducing
    else
        return :neutral
    end
end

"Pick the 'dominant' geodesic (most overlap with visited states)."
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

struct Trace
    steps::Vector{String}
end
transitions(trace::Trace) = [(trace.steps[i], trace.steps[i+1]) for i in 1:length(trace.steps)-1]
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
delta_commutator(trace::Trace) = (ts = transitions(trace); length(unique(ts)) / max(length(ts), 1))
function trace_analysis(trace_steps::Vector{String})
    trace = Trace(trace_steps)
    return (compressed = compress(trace), patterns = recurrence_patterns(trace), delta = delta_commutator(trace))
end

# =========================
# 5) Flow tracing + rollout
# =========================

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

"One run: evolve ρ for T steps; compute metrics; mark geodesic overlap."
function record_flow_trace!(
    ρ0::Vector{Float32}, T::Int,
    pf_states::Vector{String}, flat_pos::Dict{String, Tuple{Float64, Float64}}, edges::Vector{Tuple{String, String}};
    geodesics::Vector{Vector{String}}, max_moves_per_step::Int=10, run_id::String="default_run"
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

    cumulative_entropy = 0.0f0

    # Initialize geometry + buffers
    G = GeoGraphReal(pf_states, flat_pos, edges)
    buffers = init_alive_buffers!(G, pf_states)

    for _ in 1:T
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
    end

    geo_path = dominant_geodesic(trajectory, geodesics)
    for state in trajectory
        push!(on_geodesic_flags, state in geo_path)
    end

    lyap = Float32(lyapunov_exponent(ρ_series))
    action_cost = cumulative_entropy / Float32(max(T, 1))

    trace_meta = trace_analysis(trajectory) # (optional, not returned here)

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
    )
end

"Batch rollout over many initials; returns FlowTraces + a simplex view."
function rollout_batch(batch_id::String, initials, pf_states, flat_pos, edges, geodesics, T::Int)
    flow_traces = FlowTrace[]
    simplex = Vector{Vector{GeoNode}}()

    for sample in initials
        ρ0 = sample.rho
        run_id = string(sample.uuid)

        ft = record_flow_trace!(ρ0, T, pf_states, flat_pos, edges;
                                geodesics=geodesics, run_id=run_id)

        # Build a lightweight simplex of GeoNodes (per-step snapshots) for visualization/analysis
        nodes = Vector{GeoNode}()
        for t in 2:length(ft.ρ_series)
            ρ = ft.ρ_series[t]
            R = ft.R_series[t - 1]
            flux = ft.flux_series[t - 1]

            G = GeoGraphReal(pf_states, flat_pos, edges)
            update_real_geometry!(G, ρ)
            A = copy(G.anisotropy)

            node = GeoNode(
                ρ_real = copy(ρ),
                R_real = copy(R),
                A_real = copy(A),
                sheath_stress = zeros(Float32, length(ρ)),
                flux = flux,
                action_cost = ft.action_cost
            )
            push!(nodes, node)
        end

        push!(flow_traces, ft)
        push!(simplex, nodes)
    end

    return flow_traces, simplex
end

"Random partition generator for reproducible initial ρ (total mass = `total`)."
function generate_safe_random_initials(n::Int, num_bins::Int; total::Int = 100)
    batch_id = "batch_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    samples = Vector{NamedTuple{(:uuid, :counts, :rho), Tuple{UUID, Vector{Int}, Vector{Float32}}}}()

    for _ in 1:n
        cuts = sort(rand(0:total, num_bins - 1))
        counts = Vector{Int}(undef, num_bins)
        counts[1] = cuts[1]
        for i in 2:num_bins - 1
            counts[i] = cuts[i] - cuts[i - 1]
        end
        counts[num_bins] = total - cuts[end]
        rho = Float32.(counts) ./ Float32(total)
        push!(samples, (uuid=uuid4(), counts=counts, rho=rho))
    end

    @save "initials_$batch_id.bson" samples
    return batch_id, samples
end

"Persist run artifacts."
function save_run_data(batch_id::String, flow_traces, simplex)
    @save "flow_traces_$batch_id.bson" flow_traces
    @save "simplex_$batch_id.bson" simplex
    println("✔ Saved run summary:")
    println("→ FlowTraces: ", length(flow_traces))
    println("→ Simplex: ", length(simplex))
end

# =========================
# 6) Simulation config & execution
# =========================

# ---- Choose dimensionality here ----
R = 3                     # set e.g. 4, 6, 8, 10 for scale-invariance tests
pf_states, flat_pos, edges = build_modal_topology(R)
geodesics = sample_geodesics(R; M=min(6, max(2, R)))  # a small canonical set

# ---- Workload parameters (adjust as needed) ----
num_initials      = 200        # number of initial distributions (bins = 2^R)
total_molecules   = 100        # discrete mass per distribution
simulation_steps  = 200        # steps per simulation

# ---- Generate initials ----
batch_id, initials = generate_safe_random_initials(num_initials, length(pf_states); total=total_molecules)
println("✔ Generated initials for batch: $batch_id  (R = $R, states = $(length(pf_states)))")

# ---- Run rollout ----
flow_traces, simplex = rollout_batch(batch_id, initials, pf_states, flat_pos, edges, geodesics, simulation_steps)

println("→ FlowTraces: ", length(flow_traces))
println("→ Simplex:    ", length(simplex))

save_run_data(batch_id, flow_traces, simplex)
println("✔ Finished rollout and saved data for batch: $batch_id")
