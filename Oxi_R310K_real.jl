# =============================================================================
# Oxi-Shapes Modelling
# =============================================================================
using LinearAlgebra
using SparseArrays
using StaticArrays
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

pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
flat_pos = Dict(
    "000" => (0.0, 3.0), "001" => (-2.0, 2.0), "010" => (0.0, 2.0),
    "100" => (2.0, 2.0), "011" => (-1.0, 1.0), "101" => (0.0, 1.0),
    "110" => (1.0, 1.0), "111" => (0.0, 0.0)
)
edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "011"), ("001", "101"), ("010", "011"), ("010", "110"),
    ("100", "110"), ("100", "101"), ("011", "111"), ("101", "111"), ("110", "111")
]

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
            if norm(dir) > 1e-6
                bundle += (ΔR / norm(dir)^2) * dir
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


# =============================================================================
# Geometric object 2: Simplex 
# =============================================================================

# =============================================================================
# 1. Real-Flow (Oxi-Shapes Alive)
# =============================================================================

function init_alive_buffers!(G::GeoGraphReal, bitcounts::Vector{Int})
    n = G.n
    counts = Vector{Int}(undef, n)
    inflow_int = zeros(Int, n)
    outflow_int = zeros(Int, n)
    R_total = length(bitcounts)
    binom = Dict(k => binomial(R_total, k) for k in 0:R_total)
    deg_pen = Float32[1f0 / binom[bitcounts[i]] for i in 1:n]
    return (counts=counts, inflow_int=inflow_int, outflow_int=outflow_int, deg_pen=deg_pen)
end

function oxi_shapes_alive!(rho::Vector{Float32}, G::GeoGraphReal, buffers; max_moves::Int=10)
    n = G.n
    counts = buffers.counts

    # Convert rho → counts (discrete copy, rounding)
    @inbounds for i in 1:n
        counts[i] = round(Int, rho[i] * 100)
    end
    counts[n] = 100 - sum(counts[1:n-1])  # force total mass = 100

    update_real_geometry!(G, rho)

    fill!(buffers.inflow_int, 0)
    fill!(buffers.outflow_int, 0)
    total_moves = rand(0:max_moves)

    nonzero = findall(>(0), counts)

    for _ in 1:total_moves
        isempty(nonzero) && break
        i = rand(nonzero)

        if counts[i] - buffers.outflow_int[i] ≤ 0
            continue  # skip: nothing left to move
        end

        nbrs = G.neighbors[i]
        isempty(nbrs) && continue

        # Compute transition weights
        wsum = 0f0
        ws = Float32[]
        push!(ws, 0f0)  # dummy 0 for indexing offset

        for j in nbrs
            ΔS = 0.1f0 + 0.01f0 + abs(G.R_vals[j]) + buffers.deg_pen[j]
            Δf = exp(counts[i]/100) - G.R_vals[i] + ΔS

            # Compute the anisotropic penalty around j
            anisotropy_penalty = 0f0
            for k in G.neighbors[j]
            Δfk = exp(counts[j]/100) - G.R_vals[j] + buffers.deg_pen[j]  # or use a Δf-like proxy
            anisotropy_penalty += exp(-Δfk - norm(G.anisotropy[k]))
            end

            w = exp(-Δf) * anisotropy_penalty
            wsum += w
            push!(ws, w)
        end

        wsum < 1f-8 && continue  # avoid division by zero

        # Weighted random choice
        r = rand() * wsum
        cum = 0f0
        chosen = nbrs[1]
        for (k, j) in enumerate(nbrs)
            cum += ws[k+1]
            if cum >= r
                chosen = j
                break
            end
        end

        buffers.inflow_int[chosen] += 1
        buffers.outflow_int[i] += 1

        # Track active update set
        if counts[i] - buffers.outflow_int[i] == 0
            deleteat!(nonzero, findfirst(==(i), nonzero))
        end
        if counts[chosen] + buffers.inflow_int[chosen] == 1
            push!(nonzero, chosen)
        end
    end

    # Final update: apply inflow/outflow and update rho
    @inbounds for i in 1:n
        net = counts[i] + buffers.inflow_int[i] - buffers.outflow_int[i]
        counts[i] = max(0, net)
        rho[i] = counts[i] / 100f0
    end

    return rho
end

# =============================================================================
# 2. Simplex
# =============================================================================

simplex = Vector{Vector{GeoNode}}()  # Outer = runs, Inner = time-series for a run

function simulate_single_run(ρ0::Vector{Float32}, T::Int, pf_states, flat_pos, edges)
    G = GeoGraphReal(pf_states, flat_pos, edges)
    bitcounts = [count(==('1'), s) for s in pf_states]
    buffers = init_alive_buffers!(G, bitcounts)
    
    ρ = copy(ρ0)
    trajectory = Vector{GeoNode}()

    for t in 1:T
        ρ_old = copy(ρ)
        oxi_shapes_alive!(ρ, G, buffers; max_moves=10)
        update_real_geometry!(G, ρ)

        node = GeoNode(
            ρ_real = copy(ρ),
            R_real = copy(G.R_vals),
            A_real = copy(G.anisotropy),
            sheath_stress = zeros(Float32, length(ρ)),  # Placeholder
            flux = ρ .- ρ_old,
            action_cost = 0.0f0  # You can compute and track if needed
        )
        push!(trajectory, node)
    end

    return trajectory
end

function build_simplex(N_runs::Int, T::Int, pf_states, flat_pos, edges)
    simplex = Vector{Vector{GeoNode}}()
    for _ in 1:N_runs
        ρ0 = rand(Float32, length(pf_states))
        ρ0 ./= sum(ρ0)  # Normalize to sum to 1
        run = simulate_single_run(ρ0, T, pf_states, flat_pos, edges)
        push!(simplex, run)
    end
    return simplex
end

# =============================================================================
# Flow Helper
# =============================================================================

if !isdefined(Main, :geodesics)
    const geodesics = [
        ["000", "100", "101", "111"],
        ["000", "100", "110", "111"],
        ["000", "010", "110", "111"],
        ["000", "010", "011", "111"],
        ["000", "001", "101", "111"],
        ["000", "001", "011", "111"]
    ]
end

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
# Generate and save the initials
# =============================================================================

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

    # Save initials for reproducibility
    @save "initials_$batch_id.bson" samples

    return batch_id, samples
end

# =============================================================================
# Support Functions (Updated for Vector-Valued Anisotropy, No Imagination)
# =============================================================================

function rollout_batch(batch_id::String, initials, pf_states, flat_pos, edges, T::Int)
    flow_traces = FlowTrace[]
    trace_metadata = Vector{Dict{Symbol, Any}}()
    simplex = Vector{Vector{GeoNode}}()

    for sample in initials
        ρ0 = sample.rho
        run_id = string(sample.uuid)

        ft, trace_meta = record_flow_trace!(ρ0, T, pf_states, flat_pos, edges; run_id=run_id)

        nodes = Vector{GeoNode}()
        for t in 2:length(ft.ρ_series)  # start from step 2
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
                sheath_stress = zeros(Float32, length(ρ)),  # optional placeholder
                flux = flux,
                action_cost = ft.action_cost
            )
            push!(nodes, node)
        end

        push!(flow_traces, ft)
        push!(trace_metadata, Dict(pairs(trace_meta)))
        push!(simplex, nodes)
    end

    return flow_traces, trace_metadata, simplex
end

function save_run_data(batch_id::String, flow_traces, trace_metadata, simplex)
    @save "flow_traces_$batch_id.bson" flow_traces
    @save "trace_metadata_$batch_id.bson" trace_metadata
    @save "simplex_$batch_id.bson" simplex

    println("✔ Saved run summary:")
    println("→ FlowTraces: ", length(flow_traces))
    println("→ Metadata: ", length(trace_metadata))
    println("→ Simplex: ", length(simplex))
end

# =============================================================================
# Simulation Configuration and Execution
# =============================================================================

# --- Parameters ---
num_initials = 10000          # Number of initial distributions
total_molecules = 100       # Total molecules per distribution
simulation_steps = 1000      # Number of time steps per simulation

# --- Generate and Save Initials ---
batch_id, initials = generate_safe_random_initials(num_initials, length(pf_states); total=total_molecules)
println("✔ Generated initials for batch: $batch_id")

# --- Run Simulation ---
flow_traces, trace_metadata, simplex = rollout_batch(batch_id, initials, pf_states, flat_pos, edges, simulation_steps)

# ✅ Now it's safe to call:
println("→ FlowTraces: ", length(flow_traces))
println("→ Metadata: ", length(trace_metadata))
println("→ Simplex: ", length(simplex))

save_run_data(batch_id, flow_traces, trace_metadata, simplex)

println("✔ Finished rollout and saved data for batch: $batch_id")
