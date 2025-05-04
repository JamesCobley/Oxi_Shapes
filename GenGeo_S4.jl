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
# Load the brain
# =============================================================================

@load "ricci_learned_brain_<your_batch_id>.bson" brain

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
    anisotropy::Vector{Float32}
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
    anis = zeros(Float32, n)

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
        acc, cnt = 0f0, 0
        Ri = G.R_vals[i]
        for (k, j) in enumerate(G.neighbors[i])
            dist = G.d0[i][k]
            ΔR = abs(Ri - G.R_vals[j])
            if dist > 1f-6
                acc += ΔR / dist
                cnt += 1
            end
        end
        G.anisotropy[i] = cnt > 0 ? acc / cnt : 0f0
    end

    for i in 1:G.n
        vol = rho[i] + sum(rho[j] for j in G.neighbors[i])
        expected_vol = (1 + length(G.neighbors[i])) / G.n
        vol_ok = abs(vol - expected_vol) ≤ eps
        shape_ok = abs(G.R_vals[i]) ≤ eps * (1.0f0 + G.anisotropy[i])
        if !(vol_ok && shape_ok)
            push!(violated, i)
        end
    end

    return violated
end

function update_imagined_geometry!(G::GeoGraphReal, rho_imag::Vector{Float32}; eps::Float32 = 1f-3)
    violated = Int[]

    # Step 1: Lift into 3D
    @inbounds for i in 1:G.n
        G.points3D[i] = Point3(G.flat_x[i], G.flat_y[i], -rho_imag[i])
        G.R_vals[i] = 0.0f0
    end

    # Step 2: Compute scalar curvature
    @inbounds for i in 1:G.n
        pi = G.points3D[i]
        for (k, j) in enumerate(G.neighbors[i])
            d3 = norm(pi - G.points3D[j])
            G.R_vals[i] += d3 - G.d0[i][k]
        end
    end

    # Step 3: Compute anisotropy
    @inbounds for i in 1:G.n
        acc, cnt = 0f0, 0
        Ri = G.R_vals[i]
        for (k, j) in enumerate(G.neighbors[i])
            dist = G.d0[i][k]
            ΔR = abs(Ri - G.R_vals[j])
            if dist > 1f-6
                acc += ΔR / dist
                cnt += 1
            end
        end
        G.anisotropy[i] = cnt > 0 ? acc / cnt : 0f0
    end

    return nothing  # No violation tracking here (optional to add)
end

struct GeoNode
    ρ_real::Vector{Float32}
    R_real::Vector{Float32}
    A_real::Vector{Float32}
    ρ_imag::Vector{Float32}
    R_imag::Vector{Float32}
    A_imag::Vector{Float32}
    lambda::Float32
    sheath_stress::Vector{Float32}
    flux::Vector{Float32}
    action_cost::Float32
end

# =============================================================================
# Module 2: Simplex Construction and Flow
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
            w = exp(-Δf) * exp(-G.anisotropy[j])
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
    n = G.n

    # Convert imag → counts
    @inbounds for i in 1:n
        counts[i] = round(Int, field.imag[i] * 100)
    end
    counts[n] = 100 - sum(counts[1:n-1])  # Force total mass = 100

    update_real_geometry!(G, field.imag)

    fill!(buffers.inflow_int, 0)
    fill!(buffers.outflow_int, 0)
    nmoves = rand(0:moves)
    nonzero = findall(>(0), counts)

    for _ in 1:nmoves
        isempty(nonzero) && break
        i = nonzero[argmax(G.R_vals[nonzero])]
        if counts[i] - buffers.outflow_int[i] ≤ 0
            continue
        end

        nbrs = G.neighbors[i]
        isempty(nbrs) && continue

        # Greedy: choose neighbor with lowest Δf
        best_j, best_cost = nbrs[1], Inf
        for j in nbrs
            ΔS = 0.1f0 + 0.01f0 + abs(G.R_vals[j]) + buffers.deg_pen[j]
            Δf = exp(counts[i]/100) - G.R_vals[i] + ΔS
            if Δf < best_cost
                best_cost = Δf
                best_j = j
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

    @inbounds for i in 1:n
        net = counts[i] + buffers.inflow_int[i] - buffers.outflow_int[i]
        counts[i] = max(0, net)
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
builds the λ-surface where each point is -lambda.
"""
function build_simplex_surface(simplex::Vector{Vector{GeoNode}})
    n_runs = length(simplex)
    rollout_steps = maximum(length(run) for run in simplex)
    lambda_surface = fill(0f0, rollout_steps, n_runs)

    for (r, run) in enumerate(simplex)
        for (t, node) in enumerate(run)
            lambda_surface[t, r] = -node.lambda
        end
    end

    return lambda_surface
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

    D = Diagonal(vec(sum(W, dims=2)))  # ✅ Fix: convert to vector
    L = D - W

    return L
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
    mean_pred_error::Float32  # ← ✅ Add this line
end

# =============================================================================
# Prediction from Ricci-Learned Brain
# =============================================================================

function predict_with_brain_from_real(ρ_real::Vector{Float32}, brain::HypergraphBrain1)
    normed_phi = brain.phi ./ sum(brain.phi)
    ρ_pred = 1 .- normed_phi
    ρ_pred ./= sum(ρ_pred)
    return Float32.(ρ_pred)
end

# --- Record Flow Trace
function record_flow_trace!(ρ0::Vector{Float32}, T::Int, pf_states, flat_pos, edges;
                            brain::HypergraphBrain1,
                            max_moves_per_step=10, run_id::String="default_run")

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
    prediction_lambda_series = Float32[]

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

        # --- Ricci Flow Brain Prediction ---
        ρ_pred = predict_with_brain_from_real(ρ, brain)
        update_imagined_geometry!(G, ρ_pred)
        λ_pred = compute_lambda(ρ, ρ_pred)
        push!(prediction_lambda_series, λ_pred)
    end

    geo_path = dominant_geodesic(trajectory, geodesics)
    for state in trajectory
        push!(on_geodesic_flags, state in geo_path)
    end

    lyap = Float32(lyapunov_exponent(ρ_series))
    action_cost = cumulative_entropy / Float32(total_moves)
    trace_meta = trace_analysis(trajectory)

    # Final report
    mean_pred_error = mean(prediction_lambda_series)
    println("📉 Mean Ricci prediction λ-error: ", round(mean_pred_error, digits=5))

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
# Support Functions
# =============================================================================

function rollout_batch(batch_id::String, initials, pf_states, flat_pos, edges, T::Int, brain)
    flow_traces = FlowTrace[]
    trace_metadata = Vector{Dict{Symbol, Any}}()
    simplex = Vector{Vector{GeoNode}}()

    for sample in initials
        ρ0 = sample.rho
        run_id = string(sample.uuid)

        ft, trace_meta = record_flow_trace!(ρ0, T, pf_states, flat_pos, edges; brain=brain, run_id=run_id)

        nodes = Vector{GeoNode}()
        for t in 2:length(ft.ρ_series)  # start from step 2
        ρ = ft.ρ_series[t]
        R = ft.R_series[t - 1]
        flux = ft.flux_series[t - 1]
        G = GeoGraphReal(pf_states, flat_pos, edges)
        update_real_geometry!(G, ρ)
        A = copy(G.anisotropy)
        field = init_living_simplex_tensor(ρ)
        λ = step_imagination!(field, G, init_alive_buffers!(G, [count(==('1'), s) for s in pf_states]))

        node = GeoNode(copy(ρ), copy(R), copy(A), copy(field.imag), copy(G.R_vals), copy(G.anisotropy), λ,
                   zeros(Float32, length(ρ)), flux, ft.action_cost)
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
num_initials = 100          # Number of initial distributions
total_molecules = 100       # Total molecules per distribution
simulation_steps = 100      # Number of time steps per simulation

# --- Load Initials ---
@load "initials_<your_batch_id>.bson" initials
batch_id = "<your_batch_id>"
println("✔ Loaded initials from batch: $batch_id")

# --- Run Simulation ---
flow_traces, trace_metadata, simplex = rollout_batch(batch_id, initials, pf_states, flat_pos, edges, simulation_steps, brain)

# ✅ Now it's safe to call:
println("→ FlowTraces: ", length(flow_traces))
println("→ Metadata: ", length(trace_metadata))
println("→ Simplex: ", length(simplex))

save_run_data(batch_id, flow_traces, trace_metadata, simplex)

println("✔ Finished rollout and saved data for batch: $batch_id")
