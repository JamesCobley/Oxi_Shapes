# ============================================================================
# Load the packages
# ============================================================================

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

# ============================================================================
# Define the REAL geometry
# ============================================================================

function lift_to_z_plane(rho::Vector{Float32}, pf_states, flat_pos)
    return [Point3(Float32(flat_pos[s][1]), Float32(flat_pos[s][2]), -rho[i]) for (i, s) in enumerate(pf_states)]
end

function compute_R(points3D, flat_pos, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    R = zeros(Float32, length(pf_states))
    for (i, s) in enumerate(pf_states)
        p0, p3 = flat_pos[s], points3D[i]
        neighbors = [v for (u, v) in edges if u == s] ∪ [u for (u, v) in edges if v == s]
        for n in neighbors
            j, q0, q3 = idx[n], flat_pos[n], points3D[idx[n]]
            d0 = norm(Float32[p0[1] - q0[1], p0[2] - q0[2]])
            d3 = norm(p3 - q3)
            R[i] += d3 - d0
        end
    end
    return R
end

function compute_c_ricci_dirichlet(R::Vector{Float32}, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    C_R = zeros(Float32, length(pf_states))
    for (i, s) in enumerate(pf_states)
        neighbors = [v for (u, v) in edges if u == s] ∪ [u for (u, v) in edges if v == s]
        for n in neighbors
            j = idx[n]
            C_R[i] += (R[i] - R[j])^2
        end
    end
    return C_R
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

struct GeoGraphStruct
    pf_states::Vector{String}
    flat_pos::Dict{String, Tuple{Float64, Float64}}
    edges::Vector{Tuple{String, String}}
    idx_map::Dict{String, Int}
    points3D::Vector{Point3}
    R_vals::Vector{Float32}
    C_R_vals::Vector{Float32}
    anisotropy::Vector{Float32}
    adjacency::Matrix{Float32}
    stalks::Dict{String, Vector{Float32}}
    sheaf_inconsistencies::Vector{Tuple{String, String, Float32}}
end

function build_GeoGraphStruct(ρ::Vector{Float32}, pf_states, flat_pos, edges)
    idx_map = Dict(s => i for (i, s) in enumerate(pf_states))
    points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
    R_vals = compute_R(points3D, flat_pos, pf_states, edges)
    C_R_vals = compute_c_ricci_dirichlet(R_vals, pf_states, edges)
    anisotropy = compute_anisotropy_from_curvature(R_vals, pf_states, flat_pos, edges)

    g = SimpleDiGraph(length(pf_states))
    for (u, v) in edges
        add_edge!(g, idx_map[u], idx_map[v])
    end
    A = Float32.(adjacency_matrix(g))

    stalks = initialize_sheaf_stalks(flat_pos, pf_states)
    inconsistencies = sheaf_consistency(stalks, edges)

    return GeoGraphStruct(
        pf_states, flat_pos, edges, idx_map,
        points3D, R_vals, C_R_vals, anisotropy, A,
        stalks, inconsistencies
    )
end

function update_geometry_from_rho(ρ::Vector{Float32}, geo::GeoGraphStruct)
    pf_states = geo.pf_states
    flat_pos = geo.flat_pos
    edges = geo.edges

    points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
    R_vals = compute_R(points3D, flat_pos, pf_states, edges)
    C_R_vals = compute_c_ricci_dirichlet(R_vals, pf_states, edges)
    anisotropy = compute_anisotropy_from_curvature(R_vals, pf_states, flat_pos, edges)

    return points3D, R_vals, C_R_vals, anisotropy
end

# === Initialize and test ===
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

# ============================================================================
# Define the ALIVE function
# ============================================================================

function compute_entropy_cost(i::Int, j::Int, R_vals::Vector{Float32}, pf_states::Vector{String})
    baseline_DeltaE = 1.0f0
    mass_heat = 0.1f0
    reaction_heat = 0.01f0 * baseline_DeltaE
    conformational_cost = abs(R_vals[j])

    degeneracy_map = Dict(0 => 1, 1 => 3, 2 => 3, 3 => 1)
    deg = degeneracy_map[count(c -> c == '1', pf_states[j])]
    degeneracy_penalty = 1.0f0 / deg

    return mass_heat + reaction_heat + conformational_cost + degeneracy_penalty
end

function oxi_shapes_alive!(
    ρ::Vector{Float32},
    pf_states::Vector{String},
    flat_pos::Dict{String, Tuple{Float64, Float64}},
    edges::Vector{Tuple{String, String}};
    max_moves::Int = 10
)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = Dict(s => Int[] for s in pf_states)
    for (u, v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end

    # Step 1: Integer molecule approximation
    counts = round.(Int, ρ .* 100)
    counts[end] = 100 - sum(counts[1:end-1])
    ρ .= Float32.(counts) ./ 100

    # Step 2: Update curvature
    points3D, R_vals, C_R_vals, anisotropy_vals = update_geometry_from_rho(ρ, build_GeoGraphStruct(ρ, pf_states, flat_pos, edges))

    inflow = zeros(Float32, length(pf_states))
    outflow = zeros(Float32, length(pf_states))

    total_moves = rand(0:max_moves)
    candidate_indices = findall(x -> x > 0, counts)

    for _ in 1:total_moves
        isempty(candidate_indices) && break
        i = rand(candidate_indices)
        s = pf_states[i]
        nbrs = neighbor_indices[s]
        isempty(nbrs) && continue

        # Build transition probabilities
        probs = Float32[]
        for j in nbrs
            ΔS = compute_entropy_cost(i, j, R_vals, pf_states)
            Δf = exp(ρ[i]) - R_vals[i] + ΔS
            p = exp(-Δf) * exp(-anisotropy_vals[j])
            push!(probs, p)
        end

        if sum(probs) < 1f-8
            inflow[i] += 0.01f0
            continue
        end

        probs ./= sum(probs)
        chosen = sample(nbrs, Weights(probs))
        inflow[chosen] += 0.01f0
        outflow[i] += 0.01f0
    end

    for i in eachindex(pf_states)
        inflow[i] += (counts[i] / 100.0f0) - outflow[i]
    end

    # Step 3: Clamp, normalize, threshold
    inflow .= max.(inflow, 0.0f0)
    if sum(inflow) > 0.0f0
        inflow ./= sum(inflow)
    end

    for i in eachindex(inflow)
        if inflow[i] > 0.0f0 && inflow[i] < 0.01f0
            inflow[i] = 0.01f0
        end
    end

    inflow .= max.(inflow, 0.0f0)
    inflow ./= sum(inflow)
    ρ .= inflow
end

# ============================================================================
# Define the "Flow" 
# ============================================================================

# Define geodesics for classification
geodesics = [
    ["000", "100", "101", "111"],
    ["000", "100", "110", "111"],
    ["000", "010", "110", "111"],
    ["000", "010", "011", "111"],
    ["000", "001", "101", "111"],
    ["000", "001", "011", "111"]
]

# --- Geodesic classifier
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

function compute_k_distribution(ρ::Vector{Float32}, pf_states::Vector{String})
    k_counts = Dict{Int, Float32}()
    for (i, s) in enumerate(pf_states)
        k = count(==('1'), s)
        k_counts[k] = get(k_counts, k, 0.0f0) + ρ[i]
    end
    return k_counts
end

function weighted_mean_oxidation(k_dist::Dict{Int, Float32})
    return sum(Float32(k) * v for (k, v) in k_dist)
end

function shannon_entropy(p::Vector{Float32})
    p_nonzero = filter(x -> x > 0f0, p)
    return -sum(x -> x * log2(x), p_nonzero)
end

function fisher_information(ρ::Vector{Float32})
    grad = diff(ρ)
    return sum(grad .^ 2)
end

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

function classify_transition(prev::Vector{Float32}, curr::Vector{Float32})
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

struct FlowTrace
    run_id::String
    ρ_series::Vector{Vector{Float32}}
    flux_series::Vector{Vector{Float32}}
    curvature_series::Vector{Vector{Float32}}  # C_Ricci
    R_series::Vector{Vector{Float32}}          # Raw R(x)
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

function record_flow_trace!(
    ρ0::Vector{Float32}, T::Int, pf_states, flat_pos, edges;
    max_moves_per_step=10, run_id::String="default_run"
)
    ρ = copy(ρ0)
    ρ_series = [copy(ρ)]
    flux_series = Vector{Vector{Float32}}()
    curvature_series = Vector{Vector{Float32}}()
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

    for t in 1:T
        ρ_old = copy(ρ)
        oxi_shapes_alive!(ρ, pf_states, flat_pos, edges; max_moves=max_moves_per_step)

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

        _, _, R_vals, C_R_vals = update_geometry_from_rho(ρ, pf_states, flat_pos, edges)
        push!(R_series, Float32.(R_vals))
        push!(curvature_series, Float32.(C_R_vals))

        class = classify_transition(ρ_old, ρ)
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

    return FlowTrace(
        run_id,
        ρ_series,
        flux_series,
        curvature_series,
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

# ============================================================================
# Define LivingFieldSimplex (Core Memory Manifold)
# ============================================================================

mutable struct LivingFieldSimplex
    real::Vector{Float32}
    imag::Vector{Float32}
    curvature_memory::Vector{Float32}
    real_confidence::Vector{Float32}
    imag_confidence::Vector{Float32}
    curvature_confidence::Vector{Float32}
    adjacency::SparseMatrixCSC{Float32, Int}
    simplex_tensions::Dict{Tuple{Int,Int}, Float32}
    prev_real::Vector{Float32}  # 🌟 new field: store previous real field
end

# ============================================================================
# Initialization
# ============================================================================

function init_living_field_simplex(real::Vector{Float32}, adjacency::SparseMatrixCSC{Float32, Int})
    n = length(real)

    imag = zeros(Float32, n)
    curvature_memory = zeros(Float32, n)

    real_confidence = ones(Float32, n)
    imag_confidence = zeros(Float32, n)
    curvature_confidence = zeros(Float32, n)

    tensions = Dict{Tuple{Int, Int}, Float32}()
    for i in 1:n
        for j in findnz(adjacency[i, :])[2]
            if i < j
                tensions[(i, j)] = 0.01f0
            end
        end
    end

    return LivingFieldSimplex(
        real,
        imag,
        curvature_memory,
        real_confidence,
        imag_confidence,
        curvature_confidence,
        adjacency,
        tensions,
        copy(real)  # 🌟 prev_real initialized as copy
    )
end

# ============================================================================
# Basic Update Rules (Core Brain Evolution)
# ============================================================================

"Update real field using local memory + tension feedback."
function update_real_field!(field::LivingFieldSimplex)
    for i in eachindex(field.real)
        neighbors = findnz(field.adjacency[i, :])[2]
        if isempty(neighbors)
            continue
        end
        local_flux = sum(field.imag[neighbors]) / length(neighbors)
        adjustment = (local_flux - field.real[i]) * field.real_confidence[i]
        field.real[i] += 0.05f0 * adjustment  # Small adjustment
    end
    normalize!(field.real)
end

"Update imaginary prediction based on curvature and memory."
function update_imaginary_field!(field::LivingFieldSimplex)
    for i in eachindex(field.imag)
        memory_bias = field.curvature_memory[i] * field.curvature_confidence[i]
        prediction_error = field.real[i] - field.imag[i]
        update = memory_bias + prediction_error * field.imag_confidence[i]
        field.imag[i] += 0.05f0 * update
    end
    normalize!(field.imag)
end

"Update simplex tensions based on curvature differences."
function update_tensions!(field::LivingFieldSimplex)
    for ((i, j), tension) in field.simplex_tensions
        curvature_diff = abs(field.curvature_memory[i] - field.curvature_memory[j])
        tension_update = 0.01f0 * curvature_diff
        field.simplex_tensions[(i, j)] = clamp(tension + tension_update, 0.0f0, 1.0f0)
    end
end

"Update curvature memory based on real geometry."
function update_curvature_memory!(field::LivingFieldSimplex)
    for i in eachindex(field.curvature_memory)
        local_neighbors = findnz(field.adjacency[i, :])[2]
        if isempty(local_neighbors)
            continue
        end
        local_curvature = mean(abs.(field.real[i] .- field.real[local_neighbors]))
        # Slowly adjust curvature memory
        field.curvature_memory[i] += 0.01f0 * (local_curvature - field.curvature_memory[i])
    end
end

"Simple normalization helper."
function normalize!(v::Vector{Float32})
    total = sum(v)
    if total > 0
        v ./= total
    end
end

"Update confidences based on stability and prediction quality."
function update_confidences!(field::LivingFieldSimplex)
    η_real = 0.01f0         # learning rate for real confidence
    η_imag = 0.01f0         # learning rate for imag confidence
    η_curvature = 0.005f0   # learning rate for curvature confidence

    for i in eachindex(field.real)
        # --- Real Stability
        Δ_real = abs(field.real[i] - field.prev_real[i])
        if Δ_real < 0.01f0
            field.real_confidence[i] += η_real * (1.0f0 - field.real_confidence[i])
        else
            field.real_confidence[i] -= η_real * field.real_confidence[i]
        end

        # --- Imaginary Match
        Δ_imag = abs(field.imag[i] - field.real[i])
        if Δ_imag < 0.02f0
            field.imag_confidence[i] += η_imag * (1.0f0 - field.imag_confidence[i])
        else
            field.imag_confidence[i] -= η_imag * field.imag_confidence[i]
        end

        # --- Curvature Stability
        neighbors = findnz(field.adjacency[i, :])[2]
        if !isempty(neighbors)
            local_curvature = mean(abs.(field.real[i] .- field.real[neighbors]))
            Δ_curv = abs(field.curvature_memory[i] - local_curvature)
            if Δ_curv < 0.02f0
                field.curvature_confidence[i] += η_curvature * (1.0f0 - field.curvature_confidence[i])
            else
                field.curvature_confidence[i] -= η_curvature * field.curvature_confidence[i]
            end
        end
    end

    # Clamp confidences to [0,1]
    field.real_confidence .= clamp.(field.real_confidence, 0.0f0, 1.0f0)
    field.imag_confidence .= clamp.(field.imag_confidence, 0.0f0, 1.0f0)
    field.curvature_confidence .= clamp.(field.curvature_confidence, 0.0f0, 1.0f0)
end

# ============================================================================
# Single Full Brain Step (One Tick of Evolution)
# ============================================================================

"Run one full evolution step of the living field."
function evolve_living_simplex!(field::LivingFieldSimplex)
    field.prev_real .= field.real   # 🌟 save previous real before updates
    update_real_field!(field)
    update_imaginary_field!(field)
    update_curvature_memory!(field)
    update_tensions!(field)
    update_confidences!(field)       # 🌟 update confidences after everything else
end

# ============================================================================
# Random Initial Generator with UUIDs
# ============================================================================

function generate_safe_random_initials(n::Int; min_val=0.0f0)
    batch_id = "batch_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    samples = []

    while length(samples) < n
        vec = rand(Float32, 8)
        norm_vec = vec / sum(vec)
        is_safe = all(x -> x ≥ min_val || isapprox(x, 0.0f0; atol=1e-8f0), norm_vec)

        if is_safe
            sample_id = uuid4()
            push!(samples, (uuid=sample_id, rho=norm_vec))
        end
    end

    return batch_id, samples
end

# ============================================================================
# Living Rollout Engine
# ============================================================================

@kwdef struct LivingFlowConfig
    pf_states::Vector{String}
    flat_pos::Dict{String, Tuple{Float64, Float64}}
    edges::Vector{Tuple{String, String}}
    initials::Vector{NamedTuple{(:uuid, :rho), Tuple{UUID, Vector{Float32}}}}
    epochs::Int = 100
    batch_size::Int = 10
    rollout_steps::Int = 100
    model_id::String = "living_field"
end

mutable struct LivingRolloutLog
    epoch::Int
    avg_real_confidence::Float32
    avg_imag_confidence::Float32
    avg_curvature_confidence::Float32
    avg_tension::Float32
    init_uuid::UUID
end

"Run one rollout starting from an initial state."
function run_living_rollout(
    real0::Vector{Float32},
    adjacency::SparseMatrixCSC{Float32, Int};
    T::Int = 100
)::Tuple{Vector{LivingFieldSimplex}, Vector{Float32}, Float32}
    field = init_living_field_simplex(real0, adjacency)
    trajectory = LivingFieldSimplex[]
    curvature_entropy = Float32[]

    for _ in 1:T
        push!(trajectory, deepcopy(field))
        evolve_living_simplex!(field)

        ce = -sum(p -> p > 0 ? p * log2(p) : 0, normalize_copy(field.curvature_memory))
        push!(curvature_entropy, ce)
    end

    # Final MSE between real and imag
    final_mse = mean((field.real .- field.imag).^2)

    return trajectory, curvature_entropy, final_mse
end

"Simple helper: normalize copy."
function normalize_copy(v::Vector{Float32})
    s = sum(v)
    s > 0 ? v ./ s : v
end

"Main training loop across epochs and batch."
function train_living_flow_model(config::LivingFlowConfig)
    training_trace = LivingRolloutLog[]
    mse_epoch_history = Float32[]

    adjacency = build_adjacency_matrix(config.pf_states, config.edges)

    for epoch in 1:config.epochs
        real_confidences = Float32[]
        imag_confidences = Float32[]
        curvature_confidences = Float32[]
        avg_tensions = Float32[]
        mses = Float32[]

        for _ in 1:config.batch_size
            sample = config.initials[rand(1:end)]
            real0 = sample.rho

            traj, _, mse = run_living_rollout(real0, adjacency; T=config.rollout_steps)
            final_field = traj[end]

            push!(real_confidences, mean(final_field.real_confidence))
            push!(imag_confidences, mean(final_field.imag_confidence))
            push!(curvature_confidences, mean(final_field.curvature_confidence))
            push!(avg_tensions, mean(values(final_field.simplex_tensions)))
            push!(mses, mse)

            push!(training_trace, LivingRolloutLog(
                epoch = epoch,
                avg_real_confidence = mean(final_field.real_confidence),
                avg_imag_confidence = mean(final_field.imag_confidence),
                avg_curvature_confidence = mean(final_field.curvature_confidence),
                avg_tension = mean(values(final_field.simplex_tensions)),
                init_uuid = sample.uuid
            ))
        end

        println("Epoch $epoch | ⟨real⟩=$(round(mean(real_confidences), digits=4)) ⟨imag⟩=$(round(mean(imag_confidences), digits=4)) ⟨curv⟩=$(round(mean(curvature_confidences), digits=4)) | MSE=$(round(mean(mses), digits=6))")
        push!(mse_epoch_history, mean(mses))
    end

    return training_trace, mse_epoch_history
end

"Build adjacency matrix from edges."
function build_adjacency_matrix(pf_states::Vector{String}, edges::Vector{Tuple{String,String}})
    n = length(pf_states)
    idx_map = Dict(s => i for (i, s) in enumerate(pf_states))
    adj = spzeros(Float32, n, n)
    for (u, v) in edges
        adj[idx_map[u], idx_map[v]] = 1f0
        adj[idx_map[v], idx_map[u]] = 1f0
    end
    return adj
end

@save "brain_snapshot.bson" trajectory
