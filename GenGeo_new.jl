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
# Define the Living Simplex Tensor (Geometric Brain Core)
# ============================================================================

mutable struct LivingSimplexTensor
    real::Vector{Float32}             # Current real field (ρ_real)
    imag::Vector{Float32}             # Current imagined field (ρ_imag)
    prev_real::Vector{Float32}         # Previous real field (for λ computation)
    curvature::Vector{Float32}         # Current Ricci/Dirichlet curvature at nodes
    curvature_memory::Vector{Float32}  # Moving average of past curvatures
    adjacency::SparseMatrixCSC{Float32, Int} # Adjacency graph (sparse)
    shape_tensor::Dict{Tuple{Int, Int}, Float32}  # Tension memory
end

"Initialize a LivingSimplexTensor from real field and adjacency."
function init_living_field_simplex(real::Vector{Float32}, adjacency::Matrix{Float32})
    sparse_adj = sparse(adjacency)  # Convert to SparseMatrixCSC
    return init_living_simplex_tensor(real, sparse_adj)
end

function init_living_simplex_tensor(real::Vector{Float32}, adjacency::SparseMatrixCSC{Float32, Int})
    n = length(real)
    imag = zeros(Float32, n)
    prev_real = copy(real)
    curvature = zeros(Float32, n)
    curvature_memory = zeros(Float32, n)

    shape_tensor = Dict{Tuple{Int, Int}, Float32}()
    for i in 1:n
        for j in findnz(adjacency[i, :])[2]
            if i < j
                shape_tensor[(i, j)] = 0.0f0
            end
        end
    end

    return LivingSimplexTensor(
        real, imag, prev_real, curvature, curvature_memory, adjacency, shape_tensor
    )
end

function update_simplex_tensor!(simplex::LivingSimplexTensor, flat_pos, pf_states, edges)
    geo = build_GeoGraphStruct(simplex.real, pf_states, flat_pos, edges)
    simplex.curvature .= geo.R_vals

    α = 0.9f0
    simplex.curvature_memory .= α .* simplex.curvature_memory .+ (1.0f0 - α) .* simplex.curvature

    idx_map = Dict(s => i for (i, s) in enumerate(pf_states))
    for (u, v) in edges
        i, j = idx_map[u], idx_map[v]
        key = (min(i, j), max(i, j))
        tension_update = abs(simplex.real[i] - simplex.real[j])
        simplex.shape_tensor[key] = 0.9f0 * get(simplex.shape_tensor, key, 0.0f0) + 0.1f0 * tension_update
    end
end

"Compute lambda divergence between real and imagined fields."
function compute_lambda(real::Vector{Float32}, imag::Vector{Float32})
    return mean(abs.(real .- imag))
end

"Update GeoBrain memory based on the latest experience."
function update_geobrain!(geobrain::GeoBrainMemory,
                           prior_signature::Vector{Float32},
                           imagined_update::Vector{Float32},
                           lambda_divergence::Float32;
                           lambda_threshold=0.2f0)

    if lambda_divergence < lambda_threshold
        # Good prediction: Store it
        push!(geobrain.memory_keys, copy(prior_signature))
        push!(geobrain.memory_values, copy(imagined_update))
        push!(geobrain.memory_scores, 1.0f0 - lambda_divergence)  # The smaller divergence, the stronger the score
    end
end

"Recall a past imagined update from the GeoBrain if close prior is found."
function recall_from_geobrain(
    geobrain::GeoBrainMemory,
    current_prior::Vector{Float32};
    threshold::Float32 = 0.2f0
)
    if isempty(geobrain.memory_keys)
        return nothing
    end

    best_match_idx = nothing
    best_divergence = Inf

    for (i, stored_prior) in enumerate(geobrain.memory_keys)
        divergence = mean(abs.(current_prior .- stored_prior))
        if divergence < threshold && divergence < best_divergence
            best_match_idx = i
            best_divergence = divergence
        end
    end

    if isnothing(best_match_idx)
        return nothing
    else
        return geobrain.memory_values[best_match_idx]
    end
end

"Build the geometric structure of the imagined manifold from the field vector."
function build_imaginary_geometry(imag::Vector{Float32}, pf_states, flat_pos, edges)
    return build_GeoGraphStruct(imag, pf_states, flat_pos, edges)
end
"Single evolution step of a candidate imagination on the manifold."
function evolve_imagination_single!(
    imag::Vector{Float32},
    pf_states::Vector{String},
    flat_pos::Dict{String, Tuple{Float64, Float64}},
    edges::Vector{Tuple{String, String}},
    moves::Int
)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = Dict(s => Int[] for s in pf_states)
    for (u, v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end

    counts = round.(Int, imag .* 100)
    counts[end] = 100 - sum(counts[1:end-1])
    imag .= Float32.(counts) ./ 100

    # 🧠 Build the temporary imagined manifold
    geo = build_imaginary_geometry(imag, pf_states, flat_pos, edges)

    points3D = geo.points3D
    R_vals = geo.R_vals
    anisotropy_vals = geo.anisotropy

    inflow = zeros(Float32, length(pf_states))
    outflow = zeros(Float32, length(pf_states))

    candidate_indices = findall(x -> x > 0, counts)

    for _ in 1:moves
        isempty(candidate_indices) && break
        i = findmin(R_vals[candidate_indices])[2]
        i = candidate_indices[i]
        s = pf_states[i]
        nbrs = neighbor_indices[s]
        isempty(nbrs) && continue

        best_j = argmin([compute_entropy_cost(i, j, R_vals, pf_states) for j in nbrs])
        chosen = nbrs[best_j]

        inflow[chosen] += 0.01f0
        outflow[i] += 0.01f0
    end

    for i in eachindex(pf_states)
        inflow[i] += (counts[i] / 100.0f0) - outflow[i]
    end

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
    imag .= inflow
end

"Generate multiple imagination candidates based on structured and random strategies."
function structured_imagination_candidates(
    imag::Vector{Float32},
    pf_states::Vector{String},
    flat_pos::Dict{String, Tuple{Float64, Float64}},
    edges::Vector{Tuple{String, String}},
    max_moves::Int
)
    move_fractions = [0.0, 1.0, 0.5]
    n_random_candidates = 2

    candidates = Vector{Vector{Float32}}()

    # --- Structured candidates (no memory, logical pathways)
    for fraction in move_fractions
        candidate = copy(imag)
        moves = Int(round(fraction * max_moves))
        evolve_imagination_single!(candidate, pf_states, flat_pos, edges, moves)
        push!(candidates, candidate)
    end

    # --- Random exploration candidates (noisy imagination)
    for _ in 1:n_random_candidates
        candidate = copy(imag)
        moves = rand(1:max_moves)
        evolve_imagination_single!(candidate, pf_states, flat_pos, edges, moves)
        push!(candidates, candidate)
    end

    return candidates
end

"Use GeoBrain recall + structured imagination to evolve the imagination field."
function evolve_imagination_with_geobrain!(
    field::LivingSimplexTensor,
    pf_states::Vector{String},
    flat_pos::Dict{String, Tuple{Float64, Float64}},
    edges::Vector{Tuple{String, String}},
    geobrain::GeoBrainMemory;
    max_moves::Int = 10
)
    prior = copy(field.imag)
    recalled = recall_from_geobrain(geobrain, prior; threshold=0.2f0)

    candidates = Vector{Vector{Float32}}()

    if isnothing(recalled)
        # No memory found → structured + random imaginations
        candidates = structured_imagination_candidates(prior, pf_states, flat_pos, edges, max_moves)
    else
        # Memory found → start from memory and generate 2 new randoms
        push!(candidates, recalled)
        for _ in 1:2
            candidate = copy(prior)
            moves = rand(1:max_moves)
            evolve_imagination_single!(candidate, pf_states, flat_pos, edges, moves)
            push!(candidates, candidate)
        end
    end

    # Choose best imagined update according to lambda divergence
    best_candidate = candidates[1]
    best_lambda = compute_lambda(field.real, best_candidate)

    for candidate in candidates[2:end]
        λ = compute_lambda(field.real, candidate)
        if λ < best_lambda
            best_lambda = λ
            best_candidate = candidate
        end
    end

    # Update the imagined field
    copy!(field.imag, best_candidate)
end

# ============================================================================
# Run the Living GeoBrain Rollout (Core Evolution Loop)
# ============================================================================

# --- Generate config ---
batch_id, samples = generate_safe_random_initials(1000)
samples_fixed = [(s.uuid, s.rho) for s in samples]

@kwdef struct LivingFlowConfig2
    pf_states::Vector{String}
    flat_pos::Dict{String, Tuple{Float64, Float64}}
    edges::Vector{Tuple{String, String}}
    initials::Vector{Tuple{UUID, Vector{Float32}}}
    batch_size::Int
    rollout_steps::Int
    model_id::String
    max_moves_real::Int
    max_moves_imag::Int
end

config = LivingFlowConfig2(
    pf_states = pf_states,
    flat_pos = flat_pos,
    edges = edges,
    initials = samples_fixed,
    batch_size = 1000,
    rollout_steps = 100,
    model_id = batch_id,
    max_moves_real = 10,
    max_moves_imag = 10
)
