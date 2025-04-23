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
using Flux
using Flux.Optimise: update!
using BSON: @save
using Dates
using Zygote: @nograd

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

function compute_anisotropy_pure(
    C_R_vals::Vector{Float32},
    pf_states::Vector{String},
    flat_pos::Dict{String, Tuple{Float64, Float64}},
    edges::Vector{Tuple{String, String}}
)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = build_neighbor_indices(pf_states, edges, idx)

    # Compute anisotropy values using vectorized / functional style
    anisotropy = map(pf_states) do s
        i = idx[s]
        grads = map(neighbor_indices[s]) do j
            dx = Float32(flat_pos[s][1] - flat_pos[pf_states[j]][1])
            dy = Float32(flat_pos[s][2] - flat_pos[pf_states[j]][2])
            dist = sqrt(dx^2 + dy^2)
            dist > 1f-6 ? abs(C_R_vals[i] - C_R_vals[j]) / dist : 0.0f0
        end
        grads_nonzero = filter(x -> x > 0, grads)
        isempty(grads_nonzero) ? 0.0f0 : mean(grads_nonzero)
    end

    return collect(anisotropy)  # Convert from generator to Vector
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
    anisotropy = compute_anisotropy_pure(C_R_vals, pf_states, flat_pos, edges)

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
    anisotropy_vals = compute_anisotropy_pure(C_R_vals, pf_states, flat_pos, edges)

    return points3D, R_vals, C_R_vals, anisotropy_vals
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

function compute_entropy_cost(i::Int, j::Int, C_R_vals::Vector{Float32}, pf_states::Vector{String})
    baseline_DeltaE = 1.0f0
    mass_heat = 0.1f0
    reaction_heat = 0.01f0 * baseline_DeltaE
    conformational_cost = abs(C_R_vals[j])

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
            ΔS = compute_entropy_cost(i, j, C_R_vals, pf_states)
            Δf = exp(ρ[i]) - C_R_vals[i] + ΔS
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

# none of our geometry/sheaf building routines should be
# autodiff’d by Zygote — they only depend on ρ, never on model parameters.
@nograd lift_to_z_plane
@nograd compute_R
@nograd compute_c_ricci_dirichlet
@nograd build_neighbor_indices
@nograd compute_anisotropy_pure
@nograd initialize_sheaf_stalks
@nograd sheaf_consistency
@nograd build_GeoGraphStruct
@nograd update_geometry_from_rho
@nograd oxi_shapes_alive!

# ============================================================================
# Define the IMAGINARY Model & Complex functions
# ============================================================================
# ============================================================================
# Define the IMAGINARY Model & Differentiable Complex Flow
# ============================================================================

struct ComplexField
    real::Vector{Float32}
    imag::Vector{Float32}
    memory::Vector{Float32}
end

function init_complex_field_from_graph(real::Vector{Float32}, geo::GeoGraphStruct)
    degree_vec = vec(sum(geo.adjacency, dims=2))
    memory = degree_vec ./ sum(degree_vec)
    imag = zeros(Float32, length(real))
    return ComplexField(real, imag, memory)
end

"Compute normalized recall weights from memory + curvature"
function recall_weights(field::ComplexField, C_R_vals::Vector{Float32})
    W = field.memory .+ C_R_vals
    W = exp.(W .- maximum(W))
    return W ./ sum(W)
end

"β is derived from curvature — no arbitrary constants"
function beta_from_c_ricci(C_R_vals::Vector{Float32})
    return mean(C_R_vals)  # or std(C_R_vals) for sharper slope
end

"λ is a flow-dependent occupancy gate"
function dynamic_lambda(field::ComplexField, C_R_vals::Vector{Float32})
    β = beta_from_c_ricci(C_R_vals)
    divergence = norm(field.imag .- field.memory)
    return 1.0f0 / (1.0f0 + exp(β * divergence))
end

"Compute updated imaginary field (epistemic learning step)"
function updated_imaginary_field(field::ComplexField, geo::GeoGraphStruct)
    _, _, C_R_vals, _ = update_geometry_from_rho(field.real, geo)
    λ = dynamic_lambda(field, C_R_vals)
    W = recall_weights(field, C_R_vals)
    imag_new = (1 - λ) .* field.imag .+ λ .* W
    imag_new = max.(imag_new, 0.0f0)
    imag_new ./= sum(imag_new)
    return imag_new, λ
end

"Update memory as geometric trace of real field"
function updated_memory_field(field::ComplexField)
    mem = (field.memory .+ field.real) ./ 2f0
    mem ./= sum(mem)
    return mem
end

"Evolve the full complex field (real stays externally updated)"
function evolve_complex_field(field::ComplexField, geo::GeoGraphStruct)
    imag_new, λ = updated_imaginary_field(field, geo)
    mem_new = updated_memory_field(field)
    return ComplexField(field.real, imag_new, mem_new), λ
end

# ============================================================================
# Define the "Flow" Path Trace Module
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

@kwdef struct PathTrace
    run_id::String
    step::Int
    i_state::Vector{Float32}
    k_state::Dict{Int, Float32}
    flux::Vector{Float32}
    c_Ricci::Vector{Float32}
    mean_oxidation::Float32
    shannon_entropy::Float32
    fisher_information::Float32
    transition_class::Symbol
    on_geodesic::Bool
end

trace = PathTrace(i_state=ρ, transition_class=:oxidizing)

function record_path_trajectory_struct!(
    ρ0::Vector{Float32}, T::Int, pf_states, flat_pos, edges;
    max_moves_per_step=10, run_id::String="default_run"
)
    ρ = copy(ρ0)
    ρ_series = [copy(ρ)]
    metadata = Vector{PathTrace}(undef, T)
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
        k_dist = compute_k_distribution(ρ, pf_states)
        mean_oxidation = Float32(weighted_mean_oxidation(k_dist))
        entropy = Float32(shannon_entropy(ρ))
        fisher_info = Float32(fisher_information(ρ))
        cumulative_entropy += entropy

        _, _, C_R_vals, _ = update_geometry_from_rho(ρ, pf_states, flat_pos, edges)
        c_R_vec = Float32.(C_R_vals)

        class = classify_transition(ρ_old, ρ)
        class == :oxidizing && (ox_moves += 1)
        class == :reducing && (red_moves += 1)
        total_moves += 1

        metadata[t] = PathTrace(
            run_id=run_id,
            step=t,
            i_state=copy(ρ),
            k_state=deepcopy(k_dist),
            flux=copy(flux),
            c_Ricci=copy(c_R_vec),
            mean_oxidation=mean_oxidation,
            shannon_entropy=entropy,
            fisher_information=fisher_info,
            transition_class=class,
            on_geodesic=false  # will update later
        )
    end

    # === Post-process dominant geodesic label
    geo_path = dominant_geodesic(trajectory, geodesics)
    for (t, state) in enumerate(trajectory)
        metadata[t].on_geodesic = state in geo_path
    end

    lyap = Float32(lyapunov_exponent(ρ_series))
    action_cost = cumulative_entropy / Float32(total_moves)

    global_metadata = Dict(
        "run_id" => run_id,
        "initial_i_state" => copy(ρ0),
        "final_i_state" => copy(ρ),
        "i_state_series" => ρ_series,
        "step_metadata" => metadata,
        "dominant_geodesic" => geo_path,
        "lyapunov_exponent" => lyap,
        "total_moves" => total_moves,
        "oxidizing_moves" => ox_moves,
        "reducing_moves" => red_moves,
        "cumulative_entropy" => cumulative_entropy,
        "action_cost" => action_cost
    )

    return ρ_series, trajectory, geo_path, global_metadata
end

function save_path_metadata(global_metadata::Dict{String, Any}; prefix="path_run")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    run_id = get(global_metadata, "run_id", "run")
    filename = "$(prefix)_$(run_id)_$(timestamp).bson"
    BSON.@save filename global_metadata
    println("🧠 Metadata saved to: $filename")
end

# ============================================================================
# Random Initial Generator with UUIDs
# ============================================================================

@kwdef mutable struct GenGeoAlpha
    θ_geo::Vector{Float32}   # per-node geometry weights
    θ_flow::Vector{Float32}  # scalar weight for λ sensitivity
end

function learned_c_ricci_dirichlet(R::Vector{Float32}, pf_states, edges, θ_geo::Vector{Float32})
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    C_R = zeros(Float32, length(pf_states))
    for (i, s) in enumerate(pf_states)
        neighbors = [v for (u, v) in edges if u == s] ∪ [u for (u, v) in edges if v == s]
        for n in neighbors
            j = idx[n]
            C_R[i] += θ_geo[i] * (R[i] - R[j])^2
        end
    end
    return C_R
end

function learned_lambda(field::ComplexField, θ_flow::Vector{Float32})
    divergence = norm(field.imag .- field.memory)
    β = θ_flow[1]
    return 1.0f0 / (1.0f0 + exp(β * divergence))
end

function updated_imag_field(geo::GeoGraphStruct, field::ComplexField, θ_flow)
    λ = learned_lambda(field, θ_flow)
    _, _, C_R_vals, _ = update_geometry_from_rho(field.real, geo)
    W = exp.(field.memory .+ C_R_vals .- maximum(field.memory .+ C_R_vals))
    W ./= sum(W)
    imag = (1 - λ) .* field.imag .+ λ .* W
    imag .= max.(imag, 0.0f0)
    imag ./= sum(imag)
    return imag, λ
end

function evolve_step_loss(ρ0::Vector{Float32}, model::GenGeoAlpha, pf_states, flat_pos, edges)
    geo = build_GeoGraphStruct(ρ0, pf_states, flat_pos, edges)
    points3D, R_vals, _, _ = update_geometry_from_rho(ρ0, geo)
    C_R = learned_c_ricci_dirichlet(R_vals, pf_states, edges, model.θ_geo)

    φ = ComplexField(real=ρ0, imag=zeros(Float32, length(ρ0)), memory=copy(ρ0))
    φ.imag, λ = updated_imag_field(geo, φ, model.θ_flow)

    loss = sum(C_R) + λ * shannon_entropy(φ.imag)
    return loss
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

batch_id, initials = generate_safe_random_initials(100)

# ============================================================================
# λ-Based Dynamic Loss Function
# ============================================================================

function geo_flow_divergence_loss(ρ0, model::GenGeoAlpha, pf_states, flat_pos, edges)
    geo = build_GeoGraphStruct(ρ0, pf_states, flat_pos, edges)
    geo.C_R_vals .= geo.C_R_vals .* model.θ_geo

    φ = ComplexField(real=ρ0, imag=zeros(Float32, length(ρ0)), memory=copy(ρ0))
    λ = dynamic_lambda(φ; β=model.θ_flow[1])  # Loss = epistemic flow divergence

    return λ
end

# ============================================================================
# Training Loop
# ============================================================================

@kwdef struct GeoFlowConfig
    pf_states::Vector{String}
    flat_pos::Dict{String, Tuple{Float64, Float64}}
    edges::Vector{Tuple{String, String}}
    initials::Vector{Vector{Float32}}
    epochs::Int = 100
    batch_size::Int = 10
    max_moves_per_step::Int = 10
    model_id::String = "geo_flow"
end

@kwdef mutable struct GeoFlowTrack
    epoch::Int
    λ::Float32
    θ_geo::Vector{Float32}
    θ_flow::Vector{Float32}
end

function train_geo_flow_model(config::GeoFlowConfig)
    training_trace = GeoFlowTrack[]
    model = GenGeoAlpha(
        θ_geo = ones(Float32, length(config.pf_states)),
        θ_flow = [1.0f0]
    )
    ps = Flux.params(model)
    λ_epoch_history = Float32[]

    for epoch in 1:config.epochs
        λ_vals = Float32[]

        for _ in 1:config.batch_size
            ρ0 = config.initials[rand(1:end)]
            λ_val, back = Flux.withgradient(() ->
                geo_flow_divergence_loss(ρ0, model, config.pf_states, config.flat_pos, config.edges), ps)

            for p in ps
                p .-= back[p]
            end

            push!(λ_vals, λ_val)
        end

        mean_λ = mean(λ_vals)
        push!(λ_epoch_history, mean_λ)

        push!(training_trace, GeoFlowTrack(
            epoch = epoch,
            λ = mean_λ,
            θ_geo = deepcopy(model.θ_geo),
            θ_flow = deepcopy(model.θ_flow)
        ))

        println("Epoch $epoch | λ (mean) = $(round(mean_λ, digits=6))")
    end

    return model, training_trace, λ_epoch_history
end

# ============================================================================
# Save Trained GeoFlow Model
# ============================================================================
model, trace = train_geo_flow_model(config)
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
@save "trained_$(config.model_id)_$timestamp.bson" model trace confi
