# ============================================================================
# Installations and Imports
# ============================================================================
using GeometryBasics: Point2, Point3
using Interpolations
using Random
using StatsBase: sample, Weights
using LinearAlgebra
using Statistics
using ComplexityMeasures
using Distributions
using Distances
using DelimitedFiles
using Flux
using Flux:softmax
using Flux.Losses: mse
using Optimisers
using BSON
using BSON: @save, @load
using CUDA

# Set device dynamically: use GPU if available, otherwise CPU
device(x) = CUDA.functional() ? gpu(x) : x

# ============================================================================
# Manifold and Geometry Functions
# ============================================================================
# === Proteoform Setup ===
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

# --- Geometry Update Functions ---
function lift_to_z_plane(rho, pf_states, flat_pos)
    return [Point3(flat_pos[s][1], flat_pos[s][2], -rho[i]) for (i, s) in enumerate(pf_states)]
end

function compute_R(points3D, flat_pos, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    R = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        p0 = flat_pos[s]; p3 = points3D[i]
        neighbors = [v for (u, v) in edges if u == s]
        append!(neighbors, [u for (u, v) in edges if v == s])
        for n in neighbors
            j = idx[n]; q0 = flat_pos[n]; q3 = points3D[j]
            d0 = norm([p0[1] - q0[1], p0[2] - q0[2]])
            d3 = norm(p3 - q3)
            R[i] += d3 - d0
        end
    end
    return R
end

function compute_c_ricci_dirichlet(R, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    C_R = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        neighbors = [v for (u, v) in edges if u == s]
        append!(neighbors, [u for (u, v) in edges if v == s])
        for n in neighbors
            j = idx[n]
            C_R[i] += (R[i] - R[j])^2
        end
    end
    return C_R
end

function compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = Dict(s => Int[] for s in pf_states)
    for (u, v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end
    anisotropy = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        nbrs = neighbor_indices[s]
        grad_vals = Float64[]
        for j in nbrs
            dist = norm([
                flat_pos[s][1] - flat_pos[pf_states[j]][1],
                flat_pos[s][2] - flat_pos[pf_states[j]][2]
            ])
            if dist > 1e-6
                push!(grad_vals, abs(C_R_vals[i] - C_R_vals[j]) / dist)
            end
        end
        anisotropy[i] = isempty(grad_vals) ? 0.0 : sum(grad_vals) / length(grad_vals)
    end
    return anisotropy
end

function update_geometry_from_rho(ρ, pf_states, flat_pos, edges)
    points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
    R_vals = compute_R(points3D, flat_pos, pf_states, edges)
    C_R_vals = compute_c_ricci_dirichlet(R_vals, pf_states, edges)
    anisotropy_vals = compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)
    return points3D, R_vals, C_R_vals, anisotropy_vals
end

# --- Sheaf Setup Functions ---
function initialize_sheaf_stalks(flat_pos, pf_states)
    stalks = Dict{String, Vector{Float64}}()
    for s in pf_states
        stalks[s] = [flat_pos[s][1], flat_pos[s][2]]
    end
    return stalks
end

function sheaf_consistency(stalks, edges; threshold=2.5)
    inconsistencies = []
    for (u, v) in edges
        diff = norm(stalks[u] .- stalks[v])
        if diff > threshold
            push!(inconsistencies, (u, v, diff))
        end
    end
    return inconsistencies
end

# --- Entropy Cost Function (for transitions) ---
function compute_entropy_cost(i, j, C_R_vals, pf_states)
    baseline_DeltaE = 1.0
    mass_heat = 0.1
    reaction_heat = 0.01 * baseline_DeltaE
    conformational_cost = abs(C_R_vals[j])
    degeneracy_map = Dict(0 => 1, 1 => 3, 2 => 3, 3 => 1)
    deg = degeneracy_map[count(c -> c == '1', pf_states[j])]
    degeneracy_penalty = 1.0 / deg
    return mass_heat + reaction_heat + conformational_cost + degeneracy_penalty
end

# --- Alive Function ---
function create_dataset_ODE_alive(; t_span=nothing, max_samples=1000, save_every=250)
    # If t_span is not provided, create a default range (here T is number of time steps)
    if isnothing(t_span)
        t_span = 1:100
    end
    T = length(t_span)
    
    X, Y, Metadata, Geos = [], [], [], []
    geo_counter = Dict{Vector{String}, Int}()
    initials = generate_systematic_initials(num_total=50)
    
    total_samples = 0

    for vec in initials
        # For each systematic initial condition, generate a number of variants if needed.
        for _ in 1:10
            rho0 = vec
            # Run the evolution while capturing metadata at each time step:
            ρ_series, trajectory, geo, global_metadata = evolve_time_series_metadata!(rho0, T, pf_states, flat_pos, edges; max_moves_per_step=10)
            
            # Use the initial and final i-state as inputs/outputs (or modify as needed)
            final_rho = ρ_series[end]
            final_rho = final_rho ./ sum(final_rho)  # ensure normalization
            
            push!(X, rho0)
            push!(Y, final_rho)
            push!(Metadata, global_metadata)
            
            if !isnothing(geo)
                geo_counter[geo] = get(geo_counter, geo, 0) + 1
                push!(Geos, geo)
            end
            
            total_samples += 1
            
            if total_samples % save_every == 0
                println("Checkpoint: $total_samples samples generated...")
                # Save interim dataset checkpoint using BSON (or CSV)
                @save "checkpoint_$total_samples.bson" X Y Metadata Geos
            end
            
            if total_samples >= max_samples
                break
            end
        end
        
        if total_samples >= max_samples
            break
        end
    end
    
    println("Finished generating dataset with $total_samples samples.")
    println("Most traversed geodesics:")
    for (path, count) in sort(collect(geo_counter), by=x -> x[2], rev=true)
        println(join(path, " → "), " | Count: ", count)
    end
    
    # Save final dataset
    @save "final_dataset.bson" X Y Metadata Geos
    
    return X, Y, Metadata, Geos
end


# ============================================================================
# Geodesics and Tracking Functions
# ============================================================================
geodesics = [
    ["000", "100", "101", "111"],
    ["000", "100", "110", "111"],
    ["000", "010", "110", "111"],
    ["000", "010", "011", "111"],
    ["000", "001", "101", "111"],
    ["000", "001", "011", "111"]
]

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

function evolve_time_series_and_geodesic!(ρ0::Vector{Float64}, T::Int, pf_states, flat_pos, edges; max_moves_per_step=10)
    ρ = copy(ρ0)
    trajectory = String[]
    ρ_series = [copy(ρ)]
    for t in 1:T
        oxi_shapes_alive!(ρ, pf_states, flat_pos, edges; max_moves=max_moves_per_step)
        push!(ρ_series, copy(ρ))
        push!(trajectory, pf_states[argmax(ρ)])
    end
    geo = dominant_geodesic(trajectory, geodesics)
    return ρ_series, trajectory, geo
end

# ============================================================================
# Metadata Helper Functions
# ============================================================================
function compute_k_distribution(ρ, pf_states)
    k_counts = Dict{Int, Float64}()
    for (i, s) in enumerate(pf_states)
        k = count(c -> c == '1', s)
        k_counts[k] = get(k_counts, k, 0.0) + ρ[i]
    end
    return k_counts
end

function weighted_mean_oxidation(k_dist::Dict{Int, Float64})
    total = 0.0
    for (k, p) in k_dist
        total += k * p
    end
    return total
end

function shannon_entropy(p::Vector{Float64})
    p_nonzero = filter(x -> x > 0, p)
    return -sum(x -> x * log2(x), p_nonzero)
end

function fisher_information(ρ::Vector{Float64})
    grad = diff(ρ)
    return sum(grad .^ 2)
end

function lyapunov_exponent(series::Vector{Vector{Float64}})
    exponents = Float64[]
    for t in 2:length(series)
        norm_prev = norm(series[t-1])
        norm_diff = norm(series[t] - series[t-1])
        if norm_prev > 0 && norm_diff > 0
            push!(exponents, log(norm_diff / norm_prev))
        end
    end
    return mean(exponents)
end

function evolve_time_series_metadata!(ρ0::Vector{Float64}, T::Int, pf_states, flat_pos, edges; max_moves_per_step=10)
    ρ = copy(ρ0)
    ρ_series = [copy(ρ)]
    metadata = Vector{Dict{String,Any}}(undef, T)
    trajectory = String[]
    for t in 1:T
        ρ_old = copy(ρ)
        oxi_shapes_alive!(ρ, pf_states, flat_pos, edges; max_moves=max_moves_per_step)
        push!(ρ_series, copy(ρ))
        push!(trajectory, pf_states[argmax(ρ)])
        flux = ρ .- ρ_old
        k_dist = compute_k_distribution(ρ, pf_states)
        mean_oxidation = weighted_mean_oxidation(k_dist)
        entropy = shannon_entropy(ρ)
        fisher_info = fisher_information(ρ)
        # You can also grab current c-Ricci values if needed:
        _, R_vals, C_R_vals, _ = update_geometry_from_rho(ρ, pf_states, flat_pos, edges)
        metadata[t] = Dict(
            "i_state"          => copy(ρ),
            "k_state"          => deepcopy(k_dist),
            "flux"             => copy(flux),
            "c_Ricci"          => copy(C_R_vals),
            "mean_oxidation"   => mean_oxidation,
            "shannon_entropy"  => entropy,
            "fisher_information" => fisher_info,
        )
    end
    geo = dominant_geodesic(trajectory, geodesics)
    lyap = lyapunov_exponent(ρ_series)
    global_metadata = Dict(
        "initial_i_state" => copy(ρ0),
        "final_i_state"   => copy(ρ),
        "i_state_series"  => ρ_series,
        "step_metadata"   => metadata,
        "dominant_geodesic" => geo,
        "lyapunov_exponent" => lyap,
    )
    return ρ_series, trajectory, geo, global_metadata
end

# ============================================================================
# Dataset Generation Helper: Systematic Initial Conditions
# ============================================================================

# Function to generate systematic initial conditions
function generate_systematic_initials(; num_total=100, min_val=0.01)
    initials = []

    # (1) Single i-state occupancy
    for i in 1:8
        vec = zeros(8)
        vec[i] = 1.0
        push!(initials, vec)
    end

    # (2) Flat occupancy
    push!(initials, fill(1.0 / 8, 8))

    # (3) Curved within k=1 (e.g., 010 peak)
    curved_k1 = [0.0, 0.15, 0.7, 0.15, 0.0, 0.0, 0.0, 0.0]
    push!(initials, curved_k1 / sum(curved_k1))

    # (4) Curved within k=2 (e.g., 101 peak)
    curved_k2 = [0.0, 0.0, 0.0, 0.0, 0.15, 0.7, 0.15, 0.0]
    push!(initials, curved_k2 / sum(curved_k2))

    # (5) Flat in k=0 & k=1, peaked in k=2
    hybrid = [0.05, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.05]
    push!(initials, hybrid / sum(hybrid))

    # (6) Bell shape across k
    bell = [0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.2]
    push!(initials, bell / sum(bell))

    # (7) Geometric gradient (left to right in flat_pos)
    gradient = collect(range(0.1, stop=0.9, length=8))
    push!(initials, gradient / sum(gradient))

    # (8) Add safe random distributions
    function generate_safe_random_initials(n::Int)
        safe = []
        while length(safe) < n
            vec = rand(8)
            vec ./= sum(vec)
            if all(x -> x >= min_val || isapprox(x, 0.0; atol=1e-8), vec)
                push!(safe, vec)
            end
        end
        return safe
    end

    num_random = num_total - length(initials)
    append!(initials, generate_safe_random_initials(num_random))

    return initials
end

# ============================================================================
# 2. Define the Data Generation Function (with Metadata Logging)
# ============================================================================

# We assume that the following functions are already defined:
# - oxi_shapes_alive!(ρ, pf_states, flat_pos, edges; max_moves=10)
# - update_geometry_from_rho(ρ, pf_states, flat_pos, edges)
# - dominant_geodesic(trajectory, geodesics)
# - evolve_time_series_metadata!(ρ0, T, pf_states, flat_pos, edges; max_moves_per_step=10)
#
# Also, the metadata helper functions such as:
# compute_k_distribution, weighted_mean_oxidation, shannon_entropy,
# fisher_information, lyapunov_exponent, etc.

function create_dataset_ODE_alive(; t_span=nothing, max_samples=1000, save_every=250)
    # If t_span is not provided, create a default range (here T is number of time steps)
    if isnothing(t_span)
        t_span = 1:100
    end
    T = length(t_span)

    X, Y, Metadata, Geos = [], [], [], []
    geo_counter = Dict{Vector{String}, Int}()
    initials = generate_systematic_initials(num_total=50)

    total_samples = 0

    for vec in initials
        # For each systematic initial condition, generate a number of variants if needed.
        # You can change the inner loop count if you want to perturb the initial condition multiple times.
        for _ in 1:10
            rho0 = vec
            # Run the evolution while capturing metadata at each time step:
            ρ_series, trajectory, geo, global_metadata = evolve_time_series_metadata!(rho0, T, pf_states, flat_pos, edges; max_moves_per_step=10)
            
            # Use the initial and final i-state as inputs/outputs (or modify as needed)
            final_rho = ρ_series[end]
            # Re-normalize final_rho if necessary
            final_rho = final_rho ./ sum(final_rho)
            
            push!(X, rho0)
            push!(Y, final_rho)
            push!(Metadata, global_metadata)
            
            if !isnothing(geo)
                geo_counter[geo] = get(geo_counter, geo, 0) + 1
                push!(Geos, geo)
            end
            
            total_samples += 1
            
            if total_samples % save_every == 500
                println("Checkpoint: $total_samples samples generated...")
                # Optionally save interim datasets to CSV or BSON
                # For example:
                # writedlm("X_partial_$total_samples.csv", X, ',')
                # writedlm("Y_partial_$total_samples.csv", Y, ',')
            end
            
            if total_samples >= max_samples
                break
            end
        end
        
        if total_samples >= max_samples
            break
        end
    end
    
    println("Finished generating dataset with $total_samples samples.")
    println("Most traversed geodesics:")
    for (path, count) in sort(collect(geo_counter), by=x->x[2], rev=true)
        println(join(path, " → "), " | Count: ", count)
    end
    
    return X, Y, Metadata, Geos
end

# ============================================================================
# Example: Generate the Dataset
# ============================================================================

println("Simulating Oxi-Shape evolution with metadata...")
X, Y, metadata, geos = create_dataset_ODE_alive(t_span=1:100, max_samples=500, save_every=100)

# You can now inspect or save the dataset:
println("Number of samples in X: ", length(X))
println("Number of metadata records: ", length(metadata))

# ============================================================================
# Data Generation and Training Calls
# ============================================================================
# Example: Generate a dataset using our metadata evolution
println("Simulating with metadata recording...")
ρ0 = [1.0/8 for _ in 1:8]  # initial flat occupancy
T = 100  # number of time steps

ρ_series, trajectory, geo, global_metadata = evolve_time_series_metadata!(ρ0, T, pf_states, flat_pos, edges; max_moves_per_step=10)

println("Finished generating dataset with metadata")
@save "final_dataset.bson" X Y metadata geos

# ============================================================================
# Graph RNN 
# ============================================================================

struct GraphRNNCell
    net::Chain
end
Flux.@functor GraphRNNCell

function (cell::GraphRNNCell)(state::Vector{Float64})
    _, _, C_R_vals, _ = update_geometry_from_rho(state, pf_states, flat_pos, edges)

    # Only use C-Ricci mean
    feat = Float32.(vcat(state, [mean(C_R_vals)]))

    delta = cell.net(feat)
    state_phys = copy(state)
    oxi_shapes_alive!(state_phys, pf_states, flat_pos, edges; max_moves=10)

    new_state = state_phys .+ delta
    return softmax(new_state)
end

# ============================================================================
# Define the Neural Network
# ============================================================================
input_dim = 9  # i-state + mean c-Ricci only
hidden_dim = 32
output_dim = 8

net = Chain(
    Dense(input_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, output_dim)
)

cell = GraphRNNCell(net)

# ============================================================================
# Rollout graph RNN
# ============================================================================

function GraphRNN(initial_state::AbstractVector{<:Real}, cell::GraphRNNCell, T::Int)
    states = Vector{Vector{Float64}}(undef, T+1)
    states[1] = Float64.(initial_state)
    current_state = Float64.(initial_state)
    for t in 1:T
        current_state = cell(current_state)
        states[t+1] = current_state
    end
    return states
end

# ============================================================================
# Training
# ============================================================================

@load "final_dataset.bson" X Y metadata geos

T_steps = 100  # number of time steps for unrolling

# Optimizer and parameters
opt = Flux.ADAM(1e-4)
params = Flux.params(cell)

# Training loop
epochs = 50
for epoch in 1:epochs
    total_loss = 0.0
    for (x, y) in zip(X, Y)
        x_f32 = Float32.(x)
        y_f32 = Float32.(y)
        
        l, back = Flux.withgradient() do
            loss_fn(x_f32, y_f32, cell, T_steps)
        end
        total_loss += l
        Flux.Optimise.update!(opt, params, back())
    end
    println("Epoch $epoch: Loss = $(round(total_loss / length(X), digits=6))")
end

# ============================================================================
# Inference / Evaluation
# ============================================================================

# Choose a test initial state (e.g., flat or custom)
test_state = Float32[1.0/8 for _ in 1:8]

# Run the trained GraphRNN for T steps
predicted_series = GraphRNN(test_state, cell, T_steps)
predicted_final = predicted_series[end]

# Normalize (optional but usually good)
predicted_final ./= sum(predicted_final)

# Print predictions
println("\n--- Inference Result ---")
println("Initial State: ", round.(test_state, digits=3))
println("Predicted Final i-State: ", round.(predicted_final, digits=3))

# Optional: Compute metrics
function percent_oxidation(ρ::AbstractVector{<:Real})
    oxidation_levels = [(count(==('1'), s) / 3) * 100 for s in pf_states]
    return sum(ρ .* oxidation_levels)
end

println("Predicted Percent Oxidation: ", round(percent_oxidation(predicted_final), digits=2))
println("Shannon Entropy: ", round(shannon_entropy(predicted_final), digits=3))
