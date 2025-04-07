# ╔═╡ Install required packages (only needs to run once per session)
using Pkg
Pkg.activate(".")  # Optional: activate project environment
Pkg.add(["Flux", "CUDA", "Meshes", "GeometryBasics", "LinearAlgebra",
         "StatsBase", "DifferentialEquations", "Ripserer",
         "Distances", "Interpolations", "CairoMakie", "DelimitedFiles", "Distributions", "ComplexityMeasures", "BSON"])

using CairoMakie
using GeometryBasics: Point2, Point3
using Interpolations
using Random
using StatsBase: sample, Weights
using LinearAlgebra
using Ripserer
using Statistics
using ComplexityMeasures
using Distributions
using Distances
using DelimitedFiles
using Flux
using BSON
using BSON: @save  
using CUDA

CairoMakie.activate!()

# Set device dynamically: use GPU if available, otherwise CPU
device(x) = CUDA.functional() ? gpu(x) : x

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

# === Geometry Update ===
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

# === Sheaf Setup ===
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

function oxi_shapes_alive!(ρ, pf_states, flat_pos, edges; max_moves=10)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = Dict(s => Int[] for s in pf_states)
    for (u, v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end

    # Convert to molecule counts
    counts = round.(ρ * 100)
    counts[end] = 100 - sum(counts[1:end-1])
    ρ .= counts / 100

    # Update geometry from current ρ
    points3D, R_vals, C_R_vals, anisotropy_vals = update_geometry_from_rho(ρ, pf_states, flat_pos, edges)

    inflow = zeros(Float64, length(pf_states))
    outflow = zeros(Float64, length(pf_states))

    total_moves = rand(0:max_moves)
    candidate_indices = findall(x -> x > 0, counts)

    for _ in 1:total_moves
        isempty(candidate_indices) && break
        i = rand(candidate_indices)
        s = pf_states[i]
        nbrs = neighbor_indices[s]

        if isempty(nbrs)
            inflow[i] += 0.01
            continue
        end

        probs = Float64[]
        for j in nbrs
            ΔS = compute_entropy_cost(i, j, C_R_vals, pf_states)
            Δf = exp(ρ[i]) - C_R_vals[i] + ΔS
            p = exp(-Δf) * exp(-anisotropy_vals[j])
            push!(probs, p)
        end

        if sum(probs) < 1e-8
            inflow[i] += 0.01
            continue
        end

        probs ./= sum(probs)
        chosen = sample(nbrs, Weights(probs))
        inflow[chosen] += 0.01
        outflow[i] += 0.01
    end

    for i in eachindex(pf_states)
        inflow[i] += (counts[i] / 100.0) - outflow[i]
    end

    ρ .= inflow
end

# === Geodesics and Tracking ===
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

function geodesic_loss(initial::Vector{Float64}, final::Vector{Float64}, pf_states, geodesics)
    pred_path = [pf_states[argmax(initial)], pf_states[argmax(final)]]
    valid = any(all(s in g for s in pred_path) for g in geodesics)
    return valid ? 0.0 : 1.0
end

# Function to compute the persistent homology diagram
function persistent_diagram(rho::Vector{Float64})
    points = reshape(rho, :, 1)  # 8×1 matrix: 8 points in ℝ¹
    dgms = ripserer(points; dim_max=1)
    return dgms
end

# Function to compute the persistent entropy
function topological_entropy(dgm)
    if isempty(dgm) || isempty(dgm[1])
        return 0.0
    end
    lifespans = [pt.death - pt.birth for pt in dgm[1]]
    probs = lifespans ./ sum(lifespans)
    entropy = -sum(probs .* log2.(probs .+ 1e-10))
    return entropy
end

# Function to generate systematic initial conditions
function generate_systematic_initials()
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
    gradient = range(0.1, stop=0.9, length=8)
    push!(initials, gradient / sum(gradient))

    return initials
end

# Function to create dataset using ODE solver
function create_dataset_ODE_alive(; t_span=nothing, max_samples=500, save_every=50)
    if isnothing(t_span)
        t_span = range(0.0, stop=1.0, length=100)
    end

    X, Y, geos = [], [], []
    geo_counter = Dict{Vector{String}, Int}()
    initials = generate_systematic_initials()[1:10]  # Use only the first 10 distributions for speed testing

    total_samples = 0  # Sample counter

    for vec in initials
        for _ in 1:5  # Generate more variants per shape if needed
            rho0 = vec
            rho_t, geopath = evolve_time_series_and_geodesic!(rho0, length(t_span), pf_states, flat_pos, edges)
            final_rho = rho_t[end]

            # Digital enforcement
            @assert all(isapprox.(final_rho * 100, round.(final_rho * 100), atol=1e-6)) "Non-digital occupancy detected: $final_rho"

            push!(X, rho0)
            push!(Y, final_rho)

            if !isnothing(geopath)
                geo_counter[geopath] = get(geo_counter, geopath, 0) + 1
                push!(geos, geopath)
            end

            total_samples += 1  # Update sample count

            # Save intermediate files every save_every samples
            if total_samples % save_every == 0
                println("Saving checkpoint at $total_samples samples...")
                writedlm("X_partial_$total_samples.csv", X, ',')
                writedlm("Y_partial_$total_samples.csv", Y, ',')

            end
        end

        if total_samples >= max_samples
            break
        end
    end

    println("Finished data generation.")
    println("Most traversed geodesics:")
    for (path, count) in sort(collect(geo_counter), by=x->x[2], rev=true)
        println(join(path, " → "), " | Count: ", count)
    end

    return X, Y, geos
end

# Define the model
struct OxiNet
    fc1::Dense
    fc2::Dense
    fc3::Dense
end

# Constructor for OxiNet
function OxiNet(input_dim::Int=8, hidden_dim::Int=32, output_dim::Int=8)
    fc1 = Dense(input_dim, hidden_dim, relu)
    fc2 = Dense(hidden_dim, hidden_dim, relu)
    fc3 = Dense(hidden_dim, output_dim)  # No activation (raw logits)
    return OxiNet(fc1, fc2, fc3)
end

# Define the forward pass
(m::OxiNet)(x) = m.fc3(m.fc2(m.fc1(x)))

# === Geodesic Loss (no weights) ===
function geodesic_loss(predicted_final::Matrix{Float32}, initial::Matrix{Float32}, pf_states, geodesics)
    batch_size = size(predicted_final, 2)
    loss = 0.0
    for i in 1:batch_size
        start_idx = argmax(initial[:, i])
        end_idx = argmax(predicted_final[:, i])
        pred_path = [pf_states[start_idx], pf_states[end_idx]]
        valid = any(all(x -> x ∈ g, pred_path) for g in geodesics)
        loss += valid ? 0.0 : 1.0
    end
    return loss / batch_size
end

# === Training Function ===
function train_model(model, X_train, Y_train, X_val, Y_val; epochs=100, lr=1e-3, geodesics, pf_states)
    # Initialize optimizer with learning rate
    opt = Optimisers.setup(Optimisers.Adam(lr), model)

    for epoch in 1:epochs
        # Compute loss and gradients using the modern pattern
        loss, grads = Flux.withgradient(model) do m
            raw_pred = m(X_train)  # Forward pass
            pred = round.(raw_pred .* 100) ./ 100  # Optional: rounding for stability
            pred = pred ./ sum(pred; dims=1)  # Normalize predictions

            # Log geodesic loss separately
            println("Epoch $epoch - Geodesic Loss: ", geodesic_loss(pred, X_train, pf_states, geodesics))

            return Flux.Losses.mse(pred, Y_train)  # Main loss
        end

        # Update model parameters and optimizer state
        opt, model = Optimisers.update(opt, model, grads[1])
    end

    return model  # Return the trained model
end


# === Evaluation Function ===
function evaluate_model(model, X_val_mat, Y_val_mat)
    raw_pred = model(X_val_mat)
    pred = round.(raw_pred .* 100) ./ 100
    pred ./= sum(pred; dims=1)
    loss = Flux.Losses.mse(pred, Y_val_mat)
    return pred, loss
end

println("Generating dataset using systematic Oxi-Shape sampling...")
t_span = range(0.0, stop=1.0, length=100)
X, Y, geos = create_dataset_ODE_alive(t_span=t_span)

# Shuffle and split the dataset
perm = shuffle(1:length(X))
X = X[perm]
Y = Y[perm]

# Convert individual vectors to Float32
X = [Float32.(x) for x in X]
Y = [Float32.(y) for y in Y]

split = Int(round(0.8 * length(X)))
X_train, Y_train = X[1:split], Y[1:split]
X_val, Y_val = X[split+1:end], Y[split+1:end]

# Convert from vector of vectors to matrix (8, N)
X_train_mat = hcat(X_train...)
Y_train_mat = hcat(Y_train...)
X_val_mat = hcat(X_val...)
Y_val_mat = hcat(Y_val...)

# (No need to Float32.() here again, but harmless)
X_train_mat = Float32.(X_train_mat)
Y_train_mat = Float32.(Y_train_mat)
X_val_mat = Float32.(X_val_mat)
Y_val_mat = Float32.(Y_val_mat)

println("First X sample size: ", size(X[1]))
println("First Y sample size: ", size(Y[1]))
println("X[1] type: ", typeof(X[1]))

println("Building and training the neural network (OxiFlowNet)...")
# Build the model
model = Chain(
    Dense(8, 32, relu),
    Dense(32, 32, relu),
    Dense(32, 8),
    relu  # Ensures no negative outputs
) |> device

# Move training and validation data to the right device
X_train_mat = device(X_train_mat)
Y_train_mat = device(Y_train_mat)
X_val_mat   = device(X_val_mat)
Y_val_mat   = device(Y_val_mat)

train_model(model, X_train_mat, Y_train_mat, X_val_mat, Y_val_mat;
            epochs=100, lr=1e-3, geodesics=geos, pf_states=pf_states)

println("\nEvaluating on validation data...")
pred_val, val_loss = evaluate_model(model, X_val_mat, Y_val_mat)
println("Validation Loss: $(round(val_loss, digits=6))")

# Save model
BSON.@save "oxinet_model.bson" model=cpu(model) pf_states flat_pos
println("✅ Trained model saved to 'oxinet_model.bson'")

# Display sample predictions
for idx in rand(1:length(X_val), 3)
    init_occ = X_val[idx]
    true_final = Y_val[idx]
    pred_final = pred_val[:, idx]  # ← fixed line
    println("\n--- Sample ---")
    println("Initial occupancy: ", round.(init_occ, digits=3))
    println("True final occupancy: ", round.(true_final, digits=3))
    println("Predicted final occupancy: ", round.(pred_final, digits=3))
end
