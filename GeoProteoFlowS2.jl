using LinearAlgebra
using Statistics
using StatsBase
using BSON
using BSON: @load, @save


# Define proteoform states and indexing
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
state_index = Dict(s => i for (i, s) in enumerate(pf_states))

# Convert k-priors to full i-state occupancy
function kpriors_to_istate(k_priors::Vector{Float64})
    groups = Dict(
        0 => ["000"],
        1 => ["001", "010", "100"],
        2 => ["011", "101", "110"],
        3 => ["111"]
    )
    occupancy = zeros(Float64, length(pf_states))
    for k in 0:3
        for s in groups[k]
            occupancy[state_index[s]] = k_priors[k + 1] / length(groups[k])
        end
    end
    return occupancy
end

# --- Metric Functions ---

function percent_oxidation(occupancy::AbstractVector{<:Real})
    oxidation_levels = [(count(==('1'), s) / 3) * 100 for s in pf_states]
    return sum(occupancy .* oxidation_levels)
end

function shannon_entropy(occupancy::AbstractVector{<:Real})
    p = filter(x -> x > 0, occupancy)
    return -sum(p .* log2.(p))
end

function lyapunov_exponent_kbins(rho_series::Vector{<:AbstractVector{<:Real}})
    k_series = [
        sum(count(==('1'), s) * rho[i] for (i, s) in enumerate(pf_states)) / 3
        for rho in rho_series
    ]

    diffs = abs.(diff(k_series))
    diffs = replace(diffs, 0.0 => 1e-12)  # Avoid log(0)
    log_diffs = log.(diffs)

    t = collect(0:length(log_diffs)-1)
    A = hcat(t, ones(length(t)))
    coeffs = A \ log_diffs
    return coeffs[1]  # Slope is the Lyapunov exponent
end

function evolve_time_series_alive(rho0::Vector{Float64}, T::Int, pf_states, flat_pos, edges; max_moves_per_step=10)
    """
    Discrete-time evolution using oxi_shapes_alive!
    Returns a vector of occupancy vectors over T time steps.
    """
    rho = copy(rho0)
    rho_series = [copy(rho)]
    
    for _ in 1:T
        oxi_shapes_alive!(rho, pf_states, flat_pos, edges; max_moves=max_moves_per_step)
        push!(rho_series, copy(rho))
    end

    return rho_series
end

# --- Empirical priors (from experimental data)
# For k-bins: [k=0, k=1, k=2, k=3]
empirical_start_k = [0.25, 0.75, 0.0, 0.0]
empirical_end_k   = [0.06, 0.53, 0.33, 0.10]

# Convert k-priors to full occupancy (length 8) for R=3.
rho_start_full = kpriors_to_istate(empirical_start_k)
rho_target_full = kpriors_to_istate(empirical_end_k)

println("Empirical Start k-priors: ", empirical_start_k)
println("Empirical End k-priors: ", empirical_end_k)

println("Empirical Percent Oxidation (Start): ", percent_oxidation(rho_start_full))
println("Empirical Percent Oxidation (Target): ", percent_oxidation(rho_target_full))

println("Empirical Shannon Entropy (Start): ", round(shannon_entropy(rho_start_full), digits=3))
println("Empirical Shannon Entropy (Target): ", round(shannon_entropy(rho_target_full), digits=3))

# Load the trained model from BSON file
@load "oxinet_model.bson" model pf_states flat_pos

# Convert empirical start to Float32 for prediction
rho_start_empirical = Float32.(rho_start_full)

# Predict the final occupancy using the trained neural network model
predicted_final_occ = model(rho_start_empirical)

# Normalize and round the prediction
predicted_final_occ = round.(predicted_final_occ .* 100) ./ 100
predicted_final_occ ./= sum(predicted_final_occ)

# Print predicted metrics
println("\nML Predicted Final Occupancy:", round.(predicted_final_occ, digits=3))
println("Predicted Percent Oxidation:", percent_oxidation(predicted_final_occ))
println("Predicted Shannon Entropy:", round(shannon_entropy(predicted_final_occ), digits=3))

# Generate time series trajectory via model rollouts
function predict_trajectory_with_metadata(model, initial, steps=240)
    traj = [initial]
    current = initial
    total_moves = 0
    oxidizing_moves = 0
    reducing_moves = 0

    for _ in 1:steps
        pred = model(current)
        pred = round.(pred .* 100) ./ 100
        pred ./= sum(pred)

        # Compute oxidation level change (for move classification)
        k_prev = percent_oxidation(current)
        k_new = percent_oxidation(pred)

        if k_new > k_prev + 1e-4
            oxidizing_moves += 1
        elseif k_new < k_prev - 1e-4
            reducing_moves += 1
        end
        total_moves += 1

        push!(traj, pred)
        current = pred
    end

    return traj, total_moves, oxidizing_moves, reducing_moves
end

# Run trajectory prediction
traj_series, total_moves, ox_moves, red_moves = predict_trajectory_with_metadata(model, rho_start_empirical, 240)
final_occ = traj_series[end]
percent_series = [percent_oxidation(r) for r in traj_series]
entropy_final = shannon_entropy(final_occ)
lyapunov_exp = lyapunov_exponent_kbins(traj_series)

metadata = Dict(
    "initial_state" => rho_start_empirical,
    "final_state" => final_occ,
    "trajectory" => traj_series,
    "percent_series" => percent_series,
    "entropy_final" => entropy_final,
    "lyapunov" => lyapunov_exp,
    "trajectory_id" => "empirical_run_1",
    "total_moves" => total_moves,
    "oxidizing_moves" => ox_moves,
    "reducing_moves" => red_moves
)

# Compute Lyapunov exponent over trajectory
percent_series = [percent_oxidation(r) for r in traj_series]
lyap_exp_k = lyapunov_exponent_kbins(traj_series)

println("Lyapunov Exponent (k-bin trajectory): ", round(lyap_exp_k, digits=6))

# Simulate trajectories for many molecules (stochastic variation via model rollout)
num_molecules = 100
trajectories = [predict_trajectory_with_metadata(model, rho_start_empirical, 240) for _ in 1:num_molecules]

# Confidence = max occupancy in final state
final_preds = [traj[1][end] for traj in trajectories]  # traj[1] is the trajectory
confidences = [maximum(pred) for pred in final_preds]

# Get top 5 most confident predictions
top5_indices = partialsortperm(confidences, rev=true, 1:5)
top5_trajectories = [trajectories[i][1] for i in top5_indices]  # again, only the trajectory

# Compute metrics for the top 5 trajectories
for (idx, traj) in enumerate(top5_trajectories)
    final_occ = traj[end]
    perc_ox = percent_oxidation(final_occ)
    entropy_val = shannon_entropy(final_occ)
    perc_series_traj = [percent_oxidation(r) for r in traj]
    lyap_val = lyapunov_exponent_kbins(traj)

    println("\nTop Prediction $(idx):")
    println("Final occupancy: ", round.(final_occ, digits=3))
    println("Percent oxidation: ", round(perc_ox, digits=2))
    println("Shannon entropy: ", round(entropy_val, digits=3))
    println("Lyapunov exponent: ", round(lyap_val, digits=6))
end

# --- Master metadata dictionary for full export
master_metadata = Dict(
    "empirical_metadata" => metadata,
    "all_trajectories" => trajectories,
    "confidences" => confidences,
    "top5_trajectories" => top5_trajectories,
    "pf_states" => pf_states,
    "empirical_start_k" => empirical_start_k,
    "empirical_end_k" => empirical_end_k
)

# Save to BSON file with timestamp
using Dates
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
save_path = "full_simulation_metadata_$timestamp.bson"

BSON.@save save_path master_metadata
println("✅ Full simulation metadata saved to '$save_path'")
