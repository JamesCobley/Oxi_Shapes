using LinearAlgebra
using Statistics
using StatsBase
using BSON
using BSON: @load


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

# (1) Percent oxidation: average number of 1's per state × 100
function percent_oxidation(occupancy::Vector{Float64})
    oxidation_levels = [(count(==('1'), s) / 3) * 100 for s in pf_states]
    return sum(occupancy .* oxidation_levels)
end

# (2) Shannon entropy of the distribution
function shannon_entropy(occupancy::Vector{Float64})
    p = filter(x -> x > 0, occupancy)
    return -sum(p .* log2.(p))
end

# (3) Lyapunov exponent from k-bin oxidation trajectory
function lyapunov_exponent_kbins(rho_series::Vector{Vector{Float64}})
    k_series = [
        sum(count(==('1'), s) * rho[i] for (i, s) in enumerate(pf_states)) / 3
        for rho in rho_series
    ]

    diffs = abs.(diff(k_series))
    diffs = replace(diffs, 0.0 => 1e-12)
    log_diffs = log.(diffs)

    t = collect(0:length(log_diffs)-1)
    A = hcat(t, ones(length(t)))
    coeffs = A \ log_diffs
    slope = coeffs[1]

    return slope
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
function predict_trajectory(model, initial, steps=240)
    traj = [initial]
    current = initial
    for _ in 1:steps
        pred = model(current)
        pred = round.(pred .* 100) ./ 100
        pred ./= sum(pred)
        push!(traj, pred)
        current = pred
    end
    return traj
end

# Run trajectory prediction
traj_series = predict_trajectory(model, rho_start_empirical, 240)

# Compute Lyapunov exponent over trajectory
percent_series = [percent_oxidation(r) for r in traj_series]
lyap_exp_k = lyapunov_exponent_kbins(traj_series)

println("Lyapunov Exponent (k-bin trajectory): ", round(lyap_exp_k, digits=6))

# Simulate trajectories for many molecules (stochastic variation via model rollout)
num_molecules = 100
trajectories = [predict_trajectory(model, rho_start_empirical, 240) for _ in 1:num_molecules]

# Confidence = max occupancy in final state
final_preds = [traj[end] for traj in trajectories]
confidences = [maximum(pred) for pred in final_preds]

# Get top 5 most confident predictions
top5_indices = partialsortperm(confidences, rev=true, 1:5)
top5_trajectories = trajectories[top5_indices]

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

# --- Poincaré Recurrence Plot (textual/debug mode)
function poincare_recurrence(series::Vector{Float64}; threshold::Float64=1.0)
    N = length(series)
    R = falses(N, N)
    for i in 1:N
        for j in 1:N
            R[i, j] = abs(series[i] - series[j]) < threshold
        end
    end
    return R
end

rec_plot_top = poincare_recurrence([percent_oxidation(r) for r in top5_trajectories[1]]; threshold=1.0)

# Optional: plot if using Makie or other plotting lib
# display_heatmap(rec_plot_top)

# --- Fisher Information Metric
function fisher_information(occupancy_series::Vector{Vector{Float64}})
    diffs = [norm(occupancy_series[i+1] .- occupancy_series[i]) for i in 1:length(occupancy_series)-1]
    return sum(diffs .^ 2)
end

fisher_vals = [fisher_information(traj) for traj in trajectories]
fisher_info = mean(fisher_vals)

println("Fisher Information Metric (ML trajectories): ", round(fisher_info, digits=6))

# --- Feigenbaum placeholder
function estimate_feigenbaum(percent_series::Vector{Float64})
    return 4.669  # Placeholder: universal Feigenbaum delta
end

feigenbaum = estimate_feigenbaum(percent_series)
println("Estimated Feigenbaum constant (placeholder): ", feigenbaum)
