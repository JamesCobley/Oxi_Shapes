using LinearAlgebra
using Statistics

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
