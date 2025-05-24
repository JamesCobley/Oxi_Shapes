using StaticArrays
using Statistics: mean, cor, std
using BSON: @load
using SpecialFunctions: binomial

# --- Structs matching stored data ---
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

struct GeoNode
    ρ_real::Vector{Float32}
    R_real::Vector{Float32}
    A_real::Vector{SVector{3, Float32}}
    sheath_stress::Vector{Float32}
    flux::Vector{Float32}
    action_cost::Float32
end

# --- Load and cast data ---
@load "/content/flow_traces_batch_20250524_152552.bson" flow_traces
@load "/content/simplex_batch_20250524_152552.bson" simplex

flow_traces_typed = FlowTrace[ft for ft in flow_traces]
simplex_typed = Vector{GeoNode}[run for run in simplex]

# --- Degeneracy entropy function ---
function degeneracy_entropy(ρ::Vector{Float32}, pf_states::Vector{String})
    R = length(pf_states[1])
    degeneracy_map = Dict(k => binomial(R, k) for k in 0:R)
    return sum(
        ρ[i] > 0f0 ?
        ρ[i] * log(1f0 / degeneracy_map[count(==('1'), pf_states[i])]) :
        0f0 for i in 1:length(ρ)
    )
end

# --- Action integral using proper terms ---
function compute_action(trace::FlowTrace, run::Vector{GeoNode}, pf_states::Vector{String};
                        α_mass=0.01, α_geom=1.0, α_entropy=1.0)
    total_action = 0f0
    for t in 1:length(run)
        flux = trace.flux_series[t]
        R = run[t].R_real
        A = run[t].A_real
        ρ = run[t].ρ_real

        s_mass = α_mass * sum(abs.(flux))
        s_geom = α_geom * (mean(R.^2) + mean(norm.(A).^2))
        s_degeneracy = α_entropy * degeneracy_entropy(ρ, pf_states)

        total_action += s_mass + s_geom + s_degeneracy
    end
    return total_action
end

# --- Full correlation analysis ---
function geometry_vs_true_action(flow_traces::Vector{FlowTrace}, simplex::Vector{Vector{GeoNode}}, pf_states::Vector{String})
    R_means = Float32[]
    A_means = Float32[]
    true_actions = Float32[]

    for (trace, run) in zip(flow_traces, simplex)
        Rvals = vcat([node.R_real for node in run]...)
        Avals = vcat([norm(a) for node in run for a in node.A_real]...)
        push!(R_means, mean(Rvals))
        push!(A_means, mean(Avals))
        push!(true_actions, compute_action(trace, run, pf_states))
    end

    println("✔ True Action Integral Analysis")
    println("→ Corr(R_mean, Action): ", round(cor(R_means, true_actions), digits=4))
    println("→ Corr(A_mean, Action): ", round(cor(A_means, true_actions), digits=4))
    println("→ Mean Action: ", round(mean(true_actions), digits=4), " ± ", round(std(true_actions), digits=4))

    return (R_means=R_means, A_means=A_means, actions=true_actions)
end

# --- Provide your pf_states ---
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]

# --- Run the analysis ---
results = geometry_vs_true_action(flow_traces_typed, simplex_typed, pf_states)
