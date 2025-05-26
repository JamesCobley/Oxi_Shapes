using BSON: @load
using Colors
using CairoMakie
using Statistics: mean, norm

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

# --- Load traces ---
@load "flow_traces_batch_20250524_152552.bson" flow_traces
println("✔ Loaded ", length(flow_traces), " flow traces.")

# --- Use updated Lyapunov if defined ---
function lyapunov_start_end(series::Vector{Vector{Float32}})
    norm_start = norm(series[1])
    norm_diff = norm(series[end] .- series[1])
    return (norm_start > 0f0 && norm_diff > 0f0) ? log(norm_diff / norm_start) : 0f0
end

# --- Setup ---
states = ["000", "001", "010", "011", "100", "101", "110", "111"]
state_to_index = Dict(s => i-1 for (i, s) in enumerate(states))

points = ComplexF32[]
weights = Float32[]

for ft in flow_traces
    lyap = lyapunov_start_end(ft.ρ_series)
    if lyap > 0  # Only include chaotic
        for (t, ρ) in enumerate(ft.ρ_series)
            idx = argmax(ρ)
            real_part = state_to_index[states[idx]]
            im_part = mean(ft.R_series[min(t, end)])
            push!(points, ComplexF32(real_part, im_part))
            push!(weights, lyap)
        end
    end
end

# --- Grid ---
x_bins = 0:1:7
y_bins = -0.04:0.002:0.04
Z = zeros(Float32, length(x_bins), length(y_bins))

for (z, w) in zip(points, weights)
    xi = findfirst(b -> real(z) ≤ b, x_bins)
    yi = findfirst(b -> imag(z) ≤ b, y_bins)
    if xi !== nothing && yi !== nothing && xi > 1 && yi > 1
        Z[xi-1, yi-1] += w
    end
end

# Log scale
Z_log = log.(1 .+ Z)
Z_log ./= maximum(Z_log)

# --- Plot ---
fig = Figure(size = (1000, 500))
ax = Axis(fig[1, 1], 
    xlabel = "Proteoform State", 
    ylabel = "Mean Ricci Curvature",
    xticks = (0:7, states),
    yticks = y_bins[1:5:end],
    title = "Fractal-Constrained Chaos in Redox Geometry"
)

hm = heatmap!(ax, x_bins[1:end-1], y_bins[1:end-1], Z_log[1:end-1, 1:end-1]', colormap = :magma)

Colorbar(fig[1, 2], hm, label = "Normalized Lyapunov")

save("/content/chaos_geometric_signature.png", fig; px_per_unit=3)
fig
