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

# --- Recompute corrected Lyapunov exponent ---
function lyapunov_start_end(series::Vector{Vector{Float32}})
    norm_start = norm(series[1])
    norm_diff = norm(series[end] .- series[1])
    return (norm_start > 0f0 && norm_diff > 0f0) ? log(norm_diff / norm_start) : 0f0
end

lyapunov_values = [lyapunov_start_end(ft.ρ_series) for ft in flow_traces]

# --- Filter chaotic traces ---
chaotic_traces = [(ft, lyap) for (ft, lyap) in zip(flow_traces, lyapunov_values) if lyap > 0f0]
println("✔ Using ", length(chaotic_traces), " chaotic traces.")

# --- Define state space ---
states = ["000", "001", "010", "011", "100", "101", "110", "111"]
state_to_index = Dict(s => i-1 for (i, s) in enumerate(states))

# --- Project onto complex plane ---
points = ComplexF32[]
weights = Float32[]

for (ft, lyap) in chaotic_traces
    for (t, ρ) in enumerate(ft.ρ_series)
        idx = argmax(ρ)
        real_part = state_to_index[states[idx]]
        im_part = mean(ft.R_series[min(t, end)])  # Ricci curvature at step
        push!(points, ComplexF32(real_part, im_part))
        push!(weights, lyap)  # Weight by corrected Lyapunov
    end
end

# --- Create complex plane grid ---
x_bins = 0:0.05:7  # State space (000–111)
y_bins = -0.05:0.001:0.05  # Curvature range (adjust if needed)
Z = zeros(Float32, length(x_bins), length(y_bins))

for (z, w) in zip(points, weights)
    xi = findfirst(b -> real(z) ≤ b, x_bins)
    yi = findfirst(b -> imag(z) ≤ b, y_bins)
    if xi !== nothing && yi !== nothing && xi > 1 && yi > 1
        Z[xi-1, yi-1] += w
    end
end

# --- Plot ---

fig = Figure(size = (1000, 500))  # ← use `size` instead of `resolution`
ax = Axis(fig[1, 1], xlabel="State Index", ylabel="Mean Ricci Curvature")

hm = heatmap!(ax, x_bins[1:end-1], y_bins[1:end-1], Z[1:end-1, 1:end-1]', colormap = :viridis)
Colorbar(fig[1, 2], hm, label = "Lyapunov Exponent")  # ← use `hm` directly here

save("/content/chaos_geometry_map_corrected.png", fig; px_per_unit=3)
fig
