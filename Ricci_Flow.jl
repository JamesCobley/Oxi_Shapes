using BSON: @load
using PyPlot

# --- Load your trajectories ---
@load "flow_traces_batch_20250524_152552.bson" flow_traces

# === Ricci spread function ===
function ricci_spread(ρ::Vector{Float32})
    p = ρ ./ sum(ρ)  # normalize
    return sum((p .- 1/length(p)).^2)
end

# === Compute Ricci time series for each trajectory ===
ricci_series = [ [ricci_spread(ρ) for ρ in ft.ρ_series] for ft in flow_traces ]

# === Plotting ===
fig, ax = subplots(figsize=(8,5))

# plot a subset (say 50 trajectories) to show structure
for rs in ricci_series[1:50]
    ax.plot(rs, color="blue", alpha=0.2)
end

# mean + std envelope across all trajectories
len = minimum(length.(ricci_series))  # align length
mat = hcat([rs[1:len] for rs in ricci_series]...)'
mean_curve = mean(mat, dims=1)[:]
std_curve = std(mat, dims=1)[:]

ax.plot(mean_curve, color="black", lw=2, label="Mean Ricci flow")
ax.fill_between(1:len, mean_curve .- std_curve, mean_curve .+ std_curve,
    color="gray", alpha=0.3, label="±1 std")

ax.set_xlabel("Time step")
ax.set_ylabel("Ricci spread Φ(ρ)")
ax.set_title("Shape of Ricci flow chaos across trajectories")
ax.legend()

tight_layout()
savefig("ricci_flow_shape.png", dpi=300)
println("Saved: ricci_flow_shape.png")
