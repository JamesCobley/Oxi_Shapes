using BSON: @load
using DSP, PyPlot, Colors

# --- Complex projection of Ricci waves into analytic signal (Hilbert transform)
function complex_projection(rs::Vector{Float64})
    z = hilbert(rs .- mean(rs))   # analytic signal
    return z
end

# --- Project canonical examples (symmetric vs trapped)
z_sym  = complex_projection(Float64.(sym.rs))
z_trap = complex_projection(Float64.(trap.rs))

fig, axs = subplots(1, 2, figsize=(10,5))
axs[1].plot(real(z_sym), imag(z_sym), lw=0.8, color="blue")
axs[1].set_title("Complex Ricci orbit (symmetric)")
axs[1].set_xlabel("Re Φ"); axs[1].set_ylabel("Im Φ")
axs[1].axis("equal")

axs[2].plot(real(z_trap), imag(z_trap), lw=0.8, color="red")
axs[2].set_title("Complex Ricci orbit (trapped)")
axs[2].set_xlabel("Re Φ"); axs[2].set_ylabel("Im Φ")
axs[2].axis("equal")

tight_layout()
savefig("ricci_complex_projection.png", dpi=300)
println("Saved: ricci_complex_projection.png")

# ======================================================
# --- Ensemble projection: all trajectories, viridis
# ======================================================
ricci_series = [ [ricci_spread(ρ) for ρ in ft.ρ_series] for ft in flow_traces ]
ensemble_z   = [complex_projection(Float64.(rs)) for rs in ricci_series]

# Pick a scalar metric to color by: here Shannon entropy at final time
metrics = [ft.shannon_entropy_series[end] for ft in flow_traces]
norm_metrics = (metrics .- minimum(metrics)) ./ (maximum(metrics)-minimum(metrics)+1e-9)

cmap = get_cmap("viridis")

fig2, ax2 = subplots(figsize=(6,6))
for (z, m) in zip(ensemble_z, norm_metrics)
    ax2.plot(real(z), imag(z), lw=0.4, alpha=0.6, color=cmap(m))
end
ax2.set_title("Ensemble Ricci orbits in complex plane (viridis by entropy)")
ax2.set_xlabel("Re Φ"); ax2.set_ylabel("Im Φ")
ax2.axis("equal")

tight_layout()
savefig("ricci_complex_ensemble.png", dpi=300)
println("Saved: ricci_complex_ensemble.png")
