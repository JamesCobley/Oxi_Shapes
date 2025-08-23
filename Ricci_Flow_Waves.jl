using BSON: @load
using Statistics
using PyPlot
using FFTW
using DSP    # hanning, hilbert

# --- FlowTrace type so @load can reconstruct ---
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

@load "flow_traces_batch_20250524_152552.bson" flow_traces

# Ricci spread Φ and degeneracy D
const N = 8
ricci_spread(ρ::Vector{Float32}) = begin
    p = ρ ./ sum(ρ)
    s = 0.0
    u = 1.0 / length(p)
    @inbounds @simd for i in eachindex(p)
        d = p[i] - u
        s += d*d
    end
    return s
end

degeneracy_from_phi(phi::Float64) = 1.0 / (phi + 1/N)             # in [1, N]
norm_amplitude(phi::Float64) = (degeneracy_from_phi(phi) - 1) / (N - 1)  # in [0,1]

# Build time series
phi_series = [ [ricci_spread(ρ) for ρ in ft.ρ_series] for ft in flow_traces ]
len = minimum(length.(phi_series))
Φmat = hcat([Float64.(s[1:len]) for s in phi_series]...)'   # (traj × time)

# Degeneracy amplitude
Dmat = 1.0 ./ (Φmat .+ 1/N)                 # (traj × time), values in [1,8]
Amat = (Dmat .- 1.0) ./ (N - 1.0)           # normalized [0,1]

# Plot bundle (degeneracy as amplitude)
fig, ax = subplots(figsize=(8,5))
m = min(size(Dmat,1), 50)
for i in 1:m
    ax.plot(Dmat[i, :], alpha=0.25)
end
meanD = mean(Dmat, dims=1)[:]
stdD  = std(Dmat,  dims=1)[:]
ax.plot(meanD, color="black", lw=2, label="Mean degeneracy D(t)")
ax.fill_between(1:len, meanD .- stdD, meanD .+ stdD, color="gray", alpha=0.3, label="±1 sd")
ax.set_xlabel("Time step")
ax.set_ylabel("Degeneracy D(t)  (1…8)")
ax.set_title("Ricci-flow amplitude as modal degeneracy")
ax.legend()
tight_layout()
savefig("degeneracy_wave_bundle.png", dpi=300)

# Analytic signal on *degeneracy* (instantaneous frequency)
z = hilbert(meanD .- mean(meanD))
amp_D   = abs.(z)                       # envelope (still in degeneracy units)
phase_D = angle.(z)
instfreq = vcat(0.0, diff(unwrap(phase_D)) ./ (2π))   # cycles per step

fig2, (ax1, ax2) = subplots(2,1, figsize=(8,5), sharex=true)
ax1.plot(meanD, lw=1.5, label="D(t)")
ax1.plot(amp_D, lw=1.0, label="Hilbert envelope", alpha=0.8)
ax1.set_ylabel("Degeneracy")
ax1.legend(loc="best")
ax1.set_title("Degeneracy amplitude and instantaneous frequency")

ax2.plot(instfreq)
ax2.set_xlabel("Time (steps)")
ax2.set_ylabel("Inst. frequency (cycles/step)")
tight_layout()
savefig("degeneracy_wave_analytic.png", dpi=300)

println("Saved: degeneracy_wave_bundle.png, degeneracy_wave_analytic.png")
