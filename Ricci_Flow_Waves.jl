using BSON: @load
using Statistics
using PyPlot
using FFTW
using DSP  # windowing + stft + hilbert

# --- Re-declare FlowTrace so BSON can reconstruct it ---
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

# --- Load trajectories ---
@load "flow_traces_batch_20250524_152552.bson" flow_traces

# === Ricci spread function ===
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

# === Compute Ricci time series for each trajectory ===
ricci_series = [ [ricci_spread(ρ) for ρ in ft.ρ_series] for ft in flow_traces ]

# -------------------------
# 1) Waveform bundle plot
# -------------------------
fig, ax = subplots(figsize=(8,5))

m = min(length(ricci_series), 50)
for rs in ricci_series[1:m]
    ax.plot(rs, alpha=0.2)
end

len = minimum(length.(ricci_series))               # align length
mat = hcat([Float64.(rs[1:len]) for rs in ricci_series]...)'  # rows=traj, cols=time
mean_curve = mean(mat, dims=1)[:]
std_curve  = std(mat,  dims=1)[:]

ax.plot(mean_curve, lw=2, label="Mean Ricci flow", color="black")
ax.fill_between(1:len, mean_curve .- std_curve, mean_curve .+ std_curve,
                alpha=0.3, label="±1 std", color="gray")
ax.set_xlabel("Time step")
ax.set_ylabel("Ricci spread Φ(ρ)")
ax.set_title("Ricci flow as a bundle of waves")
ax.legend(loc="best")
tight_layout()
savefig("ricci_flow_shape.png", dpi=300)
println("Saved: ricci_flow_shape.png")

# -----------------------------------
# 2) Frequency basis (Fourier modes)
# -----------------------------------
L = len
w  = hanning(L)                               # mild taper to reduce leakage

power_spectrum(x::Vector{Float64}) = begin
    y = (x .- mean(x)) .* w
    Y = rfft(y)
    P = abs.(Y).^2 / length(y)                # simple power (one-sided)
    return P
end

spectra = [ power_spectrum(vec(mat[i, :])) for i in 1:size(mat,1) ]
Smat    = hcat(spectra...)                    # rows=freq bins, cols=trajectories
Pmean   = mean(Smat, dims=2)[:]
Pstd    = std(Smat,  dims=2)[:]
freqs   = collect(0:div(L,2)) ./ L            # Δt=1 ⇒ freq in 1/steps

fig2, ax2 = subplots(figsize=(8,4))
ax2.plot(freqs, Pmean, lw=2, label="Mean power")
ax2.fill_between(freqs, max.(Pmean .- Pstd, 0), Pmean .+ Pstd, alpha=0.25, label="±1 std")
ax2.set_xlim(0, maximum(freqs))
ax2.set_xlabel("Frequency (1/steps)")
ax2.set_ylabel("Power")
ax2.set_title("Ricci wave frequency basis (ensemble)")
ax2.legend(loc="best")
tight_layout()
savefig("ricci_wave_spectrum.png", dpi=300)
println("Saved: ricci_wave_spectrum.png")

# --------------------------------------------------------
# 3) Time–frequency view (STFT) on the ensemble mean curve
# --------------------------------------------------------
# Short-Time Fourier Transform using DSP.stft
wlen = min(64, L)                  # window length
hop  = max(1, Int(wlen ÷ 4))       # hop size
x    = mean_curve .- mean(mean_curve)

# stft(signal, nwin, noverlap? via hop; window=windowfunc, nfft=?, fs=?)
# Here: nwin = wlen, hop = hop, window = hanning (built-in generator)
S = stft(x, wlen, hop; window = hanning)  # returns (freq_bins × frames) complex matrix

nfbins, nframes = size(S)
tfreqs = collect(0:nfbins-1) ./ wlen      # frequency axis (1/steps)
ttimes = collect(0:nframes-1) .* hop .+ wlen/2  # time at window centers

fig3, ax3 = subplots(figsize=(8,4))
im = ax3.imshow(abs.(S); origin="lower", aspect="auto",
                extent=[ttimes[1], ttimes[end], tfreqs[1], tfreqs[end]])
ax3.set_xlabel("Time (steps)")
ax3.set_ylabel("Frequency (1/steps)")
ax3.set_title("Ricci flow spectrogram (ensemble mean)")
fig3.colorbar(im, ax=ax3, label="Magnitude")
tight_layout()
savefig("ricci_wave_spectrogram.png", dpi=300)
println("Saved: ricci_wave_spectrogram.png")


# --------------------------------------------------------
# 4) Optional: analytic signal (amplitude & phase) of mean Ricci wave
# --------------------------------------------------------
z = hilbert(mean_curve .- mean(mean_curve))        # complex Ricci wave
amp = abs.(z)
phase = angle.(z)

fig4, (ax4a, ax4b) = subplots(2, 1, figsize=(8,5), sharex=true)
ax4a.plot(amp, lw=1.5)
ax4a.set_ylabel("Amplitude")
ax4a.set_title("Analytic Ricci wave (ensemble mean)")

ax4b.plot(phase, lw=1.0)
ax4b.set_xlabel("Time (steps)")
ax4b.set_ylabel("Phase (rad)")
tight_layout()
savefig("ricci_wave_analytic.png", dpi=300)
println("Saved: ricci_wave_analytic.png")
