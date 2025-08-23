using BSON: @load
using Statistics, StatsBase
using FFTW, DSP
using PyPlot

# --- Reconstruct type so BSON can materialize it ---
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

# --- Load ---
@load "flow_traces_batch_20250524_152552.bson" flow_traces

# === Ricci spread Φ(ρ) and degeneracy D(t) ===
ricci_spread(p::Vector{<:Real}) = begin
    q = p ./ sum(p)
    u = 1/length(q)
    s = 0.0
    @inbounds @simd for i in eachindex(q)
        d = q[i] - u
        s += d*d
    end
    s
end

degeneracy(p::Vector{<:Real}) = 1 / sum((p./sum(p)).^2) # 1..8 for 8 modes

# --- Ricci series + one-sided FFT power ---
function ricci_and_spectrum(ft::FlowTrace)
    rs = [ricci_spread(ρ) for ρ in ft.ρ_series]          # Φ(t)
    x  = Float64.(rs .- mean(rs))
    L  = length(x)
    w  = hanning(L)
    y  = x .* w
    Y  = rfft(y)
    P  = abs.(Y).^2 ./ L                                 # power
    freqs = collect(0:div(L,2)) ./ L                     # 1/steps
    return rs, freqs, P
end

# --- Spectral peakedness: 1 = nearly single tone; ~0 = very spread ---
spectral_peakedness(P::AbstractVector) = begin
    pt = sum(P)
    pt == 0 ? 0.0 : maximum(P ./ pt)
end

# --- Build per-trajectory info ---
infos = map(flow_traces) do ft
    rs, f, P = ricci_and_spectrum(ft)
    (; ft, rs, f, P,
       phi_final = rs[end],
       peak = spectral_peakedness(P))
end

# --- Classify by Ricci-flow definition (final Φ): chaotic = low Φ, ordered = high Φ ---
phi_finals = getfield.(infos, :phi_final)
q_lo = quantile(phi_finals, 0.10)   # bottom decile = chaotic exemplars
q_hi = quantile(phi_finals, 0.90)   # top decile = ordered exemplars

chaotic_idxs = findall(i -> infos[i].phi_final ≤ q_lo, eachindex(infos))
ordered_idxs = findall(i -> infos[i].phi_final ≥ q_hi, eachindex(infos))

@assert !isempty(chaotic_idxs) && !isempty(ordered_idxs)

# Pick exemplars inside each class by spectral shape
chaotic_sym_idx  = argmax([infos[i].peak for i in chaotic_idxs]); chaotic_sym  = infos[chaotic_idxs[chaotic_sym_idx]]
chaotic_trap_idx = argmin([infos[i].peak for i in chaotic_idxs]); chaotic_trap = infos[chaotic_idxs[chaotic_trap_idx]]

ordered_sym_idx  = argmax([infos[i].peak for i in ordered_idxs]); ordered_sym  = infos[ordered_idxs[ordered_sym_idx]]
ordered_trap_idx = argmin([infos[i].peak for i in ordered_idxs]); ordered_trap = infos[ordered_idxs[ordered_trap_idx]]

println("Chaotic (low Φ) symmetric:  ", chaotic_sym.ft.run_id,  "  peak=", round(chaotic_sym.peak, digits=3),  "  Φ_end=", round(chaotic_sym.phi_final, digits=4))
println("Chaotic (low Φ) trapped:    ", chaotic_trap.ft.run_id, "  peak=", round(chaotic_trap.peak, digits=3), "  Φ_end=", round(chaotic_trap.phi_final, digits=4))
println("Ordered (high Φ) symmetric:  ", ordered_sym.ft.run_id,  "  peak=", round(ordered_sym.peak, digits=3),  "  Φ_end=", round(ordered_sym.phi_final, digits=4))
println("Ordered (high Φ) trapped:    ", ordered_trap.ft.run_id, "  peak=", round(ordered_trap.peak, digits=3), "  Φ_end=", round(ordered_trap.phi_final, digits=4))

# --- Helper to plot one trajectory (Φ(t) and spectrum) ---
function plot_traj(tag, rs, f, P; color="C0")
    fig = plt.figure(figsize=(10,4), constrained_layout=true)
    gs  = fig.add_gridspec(1,2, wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.plot(rs, color=color, lw=1.8)
    ax1.set_title("Ricci wave Phi(t) — "*tag)
    ax1.set_xlabel("t (steps)")
    ax1.set_ylabel("Phi(rho)")

    ax2.plot(f, P, color=color, lw=1.5)
    ax2.set_xlim(0, maximum(f))
    ax2.set_xlabel("freq (1/steps)")
    ax2.set_ylabel("Power")
    ax2.set_title("Spectrum — "*tag)

    # ASCII-only, safe filename
    outfile = replace(lowercase(tag), r"[^a-z0-9]+" => "_") * ".png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    println("Saved: ", outfile)
end

function print_summary(label, info)
    println(rpad(label, 28), "run_id=", info.ft.run_id,
            "  Phi_end=", round(info.phi_final, digits=4),
            "  peak=", round(info.peak, digits=3))
end

print_summary("Chaotic symmetric:", chaotic_sym)
print_summary("Chaotic trapped:",   chaotic_trap)
print_summary("Ordered symmetric:", ordered_sym)
print_summary("Ordered trapped:",   ordered_trap)

# --- Render four exemplars ---
plot_traj("CHAOTIC (low Φ) — symmetric", chaotic_sym.rs, chaotic_sym.f, chaotic_sym.P; color="C1")
plot_traj("CHAOTIC (low Φ) — trapped",   chaotic_trap.rs, chaotic_trap.f, chaotic_trap.P; color="C3")
plot_traj("ORDERED (high Φ) — symmetric", ordered_sym.rs, ordered_sym.f, ordered_sym.P; color="C0")
plot_traj("ORDERED (high Φ) — trapped",   ordered_trap.rs, ordered_trap.f, ordered_trap.P; color="C2")
