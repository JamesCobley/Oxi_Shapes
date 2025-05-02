# =============================================================================
# GeoBrain: Real-Flow & Imagination Pipeline
# =============================================================================
using LinearAlgebra
using SparseArrays
using GeometryBasics
using Graphs
using StatsBase
using Statistics: mean
using Random
using Wavelets
using UUIDs
using BSON: @save, @load
using Dates

# =============================================================================
# Load the relevant files with enforced types
# =============================================================================

# Declare placeholders
flow_traces = nothing
trace_metadata = nothing
simplex = nothing
features = nothing
basis = nothing

try
    # Load core simulation data
    @load "/content/flow_traces_$(batch_id).bson" flow_traces
    @load "/content/trace_metadata_$(batch_id).bson" trace_metadata
    @load "/content/simplex_$(batch_id).bson" simplex

    println("✔ Loaded core files for batch: $(batch_id)")
    println("→ FlowTraces: ", length(flow_traces))
    println("→ Metadata: ", length(trace_metadata))
    println("→ Simplex: ", length(simplex))

    # Load spectral data
    spectral_data = Dict{Symbol, Any}()
    @load "/content/trace_spectral_analysis_$(batch_id).bson" spectral_data

    basis = spectral_data[:basis]
    features = spectral_data[:features]

    println("✔ Loaded spectral analysis:")
    println("→ Spectral Features: ", length(features))
    println("→ Spectral Basis Patterns: ", length(basis))

    flow_traces = Vector{FlowTrace}(flow_traces)
    trace_metadata = Vector{Dict{Symbol, Any}}(trace_metadata)
    features = Vector{Dict{Vector{String}, Int}}(features)
    basis = Vector{Vector{String}}(basis)
    simplex = [Vector{GeoNode}(run) for run in simplex]

    # Explicit type enforcement
    trace_metadata = Vector{Dict{Symbol, Any}}(trace_metadata)
    features = Vector{Dict{Vector{String}, Int}}(features)
    basis = Vector{Vector{String}}(basis)
    simplex = [Vector{GeoNode}(run) for run in simplex]

    # Sanity checks
    @assert length(trace_metadata) == length(features) == length(flow_traces)

catch e
    @error "❌ Failed to load one or more required files for batch $(batch_id)" exception = e
end

wavelet_obj = Wavelets.wavelet(Wavelets.WT.db2)

function get_wavelet_embedding(series::Union{
    Vector{Float64}, Vector{Float32}, Vector{Vector{Float32}}, Vector{Vector{Float64}}};
    wavelet=wavelet_obj, level=3)

    if eltype(series) <: AbstractFloat
        coeffs = dwtcoeffs(collect(Float64, series), wavelet, level)
        return vcat(coeffs...)  # flatten
    elseif eltype(series) <: AbstractVector
        result = Float64[]
        for subvec in series
            coeffs = dwtcoeffs(collect(Float64, subvec), wavelet, level)
            append!(result, vcat(coeffs...))
        end
        return result
    else
        error("Unsupported series type: $(eltype(series))")
    end
end

function build_wavelet_embeddings(flow_traces::Vector{FlowTrace};
                                  wavelet=wavelet_obj, level=3)

    wavelet_features = [
        :ρ_series,
        :flux_series,
        :R_series,
        :shannon_entropy_series,
        :fisher_info_series
    ]

    embeddings = Vector{Dict{Symbol, Vector{Float64}}}(undef, length(flow_traces))
    for (i, trace) in enumerate(flow_traces)
        result = Dict{Symbol, Vector{Float64}}()
        for key in wavelet_features
            if hasproperty(trace, key)
                data = getproperty(trace, key)
                try
                    result[key] = get_wavelet_embedding(data; wavelet=wavelet, level=level)
                catch err
                    @warn "Wavelet failed on $key in trace $i" exception=err
                end
            end
        end
        embeddings[i] = result
    end
    return embeddings
end

wavelet_embeddings = build_wavelet_embeddings(flow_traces)
@save "wavelet_embeddings_$(batch_id).bson" wavelet_embeddings
println("✔ Saved wavelet embeddings for MetaLambda.")
