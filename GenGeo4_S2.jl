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
using UUIDs
using BSON: @save, @load
using Dates

# =============================================================================
# Load the relevant files with sanity checks
# =============================================================================
try
    @load "/content/flow_traces_batch_20250501_121234.bson" flow_traces
    @load "/content/trace_metadata_batch_20250501_121234.bson" trace_metadata
    @load "/content/simplex_batch_20250501_121234.bson" simplex

    println("✔ Loaded all files for batch: $batch_id")
    println("→ FlowTraces: ", length(flow_traces))
    println("→ Metadata: ", length(trace_metadata))
    println("→ Simplex: ", length(simplex))

    if length(flow_traces) == 0 || length(trace_metadata) == 0 || length(simplex) == 0
        error("❌ One or more files are empty.")
    end
catch e
    @error "Failed to load one or more required files for batch $batch_id" exception=e
end

# =============================================================================
# TraceAlgebra Module: Discrete Fourier-like Analysis for Flow Traces
# =============================================================================

struct Trace
    steps::Vector{String}
end

# Function to extract recurring patterns of a given window size from a trace
function trace_patterns(trace::Trace; window::Int=3)
    patterns = Dict{Vector{String}, Int}()
    for i in 1:(length(trace.steps) - window + 1)
        pat = trace.steps[i:i+window-1]
        patterns[pat] = get(patterns, pat, 0) + 1
    end
    return patterns
end

# Function to compute frequencies of patterns across multiple traces
function pattern_frequencies(traces::Vector{Trace}; window::Int=3)
    freq = Dict{Vector{String}, Int}()
    for trace in traces
        pats = trace_patterns(trace; window=window)
        for (pat, count) in pats
            freq[pat] = get(freq, pat, 0) + count
        end
    end
    return freq
end

# Function to project a trace onto a set of basis patterns
function project_trace(trace::Trace, basis::Vector{Vector{String}}; window::Int=3)
    pats = trace_patterns(trace; window=window)
    projection = Dict{Vector{String}, Int}()
    for b in basis
        projection[b] = get(pats, b, 0)
    end
    return projection
end

# Function to build spectral features for a set of traces
function build_trace_spectral_features(traces::Vector{Trace}; window::Int=3, top_k::Int=10)
    freq = pattern_frequencies(traces; window=window)
    sorted_patterns = sort(collect(freq), by=x -> -x[2])
    basis = [pat for (pat, _) in sorted_patterns[1:min(top_k, length(sorted_patterns))]]
    features = [project_trace(trace, basis; window=window) for trace in traces]
    return basis, features
end

# =============================================================================
# Apply TraceAlgebra to Metadata
# =============================================================================

# Converts metadata entries to Trace objects
function extract_traces_from_metadata(trace_metadata::Vector{Dict{Symbol, Any}})
    return [Trace(meta[:compressed]) for meta in trace_metadata if haskey(meta, :compressed)]
end

# Wrapper to run the spectral analysis
function run_trace_spectral_analysis(trace_metadata::Vector{Dict{Symbol, Any}}; window::Int=3, top_k::Int=10)
    traces = extract_traces_from_metadata(trace_metadata)
    println("✔ Extracted $(length(traces)) trace objects.")

    basis, features = build_trace_spectral_features(traces; window=window, top_k=top_k)
    println("✔ Built spectral basis of top $top_k patterns.")
    println("→ Example basis pattern: ", join(basis[1], " → "))

    return basis, features
end

# --- Run it ---
basis, spectral_features = run_trace_spectral_analysis(Dict{Symbol, Any}.(trace_metadata); window=3, top_k=10)

function save_trace_spectral_analysis(batch_id::String, basis::Vector{Vector{String}}, features::Vector{Dict{Vector{String}, Int}})
    @save "data/$(batch_id)/trace_spectral_analysis.bson" basis features
    println("✔ Saved spectral analysis for batch: $batch_id")
end
