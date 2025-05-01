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
