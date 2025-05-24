# =============================================================================
# GeoBrain: Real-Flow & Discrete Lie Pattern Analysis + Geometric Fields
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
# Utility Functions
# =============================================================================

function hamming(a::String, b::String)
    @assert length(a) == length(b) "Strings must be of equal length"
    return count((x_y) -> x_y[1] ≠ x_y[2], zip(a, b))
end

function is_valid_motif(motif::Vector{String})
    for i in 1:length(motif)-1
        if hamming(motif[i], motif[i+1]) ≠ 1
            return false
        end
    end
    return true
end

# =============================================================================
# Load the relevant files with sanity checks
# =============================================================================
batch_id = "20250524_152552"  # Define batch_id explicitly

try
    @load "/content/flow_traces_batch_20250524_152552.bson" flow_traces
    @load "/content/trace_metadata_batch_20250524_152552.bson" trace_metadata
    @load "/content/simplex_batch_20250524_152552.bson" simplex

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
# TraceAlgebra + Discrete Lie Analysis with Geometric Feature Integration
# =============================================================================

struct Trace
    steps::Vector{String}
end

function trace_patterns(trace::Trace; window::Int=3)
    patterns = Dict{Vector{String}, Int}()
    for i in 1:(length(trace.steps) - window + 1)
        pat = trace.steps[i:i+window-1]
        patterns[pat] = get(patterns, pat, 0) + 1
    end
    return patterns
end

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

function project_trace(trace::Trace, basis::Vector{Vector{String}}; window::Int=3)
    pats = trace_patterns(trace; window=window)
    projection = Dict{Vector{String}, Int}()
    for b in basis
        projection[b] = get(pats, b, 0)
    end
    return projection
end

function compute_geometric_means_for_patterns(simplex, basis::Vector{Vector{String}})
    avg_geometry = Dict{Vector{String}, NamedTuple{(:R_avg, :A_avg, :ρ_avg), Tuple{Float32, Float32, Float32}}}()
    for pat in basis
        r_vals, a_vals, rho_vals = Float32[], Float32[], Float32[]
        for run in simplex
            for i in 1:(length(run)-2)
                segment = [run[i].ρ_real, run[i+1].ρ_real, run[i+2].ρ_real]
                states = [findmax(ρ)[2] for ρ in segment]
                labels = ["000", "001", "010", "011", "100", "101", "110", "111"]
                motif = [labels[s] for s in states]
                if motif == pat
                    push!(r_vals, mean(vcat(run[i].R_real...)))
                    push!(a_vals, mean(norm.(run[i].A_real)))
                    push!(rho_vals, mean(run[i].ρ_real))
                end
            end
        end
        if !isempty(r_vals)
            avg_geometry[pat] = (R_avg=mean(r_vals), A_avg=mean(a_vals), ρ_avg=mean(rho_vals))  # ✅ Fixed
        end
    end
    return avg_geometry
end


function build_trace_spectral_features(traces::Vector{Trace}; window::Int=3, top_k::Int=10)
    freq = pattern_frequencies(traces; window=window)
    sorted_patterns = sort(collect(freq), by=x -> -x[2])

    # 🔍 Keep only valid geometric motifs
    valid_sorted = filter(p -> is_valid_motif(p[1]), sorted_patterns)

    basis = [pat for (pat, _) in valid_sorted[1:min(top_k, length(valid_sorted))]]
    features = [project_trace(trace, basis; window=window) for trace in traces]

    return basis, features
end

function discrete_commutator(a::Vector{String}, b::Vector{String})
    # Ensure motifs have at least two elements
    if length(a) < 2 || length(b) < 2
        return 0
    end

    # Define function to follow a path and validate first-order transitions
    function follow_path(path::Vector{String})
        for i in 1:length(path)-1
            if hamming(path[i], path[i+1]) != 1
                return nothing  # Invalid transition
            end
        end
        return path[end]  # Return final state if valid
    end

    # Create combined paths
    ab_path = vcat(a, b[2:end])  # Avoid repeating the initial state
    ba_path = vcat(b, a[2:end])

    final_ab = follow_path(ab_path)
    final_ba = follow_path(ba_path)

    # If either is invalid or leads to different outcomes → non-commutative
    return (final_ab != final_ba) ? 1 : 0
end

function commutator_matrix(basis::Vector{Vector{String}})
    n = length(basis)
    C = zeros(Int, n, n)
    for i in 1:n, j in 1:n
        C[i, j] = discrete_commutator(basis[i], basis[j])
    end
    return C
end

function adjoint_action(trace::Trace, basis::Vector{Vector{String}})
    projection = project_trace(trace, basis)
    return Dict(p => projection[p] / length(trace.steps) for p in keys(projection))
end

struct DiscreteLieAlgebra
    basis::Vector{Vector{String}}
    commutator::Matrix{Int}
    projection::Vector{Dict{Vector{String}, Int}}
    geom_features::Dict{Vector{String}, NamedTuple{(:R_avg, :A_avg, :ρ_avg), Tuple{Float32, Float32, Float32}}}
end

function build_discrete_lie_algebra(traces::Vector{Trace}, simplex; window::Int=3, top_k::Int=10)
    basis, projection = build_trace_spectral_features(traces; window=window, top_k=top_k)
    C = commutator_matrix(basis)
    geom_features = compute_geometric_means_for_patterns(simplex, basis)
    return DiscreteLieAlgebra(basis, C, projection, geom_features)
end

function extract_traces_from_metadata(trace_metadata::Vector{Dict{Symbol, Any}})
    return [Trace(meta[:compressed]) for meta in trace_metadata if haskey(meta, :compressed)]
end

function run_trace_spectral_analysis(trace_metadata::Vector{Dict{Symbol, Any}}, simplex; window::Int=3, top_k::Int=10)
    traces = extract_traces_from_metadata(trace_metadata)
    println("✔ Extracted $(length(traces)) trace objects.")

    dlie = build_discrete_lie_algebra(traces, simplex; window=window, top_k=top_k)
    println("✔ Built Discrete Lie Algebra structure with $(length(dlie.basis)) generators.")
    println("→ Example generator: ", join(dlie.basis[1], " → "))

    return dlie
end

# --- Run it ---
dlie = run_trace_spectral_analysis(Dict{Symbol, Any}.(trace_metadata), simplex; window=3, top_k=10)

function save_trace_spectral_analysis(batch_id::String, dlie::DiscreteLieAlgebra)
    filename = "trace_spectral_analysis_$(batch_id).bson"
    spectral_data = Dict(
        :basis => dlie.basis,
        :commutator => dlie.commutator,
        :projection => dlie.projection,
        :geom_features => dlie.geom_features
    )
    @save filename spectral_data
    println("✔ Saved spectral Lie analysis as Dict to: $filename")
end

save_trace_spectral_analysis(batch_id, dlie)
