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
    # Core simulation data
    @load "/content/flow_traces_$(batch_id).bson" flow_traces
    @load "/content/trace_metadata_$(batch_id).bson" trace_metadata
    @load "/content/simplex_$(batch_id).bson" simplex

    println("✔ Loaded core files for batch: $(batch_id)")
    println("→ FlowTraces: ", length(flow_traces))
    println("→ Metadata: ", length(trace_metadata))
    println("→ Simplex: ", length(simplex))

    if isempty(flow_traces) || isempty(trace_metadata) || isempty(simplex)
        error("❌ One or more core files are empty.")
    end

    # Spectral data
    spectral_data = Dict{Symbol, Any}()
    @load "/content/trace_spectral_analysis_$(batch_id).bson" spectral_data

    basis = spectral_data[:basis]
    features = spectral_data[:features]

    println("✔ Loaded spectral analysis:")
    println("→ Spectral Features: ", length(features))
    println("→ Spectral Basis Patterns: ", length(basis))

    if isempty(features) || isempty(basis)
        error("❌ Spectral analysis data is empty.")
    end

catch e
    @error "❌ Failed to load one or more required files for batch $(batch_id)" exception = e
end

# =============================================================================
# Geometric object 3: The Hypergraph learning manifold
# =============================================================================

struct HypergraphBrain
    edge_sets::Dict{Symbol, Vector{Vector{Int}}}        # Hyperedges: action_cost, curvature, spectral, etc.
    weights::Dict{Symbol, Vector{Float32}}              # One scalar per hyperedge
    phi::Vector{Float32}                                # Information potential
    psi::Vector{Float32}                                # Energy + entropy field
    lambda::Vector{Float32}                             # Real-imag divergence
    L::SparseMatrixCSC{Float32, Int}                    # Laplacian matrix from simplex lambda-surface
    modes::Matrix{Float32}                              # Laplacian eigenmodes (spectral basis)
    trace_patterns::Vector{Dict{Vector{String}, Int}}   # Recurring transition patterns per flow
    delta_commutators::Vector{Float32}                  # Non-commutativity score per flow
    fourier_features::Vector{Dict{Vector{String}, Int}} # Projection onto Fourier-like basis
    fourier_basis::Vector{Vector{String}}               # Discrete trace patterns used as basis
end

function build_hypergraph_from_simplex(simplex::Vector{Vector{GeoNode}},
                                       trace_metadata::Vector{Dict{Symbol, Any}},
                                       trace_features::Vector{Dict{Vector{String}, Int}},
                                       trace_basis::Vector{Vector{String}};
                                       curvature_bins=5,
                                       cost_bins=5,
                                       spectral_bins=5,
                                       mode_rank=10,
                                       alpha=1.0f0,
                                       beta=1.0f0)

    nodes = reduce(vcat, simplex)
    N = length(nodes)
    edge_sets = Dict{Symbol, Vector{Vector{Int}}}()

    # Bin hyperedges by action cost
    costs = [n.action_cost for n in nodes]
    edge_sets[:action_cost] = group_by_bins(StatsBase.cut(costs, cost_bins))

    # Bin hyperedges by mean curvature
    curvs = [mean(n.R_real) for n in nodes]
    edge_sets[:curvature] = group_by_bins(StatsBase.cut(curvs, curvature_bins))

    # Bin hyperedges by delta from metadata
    deltas = [m[:delta] for m in trace_metadata]
    edge_sets[:delta_commutator] = group_by_bins(StatsBase.cut(deltas, spectral_bins))

    # Compute fields
    lambda_vals = [n.λ for n in nodes]  # Still reading from field called `λ` inside `GeoNode`
    local_energy = [sum(abs.(n.R_real)) for n in nodes]
    local_entropy = [sum(abs.(n.flux)) for n in nodes]
    psi = [alpha * e + beta * s for (e, s) in zip(local_energy, local_entropy)]
    phi = zeros(Float32, N)

    # Laplacian surface
    lambda_surface = build_simplex_surface(simplex)
    L = build_simplex_laplacian(lambda_surface)
    evals, evecs = eigen(Matrix(L))
    modes = evecs[:, 1:min(mode_rank, size(evecs, 2))]

    # Spectral binning based on Laplacian modes
    add_spectral_binning!(edge_sets, modes, spectral_bins)

    # Fourier grouping by projection intensity
    trace_bin_groups = Dict{Int, Vector{Int}}()
    for (i, feats) in enumerate(trace_features)
        score = sum(values(feats))
        push!(get!(trace_bin_groups, score, Int[]), i)
    end
    edge_sets[:fourier_spectral] = collect(values(trace_bin_groups))

    # Uniform weight initialization
    weights = Dict{Symbol, Vector{Float32}}()
    for k in keys(edge_sets)
        weights[k] = ones(Float32, length(edge_sets[k]))
    end

    return HypergraphBrain(
        edge_sets,
        weights,
        phi,
        psi,
        lambda_vals,
        sparse(L),
        modes,
        [m[:patterns] for m in trace_metadata],
        deltas,
        trace_features,
        trace_basis
    )
end

# =============================================================================
# RicciFlowMetaLambda: Meta-learning and smoothing over the hypergraph
# =============================================================================

struct RicciFlowMetaLambda
    η::Float32                 # Learning rate
    λ_target::Float32         # Target divergence smoothing
    smoothing_weight::Float32 # Diffusion coefficient for φ smoothing
    curvature_weight::Float32 # Controls influence of R on φ
end

function update_info_potential!(brain::HypergraphBrain, simplex::Vector{Vector{GeoNode}};
                                η::Float32=0.1f0)

    nodes = reduce(vcat, simplex)
    N = length(nodes)
    Δφ = zeros(Float32, N)

    # Assume brain has fields: spectral_error, fourier_error, combined_error
    # These should be vectors of length N, containing the prediction errors for each node

    for (etype, edges) in brain.edge_sets
        for edge in edges
            λ_vals = [nodes[i].λ for i in edge]
            avg_λ = mean(abs, λ_vals)
            for i in edge
                # Compute weights based on inverse errors
                total_error = brain.spectral_error[i] + brain.fourier_error[i] + 1e-8f0  # Avoid division by zero
                spectral_weight = (brain.fourier_error[i] + 1e-8f0) / total_error
                fourier_weight = (brain.spectral_error[i] + 1e-8f0) / total_error

                # Combined weight
                combined_weight = 0.5f0 * (spectral_weight + fourier_weight)

                # Update Δφ with combined weight
                Δφ[i] += η * combined_weight * (avg_λ - abs(nodes[i].λ))
            end
        end
    end

    brain.φ .= brain.φ .+ Δφ
    return brain
end
