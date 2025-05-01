# =============================================================================
# GeoBrain: Real-Flow & Imagination Pipeline
# =============================================================================
using LinearAlgebra
using SparseArrays
using GeometryBasics
using Graphs
using StatsBase
import StatsBase: cut  # <-- this is the key line
using CategoricalArrays
using Statistics: mean
using Random
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

# =============================================================================
# Geometric object 3: The Hypergraph learning manifold
# =============================================================================

struct HypergraphBrain1
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
    spectral_error::Vector{Float32}                     # Per-node spectral prediction error
    fourier_error::Vector{Float32}                      # Per-node Fourier prediction error
end

function group_by_bins(bin_assignment::CategoricalArray{T,1}) where T
    bin_dict = Dict{T, Vector{Int}}()
    for (i, b) in enumerate(bin_assignment)
        push!(get!(bin_dict, b, Int[]), i)
    end
    return collect(values(bin_dict))
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
    edge_sets[:action_cost] = group_by_bins(cut(costs, cost_bins))

    # Bin hyperedges by mean curvature
    curvs = [mean(n.R_real) for n in nodes]
    edge_sets[:curvature] = group_by_bins(cut(curvs, curvature_bins))

    # Bin hyperedges by delta from metadata
    deltas = [m[:delta] for m in trace_metadata]
    edge_sets[:delta_commutator] = group_by_bins(cut(deltas, spectral_bins))

    # Compute fields
    lambda_vals = [n.lambda for n in nodes]
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

    # Placeholder errors (to be filled later with real prediction errors)
    spectral_error = zeros(Float32, N)
    fourier_error = zeros(Float32, N)

    return HypergraphBrain1(
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
        trace_basis,
        spectral_error,
        fourier_error
    )
end

brain = build_hypergraph_from_simplex(simplex, trace_metadata, features, basis)

# =============================================================================
# RicciFlowMetaLambda: Meta-learning and smoothing over the hypergraph
# =============================================================================

struct RicciFlowMetaLambda
    eta::Float32                 # Learning rate
    lambda_target::Float32       # Target divergence smoothing
    smoothing_weight::Float32    # Diffusion coefficient for phi smoothing
    curvature_weight::Float32    # Controls influence of curvature on phi
end

function update_info_potential!(brain::HypergraphBrain1, simplex::Vector{Vector{GeoNode}};
                                eta::Float32=0.1f0)

    nodes = reduce(vcat, simplex)
    N = length(nodes)
    delta_phi = zeros(Float32, N)

    # Loop over all edge types
    for (etype, edges) in brain.edge_sets
        for edge in edges
            lambda_vals = [nodes[i].lambda for i in edge]
            avg_lambda = mean(abs, lambda_vals)

            for i in edge
                total_error = brain.spectral_error[i] + brain.fourier_error[i] + 1e-8f0  # prevent div by zero
                spectral_weight = (brain.fourier_error[i] + 1e-8f0) / total_error
                fourier_weight  = (brain.spectral_error[i] + 1e-8f0) / total_error
                combined_weight = 0.5f0 * (spectral_weight + fourier_weight)

                delta_phi[i] += eta * combined_weight * (avg_lambda - abs(nodes[i].lambda))
            end
        end
    end

    brain.phi .= brain.phi .+ delta_phi
    return brain
end
