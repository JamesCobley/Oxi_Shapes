# =============================================================================
# MetaLambda + Ricci Flow Brain: Integrated Geometric Learning Framework
# =============================================================================

using LinearAlgebra
using SparseArrays
using Graphs
using Clustering
using Statistics
using BSON: @save, @load

# =============================================================================
# Load Data (assumes batch_id defined)
# =============================================================================
@load "/content/flow_traces_batch_20250504_152322.bson" flow_traces
@load "/content/wavelet_embeddings_batch_20250504_152322.bson" wavelet_embeddings
@load "/content/trace_spectral_analysis_batch_20250504_152322.bson" spectral_data
spectral_features = spectral_data[:features]
basis = spectral_data[:basis]
@load "/content/simplex_batch_20250504_152322.bson" simplex
simplex = [Vector{GeoNode}(run) for run in simplex]

# =============================================================================
# Structures
# =============================================================================

struct MetaLambdaNode
    id::Int
    wavelet::Dict{Symbol, Vector{Float64}}
    spectral::Dict{Vector{String}, Int}
    λ::Float32
    curvature::Float32
    action_cost::Float32
end

struct MetaLambdaHypergraph
    nodes::Vector{MetaLambdaNode}
    edges::Dict{Symbol, Vector{Tuple{Int, Int, Float64}}}
    clusters::Vector{Int}
end

struct HypergraphBrain1
    edge_sets::Dict{Symbol, Vector{Vector{Int}}}
    weights::Dict{Symbol, Vector{Float32}}
    phi::Vector{Float32}
    psi::Vector{Float32}
    lambda::Vector{Float32}
    L::SparseMatrixCSC{Float32, Int}
    modes::Matrix{Float32}
    trace_patterns::Vector{Dict{Vector{String}, Int}}
    delta_commutators::Vector{Float32}
    fourier_features::Vector{Dict{Vector{String}, Int}}
    fourier_basis::Vector{Vector{String}}
    spectral_error::Vector{Float32}
    fourier_error::Vector{Float32}
end

# =============================================================================
# Utility Functions
# =============================================================================
cosine_similarity(x, y) = dot(x, y) / (norm(x) * norm(y) + 1e-8)

function jaccard_similarity(a::Dict, b::Dict)
    keys_a = Set(keys(a))
    keys_b = Set(keys(b))
    inter = length(intersect(keys_a, keys_b))
    union_size = length(union(keys_a, keys_b))
    return inter / (union_size + 1e-8)
end

function group_by(assignments)
    bin_dict = Dict{Int, Vector{Int}}()
    for (i, label) in enumerate(assignments)
        push!(get!(bin_dict, label, Int[]), i)
    end
    return collect(values(bin_dict))
end

function spectral_bin_indices(modes::Matrix{T}, k::Int) where T<:AbstractFloat
    embeddings = modes[:, 1:min(k, size(modes, 2))]
    normalized = mapslices(x -> x / (norm(x) + 1e-8), embeddings; dims=1)
    result = kmeans(normalized', k; maxiter=300)
    return result.assignments
end

# =============================================================================
# Build Hypergraph Brain
# =============================================================================

function build_hypergraph_from_simplex(simplex, trace_metadata, trace_features, trace_basis;
                                       mode_rank=10, spectral_bins=5, alpha=1.0f0, beta=1.0f0)

    nodes = reduce(vcat, simplex)
    N = length(nodes)

    # Laplacian and eigenmodes
    lambda_surface = build_simplex_surface(simplex)
    L = build_simplex_laplacian(lambda_surface)
    evals, evecs = eigen(Matrix(L))
    modes = evecs[:, 1:min(mode_rank, size(evecs, 2))]

    # Fields
    lambda_vals = [n.lambda for n in nodes]
    local_energy = [sum(abs.(n.R_real)) for n in nodes]
    local_entropy = [sum(abs.(n.flux)) for n in nodes]
    psi = [alpha * e + beta * s for (e, s) in zip(local_energy, local_entropy)]
    phi = zeros(Float32, N)

    # Spectral binning
    edge_sets = Dict{Symbol, Vector{Vector{Int}}}()
    edge_sets[:curvature] = group_by(spectral_bin_indices(modes, spectral_bins))
    edge_sets[:divergence] = group_by(spectral_bin_indices(modes, spectral_bins))
    edge_sets[:action_cost] = group_by(spectral_bin_indices(modes, spectral_bins))

    # Fourier-like grouping
    trace_bin_groups = Dict{Int, Vector{Int}}()
    for (i, feats) in enumerate(trace_features)
        score = sum(values(feats))
        push!(get!(trace_bin_groups, score, Int[]), i)
    end
    edge_sets[:fourier_spectral] = collect(values(trace_bin_groups))

    # Weights
    weights = Dict{Symbol, Vector{Float32}}()
    for k in keys(edge_sets)
        weights[k] = ones(Float32, length(edge_sets[k]))
    end

    return HypergraphBrain1(
        edge_sets,
        weights,
        phi,
        psi,
        lambda_vals,
        sparse(L),
        modes,
        [m[:patterns] for m in trace_metadata],
        [m[:delta] for m in trace_metadata],
        trace_features,
        trace_basis,
        zeros(Float32, N),
        zeros(Float32, N)
    )
end

# =============================================================================
# Ricci Flow Learning (with λ convergence and φ history)
# =============================================================================

function update_info_potential!(brain::HypergraphBrain1, simplex;
                                eta::Float32 = 0.1f0)
    nodes = reduce(vcat, simplex)
    N = length(nodes)
    delta_phi = zeros(Float32, N)

    for (etype, edges) in brain.edge_sets
        for edge in edges
            lambda_vals = [nodes[i].lambda for i in edge]
            avg_lambda = mean(abs, lambda_vals)

            for i in edge
                total_error = brain.spectral_error[i] + brain.fourier_error[i] + 1e-8f0
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

function ricci_flow_learn!(brain::HypergraphBrain1, simplex;
                           eta::Float32 = 0.1f0,
                           max_steps::Int = 20,
                           tol::Float32 = 1e-4)

    nodes = reduce(vcat, simplex)
    phi_history = [copy(brain.phi)]
    prev_lambda = [abs(n.lambda) for n in nodes]

    for step in 1:max_steps
        println("→ Ricci flow step $step")
        update_info_potential!(brain, simplex; eta=eta)
        push!(phi_history, copy(brain.phi))

        curr_lambda = [abs(n.lambda) for n in nodes]
        delta = maximum(abs.(curr_lambda .- prev_lambda))
        println("   λ smoothness Δ = $delta")

        if delta < tol
            println("✅ Converged at step $step (Δλ < $tol)")
            break
        end

        prev_lambda = curr_lambda
    end

    return brain, phi_history
end

# =============================================================================
# Run Full Learning
# =============================================================================
brain = build_hypergraph_from_simplex(simplex, trace_metadata, spectral_features, basis)
brain, phi_history = ricci_flow_learn!(brain, simplex; eta=0.1f0, max_steps=20, tol=1e-4)

@save "ricci_learned_brain_$(batch_id).bson" brain phi_history
println("✔ Ricci flow learning complete and brain + φ evolution saved.")
