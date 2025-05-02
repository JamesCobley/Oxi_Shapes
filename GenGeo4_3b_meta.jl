# =============================================================================
# GeoBrain: MetaLambda Hypergraph Construction
# =============================================================================

using LinearAlgebra
using SparseArrays
using GeometryBasics
using Graphs
using StatsBase
using Base: union, intersect
using Clustering
using Statistics: mean
using Random
using UUIDs
using BSON: @save, @load
using Dates

# =============================================================================
# Load Required Data
# =============================================================================
batch_id = "batch_20250502_120445"

@load "/content/flow_traces_$(batch_id).bson" flow_traces
@load "/content/wavelet_embeddings_$(batch_id).bson" wavelet_embeddings
@load "/content/trace_spectral_analysis_$(batch_id).bson" spectral_data

spectral_features = spectral_data[:features]

println("✔ Loaded all required data files.")

# =============================================================================
# MetaLambda Data Structures
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
    edges::Dict{Symbol, Vector{Tuple{Int, Int}}}
end

# =============================================================================
# Utility: Similarity Functions
# =============================================================================

function cosine_similarity(x::Vector{Float64}, y::Vector{Float64})
    return dot(x, y) / (norm(x) * norm(y) + 1e-8)
end

function jaccard_similarity(a::Dict, b::Dict)
    keys_a = Set(keys(a))
    keys_b = Set(keys(b))
    inter = length(intersect(keys_a, keys_b))
    union_size = length(union(keys_a, keys_b))
    return inter / (union_size + 1e-8)
end

# =============================================================================
# Core: Build Nodes
# =============================================================================

function build_metalambda_nodes(flow_traces, wavelet_embeddings, spectral_features)
    nodes = MetaLambdaNode[]
    for i in 1:length(flow_traces)
        trace = flow_traces[i]
        wavelet = wavelet_embeddings[i]
        spectral = spectral_features[i]

        curvature = mean([mean(abs.(r)) for r in trace.R_series])
        λ = mean(abs.(trace.ρ_series[end] .- trace.ρ_series[1]))  # Divergence estimate
        cost = trace.action_cost

        push!(nodes, MetaLambdaNode(i, wavelet, spectral, λ, curvature, cost))
    end
    return nodes
end

# =============================================================================
# Core: Build Edges via Similarity
# =============================================================================

function build_metalambda_edges(nodes::Vector{MetaLambdaNode};
                                wavelet_thresh=0.95,
                                spectral_thresh=0.5,
                                λ_thresh=0.05)

    wavelet_edges = Tuple{Int, Int}[]
    spectral_edges = Tuple{Int, Int}[]
    λ_edges = Tuple{Int, Int}[]

    for i in 1:length(nodes), j in (i+1):length(nodes)
        node_i = nodes[i]
        node_j = nodes[j]

        sim = mean([
            cosine_similarity(node_i.wavelet[k], node_j.wavelet[k])
            for k in intersect(keys(node_i.wavelet), keys(node_j.wavelet))
        ])
        if sim > wavelet_thresh
            push!(wavelet_edges, (i, j))
        end

        spec_sim = jaccard_similarity(node_i.spectral, node_j.spectral)
        if spec_sim > spectral_thresh
            push!(spectral_edges, (i, j))
        end

        if abs(node_i.λ - node_j.λ) < λ_thresh
            push!(λ_edges, (i, j))
        end
    end

    return Dict(
        :wavelet => wavelet_edges,
        :spectral => spectral_edges,
        :divergence => λ_edges
    )
end

# =============================================================================
# MetaLambda Hypergraph Builder
# =============================================================================

function build_metalambda_hypergraph(flow_traces, wavelet_embeddings, spectral_features)
    nodes = build_metalambda_nodes(flow_traces, wavelet_embeddings, spectral_features)
    edges = build_metalambda_edges(nodes)
    return MetaLambdaHypergraph(nodes, edges)
end

# =============================================================================
# Save Output
# =============================================================================

meta = build_metalambda_hypergraph(flow_traces, wavelet_embeddings, spectral_features)
@save "metalambda_hypergraph_$(batch_id).bson" meta
println("✔ MetaLambda hypergraph constructed and saved.")
