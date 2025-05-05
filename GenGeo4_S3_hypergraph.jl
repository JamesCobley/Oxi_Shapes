# =============================================================================
# MetaLambda Ricci Flow Learner – Hypergraph Construction & Training
# =============================================================================

using LinearAlgebra
using SparseArrays
using Clustering
using Statistics
using BSON: @save, @load

# =============================================================================
# Load Data (adjust your batch ID accordingly)
# =============================================================================

struct GeoNode
    ρ_real::Vector{Float32}
    R_real::Vector{Float32}
    A_real::Vector{Float32}
    ρ_imag::Vector{Float32}
    R_imag::Vector{Float32}
    A_imag::Vector{Float32}
    lambda::Float32
    sheath_stress::Vector{Float32}
    flux::Vector{Float32}
    action_cost::Float32
end

batch_id = "20250505_080315"
@load "/content/flow_traces_batch_$batch_id.bson" flow_traces
@load "/content/trace_metadata_batch_$(batch_id).bson" trace_metadata
@load "/content/wavelet_embeddings_batch_$batch_id.bson" wavelet_embeddings
@load "/content/trace_spectral_analysis_batch_$batch_id.bson" spectral_data
spectral_features = spectral_data[:features]
basis = spectral_data[:basis]
@load "/content/simplex_batch_$batch_id.bson" simplex
simplex = [Vector{GeoNode}(run) for run in simplex]  # GeoNode must be in scope!

# =============================================================================
# Structs
# =============================================================================

struct HypergraphBrain
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
    curvature::Vector{Float32}
    anisotropy::Vector{Float32}
    action_cost::Vector{Float32}   # ✅ ADD THIS
end

# =============================================================================
# Utilities
# =============================================================================

cosine_similarity(x, y) = dot(x, y) / (norm(x) * norm(y) + 1e-8)

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

function build_simplex_surface(simplex::Vector{Vector{GeoNode}})
    n_runs = length(simplex)
    rollout_steps = maximum(length(run) for run in simplex)
    lambda_surface = fill(0f0, rollout_steps, n_runs)

    for (r, run) in enumerate(simplex)
        for (t, node) in enumerate(run)
            lambda_surface[t, r] = -node.lambda
        end
    end

    return lambda_surface
end

function build_simplex_laplacian(lambda_surface::Matrix{Float32})
    T, R = size(lambda_surface)
    n = T * R
    W = zeros(Float32, n, n)

    for t in 1:T
        for r in 1:R
            idx = (t-1)*R + r
            if t < T
                W[idx, idx+R] = 1.0f0
                W[idx+R, idx] = 1.0f0
            end
            if r < R
                W[idx, idx+1] = 1.0f0
                W[idx+1, idx] = 1.0f0
            end
        end
    end

    D = Diagonal(vec(sum(W, dims=2)))  # ✅ Fix: convert to vector
    L = D - W

    return L
end

# =============================================================================
# Build Hypergraph Brain
# =============================================================================

function build_hypergraph_from_simplex(simplex, trace_metadata, trace_features, trace_basis;
                                       mode_rank=10, spectral_bins=5, alpha=1.0f0, beta=1.0f0)
    nodes = reduce(vcat, simplex)
    N = length(nodes)

    lambda_surface = build_simplex_surface(simplex)
    L = build_simplex_laplacian(lambda_surface)
    evals, evecs = eigen(Matrix(L))
    modes = evecs[:, 1:min(mode_rank, size(evecs, 2))]

    lambda_vals = [n.lambda for n in nodes]
    local_energy = [sum(abs.(n.R_real)) for n in nodes]
    local_entropy = [sum(abs.(n.flux)) for n in nodes]
    psi = [alpha * e + beta * s for (e, s) in zip(local_energy, local_entropy)]
    phi = zeros(Float32, N)

    curvature_vals = [mean(n.R_real) for n in nodes]
    anisotropy_vals = [mean(n.A_real) for n in nodes]
    action_costs = [n.action_cost for n in nodes]  # ✅ New

    edge_sets = Dict{Symbol, Vector{Vector{Int}}}()
    edge_sets[:curvature] = group_by(spectral_bin_indices(modes, spectral_bins))
    edge_sets[:divergence] = group_by(spectral_bin_indices(modes, spectral_bins))
    edge_sets[:action_cost] = group_by(spectral_bin_indices(modes, spectral_bins))

    trace_bin_groups = Dict{Int, Vector{Int}}()
    for (i, feats) in enumerate(trace_features)
        score = sum(values(feats))
        push!(get!(trace_bin_groups, score, Int[]), i)
    end
    edge_sets[:fourier_spectral] = collect(values(trace_bin_groups))

    weights = Dict{Symbol, Vector{Float32}}()
    for k in keys(edge_sets)
        weights[k] = ones(Float32, length(edge_sets[k]))
    end
    
    return HypergraphBrain(
        edge_sets, weights, phi, psi, lambda_vals, sparse(L), modes,
        [m[:patterns] for m in trace_metadata],
        [m[:delta] for m in trace_metadata],
        trace_features, trace_basis,
        zeros(Float32, N), zeros(Float32, N),
        curvature_vals, anisotropy_vals,
        action_costs  # ✅ New final argument
    )
end

# =============================================================================
# Ricci Flow Learner
# =============================================================================

function update_info_potential!(brain::HypergraphBrain, simplex; eta::Float32=0.1f0)
    nodes = reduce(vcat, simplex)
    N = length(nodes)
    delta_phi = zeros(Float32, N)

    for (etype, edges) in brain.edge_sets
        for edge in edges
            lambda_vals = [nodes[i].lambda for i in edge]
            avg_lambda = mean(abs, lambda_vals)
            for i in edge
                total_error = brain.spectral_error[i] + brain.fourier_error[i] + 1f-8
                spectral_weight = (brain.fourier_error[i] + 1f-8) / total_error
                fourier_weight  = (brain.spectral_error[i] + 1f-8) / total_error
                combined_weight = 0.5f0 * (spectral_weight + fourier_weight)
                curvature_factor = 1f0 + abs(brain.curvature[i])
                anisotropy_factor = 1f0 + abs(brain.anisotropy[i])
                geom_weight = curvature_factor * anisotropy_factor
                delta_phi[i] += eta * combined_weight * geom_weight * (avg_lambda - abs(nodes[i].lambda))
            end
        end
    end

    brain.phi .= brain.phi .+ delta_phi
    return brain
end

function ricci_flow_learn!(brain::HypergraphBrain, simplex;
                           eta::Float32 = 0.1f0,
                           max_steps::Int = 20,
                           tol::Float32 = 0.00001f0)
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
# Run & Save
# =============================================================================

brain = build_hypergraph_from_simplex(simplex, trace_metadata, spectral_features, basis)
brain, phi_history = ricci_flow_learn!(brain, simplex; eta=0.1f0, max_steps=20, tol=0.00001f0)
@save "ricci_learned_brain_$(batch_id).bson" brain phi_history
println("✔ Ricci flow learning complete and brain + φ evolution saved.")
