# =============================================================================
# Dynamic MetaLambda Reasoner — GeoObject5
# =============================================================================

function dynamic_metalambda_reasoner!(ρ_real::Vector{Float32}, ρ_true::Vector{Float32},
                                      brain::HypergraphBrain1; top_k=10, η=0.05f0)
    sim = Float32[]
    for i in 1:length(brain.lambda)
        Δ = abs.([
            mean(ρ_real) - brain.lambda[i],
            mean(brain.phi) - brain.phi[i],
            mean(brain.psi) - brain.psi[i],
            mean(brain.curvature) - brain.curvature[i],
            mean(brain.anisotropy) - brain.anisotropy[i],
            mean(brain.action_cost) - brain.action_cost[i]
        ])
        push!(sim, 1f0 / (sum(Δ) + 1e-5f0))
    end

    idx = partialsortperm(sim, rev=true, 1:top_k)
    w = softmax(sim[idx])
    ρ_pred = zeros(Float32, length(ρ_real))

    for (j, i) in enumerate(idx)
        v = zeros(Float32, length(ρ_real))
        for (pat, x) in brain.fourier_features[i]
            k = count(==('1'), pat[1]) + 1
            k ≤ length(v) && (v[k] += Float32(x))
        end
        ρ_pred .+= w[j] * (v ./ max(sum(v), 1e-5f0))
    end

    ρ_pred ./= max(sum(ρ_pred), 1e-5f0)
    err = mean(abs.(ρ_true .- ρ_pred))

    for (j, i) in enumerate(idx)
        grad = err * w[j]
        brain.phi[i] -= η * grad
        brain.psi[i] += η * grad
    end

    return ρ_pred, err
end

function reinforced_reasoner!(
    ρ_real::Vector{Float32}, 
    ρ_true::Vector{Float32},
    brain::HypergraphBrain1; 
    top_k::Int = 10, 
    n_samples::Int = 3, 
    η::Float32 = 0.05f0
)
    # --- Compute similarity for top_k node candidates ---
    sim = Float32[]
    for i in 1:length(brain.lambda)
        Δ = abs.([
            mean(ρ_real) - brain.lambda[i],
            mean(brain.phi) - brain.phi[i],
            mean(brain.psi) - brain.psi[i],
            mean(brain.curvature) - brain.curvature[i],
            mean(brain.anisotropy) - brain.anisotropy[i]
        ])
        push!(sim, 1f0 / (sum(Δ) + 1e-5f0))
    end

    idx = partialsortperm(sim, rev=true, 1:top_k)
    weights = softmax(sim[idx])

    # --- Generate samples ---
    function sample_prediction(brain, idx, weights)
        ρ_pred = zeros(Float32, length(ρ_real))
        for (j, i) in enumerate(idx)
            for (pat, val) in brain.fourier_features[i]
                k = count(==('1'), pat[1]) + 1
                k ≤ length(ρ_pred) && (ρ_pred[k] += weights[j] * Float32(val))
            end
        end
        ρ_pred ./= max(sum(ρ_pred), 1e-5f0)
        return ρ_pred
    end

    samples = [sample_prediction(brain, idx, weights) for _ in 1:n_samples]
    errors = [mean(abs.(ρ_true .- pred)) for pred in samples]
    best_idx = argmin(errors)

    # --- Reinforce best sample ---
    for i in 1:n_samples
        sign = i == best_idx ? -1f0 : 1f0
        for (j, node_id) in enumerate(idx)
            grad = errors[i] * weights[j]
            brain.phi[node_id] += sign * η * grad
            brain.psi[node_id] -= sign * η * grad
        end
    end

    return samples[best_idx], errors[best_idx]
end

