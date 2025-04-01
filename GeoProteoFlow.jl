# --- Device Setup ---
using Flux
using CUDA
using Meshes
using GeometryBasics
using LinearAlgebra
using StatsBase
using DifferentialEquations
using Ripserer
using Distances  
using Makie                   # For 3D visualization (GLMakie or CairoMakie)

if CUDA.has_cuda()
    device = gpu
    println("Using device: GPU")
else
    device = cpu
    println("Using device: CPU")
end

# --- Differentiable Lambda: Trainable scaling for c-Ricci ---
# We store log(λ) for stability
log_lambda = param([log(1.0f0)])  # Flux param makes it trainable

# Accessor function
function get_lambda()
    return exp(log_lambda[1])  # scalar λ value
end

# --- Global RT constant ---
const RT = 1.0f0

################################################################################
# Step 1: Define the flat 2D proteoform lattice with ρ(x) -> z(x)
################################################################################

# Define the proteoform states and their flat (x, y) positions
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
flat_pos = Dict(
    "000" => Point3(0.0, 3.0, 0.0),
    "001" => Point3(-2.0, 2.0, 0.0),
    "010" => Point3(0.0, 2.0, 0.0),
    "100" => Point3(2.0, 2.0, 0.0),
    "011" => Point3(-1.0, 1.0, 0.0),
    "101" => Point3(0.0, 1.0, 0.0),
    "110" => Point3(1.0, 1.0, 0.0),
    "111" => Point3(0.0, 0.0, 0.0)
)

# Define neighbor connections (edges by Hamming distance 1)
allowed_edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "101"), ("001", "011"),
    ("010", "110"), ("010", "011"),
    ("011", "111"),
    ("100", "110"), ("100", "101"),
    ("101", "111"), ("110", "111")
]

state_index = Dict(s => i for (i, s) in enumerate(pf_states))
num_states = length(pf_states)

################################################################################
# Step 2: Define occupancy ρ(x) and build curved 3D mesh with z = ρ(x)
################################################################################

function lift_to_3d_mesh(rho::Vector{Float64})
    @assert length(rho) == num_states
    rho = rho ./ sum(rho)  # Ensure volume conservation
    lifted_points = [Point3(flat_pos[s].x, flat_pos[s].y, rho[state_index[s]]) for s in pf_states]

    # Manually define triangles for the R=3 diamond based on connectivity
    triangles = [
        Triangle(1, 2, 3), Triangle(1, 3, 4), Triangle(2, 5, 3),
        Triangle(3, 5, 6), Triangle(3, 6, 7), Triangle(4, 3, 7),
        Triangle(5, 8, 6), Triangle(6, 8, 7)
    ]

    mesh = SimpleMesh(lifted_points, triangles)
    return mesh
end

################################################################################
# Step 3: Compute cotangent Laplacian ∆_T over 3D mesh
################################################################################

function cotangent_laplacian(mesh::SimpleMesh)
    N = length(mesh.points)
    L = zeros(Float64, N, N)
    A = zeros(Float64, N)

    for tri in mesh.connectivity
        i, j, k = tri.indices
        p1, p2, p3 = mesh.points[i], mesh.points[j], mesh.points[k]

        # Edges
        u = p2 - p1
        v = p3 - p1
        w = p3 - p2

        # Angles
        angle_i = acos(clamp(dot(u, v) / (norm(u) * norm(v)), -1, 1))
        angle_j = acos(clamp(dot(-u, w) / (norm(u) * norm(w)), -1, 1))
        angle_k = acos(clamp(dot(-v, -w) / (norm(v) * norm(w)), -1, 1))

        # Cotangent weights
        cot_i = 1 / tan(angle_i)
        cot_j = 1 / tan(angle_j)
        cot_k = 1 / tan(angle_k)

        for (a, b, cot) in ((j,k,cot_i), (i,k,cot_j), (i,j,cot_k))
            L[a,b] -= cot
            L[b,a] -= cot
            L[a,a] += cot
            L[b,b] += cot
        end
    end

    return L
end

################################################################################
# Step 4: Compute c-Ricci = λ ⋅ ∆_T ρ(x)
################################################################################

function compute_c_ricci(rho::Vector{Float64}, λ::Float64 = 1.0)
    mesh = lift_to_3d_mesh(rho)
    L = cotangent_laplacian(mesh)
    rho_norm = rho ./ sum(rho)  # Ensure volume is normalized
    c_ricci = λ .* (L * rho_norm)
    return c_ricci
end

# Example run:
rho_example = rand(num_states)
c_ricci_out = compute_c_ricci(rho_example, 1.0)
println("c-Ricci curvature:", round.(c_ricci_out; digits=4))

################################################################################
# Step 5: Compute anisotropy field A(x) = ∇_T [C-Ricci(x)]
################################################################################

function compute_anisotropy(mesh::SimpleMesh, c_ricci::Vector{Float64})
    A_field = zeros(Float64, num_states)

    for i in 1:num_states
        p_i = mesh.points[i]
        grad_vals = Float64[]
        for j in 1:num_states
            if i == j
                continue
            end
            p_j = mesh.points[j]
            d = norm(p_j - p_i)
            if d > 1e-6
                push!(grad_vals, abs(c_ricci[i] - c_ricci[j]) / d)
            end
        end
        A_field[i] = isempty(grad_vals) ? 0.0 : mean(grad_vals)
    end
    return A_field
end

function update_c_ricci_and_anisotropy(rho::Vector{Float64})
    rho_norm = rho ./ sum(rho)  # Ensure volume-conserving
    λ = get_lambda()

    # Step 1: Lift to 3D mesh
    mesh = lift_to_3d_mesh(rho_norm)

    # Step 2: Cotangent Laplacian
    L = cotangent_laplacian(mesh)

    # Step 3: Compute c-Ricci = λ ⋅ ∆ₜρ(x)
    c_ricci = λ .* (L * rho_norm)

    # Step 4: Compute anisotropy ∇(c-Ricci) from neighbors
    A_field = compute_anisotropy(mesh, c_ricci)

    return c_ricci, A_field, mesh  # Also return the mesh if needed for visualization
end

################################################################################
# Step 6: Sheath theory
################################################################################

function sheaf_consistency_relative(mesh::SimpleMesh; tolerance=0.2)
    inconsistencies = []

    for (u, v) in allowed_edges
        i = state_index[u]
        j = state_index[v]
        p0_u = flat_pos[u]
        p0_v = flat_pos[v]
        d0 = norm(p0_u - p0_v)

        pt_u = mesh.points[i]
        pt_v = mesh.points[j]
        dt = norm(pt_u - pt_v)

        δ = abs(dt - d0) / d0
        if δ > tolerance
            push!(inconsistencies, (u, v, round(δ, digits=3)))
        end
    end

    return inconsistencies
end

################################################################################
# Step 7: ALIVE ODE Action LImited eVolution Engine
################################################################################

# Global Parameters
const MAX_MOVES_PER_STEP = 10
const PDE_UPDATE_INTERVAL = 1
global_pde_step_counter = Ref(0)  # mutable counter

# Degeneracy map (bitwise 1-count)
degeneracy_map = Dict(0 => 1, 1 => 3, 2 => 3, 3 => 1)

# === ALIVE ODE: Action-Limited Evolution Engine ===
function oxi_shapes_ode_alive(t, rho::Vector{Float64})
    rho = rho ./ sum(rho)  # volume conservation
    counts = round.(Int, rho .* 100)
    counts[end] = 100 - sum(counts[1:end-1])  # force normalization
    rho = counts ./ 100.0

    global_pde_step_counter[] += 1
    if global_pde_step_counter[] % PDE_UPDATE_INTERVAL == 0
        global c_ricci, anisotropy, _ = update_c_ricci_and_anisotropy(rho)
    end

    inflow = zeros(Float64, num_states)
    outflow = zeros(Float64, num_states)
    total_moves = rand(0:MAX_MOVES_PER_STEP)
    candidate_indices = findall(>(0), counts)

    for _ in 1:total_moves
        isempty(candidate_indices) && break
        i = rand(candidate_indices)
        s = pf_states[i]
        neighbors = get(neighbor_indices, s, [])
        isempty(neighbors) && (inflow[i] += 0.01; continue)

        probs = Float64[]
        for j in neighbors
            delta_S = compute_entropy_cost(i, j, rho)
            delta_f = (1.0 * exp(rho[i]) - c_ricci[i] + delta_S) / RT
            p_ij = exp(-delta_f) * exp(-anisotropy[j])
            push!(probs, p_ij)
        end

        if sum(probs) < 1e-8
            inflow[i] += 0.01
            continue
        end

        probs ./= sum(probs)
        target_idx = neighbors[sample(DiscreteNonParametric(1:length(probs), probs))]
        inflow[target_idx] += 0.01
        outflow[i] += 0.01
    end

    for i in 1:num_states
        inflow[i] += (counts[i] / 100.0) - outflow[i]
    end

    return inflow .- outflow
end

function compute_entropy_cost(i::Int, j::Int, rho::Vector{Float64})
    baseline_DeltaE = 1.0
    mass_heat = 0.1
    reaction_heat = 0.01 * baseline_DeltaE
    conformational_cost = abs(c_ricci[j])
    deg = degeneracy_map[count("1", pf_states[j])]
    degeneracy_penalty = 1.0 / deg
    return mass_heat + reaction_heat + conformational_cost + degeneracy_penalty
end

function dominant_geodesic(traj::Vector{String}, geodesics::Vector{Vector{String}})
    max_score = 0
    best_path = nothing
    for path in geodesics
        score = count(s -> s in path, traj)
        if score > max_score
            max_score = score
            best_path = path
        end
    end
    return best_path
end

function evolve_time_series_and_geodesic(rho0::Vector{Float64}, t_span)
    prob = ODEProblem(oxi_shapes_ode_alive, rho0, (t_span[1], t_span[end]))
    sol = solve(prob, Tsit5(), saveat=t_span)
    traj = [pf_states[argmax(r)] for r in sol.u]
    geo = dominant_geodesic(traj, geodesics)
    return sol, geo
end

global geodesics = [
    ["000", "100", "101", "111"],
    ["000", "100", "110", "111"],
    ["000", "010", "110", "111"],
    ["000", "010", "011", "111"],
    ["000", "001", "101", "111"],
    ["000", "001", "011", "111"]
]

function geodesic_loss(pred_final::Matrix{Float64}, initial::Matrix{Float64})
    max_idx_pred = argmax(pred_final, dims=1)
    max_idx_init = argmax(initial, dims=1)
    batch_loss = 0.0
    for i in 1:size(pred_final, 2)
        path = [pf_states[max_idx_init[i]], pf_states[max_idx_pred[i]]]
        valid = any(all(s in g for s in path) for g in geodesics)
        if !valid
            batch_loss += 1.0
        end
    end
    return batch_loss / size(pred_final, 2)
end

# --- Generate Systematic Initial Distributions in Julia ---

function generate_systematic_initials()
    initials = []

    # (1) Single occupancy states (pure i-states)
    for i in 1:8
        v = zeros(Float64, 8)
        v[i] = 1.0
        push!(initials, v)
    end

    # (2) Flat distribution
    push!(initials, fill(1.0 / 8, 8))

    # (3) Curved in k=1 (peak at "010")
    curved_k1 = [0.0, 0.15, 0.7, 0.15, 0.0, 0.0, 0.0, 0.0]
    push!(initials, curved_k1 ./ sum(curved_k1))

    # (4) Curved in k=2 (peak at "101")
    curved_k2 = [0.0, 0.0, 0.0, 0.0, 0.15, 0.7, 0.15, 0.0]
    push!(initials, curved_k2 ./ sum(curved_k2))

    # (5) Flat k=0/1 with peak in k=2
    hybrid = [0.05, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.05]
    push!(initials, hybrid ./ sum(hybrid))

    # (6) Bell-shaped curve across i-space
    bell = [0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.2]
    push!(initials, bell ./ sum(bell))

    # (7) Left-to-right gradient
    gradient = collect(range(0.1, 0.9, length=8))
    push!(initials, gradient ./ sum(gradient))

    return initials
end

# Test output
for (i, dist) in enumerate(generate_systematic_initials())
    println("Initial ", i, ": ", round.(dist; digits=3))
end
# Persistent homology diagram using Ripserer.jl
function persistent_diagram(rho::Vector{Float64})
    D = pairwise(Euclidean(), rho, dims=1)
    result = ripser(D, dim_max=1, metric=:precomputed)
    return result
end

# Compute persistent entropy from H0 diagram
function topological_entropy(diagrams)
    if isempty(diagrams) || isempty(diagrams[1])
        return 0.0
    end
    births = diagrams[1][!, :birth]
    deaths = diagrams[1][!, :death]
    lifespans = deaths .- births
    probs = lifespans ./ sum(lifespans)
    entropy = -sum(p * log2(p + 1e-10) for p in probs)
    return entropy
end

println("✅ Finished data generation.")
println("Most traversed geodesics:")
for (path, count) in sort(collect(geo_counter), by = x -> -x[2])
    println(join(path, " → "), " | Count: ", count)
end

################################################################################
# Step 8: OxiFlowNet Neural Network (Flux MLP)
################################################################################

# Define model architecture (similar to PyTorch version)
struct OxiNet
    fc1::Dense
    fc2::Dense
    fc3::Dense
end

# Constructor
function OxiNet(input_dim::Int=8, hidden_dim::Int=32, output_dim::Int=8)
    return OxiNet(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, output_dim)
    )
end

# Forward pass definition
(m::OxiNet)(x) = m.fc3(m.fc2(m.fc1(x)))

# Instantiate the model
oxinet = OxiNet()

# Example usage
x_example = rand(Float32, 8)
y_pred = oxinet(x_example)  # outputs raw vector (no softmax)

println("✅ OxiNet initialized. Sample output:")
println(round.(y_pred; digits=4))

################################################################################
# Step 10: Training and Evaluation Functions (Flux)
################################################################################

using Flux: mse, train!, ADAM

# Volume constraint: preserves ∑ρ = 1
function volume_constraint(pred, target)
    return mean((sum(pred, dims=1) .- sum(target, dims=1)).^2)
end

# Topology constraint: compare supports (binary presence of mass)
function topology_constraint(pred, target; threshold=0.05)
    support_pred = pred .> threshold
    support_true = target .> threshold
    return mean((support_pred .- support_true).^2)
end

# Geodesic constraint: ensure evolution stays on a valid path
function geodesic_loss(pred, initial, geodesics::Vector{Vector{String}}, pf_states)
    pred_idx = findmax.(eachcol(pred)) .|> x -> x[2]
    init_idx = findmax.(eachcol(initial)) .|> x -> x[2]
    loss = 0.0
    for i in 1:length(pred_idx)
        path = [pf_states[init_idx[i]], pf_states[pred_idx[i]]]
        valid = any(all(in(path), g) for g in geodesics)
        loss += valid ? 0.0 : 1.0
    end
    return loss / length(pred_idx)
end

# Training loop
function train_model!(model, X_train, Y_train, X_val, Y_val;
    epochs=100, lr=1e-3, λ_topo=0.5, λ_vol=0.5, λ_geo=0.5, geodesics=nothing
)
    opt = ADAM(lr)
    loss_log = []

    for epoch in 1:epochs
        grads = Flux.gradient(Flux.params(model)) do
            raw_pred = model(X_train)
            pred = round.(raw_pred .* 100) ./ 100
            pred ./= sum(pred; dims=1)

            loss_main = mse(pred, Y_train)
            loss_vol = volume_constraint(pred, Y_train)
            loss_topo = topology_constraint(pred, Y_train)
            loss_geo = geodesic_loss(pred, X_train, geodesics, pf_states)

            total = loss_main + λ_vol * loss_vol + λ_topo * loss_topo + λ_geo * loss_geo
            push!(loss_log, total)
            return total
        end

        Flux.Optimise.update!(opt, Flux.params(model), grads)

        if epoch % 10 == 0
            raw_val = model(X_val)
            val_pred = round.(raw_val .* 100) ./ 100
            val_pred ./= sum(val_pred; dims=1)
            val_loss = mse(val_pred, Y_val)
            println("Epoch $epoch | Loss = $(round(loss_log[end], digits=5)) | Val MSE = $(round(val_loss, digits=5))")
        end
    end
end

# Evaluation
function evaluate_model(model, X_test, Y_test)
    raw_pred = model(X_test)
    pred = round.(raw_pred .* 100) ./ 100
    pred ./= sum(pred; dims=1)
    return pred, mse(pred, Y_test)
end


