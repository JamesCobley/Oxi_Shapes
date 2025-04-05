using CairoMakie
using GeometryBasics: Point2, Point3
using Interpolations
using Random
using LinearAlgebra

CairoMakie.activate!()

# === Proteoform Setup ===
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
flat_pos = Dict(
    "000" => (0.0, 3.0), "001" => (-2.0, 2.0), "010" => (0.0, 2.0),
    "100" => (2.0, 2.0), "011" => (-1.0, 1.0), "101" => (0.0, 1.0),
    "110" => (1.0, 1.0), "111" => (0.0, 0.0)
)
edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "011"), ("001", "101"), ("010", "011"), ("010", "110"),
    ("100", "110"), ("100", "101"), ("011", "111"), ("101", "111"), ("110", "111")
]

# === Geometry Update ===
function lift_to_z_plane(rho, pf_states, flat_pos)
    return [Point3(flat_pos[s][1], flat_pos[s][2], -rho[i]) for (i, s) in enumerate(pf_states)]
end

function compute_R(points3D, flat_pos, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    R = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        p0 = flat_pos[s]; p3 = points3D[i]
        neighbors = [v for (u, v) in edges if u == s]
        append!(neighbors, [u for (u, v) in edges if v == s])
        for n in neighbors
            j = idx[n]; q0 = flat_pos[n]; q3 = points3D[j]
            d0 = norm([p0[1] - q0[1], p0[2] - q0[2]])
            d3 = norm(p3 - q3)
            R[i] += d3 - d0
        end
    end
    return R
end

function compute_c_ricci_dirichlet(R, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    C_R = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        neighbors = [v for (u, v) in edges if u == s]
        append!(neighbors, [u for (u, v) in edges if v == s])
        for n in neighbors
            j = idx[n]
            C_R[i] += (R[i] - R[j])^2
        end
    end
    return C_R
end

function compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = Dict(s => Int[] for s in pf_states)
    for (u, v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end
    anisotropy = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        nbrs = neighbor_indices[s]
        grad_vals = Float64[]
        for j in nbrs
            dist = norm([
                flat_pos[s][1] - flat_pos[pf_states[j]][1],
                flat_pos[s][2] - flat_pos[pf_states[j]][2]
            ])
            if dist > 1e-6
                push!(grad_vals, abs(C_R_vals[i] - C_R_vals[j]) / dist)
            end
        end
        anisotropy[i] = isempty(grad_vals) ? 0.0 : sum(grad_vals) / length(grad_vals)
    end
    return anisotropy
end

function update_geometry_from_rho(ρ, pf_states, flat_pos, edges)
    points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
    R_vals = compute_R(points3D, flat_pos, pf_states, edges)
    C_R_vals = compute_c_ricci_dirichlet(R_vals, pf_states, edges)
    anisotropy_vals = compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)
    return points3D, R_vals, C_R_vals, anisotropy_vals
end

# === Sheaf Setup ===
function initialize_sheaf_stalks(flat_pos, pf_states)
    stalks = Dict{String, Vector{Float64}}()
    for s in pf_states
        stalks[s] = [flat_pos[s][1], flat_pos[s][2]]
    end
    return stalks
end

function sheaf_consistency(stalks, edges; threshold=2.5)
    inconsistencies = []
    for (u, v) in edges
        diff = norm(stalks[u] .- stalks[v])
        if diff > threshold
            push!(inconsistencies, (u, v, diff))
        end
    end
    return inconsistencies
end

# === Plotting ===
function plot_c_ricci_surface_interpolated(flat_pos, field, pf_states)
    fig = Figure(size=(900, 700))
    ax = Axis3(fig[1, 1], title="Interpolated C-Ricci Surface Field", perspectiveness=0.8)

    xs = [flat_pos[s][1] for s in pf_states]
    ys = [flat_pos[s][2] for s in pf_states]

    grid_x = LinRange(minimum(xs)-0.5, maximum(xs)+0.5, 100)
    grid_y = LinRange(minimum(ys)-0.5, maximum(ys)+0.5, 100)
    grid_z = fill(NaN, length(grid_x), length(grid_y))

    for (i, s) in enumerate(pf_states)
        x_idx = findmin(abs.(grid_x .- flat_pos[s][1]))[2]
        y_idx = findmin(abs.(grid_y .- flat_pos[s][2]))[2]
        grid_z[x_idx, y_idx] = field[i]
    end

    interp_func = interpolate(grid_z, BSpline(Linear()))
    interp_func_itp = extrapolate(interp_func, NaN)
    surface!(ax, grid_x, grid_y, (x, y) -> interp_func_itp[x, y], colormap=:viridis)

    for (i, s) in enumerate(pf_states)
        x, y = flat_pos[s]
        z = field[i]
        scatter!(ax, [x], [y], [z], markersize=10, color=:black)
        text!(ax, s, position=(x, y, z + 0.05), align=(:center, :bottom), fontsize=14)
    end

    Colorbar(fig[1, 2], limits=extrema(field), colormap=:viridis, label="C-Ricci (Dirichlet)")
    fig
end

function compute_entropy_cost(i, j, C_R_vals, pf_states)
    baseline_DeltaE = 1.0
    mass_heat = 0.1
    reaction_heat = 0.01 * baseline_DeltaE
    conformational_cost = abs(C_R_vals[j])
    degeneracy_map = Dict(0 => 1, 1 => 3, 2 => 3, 3 => 1)
    deg = degeneracy_map[count(c -> c == '1', pf_states[j])]
    degeneracy_penalty = 1.0 / deg
    return mass_heat + reaction_heat + conformational_cost + degeneracy_penalty
end

function oxi_shapes_alive!(ρ, pf_states, flat_pos, edges; max_moves=20)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    neighbor_indices = Dict(s => Int[] for s in pf_states)
    for (u, v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end

    # Convert to molecule counts
    counts = round.(ρ * 100)
    counts[end] = 100 - sum(counts[1:end-1])
    ρ .= counts / 100

    # Update geometry based on current ρ
    points3D, R_vals, C_R_vals, anisotropy_vals = update_geometry_from_rho(ρ, pf_states, flat_pos, edges)

    inflow = zeros(Float64, length(pf_states))
    outflow = zeros(Float64, length(pf_states))

    total_moves = rand(0:max_moves)
    candidate_indices = findall(x -> x > 0, counts)

    for _ in 1:total_moves
        isempty(candidate_indices) && break
        i = rand(candidate_indices)
        s = pf_states[i]

        nbrs = neighbor_indices[s]
        if isempty(nbrs)
            inflow[i] += 0.01
            continue
        end

        probs = Float64[]
        for j in nbrs
            ΔS = compute_entropy_cost(i, j, C_R_vals, pf_states)
            Δf = exp(ρ[i]) - C_R_vals[i] + ΔS
            p = exp(-Δf) * exp(-anisotropy_vals[j])
            push!(probs, p)
        end

        if sum(probs) < 1e-8
            inflow[i] += 0.01
            continue
        end

        probs ./= sum(probs)
        chosen = sample(nbrs, Weights(probs))
        inflow[chosen] += 0.01
        outflow[i] += 0.01
    end

    # Static fraction remains
    for i in eachindex(pf_states)
        inflow[i] += (counts[i] / 100.0) - outflow[i]
    end

    ρ .= inflow
end

# === Geodesics and Tracking ===
geodesics = [
    ["000", "100", "101", "111"],
    ["000", "100", "110", "111"],
    ["000", "010", "110", "111"],
    ["000", "010", "011", "111"],
    ["000", "001", "101", "111"],
    ["000", "001", "011", "111"]
]

function dominant_geodesic(trajectory::Vector{String}, geodesics::Vector{Vector{String}})
    best_path, max_score = nothing, 0
    for path in geodesics
        score = count(s -> s in trajectory, path)
        if score > max_score
            max_score = score
            best_path = path
        end
    end
    return best_path
end

function evolve_time_series_and_geodesic!(ρ0::Vector{Float64}, T::Int, pf_states, flat_pos, edges; max_moves_per_step=10)
    ρ = copy(ρ0)
    trajectory = String[]
    ρ_series = [copy(ρ)]

    for t in 1:T
        oxi_shapes_alive!(ρ, pf_states, flat_pos, edges; max_moves=max_moves_per_step)
        push!(ρ_series, copy(ρ))
        push!(trajectory, pf_states[argmax(ρ)])
    end

    geo = dominant_geodesic(trajectory, geodesics)
    return ρ_series, trajectory, geo
end

function geodesic_loss(initial::Vector{Float64}, final::Vector{Float64}, pf_states, geodesics)
    pred_path = [pf_states[argmax(initial)], pf_states[argmax(final)]]
    valid = any(all(s in g for s in pred_path) for g in geodesics)
    return valid ? 0.0 : 1.0
end

# === Run Example ===
ρ = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
ρ ./= sum(ρ)

points3D, R_vals, C_R_vals, anisotropy_vals = update_geometry_from_rho(ρ, pf_states, flat_pos, edges)

println("Anisotropy field: ", round.(anisotropy_vals; digits=4))

sheaf_stalks = initialize_sheaf_stalks(flat_pos, pf_states)
inconsistencies = sheaf_consistency(sheaf_stalks, edges)
if !isempty(inconsistencies)
    println("Sheaf inconsistencies found: ", inconsistencies)
else
    println("Sheaf stalks are consistent.")
end

fig_surf = plot_c_ricci_surface_interpolated(flat_pos, C_R_vals, pf_states)
save("C_Ricci_interpolated_surface.png", fig_surf)
display(fig_surf)

# === Run Geodesic Test ===
ρ_init = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
ρ_init ./= sum(ρ_init)

ρ_series, traj, geo = evolve_time_series_and_geodesic!(ρ_init, 100, pf_states, flat_pos, edges; max_moves_per_step=10)

println("Dominant geodesic: ", geo)
println("Trajectory: ", traj)
println("Geodesic loss: ", geodesic_loss(ρ_series[1], ρ_series[end], pf_states, geodesics))
