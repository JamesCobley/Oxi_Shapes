# ╔═╡ Install required packages (only needs to run once per session)
using Pkg
Pkg.activate(".")  # Optional: activate project environment
Pkg.add(["Flux", "CUDA", "Meshes", "GeometryBasics", "LinearAlgebra",
         "StatsBase", "DifferentialEquations", "Ripserer",
         "Distances", "Interploations", "CairoMakie"])

using CairoMakie
using GeometryBasics: Point2, Point3
using Interpolations
using LinearAlgebra

CairoMakie.activate!()

# === Proteoform Setup ===
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
flat_pos = Dict(
    "000" => (0.0, 3.0),
    "001" => (-2.0, 2.0),
    "010" => (0.0, 2.0),
    "100" => (2.0, 2.0),
    "011" => (-1.0, 1.0),
    "101" => (0.0, 1.0),
    "110" => (1.0, 1.0),
    "111" => (0.0, 0.0)
)
edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "011"), ("001", "101"),
    ("010", "011"), ("010", "110"),
    ("100", "110"), ("100", "101"),
    ("011", "111"), ("101", "111"), ("110", "111")
]

# === Lifting and Curvature Computation ===
function lift_to_z_plane(rho, pf_states, flat_pos)
    [Point3(flat_pos[s][1], flat_pos[s][2], -rho[i]) for (i, s) in enumerate(pf_states)]
end

function compute_R(points3D, flat_pos, pf_states, edges)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    R = zeros(Float64, length(pf_states))
    for (i, s) in enumerate(pf_states)
        p0 = flat_pos[s]
        p3 = points3D[i]
        neighbors = [v for (u, v) in edges if u == s]
        append!(neighbors, [u for (u, v) in edges if v == s])
        for n in neighbors
            j = idx[n]
            q0 = flat_pos[n]
            q3 = points3D[j]
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

anisotropy_vals = compute_anisotropy(C_R_vals, pf_states, flat_pos, edges)
println("Anisotropy field: ", round.(anisotropy_vals; digits=4))

sheaf_stalks = initialize_sheaf_stalks(flat_pos, pf_states)
inconsistencies = sheaf_consistency(sheaf_stalks, edges)
if !isempty(inconsistencies)
    println("Sheaf inconsistencies found: ", inconsistencies)
else
    println("Sheaf stalks are consistent.")
end

function initialize_sheaf_stalks(flat_pos, pf_states)
    stalks = Dict{String, Vector{Float64}}()
    for s in pf_states
        stalks[s] = [flat_pos[s][1], flat_pos[s][2]]
    end
    return stalks
end


# === Interpolated Surface Plot ===
function plot_c_ricci_surface_interpolated(flat_pos, field, pf_states)
    fig = Figure(size=(900, 700))
    ax = Axis3(fig[1, 1], title="Interpolated C-Ricci Surface Field", perspectiveness=0.8)

    xs = [flat_pos[s][1] for s in pf_states]
    ys = [flat_pos[s][2] for s in pf_states]

    # Grid
    grid_x = LinRange(minimum(xs)-0.5, maximum(xs)+0.5, 100)
    grid_y = LinRange(minimum(ys)-0.5, maximum(ys)+0.5, 100)
    grid_z = fill(NaN, length(grid_x), length(grid_y))

    for (i, s) in enumerate(pf_states)
        x_idx = findmin(abs.(grid_x .- flat_pos[s][1]))[2]
        y_idx = findmin(abs.(grid_y .- flat_pos[s][2]))[2]
        grid_z[x_idx, y_idx] = field[i]
    end

    # Interpolation
    interp_func = interpolate(grid_z, BSpline(Linear()))
    interp_func_itp = extrapolate(interp_func, NaN)
    surface!(ax, grid_x, grid_y, (x, y) -> interp_func_itp[x, y], colormap=:viridis)

    # Overlay points
    for (i, s) in enumerate(pf_states)
        x, y = flat_pos[s]
        z = field[i]
        scatter!(ax, [x], [y], [z], markersize=10, color=:black)
        text!(ax, s, position=(x, y, z + 0.05), align=(:center, :bottom), fontsize=14)
    end

    Colorbar(fig[1, 2], limits=extrema(field), colormap=:viridis, label="C-Ricci (Dirichlet)")
    fig
end

# === Run Example ===
ρ = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
ρ ./= sum(ρ)

points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
R_vals = compute_R(points3D, flat_pos, pf_states, edges)
C_R_vals = compute_c_ricci_dirichlet(R_vals, pf_states, edges)

fig_surf = plot_c_ricci_surface_interpolated(flat_pos, C_R_vals, pf_states)
save("C_Ricci_interpolated_surface.png", fig_surf)
display(fig_surf)
