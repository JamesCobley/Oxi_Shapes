# ╔═╡ Install required packages (only needs to run once per session)
using Pkg
Pkg.activate(".")  # Optional: activate project environment
Pkg.add(["Flux", "CUDA", "Meshes", "GeometryBasics", "LinearAlgebra",
         "StatsBase", "DifferentialEquations", "Ripserer",
         "Distances", "Makie"])

using GeometryBasics: Point3
using LinearAlgebra
using CairoMakie

CairoMakie.activate!()

# === Pascal Diamond ===
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

# === Lift to z(x) = -ρ(x) ===
function lift_to_z_plane(rho::Vector{Float64}, pf_states, flat_pos)
    return [Point3(flat_pos[s][1], flat_pos[s][2], -rho[i]) for (i, s) in enumerate(pf_states)]
end

# === R(x) as deviation in edge distances ===
function compute_R_from_distances(points3D, flat_pos, pf_states, edges)
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

# === Visualization ===
function visualize_lifted(points3D, pf_states, edges)
    fig = Figure(size=(800, 600))
    ax = Axis3(fig[1,1], title="Oxi-Shape Axiom Deformation")

    xs = [p[1] for p in points3D]
    ys = [p[2] for p in points3D]
    zs = [p[3] for p in points3D]

    scatter!(ax, xs, ys, zs; markersize=15, color=:cornflowerblue)

    idx = Dict(s => i for (i, s) in enumerate(pf_states))
    for (u, v) in edges
        i, j = idx[u], idx[v]
        line = [points3D[i], points3D[j]]
        lines!(ax, getindex.(line, 1), getindex.(line, 2), getindex.(line, 3), color=:gray)
    end

    return fig
end

# === Example run ===
ρ = [0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.25, 0.6]
ρ ./= sum(ρ)

points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
R_vals = compute_R_from_distances(points3D, flat_pos, pf_states, edges)

fig = visualize_lifted(points3D, pf_states, edges)
display(fig)
save("oxi_shape.png", fig)


println("R(x): ", round.(R_vals; digits=4))
