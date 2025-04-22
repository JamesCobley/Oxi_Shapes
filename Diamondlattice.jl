using CairoMakie
using GeometryBasics: Point3
using LinearAlgebra

CairoMakie.activate!()

# === Data ===
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

# === Functions ===
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

function plot_curved_shape(points3D, pf_states, edges, field; title="Curvature Field", λ=1.0)
    fig = Figure(size=(900, 700))
    ax = Axis3(fig[1,1], title=title)
    idx = Dict(s => i for (i, s) in enumerate(pf_states))

    xs = [p[1] for p in points3D]
    ys = [p[2] for p in points3D]
    zs = [p[3] for p in points3D]

    scatter!(ax, xs, ys, zs; markersize=16, color=field, colormap=:viridis)

    for (u, v) in edges
        i, j = idx[u], idx[v]
        lines!(ax,
            [points3D[i][1], points3D[j][1]],
            [points3D[i][2], points3D[j][2]],
            [points3D[i][3], points3D[j][3]],
            color=:gray)
    end

    for (i, s) in enumerate(pf_states)
        text!(ax, s, position=(xs[i], ys[i], zs[i]+0.15), align=(:center, :bottom), fontsize=14)
    end

    Colorbar(fig[1,2], limits=extrema(field), colormap=:viridis, label="Field Value")
    fig
end

# === Example run ===
ρ = [0.1, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.9]
ρ ./= sum(ρ)

points3D = lift_to_z_plane(ρ, pf_states, flat_pos)
R_vals = compute_R(points3D, flat_pos, pf_states, edges)

fig_R = plot_curved_shape(points3D, pf_states, edges, R_vals, title="R(x) Scalar Curvature")
save("R_field_oxi_shape.png", fig_R)
display(fig_R)

# === C-Ricci ===
λ = 8.0
C_R = λ .* R_vals

fig_CR = plot_curved_shape(points3D, pf_states, edges, C_R, title="C-Ricci Field (λ=1.0)")
save("C_Ricci_oxi_shape.png", fig_CR)
display(fig_CR)
