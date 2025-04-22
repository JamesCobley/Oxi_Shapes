# === Geodesic Visualization ===
colors = [:red, :blue, :green, :purple, :orange, :turquoise]

fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1], title="Geodesic Paths on Pascal Diamond", aspect=1)

# Draw base edges
for (u, v) in edges
    p1, p2 = flat_pos[u], flat_pos[v]
    lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]], color=:gray, linewidth=1.5)
end

# Draw node points and labels
for (s, (x, y)) in flat_pos
    scatter!(ax, [x], [y], color=:black, markersize=8)
    text!(ax, s, position=(x, y + 0.15), align=(:center, :bottom), fontsize=14)
end

# Draw colored geodesic paths
for (geo, color) in zip(geodesics, colors)
    xs = [flat_pos[s][1] for s in geo]
    ys = [flat_pos[s][2] for s in geo]
    lines!(ax, xs, ys, linewidth=4, color=color)
end

save("geodesic_pascal_diamond.png", fig)
display(fig)
