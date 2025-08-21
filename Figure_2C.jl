using PyPlot

# === Define modal manifold (centered 1:3:3:1 diamond) ===
pos = Dict(
    "000" => (0.0, 0.0),            # top
    "100" => (-1.0, -1.0),
    "010" => (0.0, -1.0),
    "001" => (1.0, -1.0),
    "110" => (-1.0, -2.0),
    "101" => (0.0, -2.0),
    "011" => (1.0, -2.0),
    "111" => (0.0, -3.0)            # bottom
)

# === Geodesic paths ===
geodesics = [
    ["000","100","101","111"],
    ["000","100","110","111"],
    ["000","010","011","111"],
    ["000","010","110","111"],
    ["000","001","101","111"],
    ["000","001","011","111"]
]

# === Plot ===
fig, ax = subplots(figsize=(6,6))
ax.axis("off")

# plot nodes
for (s,(x,y)) in pos
    ax.scatter(x, y; s=600, c="lightyellow", edgecolors="k", zorder=2)
    ax.text(x, y, s; ha="center", va="center", fontsize=12, fontweight="bold", zorder=3)
end

# plot geodesics
colors = ["red","blue","green","purple","orange","brown"]
for (i,path) in enumerate(geodesics)
    for j in 1:(length(path)-1)
        x1,y1 = pos[path[j]]
        x2,y2 = pos[path[j+1]]
        ax.annotate("",
            xy=(x2,y2), xycoords="data",
            xytext=(x1,y1), textcoords="data",
            arrowprops=Dict(
                "arrowstyle"=>"-|>",
                "lw"=>2,
                "color"=>colors[i],
                "alpha"=>0.8,
                "shrinkA"=>15, "shrinkB"=>15
            )
        )
    end
end

plt.savefig("geodesics_modal_manifold_centered.png", dpi=300, bbox_inches="tight")
println("✔ Saved: geodesics_modal_manifold_centered.png")
