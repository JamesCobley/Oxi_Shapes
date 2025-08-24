#!/usr/bin/env julia
# ===============================================================
# Oxi-Shapes (R=3): Zero-Sum Curvature Visualization
# ===============================================================

using LinearAlgebra
using PyPlot

# ---------------------------------------------------------------
# States for R=3
# ---------------------------------------------------------------
states = ["000","001","010","011","100","101","110","111"]

# Pre-computed curvature fields R = Δρ from your zero-sum test
curv_fields = Dict(
    "000" => [ 3,-1,-1, 0,-1, 0, 0, 0],
    "001" => [-1, 3, 0,-1, 0,-1, 0, 0],
    "010" => [-1, 0, 3,-1, 0, 0,-1, 0],
    "011" => [ 0,-1,-1, 3, 0, 0, 0,-1],
    "100" => [-1, 0, 0, 0, 3,-1,-1, 0],
    "101" => [ 0,-1, 0, 0,-1, 3, 0,-1],
    "110" => [ 0, 0,-1, 0,-1, 0, 3,-1],
    "111" => [ 0, 0, 0,-1, 0,-1,-1, 3]
)

# ---------------------------------------------------------------
# Plot bar charts (2 rows × 4 cols)
# ---------------------------------------------------------------
fig, axs = subplots(2, 4, figsize=(14,6), squeeze=false)  # always 2D array

for (i,(label,vals)) in enumerate(curv_fields)
    row, col = divrem(i-1, 4)
    ax = axs[row+1, col+1]

    # Bar plot of curvature per state
    ax.bar(1:length(states), vals, color="royalblue")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(1:length(states))
    ax.set_xticklabels(states, rotation=45, fontsize=8)
    ax.set_title("Occupancy at $label", fontsize=10)
    ax.set_ylim(-2.5, 3.5)
    
    # Show total curvature sum in subtitle
    total = round(sum(vals), digits=6)
    ax.text(0.5, 3.0, "ΣR = $total", ha="left", va="center", fontsize=8)
end

fig.suptitle("Zero-Sum Curvature Fields for Single-Molecule Placements (R=3)", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95])
savefig("curvature_bars.png", dpi=300)
println("✔ Saved curvature_bars.png")
