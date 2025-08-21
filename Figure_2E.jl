using PyPlot

# --- Geodesics (each is a sequence of states) ---
geodesics = [
    ["000","100","110","111"],
    ["000","100","101","111"],
    ["000","010","011","111"],
    ["000","010","110","111"],
    ["000","001","101","111"],
    ["000","001","011","111"]
]

# --- Helper: which bit flipped between two states (1..3) ---
bit_index(s::String, t::String) = begin
    @assert length(s) == length(t)
    k = 0
    @inbounds for i in 1:length(s)
        if s[i] != t[i]
            k == 0 || return 0
            k = i
        end
    end
    return k
end

# --- Turn-by-bit rule (relative to current heading)
#     left (bit1) = -δ, straight (bit2) = 0, right (bit3) = +δ
δ = π/7

function turn_for_bit(b::Int)
    b == 1 && return -δ
    b == 2 && return 0.0
    b == 3 && return  δ
    return 0.0
end

# --- Colors per bit for segments ---
bit_color = Dict(1=>"crimson", 2=>"royalblue", 3=>"seagreen")

# --- Draw one geodesic as a polyline that "turns by bit" each step ---
#     start at (0,0), initial heading upwards (π/2), equal step lengths
function draw_path(ax, path; step_len=1.0, base_angle=π/2, lw=2.5, α=0.9)
    x, y = 0.0, 0.0
    θ = base_angle
    xs = [x]; ys = [y]
    bits = Int[]

    for j in 1:length(path)-1
        b = bit_index(path[j], path[j+1])
        push!(bits, b)
        θ += turn_for_bit(b)
        x += step_len * cos(θ)
        y += step_len * sin(θ)
        push!(xs, x); push!(ys, y)
    end

    # plot colored segments per bit choice
    for k in 1:length(bits)
        ax.plot(xs[k:k+1], ys[k:k+1],
                color=bit_color[bits[k]], lw=lw, alpha=α, solid_capstyle="round")
    end

    # mark endpoints
    ax.scatter(xs[1], ys[1]; s=30, c="black", zorder=3)
    ax.scatter(xs[end], ys[end]; s=30, c="black", zorder=3)

    return xs, ys, bits
end

# --- Figure ---
fig, ax = subplots(figsize=(6,6))
ax.axis("off")

# faint central spine for reference (000→111 straight)
ax.plot([0,0], [0,3.0], color="gray", lw=1.0, ls="--", alpha=0.6)

# draw all geodesics with slight radial spread so arms form a snowflake
# (rotate the initial heading evenly around the circle)
for (i, path) in enumerate(geodesics)
    θ0 = π/2 + 2π*(i-1)/length(geodesics)  # starting heading for this arm
    draw_path(ax, path; step_len=1.0, base_angle=θ0, lw=2.8, α=0.95)
end

# labels for start/end
ax.text(0, 0,  "000"; ha="center", va="top",  fontsize=10, fontweight="bold")
# place "111" near one of the arm ends (approximate radius)
ax.text(0, 3.0, "111"; ha="center", va="bottom", fontsize=10, fontweight="bold")

# legend: bit→deviation rule

Line2D = PyPlot.matplotlib.lines.Line2D
legend_lines = [Line2D([0],[0], color=bit_color[1], lw=3),
                Line2D([0],[0], color=bit_color[2], lw=3),
                Line2D([0],[0], color=bit_color[3], lw=3)]
ax.legend(legend_lines, ["bit 1 = left", "bit 2 = straight", "bit 3 = right"],
          loc="upper right", frameon=false)


ax.set_xlim(-3.4, 3.4)
ax.set_ylim(-3.4, 3.4)
ax.set_aspect("equal")

tight_layout()
savefig("geodesic_snowflake_bit_turns.png", dpi=300, bbox_inches="tight")
println("✔ Saved: geodesic_snowflake_bit_turns.png")
