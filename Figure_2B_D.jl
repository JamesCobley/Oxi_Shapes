using PyPlot

# --- Data ---
C = [0.0 0.0 0.0; 
     0.0 0.0 0.0; 
     0.0 0.0 0.0]

counts = Dict(
    (1,2) => 16, (2,1) => 12,
    (1,3) => 12, (3,1) => 16,
    (2,3) => 10, (3,2) => 1
)

labels = ["M1","M2","M3"]
pos = Dict(1 => (0.0,0.0), 2 => (1.0,0.0), 3 => (0.5,0.8))  # ✅ fixed

fig, axs = subplots(1,2, figsize=(10,4))

# --- Panel A: Heatmap of commutators ---
im = axs[1].imshow(C, cmap="Blues", vmin=0, vmax=1)
axs[1].set_xticks(0:2); axs[1].set_yticks(0:2)
axs[1].set_xticklabels(labels); axs[1].set_yticklabels(labels)
axs[1].set_title("Commutator Norms (Operator Level)")
for i in 1:3, j in 1:3
    axs[1].text(j-1, i-1, string(round(C[i,j],digits=2));
        ha="center", va="center", color="black")
end

# --- Panel B: Path asymmetries ---
ax = axs[2]
ax.set_xlim(-0.2,1.2); ax.set_ylim(-0.2,1.0); ax.axis("off")
for i in 1:3
    x,y = pos[i]
    ax.scatter(x,y; s=600, c="lightblue", edgecolors="k", zorder=2)
    ax.text(x,y,labels[i]; ha="center", va="center", fontsize=14, fontweight="bold")
end

for ((i,j),c) in counts
    x1,y1 = pos[i]; x2,y2 = pos[j]
    c_back = get(counts,(j,i),0)
    color = (c != c_back) ? "red" : "black"
    ax.annotate("",
        xy=(x2,y2), xycoords="data",
        xytext=(x1,y1), textcoords="data",
        arrowprops=Dict("arrowstyle"=>"-|>",
                        "lw"=>0.5+0.2*c,
                        "color"=>color,
                        "shrinkA"=>15,"shrinkB"=>15))
    xm, ym = (x1+x2)/2, (y1+y2)/2
    ax.text(xm, ym, string(c); fontsize=10, color=color, ha="center", va="center")
end

ax.set_title("Path Asymmetries (Realized Geodesics)")

plt.tight_layout()
plt.savefig("commutator_vs_path_asymmetry.png", dpi=300)
println("✔ Saved commutator_vs_path_asymmetry.png (300 dpi)")
