import PyPlot
const plt = PyPlot

# === Precomputed regression results ===
names = ["(Intercept)", "R", "A"]
coef  = [-78.2167, -1112.66, 2198.48]
lo95  = [-79.5936, -1208.67, 1995.19]
hi95  = [-76.8399, -1016.64, 2401.78]
pvals = ["<1e-99", "<1e-91", "<1e-82"]

# === Plot ===
fig, ax = plt.subplots(figsize=(6,4))
ypos = collect(length(names):-1:1)

ax.axvline(0, color="gray", lw=1, ls="--", zorder=0)

for (i, y) in enumerate(ypos)
    ax.errorbar(coef[i], y;
        xerr=[[coef[i]-lo95[i]], [hi95[i]-coef[i]]],
        fmt="o", color="black", ecolor="black",
        elinewidth=2, capsize=4, markersize=6
    )
    ax.text(hi95[i] + 100, y, "p = " * pvals[i],
        va="center", ha="left", fontsize=10, color="black")
end

ax.set_yticks(ypos)
ax.set_yticklabels(names, fontsize=12)
ax.set_xlabel("Coefficient (95% CI)", fontsize=12)
ax.set_title("Regression coefficients for Action Integral")

plt.tight_layout()
plt.savefig("regression_forest_with_p.png", dpi=300, bbox_inches="tight")
println("Saved: regression_forest_with_p.png (300 dpi)")
