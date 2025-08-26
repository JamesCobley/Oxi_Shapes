using Printf   # <-- needed for @printf

# === Back-calculate activation energy from bimolecular k2 ===
# GAPDH Cys152 anchor: k2 = 7 M^-1 s^-1 at 20 °C (293.15 K)

# Physical constants
R  = 8.314462618      # J mol^-1 K^-1
kB = 1.380649e-23     # J K^-1
h  = 6.62607015e-34   # J s

# Inputs (change as needed)
k2   = 7.0       # M^-1 s^-1 (rate constant)
T    = 293.15    # K (20 °C)
C0   = 1.0       # M (standard state)
kappa = 1.0      # transmission coefficient (default = 1)

# Back-calc ΔG‡ using Eyring equation
arg = (k2 * h * C0) / (kappa * kB * T)
ΔG_Jmol = -R * T * log(arg)
ΔG_kJmol = ΔG_Jmol / 1000
ΔG_kcal  = ΔG_Jmol / 4184.0

println("=== Eyring Back-Calculation (Bimolecular) ===")
println("Input: k2 = $k2 M^-1 s^-1, T = $T K, C0 = $C0 M, κ = $kappa")
@printf("ΔG‡ = %.2f kJ/mol  (%.2f kcal/mol)\n", ΔG_kJmol, ΔG_kcal)

# Optional: compare Arrhenius-style Ea for assumed pre-exponentials A2
for A2 in (1e7, 1e8, 1e9)
    Ea_Jmol = -R * T * log(k2 / A2)
    Ea_kJmol = Ea_Jmol / 1000
    @printf("Assuming A2 = %.1e M^-1 s^-1 ⇒ Ea ≈ %.2f kJ/mol\n", A2, Ea_kJmol)
end
