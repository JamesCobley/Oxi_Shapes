import numpy as np

# Kinetic parameters from the textbook:
k = 5.0               # Rate constant in M⁻¹ s⁻¹
H2O2_conc = 1e-9      # Hydrogen peroxide concentration: 1 nM = 1e-9 M

# Compute the pseudo-first-order rate constant:
k_prime = k * H2O2_conc  # in s⁻¹

# Simulation time: 600 seconds (10 minutes)
t = 600  # seconds

# Fraction of cysteine residues oxidized over time t
fraction_oxidized = 1 - np.exp(-k_prime * t)

# For a cell with 70,000 protein molecules and R = 10 cysteines per protein:
proteins_per_cell = 70000
cysteines_per_protein = 10
total_cysteines = proteins_per_cell * cysteines_per_protein

# Expected number of oxidized cysteines:
oxidized_cysteines = total_cysteines * fraction_oxidized

print("Pseudo-first order rate constant k' =", k_prime, "s⁻¹")
print("Fraction oxidized over", t, "seconds =", fraction_oxidized)
print("Total cysteines =", total_cysteines)
print("Expected number of oxidized cysteines =", oxidized_cysteines)
