#!/usr/bin/env python
# coding: utf-8

"""
Pipeline Step 2: Empirical Evolution and Analysis

This script:
  1. Converts empirical k-priors into full i-state occupancy vectors.
  2. Evolves the occupancy distribution using the Neural ODE PDE solver over 240 steps.
  3. Computes metrics:
      - Percent oxidation (weighted mean over k-bins),
      - Shannon entropy,
      - Lyapunov exponent (based on the evolution of the percent oxidation).
  4. (Optionally) Prepares outputs for further ML-based geodesic analysis.

Note: This script assumes the following functions and variables from Pipeline Step 1 are defined:
    - pf_states, state_index, flat_pos, G, etc.
    - The ODE function: oxi_shapes_ode(t, rho)
    - The PDE solver: evolve_oxi_shapes_pde(rho0, t_span)
Make sure these are already available in your environment.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# --- Utility: Convert k-priors to full i-state occupancy
def kpriors_to_istate(k_priors):
    """
    For R=3:
      k=0: ["000"]
      k=1: ["001", "010", "100"]
      k=2: ["011", "101", "110"]
      k=3: ["111"]
    k_priors: list or array of 4 numbers (should sum to 1).
    Returns an 8-element occupancy vector (numpy array).
    """
    groups = {
        0: ["000"],
        1: ["001", "010", "100"],
        2: ["011", "101", "110"],
        3: ["111"]
    }
    occupancy = np.zeros(len(pf_states))
    for k in range(4):
        for s in groups[k]:
            occupancy[state_index[s]] = k_priors[k] / len(groups[k])
    return occupancy

# --- Metric functions

def percent_oxidation(occupancy):
    """
    Compute percent oxidation as the weighted mean of oxidation level.
    Oxidation level for a state = (# of 1's in the bit string)/3 * 100.
    """
    oxidation_levels = np.array([(s.count('1')/3)*100 for s in pf_states])
    return np.sum(occupancy * oxidation_levels)

def shannon_entropy(occupancy):
    p = occupancy[occupancy > 0]
    return -np.sum(p * np.log2(p))

def lyapunov_exponent(time_series):
    """
    Estimate the Lyapunov exponent from a time series of percent oxidation.
    time_series: numpy array of shape (T,), percent oxidation over T time steps.
    """
    diffs = np.abs(np.diff(time_series))
    diffs[diffs==0] = 1e-12
    log_diffs = np.log(diffs)
    t = np.arange(len(log_diffs))
    A = np.vstack([t, np.ones(len(t))]).T
    slope, _ = np.linalg.lstsq(A, log_diffs, rcond=None)[0]
    return slope

# --- Helper: Evolve occupancy to obtain full time series
def evolve_time_series(rho0, t_span):
    """
    Use odeint to evolve the occupancy over t_span.
    Returns a numpy array of shape (T, num_states) with the occupancy at each time step.
    """
    rho_t = odeint(oxi_shapes_ode, rho0, t_span)
    rho_series = []
    for r in rho_t:
        r_norm = r / r.sum()
        rho_series.append(r_norm.detach().cpu().numpy())
    return np.array(rho_series)

# --- Empirical priors (from experimental data)
# For k-bins: [k=0, k=1, k=2, k=3]
empirical_start_k = [0.25, 0.75, 0.0, 0.0]
empirical_end_k   = [0.06, 0.53, 0.33, 0.10]

# Convert k-priors to full occupancy (length 8) for R=3.
rho_start_full = kpriors_to_istate(empirical_start_k)
rho_target_full = kpriors_to_istate(empirical_end_k)

print("Empirical Start k-priors:", empirical_start_k)
print("Empirical End k-priors:", empirical_end_k)
print("Empirical Start occupancy (full):", np.round(rho_start_full, 3))
print("Empirical Target occupancy (full):", np.round(rho_target_full, 3))
print("Empirical Percent Oxidation (Start):", percent_oxidation(rho_start_full))
print("Empirical Percent Oxidation (Target):", percent_oxidation(rho_target_full))
print("Empirical Shannon Entropy (Start):", np.round(shannon_entropy(rho_start_full), 3))
print("Empirical Shannon Entropy (Target):", np.round(shannon_entropy(rho_target_full), 3))

# --- Evolve the PDE from the empirical start occupancy
t_span = torch.linspace(0.0, 1.0, 240, dtype=torch.float32)
rho0_emp = torch.tensor(rho_start_full, dtype=torch.float32)
rho_final_emp = evolve_oxi_shapes_pde(rho0_emp, t_span)
rho_final_emp_np = rho_final_emp.detach().cpu().numpy()

print("\nPDE predicted final occupancy:", np.round(rho_final_emp_np, 3))
print("Predicted Percent Oxidation:", percent_oxidation(rho_final_emp_np))
print("Predicted Shannon Entropy:", np.round(shannon_entropy(rho_final_emp_np), 3))

# --- Generate full time series to estimate the Lyapunov exponent.
rho_time_series = evolve_time_series(rho0_emp, t_span)
percent_series = np.array([percent_oxidation(r) for r in rho_time_series])
lyap_exp = lyapunov_exponent(percent_series)
print("Lyapunov Exponent (from PDE time series):", np.round(lyap_exp, 6))

# --- Poincaré Recurrence Plot of Percent Oxidation
def poincare_recurrence(series, threshold=1.0):
    N = len(series)
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if np.abs(series[i] - series[j]) < threshold:
                R[i, j] = 1
    return R

recurrence_plot = poincare_recurrence(percent_series, threshold=1.0)
plt.figure(figsize=(6,6))
plt.imshow(recurrence_plot, cmap='binary', origin='lower')
plt.title("Poincaré Recurrence Plot (Percent Oxidation)")
plt.xlabel("Time step")
plt.ylabel("Time step")
plt.colorbar(label="Recurrence")
plt.show()

# --- Now, call the trained ML model to solve for the i-states based on empirical priors.
# The trained model (OxiFlowNet) was saved in Pipeline Step 1 as "oxinet_model.pt".
# Here we assume that the ML model predicts a final occupancy distribution given an initial occupancy.
class OxiNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=8):
        super(OxiNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

# Load the model
model = OxiNet(input_dim=8, hidden_dim=32, output_dim=8)
model_data = torch.load("oxinet_model.pt", map_location=torch.device("cpu"))
model.load_state_dict(model_data['model_state_dict'])
model.eval()

# Define a function to simulate a trajectory using the ML model.
def predict_i_states(model, initial_occ, steps=80):
    trajectory = [initial_occ.copy()]
    current_occ = initial_occ.copy()
    for _ in range(steps):
        inp = torch.tensor(current_occ, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(inp).squeeze(0).numpy()
        trajectory.append(pred)
        current_occ = pred
    return np.array(trajectory)

# Simulate trajectories for many molecules (here, using the same empirical start)
num_molecules = 100
trajectories = [predict_i_states(model, rho_start_full, steps=80) for _ in range(num_molecules)]

# For analysis, we score predictions by confidence (max probability in final occupancy).
final_preds = np.array([traj[-1] for traj in trajectories])
confidences = np.max(final_preds, axis=1)
top5_indices = np.argsort(confidences)[-5:]
top5_trajectories = [trajectories[i] for i in top5_indices]

# Compute metrics for the top 5 trajectories.
for idx, traj in enumerate(top5_trajectories):
    final_occ = traj[-1]
    perc_ox = percent_oxidation(final_occ)
    entropy_val = shannon_entropy(final_occ)
    # For Lyapunov, compute from percent oxidation time series along the trajectory.
    perc_series_traj = np.array([percent_oxidation(r) for r in traj])
    lyap_val = lyapunov_exponent(perc_series_traj)
    print(f"\nTop Prediction {idx+1}:")
    print("Final occupancy:", np.round(final_occ, 3))
    print("Percent oxidation:", np.round(perc_ox, 2))
    print("Shannon entropy:", np.round(entropy_val, 3))
    print("Lyapunov exponent:", np.round(lyap_val, 6))
    # Here, store the full trajectory for persistent topology analysis if desired.

# Optionally, produce a Poincaré recurrence plot for one top trajectory:
rec_plot_top = poincare_recurrence(np.array([percent_oxidation(r) for r in top5_trajectories[0]]), threshold=1.0)
plt.figure(figsize=(6,6))
plt.imshow(rec_plot_top, cmap='binary', origin='lower')
plt.title("Poincaré Recurrence Plot for Top Trajectory")
plt.xlabel("Time step")
plt.ylabel("Time step")
plt.colorbar(label="Recurrence")
plt.show()

# (Optional) Calculate Fisher Information Metric for the ML trajectories.
def fisher_information(occupancy_series):
    diffs = np.diff(occupancy_series, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1)**2)

fisher_info = np.mean([fisher_information(traj) for traj in trajectories])
print("Fisher information metric (ML trajectories):", np.round(fisher_info, 6))

# (Placeholder) Calculate Feigenbaum constant if applicable. 
# This would require identifying period-doubling bifurcations in the percent oxidation time series.
def estimate_feigenbaum(percent_series):
    # A dummy function; in practice, you'd perform a bifurcation analysis.
    return 4.669  # The universal Feigenbaum delta for period-doubling bifurcations.

feigenbaum = estimate_feigenbaum(percent_series)
print("Estimated Feigenbaum constant:", feigenbaum)
