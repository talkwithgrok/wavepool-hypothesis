Python"""
Wavepool Hypothesis - Minimal Simulatable Toy Model (Part 3)
A 1D implementation of the biased vibrational pump + reaction-diffusion field
From "The Wavepool Hypothesis III: Mechanics of Emergence"
Author: D. Tranberg (with assistance from Grok)
Date: April 2026
"""

import numpy as np
import matplotlib.pyplot as plt

# ========================= PARAMETERS (from WP3) =========================
dt = 0.01          # time step
T = 200.0          # total simulation time
N_steps = int(T / dt)
Nx = 256           # spatial grid points (1D for simplicity)

# Pump parameters
Gamma = 1.0
delta = 0.01       # small entropy bias (Postulate 3 from WP1)
alpha = 0.05       # expansion coupling strength (H = α * EL)

# Field parameters
D = 0.1            # diffusion coefficient
lambda0 = 0.8      # base growth rate
beta = 0.3         # coupling from low-energy background
g = 1.0            # saturation (nonlinear term)
noise_amp = 0.01   # initial quantum-foam noise

# ========================= INITIAL CONDITIONS =========================
EL = np.full(N_steps, 1.0)      # low-vibration background energy
EH = np.zeros(N_steps)          # high-vibration localized energy
a = np.zeros(N_steps)           # scale factor (starts at 1)
a[0] = 1.0

# Vibrational field ϕ(x, t) on 1D grid
x = np.linspace(0, 1.0, Nx)
phi = np.random.normal(0, noise_amp, Nx)   # tiny initial noise

# ========================= SIMULATION LOOP =========================
for n in range(1, N_steps):
    # Biased energy pump (exact discrete update)
    Gamma_LH = Gamma * (1 + delta)
    Gamma_HL = Gamma * (1 - delta)
    
    dEH = (Gamma_LH * EL[n-1] - Gamma_HL * EH[n-1]) * dt
    dEL = (-Gamma_LH * EL[n-1] + Gamma_HL * EH[n-1]) * dt
    
    EH[n] = EH[n-1] + dEH
    EL[n] = EL[n-1] + dEL
    
    # Cosmic expansion driven by low-vibration background
    H = alpha * EL[n]
    a[n] = a[n-1] * (1 + H * dt)
    
    # Vibrational field dynamics (reaction-diffusion + saturation)
    # Laplacian in 1D with periodic boundaries
    laplacian = (np.roll(phi, -1) - 2*phi + np.roll(phi, 1)) / (x[1]-x[0])**2
    
    lambda_eff = lambda0 + beta * EL[n]          # coupling to background
    dphi = (D * laplacian + lambda_eff * phi - g * phi**3) * dt
    
    # Add noise (quantum foam)
    dphi += noise_amp * np.random.normal(0, 1, Nx) * np.sqrt(dt)
    
    phi += dphi
    
    # Couple EH to integrated field intensity
    EH[n] += 0.05 * np.sum(phi**2) * dt   # weak feedback

# ========================= PLOTTING =========================
t = np.arange(N_steps) * dt

plt.figure(figsize=(12, 8))

# Energy cycling
plt.subplot(2, 2, 1)
plt.plot(t, EL, label='EL (low-vibration background)', linewidth=2)
plt.plot(t, EH, label='EH (high-vibration clumps)', linewidth=2)
plt.title('Biased Vibrational Pump – Energy Cycling')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.grid(True, alpha=0.3)

# Scale factor (expansion)
plt.subplot(2, 2, 2)
plt.plot(t, a, 'tab:orange', linewidth=2)
plt.title('Cosmic Expansion (driven by EL)')
plt.xlabel('Time')
plt.ylabel('Scale factor a(t)')
plt.grid(True, alpha=0.3)

# Field snapshots (patchy structure)
plt.subplot(2, 2, 3)
plt.plot(x, phi, 'tab:green', linewidth=1.5)
plt.title('Final Vibrational Field ϕ(x) – Patchy Structures')
plt.xlabel('Position x')
plt.ylabel('ϕ(x)')
plt.grid(True, alpha=0.3)

# Energy ratio over time
plt.subplot(2, 2, 4)
plt.plot(t, EL / (EH + 1e-8), 'tab:purple', linewidth=2)
plt.title('EL / EH Ratio (approaches (1-δ)/(1+δ) ≈ 0.98)')
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Wavepool toy model simulation complete!")
print(f"Final EL/EH ratio: {EL[-1]/EH[-1]:.4f}  (theoretical target ≈ {(1-delta)/(1+delta):.4f})")
print(f"Final scale factor a(t): {a[-1]:.2f}")
