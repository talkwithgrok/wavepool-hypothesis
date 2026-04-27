Python"""
Wavepool Hypothesis - Stable Minimal Toy Model (Part 3)
1D implementation of the biased vibrational pump + reaction-diffusion field
Stable parameters for reliable simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

# ========================= PARAMETERS =========================
dt = 0.002          # smaller time step for numerical stability
T = 120.0           # total simulation time
N_steps = int(T / dt)
Nx = 256

Gamma = 1.0
delta = 0.01
alpha = 0.05

D = 0.1
lambda0 = 0.55       # tuned for stability
beta = 0.3
g = 1.2              # slightly stronger saturation
noise_amp = 0.005

# ========================= INITIAL CONDITIONS =========================
EL = np.full(N_steps, 1.0)
EH = np.zeros(N_steps)
a = np.zeros(N_steps)
a[0] = 1.0

x = np.linspace(0, 1.0, Nx)
phi = np.random.normal(0, noise_amp, Nx)

# ========================= SIMULATION =========================
for n in range(1, N_steps):
    # Biased vibrational pump
    Gamma_LH = Gamma * (1 + delta)
    Gamma_HL = Gamma * (1 - delta)
    dEH = (Gamma_LH * EL[n-1] - Gamma_HL * EH[n-1]) * dt
    dEL = (-Gamma_LH * EL[n-1] + Gamma_HL * EH[n-1]) * dt
    EH[n] = EH[n-1] + dEH
    EL[n] = EL[n-1] + dEL

    # Expansion
    H = alpha * EL[n]
    a[n] = a[n-1] * (1 + H * dt)

    # Vibrational field dynamics
    dx = x[1] - x[0]
    laplacian = (np.roll(phi, -1) - 2*phi + np.roll(phi, 1)) / dx**2
    lambda_eff = lambda0 + beta * EL[n]
    dphi = D * laplacian + lambda_eff * phi - g * phi**3
    phi += dphi * dt + noise_amp * np.random.normal(0, 1, Nx) * np.sqrt(dt)

    # Couple field intensity back to EH
    EH[n] += 0.02 * np.sum(phi**2) * dt

# ========================= PLOTS =========================
t = np.arange(N_steps) * dt

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, EL, label='EL (background)', linewidth=2)
plt.plot(t, EH, label='EH (clumps)', linewidth=2)
plt.title('Biased Vibrational Pump – Energy Cycling')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(t, a, 'tab:orange', linewidth=2)
plt.title('Cosmic Expansion (driven by EL)')
plt.xlabel('Time')
plt.ylabel('Scale factor a(t)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(x, phi, 'tab:green', linewidth=1.5)
plt.title('Final Vibrational Field ϕ(x) – Patchy Structures')
plt.xlabel('Position x')
plt.ylabel('ϕ(x)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(t, EL / (EH + 1e-8), 'tab:purple', linewidth=2)
plt.title('EL / EH Ratio')
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Wavepool toy model ran successfully!")
print(f"Final EL/EH ratio: {EL[-1]/EH[-1]:.4f}")
print(f"Final scale factor: {a[-1]:.2f}")
