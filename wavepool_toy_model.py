"""
Wavepool Hypothesis - Stable Minimal Toy Model (Part 3)
Gentle parameters that run reliably on any computer.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========================= STABLE PARAMETERS =========================
dt = 0.005
T = 60.0
N_steps = int(T / dt)
Nx = 128

Gamma = 1.0
delta = 0.01
alpha = 0.008

D = 0.05
lambda0 = 0.25
beta = 0.12
g = 2.5
noise_amp = 0.001

# ========================= INITIAL CONDITIONS =========================
EL = np.full(N_steps, 1.0)
EH = np.zeros(N_steps)
a = np.zeros(N_steps)
a[0] = 1.0

x = np.linspace(0, 1.0, Nx)
phi = np.random.normal(0, noise_amp, Nx)

# ========================= SIMULATION =========================
for n in range(1, N_steps):
    Gamma_LH = Gamma * (1 + delta)
    Gamma_HL = Gamma * (1 - delta)
    dEH = (Gamma_LH * EL[n-1] - Gamma_HL * EH[n-1]) * dt
    dEL = (-Gamma_LH * EL[n-1] + Gamma_HL * EH[n-1]) * dt
    EH[n] = EH[n-1] + dEH
    EL[n] = EL[n-1] + dEL

    H = alpha * EL[n]
    a[n] = a[n-1] * (1 + H * dt)

    dx = x[1] - x[0]
    laplacian = (np.roll(phi, -1) - 2*phi + np.roll(phi, 1)) / dx**2
    lambda_eff = lambda0 + beta * EL[n]
    dphi = D * laplacian + lambda_eff * phi - g * phi**3
    phi += dphi * dt + noise_amp * np.random.normal(0, 1, Nx) * np.sqrt(dt)

    EH[n] += 0.005 * np.sum(phi**2) * dt

t = np.arange(N_steps) * dt

print("✅ Simulation finished successfully!")
print(f"Final EL/EH ratio: {EL[-1]/EH[-1]:.4f}")
print(f"Final scale factor: {a[-1]:.3f}")

# Save plots as PNG files
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, EL, label='EL (background)', linewidth=2)
plt.plot(t, EH, label='EH (clumps)', linewidth=2)
plt.title('Biased Vibrational Pump – Energy Cycling')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('energy_cycling.png')

plt.subplot(2, 2, 2)
plt.plot(t, a, 'tab:orange', linewidth=2)
plt.title('Cosmic Expansion (driven by EL)')
plt.xlabel('Time')
plt.ylabel('Scale factor a(t)')
plt.grid(True, alpha=0.3)
plt.savefig('expansion.png')

plt.subplot(2, 2, 3)
plt.plot(x, phi, 'tab:green', linewidth=1.5)
plt.title('Final Vibrational Field ϕ(x) – Patchy Structures')
plt.xlabel('Position x')
plt.ylabel('ϕ(x)')
plt.grid(True, alpha=0.3)
plt.savefig('field_patchy.png')

plt.subplot(2, 2, 4)
plt.plot(t, EL / (EH + 1e-8), 'tab:purple', linewidth=2)
plt.title('EL / EH Ratio')
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.grid(True, alpha=0.3)
plt.savefig('ratio.png')

plt.close()
print("📊 Four plots saved as PNG files in this folder:")
print("   energy_cycling.png")
print("   expansion.png")
print("   field_patchy.png")
print("   ratio.png")
