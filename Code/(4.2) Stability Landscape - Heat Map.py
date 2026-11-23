from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# Force numpy mode and suppress unnecessary logs
prefs.codegen.target = "numpy"
import logging

logging.getLogger('brian2').setLevel(logging.ERROR)

# --- 1. SET UP THE GRIDS ---
noise_levels = np.linspace(0.01, 0.5, 7)
inh_strengths = np.linspace(0.1, 1.2, 7)
exc_strengths = np.linspace(0.1, 1.2, 7)

R_inhibition = np.zeros((len(noise_levels), len(inh_strengths)))
R_excitation = np.zeros((len(noise_levels), len(exc_strengths)))

# --- 2. NOISE vs INHIBITION SWEEP ---
for i, sigma in enumerate(noise_levels):
    for j, w_inh in enumerate(inh_strengths):
        start_scope()
        tau = 10 * ms
        eqs = 'dv/dt = (-v + 1.5 + sigma*xi*(tau**-0.5)*tau)/tau : 1 (unless refractory)'
        E = NeuronGroup(50, eqs, threshold='v>1', reset='v=0', refractory=2 * ms, method='euler')
        I = NeuronGroup(12, eqs, threshold='v>1', reset='v=0', refractory=2 * ms, method='euler')

        # Connections
        S_EI = Synapses(E, I, on_pre='v_post += 0.1').connect(p=0.2)
        S_IE = Synapses(I, E, on_pre=f'v_post -= {w_inh}').connect(p=0.2)

        rate_mon = PopulationRateMonitor(E)
        run(300 * ms)  # Shorter run for speed

        # Calculate Stability
        start_idx = int(100 * ms / defaultclock.dt)
        R_inhibition[i, j] = np.std(rate_mon.rate[start_idx:])

# --- 3. NOISE vs EXCITATION SWEEP ---
for i, sigma in enumerate(noise_levels):
    for j, w_exc in enumerate(exc_strengths):
        start_scope()
        tau = 10 * ms
        eqs = 'dv/dt = (-v + 1.5 + sigma*xi*(tau**-0.5)*tau)/tau : 1 (unless refractory)'
        E = NeuronGroup(50, eqs, threshold='v>1', reset='v=0', refractory=2 * ms, method='euler')
        I = NeuronGroup(12, eqs, threshold='v>1', reset='v=0', refractory=2 * ms, method='euler')

        # Connections (Varying E weight instead of Extra-BrainOscillPrject)
        S_EI = Synapses(E, I, on_pre=f'v_post += {w_exc}').connect(p=0.2)
        S_IE = Synapses(I, E, on_pre='v_post -= 0.5').connect(p=0.2)

        rate_mon = PopulationRateMonitor(E)
        run(300 * ms)

        # Calculate Stability
        start_idx = int(100 * ms / defaultclock.dt)
        R_excitation[i, j] = np.std(rate_mon.rate[start_idx:])

# --- 4. PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Map 1: Inhibition
im1 = ax1.imshow(R_inhibition, origin='lower', aspect='auto', cmap='plasma',
                 extent=[inh_strengths[0], inh_strengths[-1], noise_levels[0], noise_levels[-1]])
ax1.set_title('Noise vs Inhibitory Strength')
ax1.set_xlabel('Inhibitory Weight')
ax1.set_ylabel('Noise Level (σ)')
plt.colorbar(im1, ax=ax1, label='Stability (SD of Rate)')

# Map 2: Excitation
im2 = ax2.imshow(R_excitation, origin='lower', aspect='auto', cmap='magma',
                 extent=[exc_strengths[0], exc_strengths[-1], noise_levels[0], noise_levels[-1]])
ax2.set_title('Noise vs Excitatory Strength')
ax2.set_xlabel('Excitatory Weight')
ax2.set_ylabel('Noise Level (σ)')
plt.colorbar(im2, ax=ax2, label='Stability (SD of Rate)')

plt.tight_layout()
plt.show()