from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

prefs.codegen.target = "numpy"

# --- PARAMETERS ---
noise_levels = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
#noise_levels = [0.01, 0.1, 0.2, 0.3, 0.5]

run_time = 1000 * ms
coherence_results = []

for sigma_val in noise_levels:
    start_scope()

    # NEURON MODEL
    tau = 10 * ms
    eqs = '''
    dv/dt = (-v + Extra-BrainOscillPrject + sigma_val*xi*(tau**-0.5)*tau)/tau : 1 (unless refractory)
    Extra-BrainOscillPrject : 1
    '''

    E = NeuronGroup(100, eqs, threshold='v>1', reset='v=0', refractory=2 * ms, method='euler')
    E.v = 'rand()'
    E.I = 1.5

    # We use a simple clock-driven synapse to test coherence
    # (Or keep your E-Extra-BrainOscillPrject loop from before)
    S = Synapses(E, E, on_pre='v_post += 0.1')
    S.connect(p=0.2)

    spikes = SpikeMonitor(E)
    run(run_time)

    # --- CALCULATE COHERENCE (Kuramoto-style) ---
    # We calculate the variance of the population activity as a proxy for coherence
    # Or use the Chi-Square synchrony metric
    if len(spikes.t) > 10:
        # Bin the spikes to get a population activity profile
        bin_size = 5 * ms
        counts, bin_edges = np.histogram(spikes.t / ms, bins=np.arange(200, 1000, 5))

        # Synchrony Index (Normalized variance)
        # Var(Sum of inputs) / Sum(Var of individual inputs)
        # For simplicity in this project, we use the normalized standard deviation of the rate
        sync_index = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
    else:
        sync_index = 0

    coherence_results.append(sync_index)

# --- PLOTTING ---
plt.figure(figsize=(8, 5))
plt.plot(noise_levels, coherence_results, 'o-', color='darkblue', linewidth=2)
plt.fill_between(noise_levels, coherence_results, alpha=0.2, color='purple')

plt.title("Phase Coherence vs. Neural Noise")
plt.xlabel("Noise Level (σ)")
plt.ylabel("Synchrony Index (Coherence)")
plt.grid(True, linestyle='--', alpha=0.6)

# Adding the Hypothesis Proof labels
plt.annotate('Stable Rhythms', xy=(0.01, coherence_results[0]), xytext=(0.1, coherence_results[0] + 0.2),
             arrowprops=dict(facecolor='orange', shrink=0.05))
plt.annotate('Noise Interference', xy=(0.4, coherence_results[-1]), xytext=(0.3, coherence_results[-1] + 0.5),
             arrowprops=dict(facecolor='orange', shrink=0.05))

plt.show()

# --- PRINT DATA ---
print("\nNoise σ | Coherence Index")
print("-" * 25)
for n, c in zip(noise_levels, coherence_results):
    print(f"{n:<8.2f} | {c:<15.4f}")