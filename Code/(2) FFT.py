from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Force Numpy mode to avoid C++ compiler errors
prefs.codegen.target = "numpy"

# --- PARAMETERS ---
noise_levels = [ 0.0, 0.1, 0.2, 0.3,0.4,  0.5]
run_time = 1200 * ms
data_for_table = [] #Stores quantitative results

plt.figure(figsize=(10, 6))

for sigma_val in noise_levels:
    start_scope()

    # NEURON MODEL
    tau = 10 * ms  ## tau = r = time constant, determining how quickly v approacches 1
    #Determines how fast neurons respond- Real neurons also have time constants (~10–20 ms)

    eqs = '''
    dv/dt = (-v + Extra-BrainOscillPrject + sigma_val*xi*(tau**-0.5)*tau)/tau : 1 (unless refractory)
    Extra-BrainOscillPrject : 1
    '''

    E = NeuronGroup(80, eqs, threshold='v>1', reset='v=0', refractory=2 * ms, method='euler')
    I_grp = NeuronGroup(20, eqs, threshold='v>1', reset='v=0', refractory=2 * ms, method='euler')

    E.v, I_grp.v = 'rand()', 'rand()' ## neurons start at random states - Prevents artificial synchronization
    E.I, I_grp.I = 1.5, 1.0

    # 2. FIXED SYNAPSES: just shows the neuronal firing and stoping repetition creates balanced oscillations and brain rhythms
    S_EE = Synapses(E, E, on_pre='v_post += 0.1')
    S_EE.connect(p=0.2)

    S_EI = Synapses(E, I_grp, on_pre='v_post += 0.1')
    S_EI.connect(p=0.2)

    S_IE = Synapses(I_grp, E, on_pre='v_post -= 0.4')
    S_IE.connect(p=0.2)

    S_II = Synapses(I_grp, I_grp, on_pre='v_post -= 0.4')
    S_II.connect(p=0.2)

    rate_mon = PopulationRateMonitor(E)  ## tracks how many neurons fire over time, represents collective brain activity

    run(run_time)

    # 3. FFT ANALYSIS : removes first 200 ms, early activity is unstable so it doesn't produce real oscillations
    start_idx = int(200 * ms / defaultclock.dt)
    sig = rate_mon.rate[start_idx:]
    fs = 1.0 / defaultclock.dt     ## Determines how finely time is measured, needed for accurate frequency calculation

    # Welch's Method
    freqs, psd = signal.welch(sig, fs=fs, nperseg=2048)
    #Converts time signal --> frequency signal
    #calculates PSD (Power Spectral Density) - how strong each frequency is

    # 4. FIXED COMPARISON: Added '* Hz' to match dimensions
    gamma_mask = (freqs >= 30 * Hz) & (freqs <= 90 * Hz)
        #Establishes gamma band frequency as 30-90 Hz


## What is the brain's dominant rhythm
    if any(gamma_mask):
        peak_idx = np.argmax(psd[gamma_mask])
        peak_freq = freqs[gamma_mask][peak_idx] #finds strongest frequency
        peak_pwr = psd[gamma_mask][peak_idx] #finds how strong it is
    else:
        peak_freq, peak_pwr = 0 * Hz, 0
        ## no oscillation

    data_for_table.append((sigma_val, peak_freq, peak_pwr))

    # --- PLOTTING ---
    plt.semilogy(freqs / Hz, psd, label=f'Noise: {sigma_val}')

# --- PRINTING THE DATA TABLE ---
print("\n" + "=" * 60)
print(f"{'Noise (σ)':<10} | {'Peak Freq (Hz)':<15} | {'Peak Power':<10}")
print("-" * 60)
for noise, freq, pwr in data_for_table:
    # Convert pwr to float for a cleaner print
    p_val = float(pwr)
    stability = "Stable" if p_val > 50 else "Unstable"
    print(f"{noise:<10.2f} | {freq / Hz:<15.2f} | {p_val:<10.2f} ({stability})")
print("=" * 60 + "\n")

plt.title("FFT Analysis: Stability of Neural Oscillations")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.xlim(10, 100)
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()
