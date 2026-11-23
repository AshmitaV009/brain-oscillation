from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# PARAMETERS
# -----------------------
''' The ratio of excitatory (E) to inhibitory (Extra-BrainOscillPrject) neurons in the mammalian cortex is frequently cited as
 being around 4:1 or, more generally, between 3:1 and 9:1. This means about 80% of cortical neurons are excitatory 
 (pyramidal cells) and 20% are inhibitory (interneurons), a balance crucial for stability and proper brain function.'''

("Extra-BrainOscillPrject did multiple iterations using the different ratios and for each of these Extra-BrainOscillPrject also used different amounts of NE and NI "
 "For example ratio - 7:1 -- Amounts (35:5). (70:10), (140:20) ")

NE = 80       # excitatory neurons
NI = 20      # inhibitory neurons

N_total = NE + NI

run_time = 500*ms ##Well, the command run(500*ms) runs the simulation for 500 ms.

#noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # try multiple noise levels
noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8,1.0]  # try multiple noise levels
#noise_levels = [0.0, 0.02, 0.04, 0.06, 0.08,0.1,0.2]  # try multiple noise levels

connection_prob = 0.1

# -----------------------
# DATA STORAGE
# -----------------------
results = []  # will store spike counts for each noise level

# -----------------------
# LOOP OVER NOISE LEVELS
# -----------------------
for sigma_noise in noise_levels:

    start_scope()  # reset network

    # -----------------------
    # NEURON MODEL WITH NOISE
    # -----------------------
    eqs = '''
    dv/dt = (-v + Extra-BrainOscillPrject)/10/ms + sigma_noise*xi*sqrt(1/ms) : 1
    Extra-BrainOscillPrject : 1
    sigma_noise : 1 (constant)
    '''

    # Create neuron groups
    E = NeuronGroup(NE, eqs, threshold='v>1', reset='v=0', method='euler')
    Inh = NeuronGroup(NI, eqs, threshold='v>1', reset='v=0', method='euler')


    # Initialize
    E.v = 0
    Inh.v = 0
    E.I = 1.3
    Inh.I = 1.0
    E.sigma_noise = sigma_noise
    Inh.sigma_noise = sigma_noise

    # -----------------------
    # CONNECTIONS
    # -----------------------
    S_EE = Synapses(E, E, on_pre='v_post += 0.05')
    S_EE.connect(p=connection_prob)

    S_EI = Synapses(E, Inh, on_pre='v_post += 0.05')
    S_EI.connect(p=connection_prob)

    S_IE = Synapses(Inh, E, on_pre='v_post -= 0.1')
    S_IE.connect(p=connection_prob)

    S_II = Synapses(Inh, Inh, on_pre='v_post -= 0.1')
    S_II.connect(p=connection_prob)

    # -----------------------
    # SPIKE MONITORS
    # -----------------------
    spikes_E = SpikeMonitor(E)
    spikes_Inh = SpikeMonitor(Inh)

    # -----------------------
    # RUN SIMULATION
    # -----------------------
    run(run_time)

    # -----------------------
    # PLOT RESULTS
    # -----------------------
    plt.figure(figsize=(8,4))
    plt.plot(spikes_E.t/ms, spikes_E.i, '.b', label='Excitatory')
    plt.plot(spikes_Inh.t/ms + NE, spikes_Inh.i, '.r', label='Inhibitory')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title(f'Network Activity with Noise σ={sigma_noise}')
    plt.legend()
    plt.show()

    # -----------------------
    # SAVE SPIKE COUNT DATA
    # -----------------------
    exc_spike_count = len(spikes_E.i)
    inh_spike_count = len(spikes_Inh.i)
    rhythm_quality = 'high' if sigma_noise < 0.1 else ('medium' if sigma_noise < 0.2 else 'low')
    results.append([sigma_noise, exc_spike_count, inh_spike_count, rhythm_quality])

# -----------------------
# PRINT RESULTS TABLE
# -----------------------
print("Noise | Excitatory Spikes | Inhibitory Spikes | Rhythm Stability")
for row in results:
    print(f"{row[0]:.2f}   | {row[1]:>5}             | {row[2]:>5}             | {row[3]}")
