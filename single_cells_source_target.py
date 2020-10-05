import pyNN.nest as pynn
import numpy as np
import matplotlib.pyplot as plt

pynn.setup(timestep=0.1, min_delay=2.0)
sim_duration = 1000.0

if_cell = pynn.create(pynn.IF_cond_exp, {'i_offset':0.11, 'tau_refrac':3.0, 'v_thresh':-51.0})
source = pynn.create(pynn.SpikeSourcePoisson, {'rate':10})

pynn.connect(source, if_cell, weight=0.006, receptor_type='excitatory', delay=2.0)

if_cell.record('v')
source.record('spikes')

pynn.run(sim_duration)
pynn.end()

if_cell_v = if_cell.get_data('v').segments[0].analogsignals[0]
source_spiketrain = source.get_data('spikes').segments[0].spiketrains[0]

fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,4))
ax.plot(if_cell_v.times, if_cell_v, label='IF-cell membrane voltage')
y_lims = ax.get_ylim()
ax.vlines(source_spiketrain.times, ymin=y_lims[0], ymax=y_lims[1], color='black', alpha=0.3, label='Pre-synaptic spike', linestyle='--')
ax.set_ylim(y_lims)
ax.set_xlim(0,sim_duration)
ax.set_xlabel('Time (ms)', fontsize='x-large')
ax.set_ylabel('Membrane Voltage (mV)', fontsize='x-large')
ax.tick_params(axis='both', labelsize='large')
[ax.spines[l].set_visible(False) for l in ['top','right']]
ax.legend(fontsize='large')
ax.set_title('Pre-synaptic cell firing rate = ' + str(source.get('rate')) + 'Hz', fontsize='x-large')
plt.tight_layout()
plt.show(block=False)
