import numpy as np
import matplotlib.pyplot as plt

def plotInhibExcitSpikes(ax, y_lims, inhib_source_spiketrains, excit_source_spiketrains):
    """
    For plotting inhibitory and excitatory spikes in orange and blue.
    Arguments:  ax, the axis of the figure we're working on.
                y_lims, limits of figure's y axis.
                inhib_source_spiketrains, inhibitory spike trains
                excit_source_spiketrains, excitatory spike trains
    Returns:    nothing
    """
    line_upper_lim = 0.05*(y_lims[1] - y_lims[0]) + y_lims[0]
    for inhib_spiketrain in inhib_source_spiketrains:
        ax.vlines(inhib_spiketrain.times, ymin=y_lims[0], ymax=line_upper_lim, color='darkorange')
    for excit_spiketrain in excit_source_spiketrains:
        ax.vlines(excit_spiketrain.times, ymin=y_lims[0], ymax=line_upper_lim, color='blue')

def plotTargetVWithInhibExcitSpikes(target_cell_v, inhib_source_spiketrains, excit_source_spiketrains):
    """
    For plotting the target membrane voltage with the inhibitory and excitatory spikes.
    Arguments:  target_cell_v, membrane voltage
                inhib_source_spiketrains,
                excit_source_spiketrains,
    Returns:    nothing
    """
    num_inhib = len(inhib_source_spiketrains)
    num_excit = len(excit_source_spiketrains)
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,4))
    ax.plot(target_cell_v.times, target_cell_v, label='target membrane voltage', color='green')
    y_lims = ax.get_ylim()
    plotInhibExcitSpikes(ax, y_lims, inhib_source_spiketrains, excit_source_spiketrains)
    ax.set_ylim(y_lims)
    ax.set_xlim(target_cell_v.times[0],target_cell_v.times[-1])
    ax.set_xlabel('Time (ms)', fontsize='x-large')
    ax.set_ylabel('Membrane Voltage (mV)', fontsize='x-large')
    ax.tick_params(axis='both', labelsize='large')
    [ax.spines[l].set_visible(False) for l in ['top','right']]
    ax.legend(fontsize='large')
    ax.set_title('Inhibitory cells = ' + str(num_inhib) + ', Excitatory cells = ' + str(num_excit), fontsize='x-large')
    plt.tight_layout()

def plotTargetInhibExcitSpikes(target_spikes, inhib_source_spiketrains, excit_source_spiketrains, duration):
    """
    For plotting a raster plot of all the cells, grouped into target, excit, and inhib.
    Arguments:  target_spikes,
                inhib_source_spiketrains,
                excit_source_spiketrains,
                duration,
    Returns:    nothing
    """
    num_inhib = len(inhib_source_spiketrains)
    num_excit = len(excit_source_spiketrains)
    num_targets = len(target_spikes)
    total_num_spikes = num_inhib + num_excit + num_targets
    all_spikes = inhib_source_spiketrains + excit_source_spiketrains + target_spikes
    all_colours = ['orange']*num_inhib + ['blue']*num_excit + ['green']*num_targets
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,4))
    ax.eventplot(all_spikes, colors=all_colours)
    ax.set_xlabel('Time (ms)', fontsize='x-large')
    ax.set_ylabel('Cells', fontsize='x-large')
    ax.set_ylim(-0.5, -0.5 + total_num_spikes)
    ax.set_xlim(0, duration)
    ax.tick_params(axis='x', labelsize='large')
    ax.set_yticks([])
    [ax.spines[l].set_visible(False) for l in ['top','right']]
    plt.tight_layout()

def plotInhExcSynapticStrengths(target_cell_gsyn_exc, target_cell_gsyn_inh, duration):
    """
    For plotting the excitatory/inhibitory synaptic strengths over time. (I believe these are aggregates.)
    Arguments:  target_cell_gsyn_exc, floats
                target_cell_gsyn_inh, floats
                duration, the duration of the simulation
    Returns:    nothing
    """
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,4))
    ax.plot(target_cell_gsyn_exc.times, target_cell_gsyn_exc, color='blue', label=r'excitatory g$_{syn}$')
    ax.plot(target_cell_gsyn_inh.times, target_cell_gsyn_inh, color='darkorange', label=r'inhibatory g$_{syn}$')
    ax.set_xlabel('Time (ms)', fontsize='x-large')
    ax.set_ylabel(r'Strength ($\mu$S)', fontsize='x-large')
    ax.legend(fontsize='large')
    ax.set_xlim(0, duration)
    ax.tick_params(axis='both', labelsize='large')
    [ax.spines[l].set_visible(False) for l in ['top','right']]
    plt.tight_layout()

def plotWeightsOverTime(weights_time_series, title=''):
    """
    For plotting the change in weights over time.
    Arguments:  weights_time_series, numpy array (num time points, num weights)
    Returns:    nothing
    """
    num_weights = weights_time_series.shape[1]
    plotting_weights = (weights_time_series/weights_time_series.mean(axis=0)) + np.arange(num_weights)
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,4))
    plt.plot(weights_time_series.times, plotting_weights)
    ax.set_xlabel('Time (ms)', fontsize='x-large')
    ax.set_ylabel('Weights', fontsize='x-large')
    ax.set_xlim(weights_time_series.times[0], weights_time_series.times[-1])
    ax.tick_params(axis='x', labelsize='large')
    ax.set_yticks([])
    [ax.spines[l].set_visible(False) for l in ['top','right']]
    ax.set_title(title, fontsize='x-large')
    plt.tight_layout()
