import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
    for i,inhib_spiketrain in enumerate(inhib_source_spiketrains):
        if i == 0:
            ax.vlines(inhib_spiketrain.times, ymin=y_lims[0], ymax=line_upper_lim, color='darkorange', label='Inhibitory spikes')
        else:
            ax.vlines(inhib_spiketrain.times, ymin=y_lims[0], ymax=line_upper_lim, color='darkorange')
    for i,excit_spiketrain in enumerate(excit_source_spiketrains):
        if i == 0:
            ax.vlines(excit_spiketrain.times, ymin=y_lims[0], ymax=line_upper_lim, color='blue', label='Excitatory spikes')
        else:
            ax.vlines(excit_spiketrain.times, ymin=y_lims[0], ymax=line_upper_lim, color='blue')

def plotTargetVWithInhibExcitSpikes(target_pop_v, inhib_source_spiketrains, excit_source_spiketrains):
    """
    For plotting the target membrane voltage with the inhibitory and excitatory spikes.
    Arguments:  target_pop_v, membrane voltage
                inhib_source_spiketrains,
                excit_source_spiketrains,
    Returns:    nothing
    """
    num_inhib = len(inhib_source_spiketrains)
    num_excit = len(excit_source_spiketrains)
    num_targets = target_pop_v.shape[1]
    fig,axes = plt.subplots(nrows=num_targets, ncols=1, figsize=(5,4), squeeze=False)
    for i,v in enumerate(target_pop_v.T):
        ax = axes[i,0]
        if i == 0:
            ax.set_title('Inhibitory cells = ' + str(num_inhib) + ', Excitatory cells = ' + str(num_excit), fontsize='x-large')
        ax.plot(target_pop_v.times, v, label='target membrane voltage', color='green')
        ax.set_xlim(target_pop_v.times[0],target_pop_v.times[-1])
        if i == np.floor(num_targets/2).astype(int):
            ax.set_ylabel('Membrane Voltage (mV)', fontsize='x-large')
        if i == (num_targets - 1):
            y_lims = ax.get_ylim()
            plotInhibExcitSpikes(ax, y_lims, inhib_source_spiketrains, excit_source_spiketrains)
            ax.set_ylim(ax.get_ylim()[0], v.max().magnitude.sum())
            ax.set_xlabel('Time (ms)', fontsize='x-large')
            ax.tick_params(axis='both', labelsize='large')
            [ax.spines[l].set_visible(False) for l in ['top','right']]
            ax.legend(fontsize='large')
        else:
            ax.set_ylim(v.min(), v.max())
            ax.tick_params(axis='y', labelsize='large')
            ax.set_xticks([])
            [ax.spines[l].set_visible(False) for l in ['top','right','bottom']]
    # ax.set_ylim(y_lims)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.075)

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

def plotInhExcSynapticStrengths(target_pop_gsyn_exc, target_pop_gsyn_inh, duration):
    """
    For plotting the excitatory/inhibitory synaptic strengths over time. (I believe these are aggregates.)
    Arguments:  target_cell_gsyn_exc, floats
                target_cell_gsyn_inh, floats
                duration, the duration of the simulation
    Returns:    nothing
    """
    num_excit = target_pop_gsyn_exc.shape[1]
    num_inhib = target_pop_gsyn_inh.shape[1]
    num_rows = num_excit + num_inhib
    fig,axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(5,4))
    for i,gsyn_exc in enumerate(target_pop_gsyn_exc.T):
        ax = axes[i]
        ax.plot(target_pop_gsyn_exc.times, gsyn_exc, color='blue', label=r'excitatory g$_{syn}$')
        ax.set_xlim(0, duration)
        ax.legend(fontsize='large') if i == 0 else None
        ax.set_ylabel(r'Strength ($\mu$S)', fontsize='x-large') if i == np.floor(num_rows/2).astype(int) else None
        ax.set_xticks([])
        [ax.spines[l].set_visible(False) for l in ['top','right','bottom']]
    for i,gsyn_inh in enumerate(target_pop_gsyn_inh.T):
        ax = axes[i + num_excit]
        ax.plot(target_pop_gsyn_inh.times, gsyn_inh, color='darkorange', label=r'inhibatory g$_{syn}$')
        ax.set_xlim(0, duration)
        ax.legend(fontsize='large') if i == 0 else None
        if i == num_inhib-1:
            ax.set_xlabel('Time (ms)', fontsize='x-large')
            ax.tick_params(axis='both', labelsize='large')
            [ax.spines[l].set_visible(False) for l in ['top','right']]
        else:
            [ax.spines[l].set_visible(False) for l in ['top','right','bottom']]
            ax.set_xticks([])
        ax.set_ylabel(r'Strength ($\mu$S)', fontsize='x-large') if i == np.floor(num_rows/2).astype(int) else None
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.075)

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
    ax.set_title(title, fontsize='x-large') if title != '' else None
    plt.tight_layout()

def plotConnectionWeights(weights, title=''):
    """
    For plotting a matrix of lateral or feed-forward weights.
    Arguments:  weights, the weight matrix
    Returns:    None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
    im = ax.imshow(weights, cmap=cm.get_cmap('Blues'))
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_title(title, fontsize='x-large') if title != '' else None
    plt.tight_layout()
