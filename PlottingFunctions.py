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

def plotBrightDarkTimeBins(ax, num_stim, pres_duration, num_pres_per_stim, is_bright):
    """
    For shading in the dark presentations. Input axis required
    Arguments:  ax, the plt axis that we're already plotting on
                num_stim, number of stimuli
                pres_duration, duration of a given presentation in classification
                num_pres_per_stim,
                is_bright,
    Returns:    nothing
    """
    pres_starts = np.arange(0,num_stim*pres_duration*num_pres_per_stim, pres_duration)
    ylims = ax.get_ylim()
    dark_intervals = np.vstack([pres_starts, pres_starts + pres_duration]).T[np.invert(is_bright)]
    for di in dark_intervals:
        ax.fill_between(di, y1=ylims[0], y2=ylims[1], color='black', alpha=0.15)
    ax.set_ylim(ylims)

def rasterMultiPopulations(spike_train_collections, colours, num_stim, pres_duration, num_pres_per_stim, is_bright, file_name=None):
    """
    For raster plotting spike trains from different populations.
    Arguments:  spike_train_collections, list of lists of spike trains
                colours, list of colours, should be same length as spike_train_collections
                num_stim, number of stimuli
                pres_duration, duration of a given presentation in classification
                num_pres_per_stim,
                is_bright,
                file_name
    Returns:    nothing
    """
    duration = num_stim*pres_duration*num_pres_per_stim
    all_trains = []
    all_colours = []
    for i,spike_train_col in enumerate(spike_train_collections):
        all_trains = all_trains + spike_train_col
        all_colours = all_colours + [colours[i]]*len(spike_train_col)
    total_num_cells = len(all_trains)
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,4))
    ax.eventplot(all_trains, colors=all_colours)
    plotBrightDarkTimeBins(ax, num_stim, pres_duration, num_pres_per_stim, is_bright)
    ax.set_xlabel('Time (ms)', fontsize='x-large')
    ax.set_ylabel('Cells', fontsize='x-large')
    ax.set_ylim(-0.5, -0.5 + total_num_cells)
    ax.set_xlim(0, duration)
    ax.tick_params(axis='x', labelsize='large')
    ax.set_yticks([])
    [ax.spines[l].set_visible(False) for l in ['top','right']]
    plt.tight_layout()
    plt.savefig(file_name) if file_name != None else None

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

def plotWeightsOverTime(weights_time_series, title='', times=None):
    """
    For plotting the change in weights over time.
    Arguments:  weights_time_series, numpy array (num time points, num weights)
    Returns:    nothing
    """
    num_weights = weights_time_series.shape[1]
    plotting_weights = (weights_time_series/weights_time_series.mean(axis=0)) + np.arange(num_weights)
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,4))
    if np.all(times == None):
        plt.plot(weights_time_series.times, plotting_weights)
        ax.set_xlim(weights_time_series.times[0], weights_time_series.times[-1])
    else:
        plt.plot(times, weights_time_series, plotting_weights)
        ax.set_xlim(times[0], times[-1])
    ax.set_xlabel('Time (ms)', fontsize='x-large')
    ax.set_ylabel('Weights', fontsize='x-large')
    ax.tick_params(axis='x', labelsize='large')
    ax.set_yticks([])
    [ax.spines[l].set_visible(False) for l in ['top','right']]
    ax.set_title(title, fontsize='x-large') if title != '' else None
    plt.tight_layout()

def plotConnectionWeights(weights, title='', file_name=None):
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
    plt.savefig(file_name) if file_name != None else None

def plotWeightSpreadOverTime(weights_time_series, title='', colour='lightblue', mean_colour='blue', times=None, include_mean=True, file_name=None):
    """
    For plotting the evolution of the weights over time on top of each other.
    Arguments:  weights_time_series,
                title,
                colour, the colour for the individual weights
                mean_colour, the colour for the mean, ideally a similar colour to 'colour' but darker
                times, x-axis (weights_time_series may sometimes have a 'time' attribute)
                include_mean, flag to include the mean of the weights
                file_name, if not none, save the figure here
    Returns:    nothing
    """
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,4))
    times = weights_time_series.times if np.all(times == None) else times
    ax.plot(times, weights_time_series, color=colour)
    ax.plot(times, weights_time_series.mean(axis=1), color=mean_colour) if include_mean else None
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel('Time (ms)', fontsize='x-large')
    ax.set_ylabel('Weights', fontsize='x-large')
    ax.tick_params(axis='both', labelsize='large')
    [ax.spines[l].set_visible(False) for l in ['top','right']]
    ax.set_title(title, fontsize='x-large') if title != '' else None
    plt.tight_layout()
    plt.savefig(file_name) if file_name != None else None

def plotColumnsFromResults(results_frame, col1, col2, x_lims, y_lims, var_label, x_label, y_label, chance=None):
    """
    For plotting something from the results dataframe
    """
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,4))
    ax.plot(results_frame[col1], results_frame[col2], label=var_label, color='orange')
    ax.hlines(xmin=-5,xmax=5,y=0.5,color='blue',label='chance', linestyle='--') if chance != None else None
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    [ax.spines[l].set_visible(False) for l in ['top','right']]
    ax.set_xlabel(x_label, fontsize='x-large')
    ax.set_ylabel(y_label, fontsize='x-large')
    ax.tick_params(axis='both', labelsize='large')
    plt.legend(fontsize='large')
    plt.tight_layout()
