import argparse, os, sys, shutil, h5py
import pyNN.nest as pynn
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
from PlottingFunctions import *
from PoissonWeightVariation import *
from utilities import *

# eventually this script will evaluate the performance of the network
parser = argparse.ArgumentParser(description='Load in the results of training, and have a look.')
parser.add_argument('-f', '--file_path_name', help='The h5 file for loading.', default=os.path.join(os.environ['HOME'],'SNN_practice','h5','training_results.h5'), type=str)
parser.add_argument('-p', '--pres_duration', help='the duration of the presentation of each stimulus.', default=100, type=int)
parser.add_argument('-n', '--num_pres_per_stim', help='number of presentations of each stimulus.', default=100, type=int)
parser.add_argument('-s', '--numpy_seed', help='For seeding random numbers', default=1798, type=int)
parser.add_argument('-a', '--lat_weight_adjustment', help='multiplicative coefficient applied to mean lat weight, to make additional factor for those weights', default=0, type=float)
parser.add_argument('--debug', help='enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

np.random.seed(args.numpy_seed)
pd.set_option('max_rows', 30)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

proj_dir = os.path.join(os.environ['HOME'], 'SNN_practice')
h5_dir = os.path.join(proj_dir, 'h5')
image_dir = os.path.join(proj_dir, 'images')
csv_dir = os.path.join(proj_dir, 'csv')

pynn.setup(timestep=0.1, min_delay=2.0) # different

def extractInfoFromH5File(file_path_name):
    """
    For extracting all the available info from a h5 file.
    Arguments:  file_path_name, str, the path and file_name
    Returns:    duration, the duration of training
                num_source, number of cells in source layers
                num_target, number of cells in target layer
                bright_on_weights, the trained weights of the synapses from the on source layer to the bright target layer
                bright_off_weights, the trained weights of the synapses from the off source layer to the bright target layer
                bright_lat_weights, the trained weights of the lateral connections within the bright target layer
                dark_on_weights, the trained weights of the synapses from the on source layer to the dark target layer
                dark_off_weights, the trained weights of the synapses from the off source layer to the dark target layer
                dark_lat_weights, the trained weights of the synapses from the off source layer to the dark target layer
                weight_times, the times at which the weights were sampled during training
                bright_ff_on_time, the evolution of the bright on weights over time
                bright_ff_off_time, the evolution of the bright off weights over time
                bright_lat_time, the evolution of the bright_lat_weights over time
                dark_ff_on_time, the evolution of the dark on weights over time
                dark_ff_off_time, the evolution of the dark off weights over time
                dark_lat_time, the evolution of the dark_lat_weights over time
    """
    h5_file = h5py.File(args.file_path_name, 'r')
    duration = h5_file.get('duration')[()]
    num_source = h5_file.get('num_source')[()]
    num_target = h5_file.get('num_target')[()]
    bright = h5_file.get('bright')
    dark = h5_file.get('dark')
    bright_on_weights = bright.get('ff_on_weights')[()]
    bright_off_weights = bright.get('ff_off_weights')[()]
    bright_lat_weights = bright.get('lat_weights')[()]
    dark_on_weights = dark.get('ff_on_weights')[()]
    dark_off_weights = dark.get('ff_off_weights')[()]
    dark_lat_weights = dark.get('lat_weights')[()]
    weight_times = bright.get('weight_times')[()]
    bright_ff_on_time = bright.get('ff_on_weights_over_time')[()]
    bright_ff_off_time = bright.get('ff_off_weights_over_time')[()]
    bright_lat_time = bright.get('lat_weights_over_time')[()]
    dark_ff_on_time = dark.get('ff_on_weights_over_time')[()]
    dark_ff_off_time = dark.get('ff_off_weights_over_time')[()]
    dark_lat_time = dark.get('lat_weights_over_time')[()]
    return duration, num_source, num_target, bright_on_weights, bright_off_weights, bright_lat_weights, dark_on_weights, dark_off_weights, dark_lat_weights, weight_times, bright_ff_on_time, bright_ff_off_time, bright_lat_time, dark_ff_on_time, dark_ff_off_time, dark_lat_time

def getAdjustedLateralWeights(lat_weight_adjustment, bright_lat_weights, dark_lat_weights):
    """
    For adjusting the lateral weights. Only adjusts non-nan weights.
    Arguments:  lat_weight_adjustment, the coefficient, can be positive or negative, applied to the mean weights
                bright_lat_weights,
                dark_lat_weights
    Returns:    adjusted_bright_lat_weights, adjusted_dark_lat_weights
    """
    if args.lat_weight_adjustment != 0:
        adjusted_dark_lat_weights = dark_lat_weights + args.lat_weight_adjustment * np.nanmean(dark_lat_weights)
        adjusted_bright_lat_weights = bright_lat_weights + args.lat_weight_adjustment * np.nanmean(bright_lat_weights)
    else:
        adjusted_dark_lat_weights = dark_lat_weights
        adjusted_bright_lat_weights = bright_lat_weights
    return adjusted_bright_lat_weights, adjusted_dark_lat_weights


def getPresentationRatesForCallback(num_stim, num_source, num_pres_per_stim):
    """
    For getting an iterator of rates to be used by the callback usied by the simulation.
    Arguments:  num_source, the number of cells in the source layers.
                num_pres_per_stim, the number of presentations per stimulus
    Returns:    is_bright, boolean array, 0 indicates dark, 1 indicates bright
                random on rates, iterator, num_source x num_pres_per_stim x num_stim
                random off rates, iterator, num_source x num_pres_per_stim x num_stim
    """
    bright_on_rates, bright_off_rates = getOnOffSourceRates((num_source, num_pres_per_stim), 'bright')
    dark_on_rates, dark_off_rates = getOnOffSourceRates((num_source, num_pres_per_stim), 'dark')
    on_rates = np.hstack([dark_on_rates, bright_on_rates])
    off_rates = np.hstack([dark_off_rates, bright_off_rates])
    random_inds = np.random.permutation(np.arange(num_stim * num_pres_per_stim))
    is_bright = random_inds >= num_pres_per_stim
    random_on_rates = iter(on_rates[:,random_inds].T)
    random_off_rates = iter(off_rates[:, random_inds].T)
    return is_bright, random_on_rates, random_off_rates

def presentStimuli(pres_duration, num_pres_per_stim, num_source, num_target, bright_on_weights, bright_off_weights, bright_lat_weights, dark_on_weights, dark_off_weights, dark_lat_weights):
    """
    For presenting a stimulus to the target network. Callback is used to switch between presentation rates.
    Arguments:  num_source
                num_target
                num_pres_per_stim,
                pres_duration
    """
    num_stim = 2 # two stimuli 'bright' and 'dark'
    total_duration = num_stim * num_pres_per_stim * pres_duration

    source_on_pop = pynn.Population(num_source, pynn.SpikeSourcePoisson(), label='source_on_pop')
    source_off_pop = pynn.Population(num_source, pynn.SpikeSourcePoisson(), label='source_off_pop')
    is_bright, random_on_rates, random_off_rates = getPresentationRatesForCallback(num_stim, num_source, num_pres_per_stim)

    bright_target_pop = pynn.Population(num_target, pynn.IF_cond_exp, {'i_offset':0.11, 'tau_refrac':3.0, 'v_thresh':-51.0}, label='target_pop')
    dark_target_pop = pynn.Population(num_target, pynn.IF_cond_exp, {'i_offset':0.11, 'tau_refrac':3.0, 'v_thresh':-51.0}, label='target_pop')

    bright_on_conn = pynn.Projection(source_on_pop, bright_target_pop, connector=pynn.AllToAllConnector(), synapse_type=pynn.StaticSynapse(weight=bright_on_weights), receptor_type='excitatory')
    bright_off_conn = pynn.Projection(source_off_pop, bright_target_pop, connector=pynn.AllToAllConnector(), synapse_type=pynn.StaticSynapse(weight=bright_off_weights), receptor_type='excitatory')
    bright_lat_conn = pynn.Projection(bright_target_pop, bright_target_pop, connector=pynn.AllToAllConnector(), synapse_type=pynn.StaticSynapse(weight=bright_lat_weights), receptor_type='inhibitory')
    dark_on_conn = pynn.Projection(source_on_pop, dark_target_pop, connector=pynn.AllToAllConnector(), synapse_type=pynn.StaticSynapse(weight=dark_on_weights), receptor_type='excitatory')
    dark_off_conn = pynn.Projection(source_off_pop, dark_target_pop, connector=pynn.AllToAllConnector(), synapse_type=pynn.StaticSynapse(weight=dark_off_weights), receptor_type='excitatory')
    dark_lat_conn = pynn.Projection(dark_target_pop, dark_target_pop, connector=pynn.AllToAllConnector(), synapse_type=pynn.StaticSynapse(weight=dark_lat_weights), receptor_type='inhibitory')

    source_on_pop.record('spikes')
    source_off_pop.record('spikes')
    bright_target_pop.record(['spikes'])
    dark_target_pop.record(['spikes'])

    pynn.run(total_duration, callbacks=[PoissonWeightVariation(source_on_pop, random_on_rates, pres_duration), PoissonWeightVariation(source_off_pop, random_off_rates, pres_duration)])
    pynn.end()

    source_on_spikes = source_on_pop.get_data('spikes').segments[0].spiketrains
    source_off_spikes = source_off_pop.get_data('spikes').segments[0].spiketrains
    bright_spikes = bright_target_pop.get_data('spikes').segments[0].spiketrains
    dark_spikes = dark_target_pop.get_data('spikes').segments[0].spiketrains
    return is_bright, source_on_spikes, source_off_spikes, bright_spikes, dark_spikes

def binSpikeTimes(num_stim, pres_duration, num_pres_per_stim, is_bright, source_on_spikes, source_off_spikes, bright_spikes, dark_spikes):
    """
    For binning the source and target spikes into spike counts by presentations.
    Arguments:  num_stim,
                pres_duration,
                num_pres_per_stim,
                is_bright,
                source_on_spikes,
                source_off_spikes,
                bright_spikes,
                dark_spikes
    Returns:    binned_source_on_spikes, binned_source_off_spikes, binned_bright_spikes, binned_dark_spikes
    """
    trial_borders = args.pres_duration * np.arange(num_stim*args.num_pres_per_stim + 1)
    binned_source_on_spikes, binned_source_off_spikes = np.zeros(trial_borders.size-1), np.zeros(trial_borders.size-1)
    binned_bright_spikes, binned_dark_spikes = np.zeros(trial_borders.size-1), np.zeros(trial_borders.size-1)
    for on_st, off_st in zip(source_on_spikes,source_off_spikes):
        binned_source_on_spikes += np.histogram(on_st, bins=trial_borders)[0]
        binned_source_off_spikes += np.histogram(off_st, bins=trial_borders)[0]
    for bright_st, dark_st in zip(bright_spikes, dark_spikes):
        binned_bright_spikes += np.histogram(bright_st, bins=trial_borders)[0]
        binned_dark_spikes += np.histogram(dark_st, bins=trial_borders)[0]
    return binned_source_on_spikes, binned_source_off_spikes, binned_bright_spikes, binned_dark_spikes

def quickSpikeCountAnalysis(binned_source_on_spikes, binned_source_off_spikes, binned_bright_spikes, binned_dark_spikes, is_bright):
    """
    For doing some quick analysis of the spike counts resulting from stimulus presentation.
    Arguments:  binned_source_on_spikes,
                binned_source_off_spikes,
                binned_bright_spikes,
                binned_dark_spikes
                is_bright
    Returns:    nothing
    """
    on_bright_mean_rate = binned_source_on_spikes[is_bright].mean()
    on_dark_mean_rate = binned_source_on_spikes[np.invert(is_bright)].mean()
    off_bright_mean_rate = binned_source_off_spikes[is_bright].mean()
    off_dark_mean_rate = binned_source_off_spikes[np.invert(is_bright)].mean()
    bright_bright_mean_rate = binned_bright_spikes[is_bright].mean()
    bright_dark_mean_rate = binned_bright_spikes[np.invert(is_bright)].mean()
    dark_bright_mean_rate = binned_dark_spikes[is_bright].mean()
    dark_dark_mean_rate = binned_dark_spikes[np.invert(is_bright)].mean()
    source_stim_agree_prop = np.sum(is_bright == (binned_source_on_spikes > binned_source_off_spikes))/binned_source_on_spikes.size
    target_stim_agree_prop = np.sum(is_bright == (binned_bright_spikes > binned_dark_spikes))/binned_source_on_spikes.size
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Source on firing rate: ' + str(on_bright_mean_rate) + ' and ' + str(on_dark_mean_rate) + ' for bright and dark.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Source off firing rate: ' + str(off_bright_mean_rate) + ' and ' + str(off_dark_mean_rate) + ' for bright and dark.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Target bright firing rate: ' + str(bright_bright_mean_rate) + ' and ' + str(bright_dark_mean_rate) + ' for bright and dark.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Target dark firing rate: ' + str(dark_bright_mean_rate) + ' and ' + str(dark_dark_mean_rate) + ' for bright and dark.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Source layers agreement with stimulus: ' + str(source_stim_agree_prop))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Target layers agreement with stimulus: ' + str(target_stim_agree_prop))
    return on_bright_mean_rate, on_dark_mean_rate, off_bright_mean_rate, off_dark_mean_rate, bright_bright_mean_rate, bright_dark_mean_rate, dark_bright_mean_rate, dark_dark_mean_rate, source_stim_agree_prop, target_stim_agree_prop

def recordRunResults(file_path_name, duration, num_source, num_target, lat_weight_adjustment, pres_duration, num_pres_per_stim, on_bright_mean_rate, off_bright_mean_rate, on_dark_mean_rate, off_dark_mean_rate, bright_bright_mean_rate, bright_dark_mean_rate, dark_bright_mean_rate, dark_dark_mean_rate, source_stim_agree_prop, target_stim_agree_prop):
    """
    Append the results of the classification to a csv file, or create the csv file if it doesn't exist.
    Arguments:  file_path_name,
                duration,
                num_source,
                num_target,
                lat_weight_adjustment,
                pres_duration,
                num_pres_per_stim,
                on_bright_mean_rate,
                off_bright_mean_rate,
                on_dark_mean_rate,
                off_dark_mean_rate,
                bright_bright_mean_rate,
                bright_dark_mean_rate,
                dark_bright_mean_rate,
                dark_dark_mean_rate,
                source_stim_agree_prop,
                target_stim_agree_prop,
    Returns     csv_file_name
    """
    csv_file_name = os.path.join(csv_dir, 'light_dark_class_results.csv')
    file_exists = os.path.isfile(csv_file_name)
    record_to_add_dict = {'file_path_name':file_path_name, 'duration':duration, 'num_source':num_source, 'num_target':num_target, 'lat_weight_adjustment':lat_weight_adjustment, 'pres_duration':pres_duration, 'num_pres_per_stim':num_pres_per_stim, 'on_bright_mean_rate':on_bright_mean_rate, 'off_bright_mean_rate':off_bright_mean_rate, 'on_dark_mean_rate':on_dark_mean_rate, 'off_dark_mean_rate':off_dark_mean_rate, 'bright_bright_mean_rate':bright_bright_mean_rate, 'bright_dark_mean_rate':bright_dark_mean_rate, 'dark_bright_mean_rate':dark_bright_mean_rate, 'dark_dark_mean_rate':dark_dark_mean_rate, 'source_stim_agree_prop':source_stim_agree_prop, 'target_stim_agree_prop':target_stim_agree_prop}
    if file_exists:
        loaded_res_frame = pd.read_csv(csv_file_name)
        loaded_res_frame.append(pd.DataFrame.from_records([record_to_add_dict]), ignore_index=True)
        loaded_res_frame.to_csv(csv_file_name, index_label='row_num')
    else:
        res_frame = pd.DataFrame.from_records([record_to_add_dict])
        res_frame.to_csv(csv_file_name, index_label='row_num')
    return csv_file_name

if not args.debug:
    duration, num_source, num_target, bright_on_weights, bright_off_weights, bright_lat_weights, dark_on_weights, dark_off_weights, dark_lat_weights, weight_times, bright_ff_on_time, bright_ff_off_time, bright_lat_time, dark_ff_on_time, dark_ff_off_time, dark_lat_time = extractInfoFromH5File(args.file_path_name)
    num_stim = 2
    adjusted_bright_lat_weights, adjusted_dark_lat_weights = getAdjustedLateralWeights(args.lat_weight_adjustment, bright_lat_weights, dark_lat_weights)
    is_bright, source_on_spikes, source_off_spikes, bright_spikes, dark_spikes = presentStimuli(args.pres_duration, args.num_pres_per_stim, num_source, num_target, bright_on_weights, bright_off_weights, adjusted_bright_lat_weights, dark_on_weights, dark_off_weights, adjusted_dark_lat_weights)
    binned_source_on_spikes, binned_source_off_spikes, binned_bright_spikes, binned_dark_spikes = binSpikeTimes(num_stim, args.pres_duration, args.num_pres_per_stim, is_bright, source_on_spikes, source_off_spikes, bright_spikes, dark_spikes)
    on_bright_mean_rate, on_dark_mean_rate, off_bright_mean_rate, off_dark_mean_rate, bright_bright_mean_rate, bright_dark_mean_rate, dark_bright_mean_rate, dark_dark_mean_rate, source_stim_agree_prop, target_stim_agree_prop = quickSpikeCountAnalysis(binned_source_on_spikes, binned_source_off_spikes, binned_bright_spikes, binned_dark_spikes, is_bright)
    csv_file_name = recordRunResults(args.file_path_name, duration, num_source, num_target, args.lat_weight_adjustment, args.pres_duration, args.num_pres_per_stim, on_bright_mean_rate, off_bright_mean_rate, on_dark_mean_rate, off_dark_mean_rate, bright_bright_mean_rate, bright_dark_mean_rate, dark_bright_mean_rate, dark_dark_mean_rate, source_stim_agree_prop, target_stim_agree_prop)
    rasterMultiPopulations([source_on_spikes, source_off_spikes, bright_spikes, dark_spikes], ['blue', 'darkblue', 'green', 'darkorange'], num_stim, args.pres_duration, args.num_pres_per_stim, is_bright)
    file_name = os.path.join(image_dir, 'bright_ff_over_time', os.path.basename(args.file_path_name).replace('.h5','_ff_over_time.png'))
    plotWeightSpreadOverTime(bright_ff_on_time, title='bright on', colour='lightblue', mean_colour='blue', times=weight_times, include_mean=True, file_name=file_name)
    plotWeightSpreadOverTime(bright_ff_off_time, title='bright off', colour='magenta', mean_colour='darkviolet', times=weight_times, include_mean=True)
    plotWeightSpreadOverTime(bright_lat_time, title='bright lat', colour='green', mean_colour='darkgreen', times=weight_times, include_mean=True)
    plotWeightSpreadOverTime(dark_ff_on_time, title='dark on', colour='cyan', mean_colour='teal', times=weight_times, include_mean=True)
    plotWeightSpreadOverTime(dark_ff_off_time, title='dark off', colour='gold', mean_colour='darkgoldenrod', times=weight_times, include_mean=True)
    plotWeightSpreadOverTime(dark_lat_time, title='dark lat', colour='orangered', mean_colour='red', times=weight_times, include_mean=True)
    plotConnectionWeights(bright_on_weights, title='bright on')
    plotConnectionWeights(dark_on_weights, title='dark on')
    plotConnectionWeights(bright_off_weights, title='bright off')
    plotConnectionWeights(dark_off_weights, title='dark off')
    plotConnectionWeights(bright_lat_weights, title='bright lat')
    plotConnectionWeights(dark_lat_weights, title='dark lat')
