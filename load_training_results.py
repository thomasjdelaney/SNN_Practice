import argparse, os, sys, shutil, h5py
import pyNN.nest as pynn
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from PlottingFunctions import *
from PoissonWeightVariation import *
from utilities import *

# eventually this script will evaluate the performance of the network
parser = argparse.ArgumentParser(description='Load in the results of training, and have a look.')
parser.add_argument('-f', '--file_path_name', help='The h5 file for loading.', default=os.path.join(os.environ['HOME'],'SNN_practice','h5','training_results.h5'), type=str)
parser.add_argument('-p', '--pres_duration', help='the duration of the presentation of each stimulus.', default=100, type=int)
parser.add_argument('-n', '--num_pres_per_stim', help='number of presentations of each stimulus.', default=100, type=int)
parser.add_argument('-s', '--numpy_seed', help='For seeding random numbers', default=1798, type=int)
parser.add_argument('--debug', help='enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

np.random.seed(args.numpy_seed)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

proj_dir = os.path.join(os.environ['HOME'], 'SNN_practice')
h5_dir = os.path.join(proj_dir, 'h5')

pynn.setup(timestep=0.1, min_delay=2.0) # different

def getPresentationRatesForCallback(num_source, num_pres_per_stim):
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
    random_inds = np.random.permutation(np.arange(200))
    is_bright = random_inds > 99
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
    is_bright, random_on_rates, random_off_rates = getPresentationRatesForCallback(num_source, num_pres_per_stim)

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

if not args.debug:
    num_stim = 2
    is_bright, source_on_spikes, source_off_spikes, bright_spikes, dark_spikes = presentStimuli(args.pres_duration, args.num_pres_per_stim, num_source, num_target, bright_on_weights, bright_off_weights, bright_lat_weights, dark_on_weights, dark_off_weights, dark_lat_weights)
    binned_source_on_spikes, binned_source_off_spikes, binned_bright_spikes, binned_dark_spikes = binSpikeTimes(num_stim, args.pres_duration, args.num_pres_per_stim, is_bright, source_on_spikes, source_off_spikes, bright_spikes, dark_spikes)
