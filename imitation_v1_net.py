"""
Inspired by Hopkins et al (2018): http://dx.doi.org/10.1098/rsfs.2018.0007

A model for simulating a layer within V1. Feed-forward excitatory connections from source to target.
Target has lateral inhibitory connections.
Aim to 'train' the network to distinguish between light and dark.
"""
import argparse, os, sys, shutil, h5py
import pyNN.nest as pynn
import numpy as np
import matplotlib.pyplot as plt
from WeightRecorder import *
from PlottingFunctions import *
from utilities import *

parser = argparse.ArgumentParser(description='Feed-forward Poisson spike input. Lateral inhibitory connections')
parser.add_argument('-b', '--num_target', help='number of target cells.', default=5, type=int)
parser.add_argument('-f', '--num_source', help='number of source/feed-forward cells.', default=10, type=int)
parser.add_argument('-p', '--source_rates_params', help='params for the gamma distribution.', default=[10,0.5], type=float, nargs=2)
# parser.add_argument('-c', '--ff_conn_prob', help='feed-forward connection prob.', default=0.75, type=float)
# parser.add_argument('-l', '--lat_conn_prob', help='lateral connection prob.', default=0.75, type=float)
parser.add_argument('-d', '--duration', help='duration of simulation.', default=3000.0, type=float)
parser.add_argument('-i', '--num_iterations', help='the number of iterations to perform.', default=10, type=int)
parser.add_argument('-u', '--use_stdp', help='use STDP', default=False, action='store_true')
parser.add_argument('-r', '--record_source_spikes', help='Record the source spikes', default=False, action='store_true')
parser.add_argument('-t', '--stim_type', help='bright or dark', default='bright', choices=['bright', 'dark'])
parser.add_argument('-s', '--numpy_seed', help='RNG seed', default=1798, type=int)
parser.add_argument('-m', '--make_plots', help='make the plots, or not.', default=False, action='store_true')
parser.add_argument('--lat_conn_strength_params', help='params for distribution of strengths of later connections', default=[0.015,0.025], type=float, nargs=2)
parser.add_argument('--debug', help='enter debug mode', default=False, action='store_true')
args = parser.parse_args()

np.random.seed(args.numpy_seed)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

pynn.setup(timestep=0.1, min_delay=2.0) # different

proj_dir = os.path.join(os.environ['HOME'], 'SNN_practice')
h5_dir = os.path.join(proj_dir, 'h5')

def getOnOffSourceRates(num_source, on_bright_params=[20.0,1.0], on_dark_params=[10.0, 0.5], off_bright_params=[10.0, 0.5], off_dark_params=[20,1.0]):
    """
    Get rates for the on and off source populations. Need rates for both bright and dark stimuli. Param args are for gamma distributions.
    Arguments:  num_source, number of cells in the source populations.
                on_bright_params, parameters for a gamma distribution
    """
    on_bright_rates = np.random.gamma(on_bright_params[0], on_bright_params[1], size=num_source)
    on_dark_rates = np.random.gamma(on_dark_params[0], on_dark_params[1], size=num_source)
    off_bright_rates = np.random.gamma(off_bright_params[0], off_bright_params[1], size=num_source)
    off_dark_rates = np.random.gamma(off_dark_params[0], off_dark_params[1], size=num_source)
    return on_bright_rates, on_dark_rates, off_bright_rates, off_dark_rates

if not args.debug:
    on_bright_rates, on_dark_rates, off_bright_rates, off_dark_rates = getOnOffSourceRates(args.num_source)
    # DO BRIGHT FOR NOW
    source_on_pop = pynn.Population(args.num_source, pynn.SpikeSourcePoisson(rate=on_bright_rates), label='source_on_pop')
    source_off_pop = pynn.Population(args.num_source, pynn.SpikeSourcePoisson(rate=off_bright_rates), label='source_off_pop')
    target_pop = pynn.Population(args.num_target, pynn.IF_cond_exp, {'i_offset':0.11, 'tau_refrac':3.0, 'v_thresh':-51.0}, label='target_pop')
    # stdp
    weight_distn = pynn.random.RandomDistribution('uniform',args.lat_conn_strength_params)
    stdp = pynn.STDPMechanism(weight=weight_distn,
        timing_dependence=pynn.SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.012),
        weight_dependence=pynn.AdditiveWeightDependence(w_min=0, w_max=0.1))
    synapse_to_use = stdp if args.stdp else pynn.StaticSynapse(weight=0.02)
    # feed forward connections
    ff_on_conn = pynn.Projection(source_on_pop, target_pop, connector=pynn.AllToAllConnector(), synapse_type=synapse_to_use, receptor_type='excitatory')
    ff_off_conn = pynn.Projection(source_off_pop, target_pop, connector=pynn.AllToAllConnector(), synapse_type=synapse_to_use, receptor_type='excitatory')
    # lateral connections (STOP SELF CONNECTIONS)
    lat_conn = pynn.Projection(target_pop, target_pop, connector=pynn.AllToAllConnector(), synapse_type=synapse_to_use, receptor_type='inhibitory')
    # instruct the network what to record
    target_pop.record(['spikes'])
    if args.record_source_spikes:
        source_on_pop.record('spikes')
        source_off_pop.record('spikes')
    # record the weights
    ff_on_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=ff_on_conn)
    ff_off_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=ff_off_conn)
    lat_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=lat_conn)
    # run the sims and collect the results (spikes, final weights)
    # initialise arrays to collect results from each iteration
    target_spikes_by_iter = [] # can't initialise this as spike time arrays are of different lengths
    if args.record_source_spikes:
        source_on_spikes_by_iter, source_off_spikes_by_iter = [], []
    ff_on_weights_by_iter = np.zeros([args.num_iterations, args.num_source, args.num_target], dtype=float)
    ff_off_weights_by_iter = np.zeros([args.num_iterations, args.num_source, args.num_target], dtype=float)
    lat_weights_by_iter = np.zeros([args.num_iterations, args.num_target, args.num_target], dtype=float)

    for i in range(args.num_iterations):
        pynn.run(args.duration, callbacks=[ff_on_weight_recorder, ff_off_weight_recorder, lat_weight_recorder])
        pynn.end()

        target_spikes = target_pop.get_data('spikes').segments[0].spiketrains
        if args.record_source_spikes:
            source_on_spikes = source_on_pop.get_data('spikes').segments[0].spiketrains
            source_off_spikes = source_off_pop.get_data('spikes').segments[0].spiketrains
        ff_on_weights = ff_on_conn.get('weight', format='array')
        ff_off_weights = ff_off_conn.get('weight', format='array')
        lat_weights = lat_conn.get('weight', format='array')

        target_spikes_by_iter.append(target_spikes)
        if args.record_source_spikes:
            source_on_spikes_by_iter.append(source_on_spikes)
            source_off_spikes_by_iter.append(source_off_spikes)
        ff_on_weights_by_iter[i,:,:] = ff_on_weights
        ff_off_weights_by_iter[i,:,:] = ff_off_weights
        lat_weights_by_iter[i,:,:] = lat_weights

    ff_on_weights_over_time = ff_on_weight_recorder.get_weights()
    ff_off_weights_over_time = ff_off_weight_recorder.get_weights()
    lat_weights_over_time = lat_weight_recorder.get_weights()
    # save the results somewhere (need to learn how to save a projection object.)

    # TODO need to save some info about the experiment itself. duration, num cells,
    results_file_name = os.path.join(h5_dir, 'training_results.h5')
    results_file = h5py.File(results_file_name, 'rw')
    results_file.create_dataset('duration', data=args.duration)
    results_file.create_dataset('num_source', data=args.num_source)
    results_file.create_dataset('num_target', data.args.num_target)
    stim_group = results_file.create_group(args.stim_type)
    stim_group.create_dataset('target_spikes', data=target_spikes)
    stim_group.create_dataset('ff_on_weights', data=ff_on_weights)
    stim_group.create_dataset('ff_off_weights', data=ff_off_weights)
    stim_group.create_dataset('lat_weights', data=lat_weights)
    stim_group.create_dataset('ff_on_weights_over_time', data=ff_on_weights_over_time)
    stim_group.create_dataset('ff_off_weights_over_time', data=ff_off_weights_over_time)
    stim_group.create_dataset('lat_weights_over_time', data=lat_weights_over_time)
