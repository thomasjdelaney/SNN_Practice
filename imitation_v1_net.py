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
import datetime as dt
from WeightRecorder import *
from PlottingFunctions import *
from utilities import *

parser = argparse.ArgumentParser(description='Feed-forward Poisson spike input. Lateral inhibitory connections')
parser.add_argument('-f', '--file_path_name', help='name of the file to save to.', default='training_results.h5', type=str)
parser.add_argument('-b', '--num_target', help='number of target cells.', default=5, type=int)
parser.add_argument('-n', '--num_source', help='number of source/feed-forward cells.', default=10, type=int)
parser.add_argument('-p', '--source_rates_params', help='params for the gamma distribution.', default=[10,0.5], type=float, nargs=2)
# parser.add_argument('-c', '--ff_conn_prob', help='feed-forward connection prob.', default=0.75, type=float)
# parser.add_argument('-l', '--lat_conn_prob', help='lateral connection prob.', default=0.75, type=float)
parser.add_argument('-d', '--duration', help='duration of simulation.', default=3000.0, type=float)
parser.add_argument('-u', '--use_stdp', help='use STDP', default=False, action='store_true')
parser.add_argument('-r', '--record_source_spikes', help='Record the source spikes', default=False, action='store_true')
parser.add_argument('-s', '--numpy_seed', help='RNG seed', default=1798, type=int)
parser.add_argument('--lat_conn_strength_params', help='params for distribution of strengths of later connections', default=[0.015,0.025], type=float, nargs=2)
parser.add_argument('--debug', help='enter debug mode', default=False, action='store_true')
args = parser.parse_args()

np.random.seed(args.numpy_seed)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

pynn.setup(timestep=0.1, min_delay=2.0) # different

proj_dir = os.path.join(os.environ['HOME'], 'SNN_practice')
h5_dir = os.path.join(proj_dir, 'h5')
args.file_path_name = os.path.join(h5_dir, args.file_path_name)

def runSimGivenStim(stim_type, lat_conn_strength_params, num_source, num_target, duration, use_stdp, record_source_spikes):
    """
    For running the simulation and returning the required results.
    Arguments:  stim_type, 'bright' or 'dark'
                lat_conn_strength_params, continuous uniform distribution params
                num_source, number of cells in the source layers
                num_target, number of cells in the target layer
                duration, float
                use_stdp,
                record_source_spikes,
    """
    on_rates, off_rates = getOnOffSourceRates(num_source, stim_type) # this will have to take more arguments soon.
    stdp_weight_distn = pynn.random.RandomDistribution('uniform',lat_conn_strength_params)
    source_on_pop = pynn.Population(num_source, pynn.SpikeSourcePoisson(rate=on_rates), label='source_on_pop')
    source_off_pop = pynn.Population(num_source, pynn.SpikeSourcePoisson(rate=off_rates), label='source_off_pop')
    target_pop = pynn.Population(num_target, pynn.IF_cond_exp, {'i_offset':0.11, 'tau_refrac':3.0, 'v_thresh':-51.0}, label='target_pop')
    stdp = pynn.STDPMechanism(weight=stdp_weight_distn,
        timing_dependence=pynn.SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.012),
        weight_dependence=pynn.AdditiveWeightDependence(w_min=0, w_max=0.1))
    synapse_to_use = stdp if use_stdp else pynn.StaticSynapse(weight=0.02)
    ff_on_conn = pynn.Projection(source_on_pop, target_pop, connector=pynn.AllToAllConnector(), synapse_type=synapse_to_use, receptor_type='excitatory')
    ff_off_conn = pynn.Projection(source_off_pop, target_pop, connector=pynn.AllToAllConnector(), synapse_type=synapse_to_use, receptor_type='excitatory')
    lat_conn = pynn.Projection(target_pop, target_pop, connector=pynn.AllToAllConnector(allow_self_connections=False), synapse_type=synapse_to_use, receptor_type='inhibitory')
    target_pop.record(['spikes'])
    [source_on_pop.record('spikes'), source_off_pop.record('spikes')] if args.record_source_spikes else None
    ff_on_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=ff_on_conn)
    ff_off_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=ff_off_conn)
    lat_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=lat_conn)
    pynn.run(duration, callbacks=[ff_on_weight_recorder, ff_off_weight_recorder, lat_weight_recorder])
    pynn.end()
    target_spikes = target_pop.get_data('spikes').segments[0].spiketrains
    if record_source_spikes:
        source_on_spikes = source_on_pop.get_data('spikes').segments[0].spiketrains
        source_off_spikes = source_off_pop.get_data('spikes').segments[0].spiketrains
    ff_on_weights = ff_on_conn.get('weight', format='array')
    ff_off_weights = ff_off_conn.get('weight', format='array')
    lat_weights = lat_conn.get('weight', format='array')
    ff_on_weights_over_time = ff_on_weight_recorder.get_weights()
    ff_off_weights_over_time = ff_off_weight_recorder.get_weights()
    lat_weights_over_time = lat_weight_recorder.get_weights()
    pynn.reset()
    return target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time

def saveTrainingResults(stim_type, duration, num_source, num_target, target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time, file_path_name):
    """
    For saving down the results of a simulation. Used separately for on and off runs.
    Arguments:  stim_type,
                duration,
                num_source,
                num_target,
                target_spikes,
                ff_on_weights,
                ff_off_weights,
                lat_weights,
                ff_on_weights_over_time,
                ff_off_weights_over_time,
                lat_weights_over_time,
                file_name, default None, if given, assume that one run has happened and another is occurring.
    Returns:    file_path_name, str, full file name including path.
    """
    file_exists = os.path.isfile(file_path_name)
    results_file = h5py.File(file_path_name, 'a')
    if file_exists:
        has_duration = 'duration' in results_file.keys()
        has_num_source = 'num_source' in results_file.keys()
        has_num_target = 'num_target' in results_file.keys()
    else:
        has_duration, has_num_source, has_num_target = False, False, False
    results_file.create_dataset('duration', data=args.duration) if not has_duration else None
    results_file.create_dataset('num_source', data=args.num_source) if not has_num_source else None
    results_file.create_dataset('num_target', data=args.num_target) if not has_num_target else None
    stim_group = results_file.create_group(stim_type)
    stim_group.create_dataset('ff_on_weights', data=ff_on_weights)
    stim_group.create_dataset('ff_off_weights', data=ff_off_weights)
    stim_group.create_dataset('lat_weights', data=lat_weights)
    stim_group.create_dataset('ff_on_weights_over_time', data=ff_on_weights_over_time)
    stim_group.create_dataset('ff_off_weights_over_time', data=ff_off_weights_over_time)
    stim_group.create_dataset('lat_weights_over_time', data=lat_weights_over_time)
    stim_group.create_dataset('weight_times', data=ff_on_weights_over_time.times)
    target_spikes_group = stim_group.create_group('target_spikes')
    for i,st in enumerate(target_spikes):
        target_spikes_group.create_dataset(str(i), data=st.as_array())
    results_file.close()
    return file_path_name

if not args.debug:
    print(dt.datetime.now().isoformat() + ' INFO: Starting main function...')
    target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time = runSimGivenStim('bright', args.lat_conn_strength_params, args.num_source, args.num_target, args.duration, args.use_stdp, args.record_source_spikes)
    file_path_name = saveTrainingResults('bright', args.duration, args.num_source, args.num_target, target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time, args.file_path_name)
    target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time = runSimGivenStim('dark', args.lat_conn_strength_params, args.num_source, args.num_target, args.duration, args.use_stdp, args.record_source_spikes)
    file_path_name = saveTrainingResults('dark', args.duration, args.num_source, args.num_target, target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time, args.file_path_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_path_name + ' saved.')
