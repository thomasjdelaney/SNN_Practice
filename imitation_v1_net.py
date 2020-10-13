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
parser.add_argument('-p', '--source_rates_params', help='params for the gamma distribution.', default=[20.0,1.0, 10.0,0.5, 10.0,0.5, 20.0,1.0], type=float, nargs=8)
parser.add_argument('-c', '--conn_type', help='connection type, [all_to_all, fixed_prob]', default='all_to_all', type=str, choices=['fixed_prob', 'all_to_all'])
parser.add_argument('-d', '--duration', help='duration of simulation.', default=3000.0, type=float)
parser.add_argument('-u', '--use_stdp', help='use STDP', default=False, action='store_true')
parser.add_argument('-r', '--record_source_spikes', help='Record the source spikes', default=False, action='store_true')
parser.add_argument('-s', '--numpy_seed', help='RNG seed', default=1798, type=int)
parser.add_argument('-w', '--w_max', help='The maximum additive weight dependence for an STDP synapse', default=1.0, type=float)
parser.add_argument('--lat_conn_strength_params', help='params for distribution of strengths of later connections', default=[0.015,0.025], type=float, nargs=2)
parser.add_argument('--feed_forward_connection_prob', help='The probability of connection for feed forward connections.', default=None, type=float)
parser.add_argument('--lateral_connection_prob', help='The probability of connection for lateral connections.', default=None, type=float)
parser.add_argument('--debug', help='enter debug mode', default=False, action='store_true')
args = parser.parse_args()

np.random.seed(args.numpy_seed)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

pynn.setup(timestep=0.1, min_delay=2.0) # different

proj_dir = os.path.join(os.environ['HOME'], 'SNN_practice')
h5_dir = os.path.join(proj_dir, 'h5')
args.file_path_name = os.path.join(h5_dir, args.file_path_name)
args.source_rates_params = np.array(args.source_rates_params).reshape(4,2)

def getConnectorType(conn_type, ff_prob=None, lat_prob=None):
    """
    For getting the feed-forward and lateral connection types.
    Arguments:  conn_type, str, choices = ['all_to_all', 'fixed_prob']
                ff_prob, float, probability of connection for feed-forward
                lat_prob, float, probability of connection for lateral
    Returns:    ff_conn, lat_conn
    """
    if conn_type == 'all_to_all':
        ff_conn = pynn.AllToAllConnector(rng=pynn.random.NumpyRNG(seed=1798))
        lat_conn = pynn.AllToAllConnector(allow_self_connections=False, rng=pynn.random.NumpyRNG(seed=1916))
    elif conn_type == 'fixed_prob':
        if (ff_prob == None) or (lat_prob == None):
            print(dt.datetime.now().isoformat() + ' ERROR: ' + 'One of the connections probabilities is "None".')
            sys.exit(2)
        else:
            ff_conn = pynn.FixedProbabilityConnector(ff_prob, rng=pynn.random.NumpyRNG(seed=1798))
            lat_conn = pynn.FixedProbabilityConnector(lat_prob, allow_self_connections=False, rng=pynn.random.NumpyRNG(seed=1916))
    else:
        print(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unrecognised connection type.')
        sys.exit(2)
    return ff_conn, lat_conn

def getSynapseType(lat_conn_strength_params, use_stdp, tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.012, w_min=0, w_max=1.0, static_weight=0.02):
    """
    For getting the required synapse type, fixed or STDP
    Arguments:  lat_conn_strength_params, parameters for the uniform distribution. The connections evolve during training anyway.
                use_stdp, flag
                the other Arguments are defaults.
    Return:     the synapse_type to use
    """
    stdp_weight_distn = pynn.random.RandomDistribution('uniform',lat_conn_strength_params)
    stdp = pynn.STDPMechanism(weight=stdp_weight_distn,
        timing_dependence=pynn.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus, A_plus=A_plus, A_minus=A_minus),
        weight_dependence=pynn.AdditiveWeightDependence(w_min=w_min, w_max=w_max))
    synapse_to_use = stdp if use_stdp else pynn.StaticSynapse(weight=static_weight)
    return synapse_to_use

def runSimGivenStim(stim_type, num_source, num_target, duration, use_stdp, record_source_spikes, source_rates_params, synapse_to_use, ff_conn, lat_conn):
    """
    For running the simulation and returning the required results.
    Arguments:  stim_type, 'bright' or 'dark'
                num_source, number of cells in the source layers
                num_target, number of cells in the target layer
                duration, float
                use_stdp,
                record_source_spikes,
                source_rates_params, params for 8 Gamma distributions
                synapse_to_use, either STDPMechanism or StaticSynapse
                ff_conn, either AllToAllConnector or FixedProbabilityConnector
                lat_conn, same as ff_conn but with a different probability, maybe
    Returns:    target_spikes,
                ff_on_weights,
                ff_off_weights,
                lat_weights,
                ff_on_weights_over_time,
                ff_off_weights_over_time,
                lat_weights_over_time
    """
    on_rates, off_rates = getOnOffSourceRates(num_source, stim_type, on_bright_params=args.source_rates_params[0], on_dark_params=args.source_rates_params[1], off_bright_params=args.source_rates_params[2], off_dark_params=args.source_rates_params[3])
    source_on_pop = pynn.Population(num_source, pynn.SpikeSourcePoisson(rate=on_rates), label='source_on_pop')
    source_off_pop = pynn.Population(num_source, pynn.SpikeSourcePoisson(rate=off_rates), label='source_off_pop')
    target_pop = pynn.Population(num_target, pynn.IF_cond_exp, {'i_offset':0.11, 'tau_refrac':3.0, 'v_thresh':-51.0}, label='target_pop')
    ff_on_proj = pynn.Projection(source_on_pop, target_pop, connector=ff_conn, synapse_type=synapse_to_use, receptor_type='excitatory')
    ff_off_proj = pynn.Projection(source_off_pop, target_pop, connector=ff_conn, synapse_type=synapse_to_use, receptor_type='excitatory')
    lat_proj = pynn.Projection(target_pop, target_pop, connector=lat_conn, synapse_type=synapse_to_use, receptor_type='inhibitory')
    target_pop.record(['spikes'])
    [source_on_pop.record('spikes'), source_off_pop.record('spikes')] if args.record_source_spikes else None
    ff_on_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=ff_on_proj)
    ff_off_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=ff_off_proj)
    lat_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=lat_proj)
    pynn.run(duration, callbacks=[ff_on_weight_recorder, ff_off_weight_recorder, lat_weight_recorder])
    pynn.end()
    target_spikes = target_pop.get_data('spikes').segments[0].spiketrains
    if record_source_spikes:
        source_on_spikes = source_on_pop.get_data('spikes').segments[0].spiketrains
        source_off_spikes = source_off_pop.get_data('spikes').segments[0].spiketrains
    ff_on_weights = ff_on_proj.get('weight', format='array')
    ff_off_weights = ff_off_proj.get('weight', format='array')
    lat_weights = lat_proj.get('weight', format='array')
    ff_on_weights_over_time = ff_on_weight_recorder.get_weights()
    ff_off_weights_over_time = ff_off_weight_recorder.get_weights()
    lat_weights_over_time = lat_weight_recorder.get_weights()
    pynn.reset()
    return target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time

def saveTrainingResults(stim_type, duration, num_source, num_target, source_rates_params, target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time, file_path_name):
    """
    For saving down the results of a simulation. Used separately for on and off runs.
    Arguments:  stim_type,
                duration,
                num_source,
                num_target,
                source_rates_params,
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
        has_source_params = 'source_rates_params' in results_file.keys()
    else:
        has_duration, has_num_source, has_num_target, has_source_params = False, False, False, False
    results_file.create_dataset('duration', data=duration) if not has_duration else None
    results_file.create_dataset('num_source', data=num_source) if not has_num_source else None
    results_file.create_dataset('num_target', data=num_target) if not has_num_target else None
    results_file.create_dataset('source_rates_params', data=source_rates_params) if not has_source_params else None
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
    ff_conn, lat_conn = getConnectorType(args.conn_type, ff_prob=args.feed_forward_connection_prob, lat_prob=args.lateral_connection_prob)
    synapse_to_use = getSynapseType(args.lat_conn_strength_params, args.use_stdp, w_max=args.w_max)
    target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time = runSimGivenStim('bright', args.num_source, args.num_target, args.duration, args.use_stdp, args.record_source_spikes, args.source_rates_params, synapse_to_use, ff_conn, lat_conn)
    file_path_name = saveTrainingResults('bright', args.duration, args.num_source, args.num_target, args.source_rates_params, target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time, args.file_path_name)
    target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time = runSimGivenStim('dark', args.num_source, args.num_target, args.duration, args.use_stdp, args.record_source_spikes, args.source_rates_params, synapse_to_use, ff_conn, lat_conn)
    file_path_name = saveTrainingResults('dark', args.duration, args.num_source, args.num_target, args.source_rates_params, target_spikes, ff_on_weights, ff_off_weights, lat_weights, ff_on_weights_over_time, ff_off_weights_over_time, lat_weights_over_time, args.file_path_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_path_name + ' saved.')
