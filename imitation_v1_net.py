"""
Inspired by Hopkins et al (2018): http://dx.doi.org/10.1098/rsfs.2018.0007

A model for simulating a layer within V1. Feed-forward excitatory connections from source to target.
Target has lateral inhibitory connections.
Aim to 'train' the network to distinguish between light and dark.
"""
import argparse
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
parser.add_argument('-t', '--duration', help='duration of simulation.', default=3000.0, type=float)
parser.add_argument('-i', '--num_iterations', help='the number of iterations to perform.', default=10, type=int)
parser.add_argument('-s', '--stdp', help='use STDP', default=False, action='store_true')
parser.add_argument('-e', '--record_source_spikes', help='Record the source spikes', default=False, action='store_true')
parser.add_argument('-m', '--make_plots', help='make the plots, or not.', default=False, action='store_true')
parser.add_argument('-d', '--debug', help='enter debug mode', default=False, action='store_true')
args = parser.parse_args()

pynn.setup(timestep=0.1, min_delay=2.0) # different

def getOnOffSourceRates(num_source):
    """
    Get rates for the on and off source populations. Need rates for both bright and dark stimuli.
    TODO:       allow for some variations (Gamma)
    Arguments:  num_source, number of cells in the source populations.
    """
    on_bright = 20
    on_dark = 3
    off_bright = 3
    off_dark = 20
    # source_rates = np.random.gamma(args.source_rates_params[0], args.source_rates_params[1], size=args.num_source)
    return on_bright, on_dark, off_bright, off_dark

if not args.debug:
    on_bright, on_dark, off_bright, off_dark = getOnOffSourceRates(args.num_source)
# DO BRIGHT FOR NOW
    source_on_pop = pynn.Population(args.num_source, pynn.SpikeSourcePoisson(rate=on_bright), label='source_on_pop')
    source_off_pop = pynn.Population(args.num_source, pynn.SpikeSourcePoisson(rate=off_bright), label='source_off_pop')
    # different default values here.
    target_pop = pynn.Population(args.num_target, pynn.IF_cond_exp, {'i_offset':0.11, 'tau_refrac':3.0, 'v_thresh':-51.0}, label='target_pop')
    # stdp
    stdp = pynn.STDPMechanism(weight=0.02, # this is the initial value of the weight, could mess around with these
        timing_dependence=pynn.SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.012),
        weight_dependence=pynn.AdditiveWeightDependence(w_min=0, w_max=0.04))
    synapse_to_use = stdp if args.stdp else pynn.StaticSynapse(weight=0.02)
    # feed forward connections
    ff_on_conn = pynn.Projection(source_on_pop, target_pop, connector=pynn.AllToAllConnector(), synapse_type=synapse_to_use, receptor_type='excitatory')
    ff_off_conn = pynn.Projection(source_off_pop, target_pop, connector=pynn.AllToAllConnector(), synapse_type=synapse_to_use, receptor_type='excitatory')
    # lateral connections
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
    target_spikes_by_iter = []
    if args.record_source_spikes:
        source_on_spikes_by_iter = []
        source_off_spikes_by_iter = []
    ff_on_weights_by_iter = []
    ff_off_weights_by_iter = []
    for i in range(num_iterations):
        pynn.run(args.duration, callbacks=[ff_on_weight_recorder, ff_off_weight_recorder, lat_weight_recorder])
        pynn.end()
    # save the results somewhere (need to learn how to save a projection object.)

    # what can we make of them?
