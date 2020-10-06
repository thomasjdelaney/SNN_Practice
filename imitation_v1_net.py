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
parser.add_argument('-c', '--ff_conn_prob', help='feed-forward connection prob.', default=0.75, type=float)
parser.add_argument('-l', '--lat_conn_prob', help='lateral connection prob.', default=0.75, type=float)
parser.add_argument('-t', '--duration', help='duration of simulation.', default=500.0, type=float)
parser.add_argument('-s', '--stdp', help='use STDP', default=False, action='store_true')
parser.add_argument('-a', '--record_target_spikes', help='Record the target spikes.', default=False, action='store_true')
parser.add_argument('-m', '--make_plots', help='make the plots, or not.', default=False, action='store_true')
parser.add_argument('-d', '--debug', help='enter debug mode', default=False, action='store_true')
args = parser.parse_args()

pynn.setup(timestep=0.1, min_delay=2.0) # different

if not args.debug:
    source_rates = np.random.gamma(args.source_rates_params[0], args.source_rates_params[1], size=args.num_source)
    source_pop = pynn.Population(args.num_source, pynn.SpikeSourcePoisson(rates=source_rates), label='source_pop')
    # different default values here.
    target_pop = pynn.Population(args.num_target, pynn.IF_cond_exp, {'i_offset':0.11, 'tau_refrac':3.0, 'v_thresh':-51.0}, label='target_pop')
    # stdp
    stdp = pynn.STDPMechanism(weight=0.02, # this is the initial value of the weight, could mess around with these
        timing_dependence=pynn.SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.012),
        weight_dependence=pynn.AdditiveWeightDependence(w_min=0, w_max=0.04))
    synapse_to_use = stdp if args.stdp else pynn.StaticSynapse(weight=0.02)
    # feed forward connections
    ff_conn = pynn.Projection(source_pop, target_pop, connector=pynn.FixedProbabilityConnector(args.ff_conn_prob), synapse_type=synapse_to_use, receptor_type='excitatory')
    # lateral connections
    lat_conn = pynn.Projection(target_pop, target_pop, connector=pynn.FixedProbabilityConnector(args.lat_conn_prob), synapse_type=synapse_to_use, receptor_type='inhibitory')
