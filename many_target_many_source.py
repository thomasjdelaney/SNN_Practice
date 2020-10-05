import argparse
import pyNN.nest as pynn
import numpy as np
import matplotlib.pyplot as plt
from WeightRecorder import *
from PlottingFunctions import *
from utilities import *

parser = argparse.ArgumentParser(description='Many Poisson spike source inputs, excitatory and inhibitory, one target. With or without STDP.')
parser.add_argument('-b', '--num_target', help='number of target cells.', default=1, type=int)
parser.add_argument('-i', '--num_inhib', help='number of inhibitory cells.', default=1, type=int)
parser.add_argument('-r', '--inhib_rates_lower', help='lower boundary of inhibitory rates (Hz).', default=8.0, type=float)
parser.add_argument('-e', '--num_excit', help='number of excitatory cells.', default=1, type=int)
parser.add_argument('-p', '--excit_rates_lower', help='lower boundary of inhibitory rates (Hz).', default=5.0, type=float)
parser.add_argument('-t', '--duration', help='duration of simulation.', default=500.0, type=float)
parser.add_argument('-s', '--stdp', help='use STDP', default=False, action='store_true')
parser.add_argument('-a', '--record_target_spikes', help='Record the target spikes.', default=False, action='store_true')
parser.add_argument('-m', '--make_plots', help='make the plots, or not.', default=False, action='store_true')
parser.add_argument('-d', '--debug', help='enter debug mode', default=False, action='store_true')
args = parser.parse_args()

pynn.setup(timestep=0.1, min_delay=2.0)

if not args.debug:
    # define target cell
    target_pop = pynn.Population(args.num_target, pynn.IF_cond_exp, {'i_offset':0.11, 'tau_refrac':3.0, 'v_thresh':-51.0})

    # define inhibitory and excitatory populations
    inhib_rates, excit_rates = getRatesForInhibExcit(args.inhib_rates_lower, args.excit_rates_lower, args.num_inhib, args.num_excit)
    inhib_source = pynn.Population(args.num_inhib, pynn.SpikeSourcePoisson(rate=inhib_rates), label="inhib_input")
    excit_source = pynn.Population(args.num_excit, pynn.SpikeSourcePoisson(rate=excit_rates), label="excit_input")

    # define stdp rules, parameters could be messed around with here.
    stdp = pynn.STDPMechanism(weight=0.02, # this is the initial value of the weight
            timing_dependence=pynn.SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.012),
            weight_dependence=pynn.AdditiveWeightDependence(w_min=0, w_max=0.04))
    synapse_to_use = stdp if args.stdp else pynn.StaticSynapse(weight=0.02)

    # connect inhibitory and excitatory sources to target. Could connect inhib to excit?
    inhib_conn = pynn.Projection(inhib_source, target_pop, connector=pynn.AllToAllConnector(), synapse_type=synapse_to_use, receptor_type='inhibitory')
    excit_conn = pynn.Projection(excit_source, target_pop, connector=pynn.AllToAllConnector(), synapse_type=synapse_to_use, receptor_type='excitatory')

    # tell the sim what to record
    target_pop.record(['spikes', 'v', 'gsyn_exc', 'gsyn_inh']) if args.record_target_spikes else target_pop.record(['v', 'gsyn_exc', 'gsyn_inh'])
    inhib_source.record('spikes')
    excit_source.record('spikes')

    # record the weights
    inhib_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=inhib_conn)
    excit_weight_recorder = WeightRecorder(sampling_interval=1.0, projection=excit_conn)

    # run the simulation
    pynn.run(args.duration, callbacks=[inhib_weight_recorder, excit_weight_recorder])
    pynn.end()

    # extract the data
    target_pop_v = target_pop.get_data('v').segments[0].analogsignals[0]
    target_spikes = target_pop.get_data('spikes').segments[0].spiketrains if args.record_target_spikes else None
    target_pop_gsyn_exc = target_pop.get_data('gsyn_exc').segments[0].analogsignals[0]
    target_pop_gsyn_inh = target_pop.get_data('gsyn_inh').segments[0].analogsignals[0]
    inhib_source_spiketrains = inhib_source.get_data('spikes').segments[0].spiketrains
    excit_source_spiketrains = excit_source.get_data('spikes').segments[0].spiketrains

    # do some plotting
    if args.make_plots:
        plotTargetVWithInhibExcitSpikes(target_pop_v, inhib_source_spiketrains, excit_source_spiketrains)
        plotInhExcSynapticStrengths(target_pop_gsyn_exc, target_pop_gsyn_inh, args.duration)
        plotWeightsOverTime(inhib_weight_recorder.get_weights(), 'Inhibitory weights')
        plotWeightsOverTime(excit_weight_recorder.get_weights(), 'Excitatory weights')
        if args.record_target_spikes:
            plotTargetInhibExcitSpikes(target_spikes, inhib_source_spiketrains, excit_source_spiketrains, args.duration)
        plt.show(block=False)

# TODO: function and options for plotting synapse strengths
