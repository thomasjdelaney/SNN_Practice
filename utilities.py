import numpy as np
import datetime as dt

def getRatesForInhibExcit(inhib_rates_lower, excit_rates_lower, num_inhib, num_excit):
    """
    Sample the firing rates for the inhibitory and excitatory cells.
    Arguments:  inhib_rates_lower, lower boundary for inhib_rates
                excit_rates_lower,
                num_inhib,
                num_excit
    Returns:    inhib_rates, numpy array float (num_inhib)
                excit_rates
    """
    inhib_rates = np.random.uniform(low=inhib_rates_lower, high=inhib_rates_lower+2, size=num_inhib)
    excit_rates = np.random.uniform(low=excit_rates_lower, high=excit_rates_lower+2, size=num_excit)
    return inhib_rates, excit_rates

def getOnOffSourceRates(num_source, stim_type, on_bright_params=[20.0,1.0], on_dark_params=[10.0, 0.5], off_bright_params=[10.0, 0.5], off_dark_params=[20,1.0]):
    """
    Get rates for the on and off source populations. Need rates for both bright and dark stimuli. Param args are for gamma distributions.
    Arguments:  num_source, number of cells in the source populations.
                stim_type, 'bright' or 'dark'
                on_bright_params, parameters for a gamma distribution
    Returns:    on and off rates, np arrays float
    """
    if stim_type == 'bright':
        on_rates = np.random.gamma(on_bright_params[0], on_bright_params[1], size=num_source)
        off_rates = np.random.gamma(off_bright_params[0], off_bright_params[1], size=num_source)
    elif stim_type == 'dark':
        on_rates = np.random.gamma(on_dark_params[0], on_dark_params[1], size=num_source)
        off_rates = np.random.gamma(off_dark_params[0], off_dark_params[1], size=num_source)
    else:
        print(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unrecognised stim_type. Exiting')
        error()
    return on_rates, off_rates
