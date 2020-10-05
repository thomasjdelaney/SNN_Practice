import numpy as np

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
