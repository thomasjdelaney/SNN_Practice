import pyNN.nest as pynn

class PoissonWeightVariation(object):
    """
    An class of objects for callback which changes the firing rate of a given population of poisson
    processes at a fixed interval.
    """
    def __init__(self, population, rate_generator, interval=20.0):
        assert isinstance(population.celltype, pynn.SpikeSourcePoisson)
        self.population = population
        self.interval = interval
        self.rate_generator = rate_generator

    def __call__(self, t):
        try:
          self.population.set(rate=next(self.rate_generator))
        except StopIteration:
            pass
        return t + self.interval
