"""
A simulated annealing module,
containing an implementation of the simulated annealing algorithm for finding the optimal
hypothesis in a large conjecture space.

Classes:
SimulatedAnnealing:
    This class implements the simulated annealing algorithm.
    1) constructor:
        receives an initial temperature, a threshold and a cooling rate (in that order).
    2) run:
        receives an initial hypothesis (= initial permutation),
        an encrypted message
        and a language model,
        and returns the output of the simulated annealing loop.
        It uses the exp function from the math module, which takes a value v and returns e^v
        Also, it randomly chooses a number r between 0 and 1 and chooses whether to
        move to a neighbour accordingly:
            for a probability p,
                If r < p: switch to a neighbour hypothesis
                Else: stay with the current hypothesis
"""
from random import random
