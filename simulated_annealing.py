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
        receives an initial hypothesis (= some permutation),
        an encrypted message
        and a language model,
        and returns the output of the simulated annealing loop (the best permutation found).
        It uses the exp function from the math module, which takes a value v and returns e^v
        Also, it randomly chooses a number r between 0 and 1 and chooses whether to
        move to a neighbour accordingly:
            for a probability p,
                • If r < p: switch to a neighbour hypothesis
                • Else: stay with the current hypothesis
"""
import math
from random import random

from language_model import LanguageModel
from permutation import Permutation


class SimulatedAnnealing:
    def __init__(self, init_temp: int, threshold: float, cool_rate: float):
        self._init_temp: int = init_temp
        self._threshold: float = threshold
        self._cool_rate: float = cool_rate

    def run(self, initial_perm: Permutation, enc_msg: str, lang_model: LanguageModel) -> Permutation:
        h = initial_perm
        t = self._init_temp
        while t > self._threshold:
            new_h = h.get_neighbor()
            delta = SimulatedAnnealing._get_delta(h, new_h, enc_msg, lang_model)
            p = 1 if delta < 0 else math.exp(-(delta / t))
            h = SimulatedAnnealing._choose(h, new_h, p)
            t = t * self._cool_rate
        return h

    @staticmethod
    def _get_delta(old_perm: Permutation, new_perm: Permutation, msg: str, lang_model: LanguageModel):
        return new_perm.get_energy(msg, lang_model) - old_perm.get_energy(msg, lang_model)

    @staticmethod
    def _choose(old_perm: Permutation, new_perm: Permutation, p: float) -> Permutation:
        r = random()
        return new_perm if r < p else old_perm
