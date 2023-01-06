"""
This module contains one class: Permutation, that represents a letter permutation
(a bijective function from Σ to itself)

Classes:

1. Permutation:
    1) Constructor:
        The constructor receives some mapping between the characters # TODO: how to map?
    2)  get_neighbor:
         returns a random neighbor of the current permutation instance.
         A neighbor of a permutation is defined as a new permutation that
         replaces the mapping of two randomly chosen characters from Σ.
    3)  translate:
        receives an input string, and returns the translation of that string according to the
        current permutation instance (i.e. takes each character of the string and replaces it
        with its corresponding character according to the permutation).
    4) get energy:
        received an encrypted message and a language model,
        and returns the "energy" of the current permutation.
        Note: energy = the result of translating the encrypted message according
        to the permutation key, and evaluating the probability of this translated sequence
        of characters according to the language model.
"""
from random import choice  # TODO: use in get_neighbor to get the thing
from language_model import KNOWN_CHARACTERS, LanguageModel


class Permutation:
    def __init__(self, perm: dict[str, str]):
        self.perm: dict[str, str] = perm

    def get_neighbor(self) -> 'Permutation':
        raise NotImplementedError()

    def translate(self, string: str) -> str:
        raise NotImplementedError()

    def get_energy(self, enc_message: str, lang_module: LanguageModel):
        raise NotImplementedError()
