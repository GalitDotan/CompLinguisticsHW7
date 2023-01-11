"""
This module contains one class: Permutation, that represents a letter permutation
(a bijective function from Σ to itself)

Classes:

1. Permutation:
    1) Constructor:
        The constructor receives some mapping between the characters
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
import random
from random import choice  # TODO: use in get_neighbor to get the thing
from language_model import KNOWN_CHARACTERS, LanguageModel


class Permutation:
    _DEFAULT_PERM = {i: i for i in KNOWN_CHARACTERS}

    def __init__(self, perm: dict[str, str] = None, enc_message: str = "", swapped: tuple[str, str] = ()):
        self._perm: dict[str, str] = perm if perm is not None else Permutation._DEFAULT_PERM
        self._enc_message = enc_message
        self._dec_message = self.translate(enc_message)
        self._swapped = swapped

    def __repr__(self):
        return str(self._perm)

    def get_neighbor(self) -> 'Permutation':
        keys = list(self._perm.keys())
        key1 = random.choice(keys)
        key2 = random.choice(keys)
        return Permutation(self._swap(key1, key2), swapped=(key1, key2))

    def _swap(self, key1: str, key2: str) -> dict:
        new_perm = self._perm.copy()
        new_perm.update({
            key1: new_perm[key2],
            key2: new_perm[key1]
        })
        return new_perm

    def translate(self, string: str) -> str:
        """
        Translate a string by the current permutation

        :param string: the string to translate
        :return: the translated string
        """
        return "".join([self._perm.get(c, c) for c in string])

    def get_energy(self, enc_message: str, lang_module: LanguageModel) -> float:
        """
        Get the result of translating the encrypted message according
        to the permutation key, and evaluating the probability of this translated sequence
        of characters according to the language model.
        :param enc_message: the message to encrypt
        :param lang_module: the language model
        :return: for a decrypted message (w1,w2,...,wn) it would return (-log_2(P(w1) - ... -log_2(P(w_n|w_1))
        """
        dec_message = list(self.translate(enc_message))
        energy = -lang_module.get_mle_unigram(dec_message[0])
        for i in range(1, len(dec_message)):
            words = (dec_message[i], dec_message[i - 1])
            energy -= lang_module.get_mle_bigram(words)
        return energy
