"""
A language model module, containing the functionality needed in order to read a corpus from
an online resource and build an n-gram language model (in particular, a bigram language model)
using that corpus.

This module contains two classes:
1. CorpusReader
    This class contains all relevant data structures and methods needed to read an online corpus
    from a given URL, and prepare it for the statistical counts.
    It has a constructor which receives a URL indicating the location of the online corpus.
    The constructor filters only specific characters. It also changes every letter to lowercase.
2. LanguageModel
    This class contains all relevant data structures and methods needed to perform statistical
    counts (unigrams, bigrams, MLE etc), given a filtered corpus.
    It has a constructor which receives a filtered corpus (provided as an initialized
    instance of the CorpusReader class), and:
    * gathers unigram and bigram raw count
    * obtains MLE unigram and bigram probabilities and applies Laplace smoothing.
    * The probabilities are saved as instance variables
"""
import string
import urllib  # TODO: use this in the constructor of CorpusReader

KNOWN_CHARACTERS = tuple(string.ascii_lowercase.split("") + [",", ".", ":", "\n", "#", "(", ")", "!", "?", "'", '"'])


class CorpusReader:
    def __init__(self, url: str):
        unfiltered_content = self._get_content(url)
        self._message = self._filter(unfiltered_content)

    def _get_content(self, url: str) -> str:
        raise NotImplementedError()

    def _filter(self, message: str, alphabet: list[str] = KNOWN_CHARACTERS, to_lower: bool = True) -> str:
        raise NotImplementedError()


class LanguageModel:
    def __init__(self, corpus_reader: CorpusReader):
        _corpus_reader = corpus_reader

    def _gather_unigram_raw_count(self):
        raise NotImplementedError()

    def _gather_bigram_raw_count(self):
        raise NotImplementedError()

    def _obtain_probabilities(self):
        raise NotImplementedError()

    def _apply_laplace(self):
        raise NotImplementedError()
