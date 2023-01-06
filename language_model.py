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
from urllib import request

KNOWN_CHARACTERS = tuple(list(string.ascii_lowercase) + [",", ".", ":", "\n", "#", "(", ")", "!", "?", "'", '"'])


class CorpusReader:
    def __init__(self, url: str, alphabet: list[str] = KNOWN_CHARACTERS):
        self._alphabet = alphabet
        unfiltered_content = CorpusReader._get_content(url)
        self._corpus = self._filter(unfiltered_content)

    @staticmethod
    def _get_content(url: str) -> str:
        with request.urlopen(url) as response:
            return response.read()

    def _filter(self, text: str, to_lower: bool = True) -> str:
        if to_lower:
            text = text.lower()
        return "".join([c for c in [chr(n) for n in [*text]] if c in self._alphabet])

    def get_corpus(self) -> str:
        return self._corpus


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
