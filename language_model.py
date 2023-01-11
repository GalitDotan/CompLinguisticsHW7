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
        • gathers unigram and bigram raw count
        • obtains MLE unigram and bigram probabilities and applies Laplace smoothing.
        • The probabilities are saved as instance variables
"""
import math
import string
from time import sleep
from typing import Any, Callable
from urllib import request

KNOWN_CHARACTERS = tuple(list(string.ascii_lowercase) + [",", ".", ":", "\n", "#", "(", ")", "!", "?", "'", '"'])
KNOWN_CHARACTERS_WITH_SPACE = KNOWN_CHARACTERS + (" ",)


class CorpusReader:
    def __init__(self, url: str, alphabet: list[str] = KNOWN_CHARACTERS_WITH_SPACE):
        self._extended_alphabet = alphabet
        try:
            unfiltered_content = CorpusReader._get_raw_corpus(url)
        except Exception:
            unfiltered_content = CorpusReader._get_raw_corpus_offline()
        self._corpus_str = self._filter_corpus(unfiltered_content)
        self._corpus_words = self._corpus_str.split()

    @staticmethod
    def _get_raw_corpus(url: str) -> str:
        """
        Get a corpus from a given URL

        :param url: the link to the corpus
        :return: the text
        """
        with request.urlopen(url) as response:
            return response.read()

    @staticmethod
    def _get_raw_corpus_offline():
        with open('corpus.txt', 'br') as f:
            return f.read()

    def _filter_corpus(self, corpus: str, to_lower: bool = True) -> str:
        """
        Filter a given corpus to include only known characters and change all characters to lowercase (if needed)

        :param corpus: some text
        :param to_lower: whether to replace all characters to lowercase
        :return: the corpus filtered (will contain only the known characters in the model)
        """
        unfiltered_chars = [chr(x).lower() if to_lower else chr(x) for x in list(corpus)]
        return ''.join([c for c in unfiltered_chars if c in self._extended_alphabet])

    def get_corpus(self) -> list[str]:
        return self._corpus_words


class LanguageModel:
    def __init__(self, corpus_reader: CorpusReader, known_characters: tuple[str] = KNOWN_CHARACTERS):
        self._corpus_reader: CorpusReader = corpus_reader
        self._known_chars: tuple[str] = known_characters

        self._corpus: list[str] = self._corpus_reader.get_corpus()
        self._corpus_size: int = len(self._corpus)

        self._unigram_cnt: dict[str, int] = self._gather_unigram_raw_count()
        self._bigram_cnt: dict[tuple[str, str], int] = self._gather_bigram_raw_count()

        self._vocabulary_size: int = len(self._unigram_cnt.keys())

        self._mle_unigram = self._calc_mle(self._unigram_cnt, self._log_prob_of_w)
        self._mle_bigram = self._calc_mle(self._bigram_cnt, self._log_prob_of_w2_given_w1)

    def _log_prob_of_w(self, w: str):
        """
        Get probability P(w) with Laplace smoothing.

        :param w: a word.
        :return: P(w) with Laplace smoothing.
        """
        c_w = self._unigram_cnt.get(w, 0)
        n = self._corpus_size
        v = self._vocabulary_size
        return math.log2((c_w + 1) / (n + v))

    def _log_prob_of_w2_given_w1(self, words: tuple[str, str]):
        """
        Get probability P(w2 | w1) with Laplace smoothing.

        :param w2: a word.
        :param w1: a word.
        :return: P(w2 | w1) with Laplace smoothing.
        """
        w2, w1 = words
        c_w2_then_w1 = self._bigram_cnt.get((w2, w1), 0)
        c_w1 = self._unigram_cnt.get(w1, 0)
        v = self._vocabulary_size
        return math.log2((c_w2_then_w1 + 1) / (c_w1 + v))

    def get_mle_unigram(self, w: str) -> float:
        return self._mle_unigram.get(w, self._log_prob_of_w(w))

    def get_mle_bigram(self, words: tuple[str, str]) -> float:
        w2, w1 = words
        return self._mle_bigram.get(words, self._log_prob_of_w2_given_w1((w2, w1)))

    def _gather_unigram_raw_count(self) -> dict[str, int]:
        unigram_cnt = {}
        for word in self._corpus:
            curr_cnt = unigram_cnt.setdefault(word, 0)
            unigram_cnt[word] = curr_cnt + 1
        return unigram_cnt

    def _gather_bigram_raw_count(self) -> dict[tuple[str, str], int]:
        bigram_cnt = {}
        for i in range(1, len(self._corpus)):
            bigram = (self._corpus[i], self._corpus[i - 1])
            curr_cnt = bigram_cnt.setdefault(bigram, 0)
            bigram_cnt[bigram] = curr_cnt + 1
        return bigram_cnt

    def _calc_mle(self, counts: dict[Any, int], prob_func: Callable) -> dict[Any, float]:
        return {key: prob_func(key) for key in counts}
