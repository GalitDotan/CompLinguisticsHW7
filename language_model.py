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
from typing import Any, Callable
from urllib import request

KNOWN_CHARACTERS = tuple(list(string.ascii_lowercase) + [",", ".", ":", "\n", "#", "(", ")", "!", "?", "'", '"'])


class CorpusReader:
    def __init__(self, url: str, alphabet: list[str] = KNOWN_CHARACTERS):
        self._alphabet = alphabet
        unfiltered_content = CorpusReader._get_content(url)
        self._corpus_str = self._filter_text(unfiltered_content)
        self._corpus_words = " ".split(self._corpus_str)

    @staticmethod
    def _get_content(url: str) -> str:
        with request.urlopen(url) as response:
            return response.read()

    def _filter_text(self, text: str, to_lower: bool = True) -> str:
        if to_lower:
            text = text.lower()
        return "".join([c for c in [chr(n) for n in [*text]] if c in self._alphabet])

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

    def _log_prob_of_w2_given_w1(self, w2: str, w1: str):
        """
        Get probability P(w2 | w1) with Laplace smoothing.

        :param w2: a word.
        :param w1: a word.
        :return: P(w2 | w1) with Laplace smoothing.
        """
        c_w2_then_w1 = self._bigram_cnt.get((w2, w1), 0)
        c_w1 = self._unigram_cnt.get(w1, 0)
        v = self._vocabulary_size
        return math.log2((c_w2_then_w1 + 1) / (c_w1 + v))

    def get_mle_unigram(self, w: str) -> float:
        return self._mle_unigram.get(w, self._log_prob_of_w(w))

    def get_mle_bigram(self, w2: str, w1: str) -> float:
        return self._mle_bigram.get((w2, w1), self._log_prob_of_w2_given_w1(w2, w1))

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
        return {word: prob_func(word) for word in counts}
