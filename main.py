"""
A main module, connecting all the logical parts:

The main function does the following logic:
    1. Reads the online corpus `http://www.gutenberg.org/files/76/76-0.txt` (from project Gutenberg).
    2. Creates a bigram language model.
    3. Reads the encrypted message from `problemset_07_encrypted_input.txt`.
    4. Creates an initial permutation hypothesis: mapping each letter in Σ to itself.
    5. Runs the simulated annealing algorithm with the relevant parameters.
        initial temperature = 10
        cooling schedule = 0.95
        threshold = 10^-1
    6. Prints out:
        • The winning permutation.
        • The initial temperature, threshold and cooling rate used.
        • The content of the deciphered message.
"""
import language_model
import permutation
import simulated_annealing

URL = 'http://www.gutenberg.org/files/76/76-0.txt'


def main():
    raise NotImplementedError()

    # 1. read the corpus

    # 2. create a language model

    # 3. read the encrypted message

    # 4. create initial hypothesis

    # 5. run simulated annealing

    # 6. print results


if __name__ == '__main__':
    main()
