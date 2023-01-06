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
from language_model import CorpusReader, LanguageModel
from permutation import Permutation
from simulated_annealing import SimulatedAnnealing

URL = 'http://www.gutenberg.org/files/76/76-0.txt'
INITIAL_TEMPERATURE = 10
COOLING_RATE = 0.95
THREASHOLD = pow(10, -1)
ENC_MESSAGE_FILE = 'problemset_07_encrypted_input.txt'


def _read_msg_from_file(filename: str = ENC_MESSAGE_FILE):
    with open(filename) as f:
        return f.read()


def _format_results(perm: Permutation, init_temp: int, threashold: float, cooling_rate: float, enc_msg: str,
                    dec_msg: str):
    return f"""
    ######################################################################################
    ### RESULTS: ###
    
    The winning permutation is {perm}
    
    The parameters used in the simulated annealing were:
        Initial temperature = {init_temp}
        Threashold = {threashold}
        Cooling rate = {cooling_rate} 
    
    Encrypted message:
    {enc_msg}
    
    Decrypted message:
    {dec_msg}
    ######################################################################################
    
    """


def main():
    # 1. read the corpus
    corpus = CorpusReader(url=URL)
    print(corpus.get_corpus())

    # 2+3. create a language model and read the encrypted message
    lang_model = LanguageModel(corpus)

    # 4. create initial hypothesis
    initial_hypothesis = Permutation()

    # 5. run simulated annealing
    sim_annealing = SimulatedAnnealing(init_temp=INITIAL_TEMPERATURE,
                                       threshold=THREASHOLD,
                                       cool_rate=COOLING_RATE)
    encrypted_message = _read_msg_from_file()
    winning_perm = sim_annealing.run(initial_perm=initial_hypothesis,
                                     enc_msg=encrypted_message,
                                     lang_model=lang_model)
    deciphered_message = winning_perm.decipher(encrypted_message)

    # 6. print results
    print(_format_results(perm=winning_perm,
                          init_temp=INITIAL_TEMPERATURE,
                          threashold=THREASHOLD,
                          cooling_rate=COOLING_RATE,
                          enc_msg=encrypted_message,
                          dec_msg=deciphered_message))


if __name__ == '__main__':
    main()
