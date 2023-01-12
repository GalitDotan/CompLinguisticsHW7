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
INITIAL_TEMPERATURE = 100
COOLING_RATE = 0.9995
THREASHOLD = pow(10, -3)
ENC_MESSAGE_FILE = 'problemset_07_encrypted_input.txt'


def _read_msg_from_file(filename: str = ENC_MESSAGE_FILE):
    """
    Given a file, read and return its content

    :param filename: the filename (location)
    :return: file's content
    """
    with open(filename) as f:
        return f.read()


def _format_results(winning_perm: Permutation, init_temp: int, threashold: float, cooling_rate: float, enc_msg: str,
                    dec_msg: str):
    """
    Receive the results of the experiment and format them to a printable string, that shows:
    The winning permutation, the simulated annealing parameters, the encrypted message and decrypted message.

    :param winning_perm: the winning permutation
    :param init_temp: the initial temperature
    :param threashold: the threashold
    :param cooling_rate: the cooling rate
    :param enc_msg: the encrypted message
    :param dec_msg: the decrypted message
    :return: the formatted string
    """
    return f"""
    ######################################################################################
    
    ### RESULTS: ###
    
    # The winning permutation #
    {winning_perm}
    
    # The parameters used in the simulated annealing #
        Initial temperature = {init_temp}
        Threashold = {threashold}
        Cooling rate = {cooling_rate} 
    
    # The encrypted message #
    {enc_msg}
    
    # The decrypted message #
    {dec_msg}
    
    ######################################################################################
    """


def main():
    # 1. read the corpus
    corpus = CorpusReader(url=URL)

    # 2+3. create a language model and read the encrypted message
    lang_model = LanguageModel(corpus)

    # 4. create initial hypothesis
    initial_hypothesis = Permutation()

    # 5. run simulated annealing
    sim_annealing = SimulatedAnnealing(init_temp=INITIAL_TEMPERATURE,
                                       threshold=THREASHOLD,
                                       cool_rate=COOLING_RATE)
    try:
        encrypted_message = _read_msg_from_file()
    except Exception:
        print(f'Missing the file "{ENC_MESSAGE_FILE}" from the work directory.')
        return
    winning_perm = sim_annealing.run(initial_hypothesis=initial_hypothesis,
                                     enc_msg=encrypted_message,
                                     lang_model=lang_model)
    deciphered_message = winning_perm.translate(encrypted_message)

    # 6. print results
    print(_format_results(winning_perm=winning_perm,
                          init_temp=INITIAL_TEMPERATURE,
                          threashold=THREASHOLD,
                          cooling_rate=COOLING_RATE,
                          enc_msg=encrypted_message,
                          dec_msg=deciphered_message))


if __name__ == '__main__':
    main()
