"""
Defines utility functions for saving and loading pickles and tensorflow networks.
"""

import pickle


def pickle_to_file(file_name, data, protocol = pickle.HIGHEST_PROTOCOL):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol)


def unpickle_from_file(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)