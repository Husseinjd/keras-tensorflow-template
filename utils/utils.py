import argparse
import pickle
import os

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def save_as_pickle(a, path, filename):
        """
        Save an object as a pickle file
        :param object: The python object. Can be list, dict etc.
        :param path: The path where to save.
        :param filename: The filename
        """

        with open(os.path.join(path,filename), 'wb') as handle:
                pickle.dump(a, handle)
        print("Save "+ filename +" successfully.")