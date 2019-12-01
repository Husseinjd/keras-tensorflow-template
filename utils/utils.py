import argparse
import pickle
import os

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-cd', '--config-dir',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')

    argparser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        metavar='C',
        default='false',
        help='Option to evaluate on test data')


    argparser.add_argument(
        '-s', '--submit',
        dest='submit',
        metavar='C',
        default='false',
        help='Option to submit the predictions to competition')

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

def load_pickle(path_to_obj):    
    # open a file, where you stored the pickled data
    file = open(path_to_obj, 'rb')
    # dump information to that file
    obj = pickle.load(file)
    # close the file
    file.close()
    
    return obj
