"""
The script evaluates models on testing data .
needed for selection of models for the final prediction task.
"""
from utils.utils import *
import pandas as pd
import numpy as np
import os
import argparse

MODEL_EVALUATE_LIST_NAME = 'val_acc.list'

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
        '-d', '--models-dir',
        dest='exp_dir',
        metavar='Dir',
        default='None',
        help='The directory where the checkpoints for each model exist')

args = argparser.parse_args()


models_list_num = [] #list containing the index of the best performance models on val data

checkpoints_directory = args.exp_dir #e.g. date of the experiment 

#find the training and validation lists get the argmax val 
list_config_results  = os.listdir(checkpoints_directory) 

print('-- Validation ')
for checkp in list_config_results:
    #check the results validation results of every config result
        #find the validation pickle list 
    path_checkpoint = os.path.join(checkpoints_directory,checkp)
    #list of results 
    results_list = os.listdir(path_checkpoint)
    if MODEL_EVALUATE_LIST_NAME in results_list:
        full_path_list = os.path.join(path_checkpoint,MODEL_EVALUATE_LIST_NAME)
        val_list = load_pickle(full_path_list)
        #find the argmax and load the model associated with that result to evaluate
        best_perf_index = np.argmax(val_list) + 1 #epochs starts at 01
        models_list_num.append(best_perf_index)
        #find the model to load and evaluate 
        for f in results_list:
            if '-' in f:
                model_epoch_num = f.split('-')[1]
                if int(model_epoch_num) == best_perf_index:
                    print('Model:{} -  Best Epoch --> {}'.format(checkp,f))
    else:
        print('Missing validation list pickle file')

print('-- Testing')