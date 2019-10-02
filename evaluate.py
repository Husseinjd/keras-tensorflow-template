"""
The script evaluates models on testing data .
needed for selection of models for the final prediction task.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from utils.utils import *
from utils import factory
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow.keras.models import load_model





def evaluate_test(checkpoint_dir,test_ds):
    MODEL_EVALUATE_LIST_NAME = 'val_acc.list'

    checkpoints_directory = checkpoint_dir #e.g. date of the experiment 

    #find the training and validation lists get the argmax val 
    list_config_results  = os.listdir(checkpoints_directory) 

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
            #find the model to load and evaluate 
            for f in results_list:
                if '-' in f:
                    model_epoch_num = f.split('-')[1]
                    if int(model_epoch_num) == best_perf_index:
                        print('--Validation: Model:{} -  Best Epoch --> {}'.format(checkp,f))
                        #evaluate model on test data  
                        model_path = os.path.join(path_checkpoint,f)
                        print('Loading model..',model_path)
                        model =  tf.keras.models.load_model(model_path)
                        print('evaluating model..')
                        loss, accuracy = model.evaluate(test_ds)
                        print('--Test: Acc {} Loss: {}'.format(loss,accuracy))
        else:
            print('Missing validation list pickle file')



