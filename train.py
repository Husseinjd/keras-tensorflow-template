from __future__ import absolute_import, division, print_function, unicode_literals
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils import factory
from evaluate import *
import sys
import os
import h5py

def main():

    try:
        args = get_args()
        list_config_files = os.listdir(args.config)
        if len(list_config_files) == 0:
            raise Exception('Empty config directory !')
        
        #for each config train
        for cf in list_config_files:
            print(f'Processing Config File : {cf}')
            path_to_config = os.path.join(args.config,cf)
            config = process_config(path_to_config)

            create_dirs([config.callbacks.checkpoint_dir])

            #print('Create the data generator.')
            data_loader = factory.create("data_loader."+config.data_loader.name)(config,config.exp.data_dir)
            #print('Create the model.')
            model = factory.create("models."+config.model.name)(config,data_loader.feature_columns)
            #print('Create the trainer')
            trainer = factory.create("trainers."+config.trainer.name)(model.model, data_loader.get_train_data(),data_loader.get_validation_data(), config)
            #print('Start training the model.')
            trainer.train()
            #print('Max val acc: ',max(trainer.val_acc))
        if args.evaluate == 't':
                #print('Evaluating..')
                path_to_models = config.callbacks.exp_dir
                print(path_to_models)
                test_x, test_y = data_loader.get_testing_data()
                best_model_path = evaluate_test(path_to_models,test_x, test_y)
        
        #submit kaggle predictions
        if args.submit == 't':
                pass

    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
