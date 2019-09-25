from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils import factory
import sys

def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        #run a loop over all config file available or to set a folder for each type of configs - High mid low baseline etc..
        config = process_config(args.config)


        #if args -high 
        #loop over all config files
        #

        #-------------
        # create the experiments dirs
        create_dirs([config.callbacks.checkpoint_dir])
        print('Create the data generator.')
        data_loader = factory.create("data_loader."+config.data_loader.name)(config,config.exp.data_dir)
        config.model.input_shape = data_loader.train_shape[0] #adding input shape to config to be used to build and show model summary
        print('Create the model.')
        model = factory.create("models."+config.model.name)(config,data_loader.feature_columns)
        print('Create the trainer')
        trainer = factory.create("trainers."+config.trainer.name)(model.model, data_loader.get_train_data(),data_loader.get_validation_data(), config)
        print('Start training the model.')
        trainer.train()
        print('Max val acc: ' ,max(trainer.val_acc))

    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
