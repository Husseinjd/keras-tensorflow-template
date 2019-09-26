from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils import factory
import sys
import os

def main():
    try:
        args = get_args()
        list_config_files = os.listdir(args.config)
        if len(list_config_files) == 0:
            raise Exception('Empty config directory !')
        
        #for each config train
        for cf in list_config_files:
            path_to_config = os.path.join(args.config,cf)
            config = process_config(path_to_config)

            create_dirs([config.callbacks.checkpoint_dir])

            print('Create the data generator.')
            data_loader = factory.create("data_loader."+config.data_loader.name)(config,config.exp.data_dir)
            print('Create the model.')
            model = factory.create("models."+config.model.name)(config,data_loader.feature_columns)
            print('Create the trainer')
            trainer = factory.create("trainers."+config.trainer.name)(model.model, data_loader.get_train_data(),data_loader.get_validation_data(), config)
            print('Start training the model.')
            trainer.train()
            print('Max val acc: ',max(trainer.val_acc))

    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
