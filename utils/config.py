import json
from dotmap import DotMap
import os
import time


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict,json_file


def process_config(json_file):
    config, _,json_file = get_config_from_json(json_file)
    json_string_name = json_file.rsplit('_',1)[1].rsplit('.',1)[0] #e.g. mid-01
    dataloader_string_name =  'dl' + config.data_loader.name.rsplit('_',1)[1].rsplit('.',1)[0] #e.g. dl01
    model_string_name = 'm' + config.model.name.rsplit('_',1)[1].rsplit('.',1)[0] #e.g. m01
    config.callbacks.exp_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()))
    config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()),"{}-{}-{}-checkpoints/".format(
          model_string_name,
          json_string_name,
          dataloader_string_name
          ))    
    return config
