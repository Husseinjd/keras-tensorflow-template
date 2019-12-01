
# Tensorflow 2.0 Project Template
A tensorflow 2.0 project template that is redesigned to automate model creation using Keras (more to be added later) for competitions. Thus Faster
creation, training and evaluation of models. 


# Acknowledgments
Thanks to Mahmoud Gemy for developing the template. The template is a combination of the templates provided here:
https://github.com/MrGemy95/Tensorflow-Project-Template, Mahmoud Gemy
https://github.com/Ahmkel/Keras-Project-Template , Ahmed Hamada Mohamed Kamel El-Hinidy
with added improvements and automation for faster training and evaluating models based on configs provided.

**template files are provided with an example and sample configs, data files from the titanic kaggle competitions were used**

Folder structure
--------------

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_data_loader.py   - this file contains the abstract class of the data loader.
│   └── base_trainer.py   - this file contains the abstract class of the trainer.
│
│
├── models              - this folder contains the models for the project.
│   └── model_01.py
│
│
├── trainers             - this folder contains trainers of your project.
│   └── trainer.py
│   
│  
├──  data _loader  
│    └── data_loader_01.py  - data loader responsible for handling data generation and preprocessing
│
│
├── train.py --  main used to run the training across different config files and models
│
├── evaluate.py --  files responsible for the evaluation of different models. Loading and selecting the best model
│ 
├── train_bash.sh --  example bash script to run the training with different arguments
│
└── utils
     ├── dirs.py
     └── factory.py
     └── config.py
     └── utils.py


```

# Configuration
Config files are used to choose the models, trainers, and dataloader files for each project. so multiple project can co-exist in this template.
Check the config file contents for more info.


# Future Work
- Add tensorflow dataset API and feature columns 
- randomized config file generator 
- Kaggle submission availability
- More examples on image and text data

# More Info
For further info  on how the template is built and more about its core components please check the link in the Acknowledgments
 