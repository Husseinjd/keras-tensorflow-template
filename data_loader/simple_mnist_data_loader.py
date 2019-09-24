import tensorflow as tf
from tensorflow import keras
from base.base_data_loader import BaseDataLoader
from sklearn.model_selection import train_test_split

class TitanicDataLoader(BaseDataLoader):
    def __init__(self, config,train_dir,test_dir):
        super(SimpleMnistDataLoader, self).__init__(config)
        self.train_dir = train_dir 
        self.test_dir = test_dir
        self.preprocess()
        
    def preprocess(self):
        

    
    
    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
