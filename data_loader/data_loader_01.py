import tensorflow as tf
from tensorflow import keras,feature_column
from base.base_data_loader import BaseDataLoader
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

class DataLoader(BaseDataLoader):
    def __init__(self, config,data_dir):
        super(DataLoader, self).__init__(config)
        self.data_dir = data_dir
        self.feature_columns = self.preprocess()
        
    def preprocess(self,val_size=0.4 ,test_size=0.2):
        """
            data preprocessing -- returning a list of tensorflow feature columns 
        """
        train_dir = os.path.join(self.data_dir,'train.csv')
        test_dir = os.path.join(self.data_dir,'test.csv')
        df_train = pd.read_csv(train_dir)
        df_test = pd.read_csv(test_dir)
        
        #remove cols
        drop_col = ['Cabin','Name','Ticket','PassengerId']
        df_train_clean =  df_train.drop(drop_col,axis=1)
        
        #remove nans for now 
        df_train_clean = df_train_clean.dropna(axis=0)
        train, val_test = train_test_split(df_train_clean,test_size=val_size)
        val,test = train_test_split(df_train_clean,test_size=val_size)

        #shape used for buffer size if shuffle is True
        self.train_shape  = train.shape
        self.val_shape  = val.shape
        self.test_shape = test.shape

        label = 'Survived'

        self.train = self._df_to_dataset(train,label)
        self.val = self._df_to_dataset(val,label)
        self.test = self._df_to_dataset(test,label)
        #---------------------------------------------------------------------
        
        #if need feature columns #for now there is a bug in loading the model using 
        #feature columns 
        #set up feature columns
        feature_columns = []

        #numeric
        for header in ['Age', 'Fare','SibSp','Parch','Pclass']:
            feature_columns.append(feature_column.numeric_column(header))
        #other
        sex = feature_column.categorical_column_with_vocabulary_list(
            'Sex', ['male', 'female'])
        feature_columns.append(feature_column.indicator_column(sex))
        feature_columns.append(feature_column.bucketized_column(feature_column.numeric_column('Age'),boundaries=[18, 30, 50, 70]))
        return feature_columns

    def get_train_data(self,shuffle=True):
        """
        Returns a tf.Dataset training dataset
        """
        if shuffle:
            self.train = self.train.shuffle(buffer_size=self.train_shape[0])
        if self.config.trainer.batch_size:
            self.train = self.train.batch(self.config.trainer.batch_size)
        return self.train


    def get_validation_data(self,shuffle=False):
        """
        Returns a tf.Dataset validation dataset
        """
        if shuffle:
            self.val = self.val.shuffle(buffer_size=self.val_shape[0])
        if self.config.trainer.batch_size:
            self.val = self.val.batch(self.config.trainer.batch_size)
        return self.val

    
    def get_testing_data(self):
        """
        Returns a tf.Dataset testing dataset
        """
        return self.test

    

    def _df_to_dataset(self,dataframe,label_column):
        """
        Method used to set up the tensorflow dataset

        """
        dataframe = dataframe.copy()
        labels = dataframe.pop(label_column)
        ds = tf.data.Dataset.from_tensor_slices((dataframe.values, labels.values))
        return ds