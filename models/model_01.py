from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


class Model(BaseModel):
    def __init__(self, config,feature_columns):
        super(Model, self).__init__(config)
        #self.feature_columns = feature_columns disabled until bugs fixed in tensorflow
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        
        #add the feature layer
        #self.features_layer = tf.keras.layers.DenseFeatures(self.feature_columns)  
        #self.model.add(self.features_layer)
        #--------------------------------------------
        
        self.model.add(Dense(self.config.model.first_layers_dense, activation='relu'))
        
        #middle layers number
        for n in range(self.config.model.midlayer_num):
            self.model.add(Dense(self.config.model.middle_layers_dense, activation='relu'))
        
        #last layer
        self.model.add(Dense(self.config.model.last_layer_dense, activation=self.config.model.last_activation))
        

        self.model.compile(
            loss=self.config.model.loss,
            optimizer=self.config.model.optimizer,
            metrics=['acc'],
            run_eagerly=True
        )
        
