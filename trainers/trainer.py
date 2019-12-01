from base.base_trainer import BaseTrain
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.utils import *
import numpy as np

class ModelTrainer(BaseTrain):
    def __init__(self, model, data_train,data_validate, config):
        super(ModelTrainer, self).__init__(model, data_train,data_validate,config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_acc:.4f}.h5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                save_format="h5",
            )
        )
    def train(self):
        self.train_x , self.train_y  = self.train_data
        self.val_x , self.val_y  = self.val_data

        history = self.model.fit(self.train_x.values , self.train_y.values,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            validation_data=(self.val_x.values , self.val_y.values),
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
        
        #show model summary
        print(self.model.summary())
        #save train and validation metric lists as pickle files
        if self.config.trainer.save_pickle:
            print('Saving training..')
            save_as_pickle(self.acc, self.config.callbacks.checkpoint_dir, 'train_acc.list')
            print('Saving validation..')
            save_as_pickle(self.val_acc, self.config.callbacks.checkpoint_dir,'val_acc.list')
