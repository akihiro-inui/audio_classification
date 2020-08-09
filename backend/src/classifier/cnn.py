#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on sleepy midnight in Aug
@author: Akihiro Inui
"""
import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import model_from_json
import os
import numpy as np
import matplotlib.pyplot as plt


class CNN:
    def __init__(self, validation_rate, num_classes):
        self.validation_rate = validation_rate
        self.num_classes = num_classes
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(22, 128, 1)))
        self.model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(num_classes, activation='softmax'))

        # Compile
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer="adam",
                           loss=self.loss_function,
                           metrics=['accuracy'])
        print(self.model.summary())

    def training(self, train_data, train_label, visualize=None):
        # Train and Test The Model
        model_graph = self.model.fit(train_data,
                                     train_label,
                                     batch_size=4,
                                     epochs=10,
                                     verbose=1,
                                     validation_split=0.2)

        # Visualize training result
        if visualize is True:
            # Show accuracy
            plt.plot(model_graph.history['accuracy'])
            plt.plot(model_graph.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

            # Show loss
            plt.plot(model_graph.history['loss'])
            plt.plot(model_graph.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        return self.model

    def test(self, model, test_data, test_label):
        # Test model
        loss, accuracy = model.evaluate(test_data, test_label, verbose=1)
        print('Test loss:', loss)
        print('Test accuracy:', accuracy)
        return accuracy*100

    def predict(self, model, test_data):
        """
        Make prediction to a given target data and return the prediction result with accuracy for each sample
        :param  model: trained model
        :param  test_data: test data
        :return prediction array with probability
        """
        # Make prediction to the target data
        prediction = model.predict(test_data, verbose=1)
        return np.array(prediction)

    def predict_one(self, model, target_data):
        """
        Make prediction to a given target data and return the prediction result with accuracy for each sample
        :param  model: trained model
        :param  target_data: test data as numpy array
        :return prediction array with probability
        """
        # Forward data and calculate loss
        prediction = model.predict(target_data, verbose=1)
        return np.array(prediction)

    def load_model(self, model_file_path: str, weight_file_path: str):
        """
        Load trained model
        :param  model_file_path: File path to model
        :param  weight_file_path: File path to model weight
        :return model: trained model
        """
        # Load model if it exists
        assert os.path.exists(model_file_path), "Model does not exist"

        # Load model
        with open(model_file_path, 'r') as f:
            model = model_from_json(f.read())

        # Load weights
        model.load_weights(weight_file_path)

        # Compile model
        model.compile(loss=categorical_crossentropy,
                      optimizer=keras.optimizers.adam(),
                      metrics=['accuracy'])
        return model

    def save_model(self, model, output_directory: str):
        """
        Save model
        :param  model: trained model
        :param  output_directory: output directory path
        """
        if not model:
            model = self.model

        # json形式でモデルを保存
        json_string = model.to_json()
        open(os.path.join(output_directory, 'model.json'), 'w').write(json_string)
        model.save_weights(os.path.join(output_directory, 'weights.hdf5'))

