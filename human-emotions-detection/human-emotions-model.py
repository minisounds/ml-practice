import tensorflow as tf
import tensorflow_datasets as tfds # for creating tensorflow dataset objects 
from packaging import version
import os
import wandb # experiment tracking + hyperparameter tuning
from wandb.keras import WandbCallback
import keras
import cv2 # computer vision 2 for visualizing and working with images 
import datetime # for tensorboard logging custom times 
import io # for saving images to tensorboard 
import albumentations as A # for data augmentation
from keras import layers, losses, optimizers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dropout, Resizing, Rescaling
from tensorflow.keras.metrics import AUC, BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras.regularizers import L2
import tensorflow_probability as tfp 
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorboard.plugins.hparams import api as hp
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import data


# USEFUL CONSTANTS: 
train_directory = "./dataset/Emotions Dataset/Emotions Dataset/train"
validation_directory = "./dataset/Emotions Dataset/Emotions Dataset/test"
CLASS_NAMES = ["angry", "happy", "sad"]

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
    "LEARNING_RATE": 0.001,
    "N_EPOCHS": 11,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 100,
    "N_DENSE_2": 10,
    "NUM_CLASSES": 3,
}

# CREATE TENSORFLOW DATASET OBJECTS FOR TESTING AND VALIDATION 
train_dataset = keras.utils.image_dataset_from_directory(
    directory=train_directory,
    labels='inferred', # since emotions dataset sections it's images under different classes (happy, angry, sad)
    label_mode='int', # angry = 0, happy = 1, sad = 2. use sparse_categorical_crossentropy as a loss function
    class_names=CLASS_NAMES,
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True, 
    seed=None, # allows you to have the same shuffling each time
    validation_split=None, # allows you to split dataset into two parts for validation & training. but we don't need since the kaggle dataset already segments it into test and training for us
)

val_dataset = keras.utils.image_dataset_from_directory(
    directory=validation_directory,
    labels='inferred', # since emotions dataset sections it's images under different classes (happy, angry, sad)
    label_mode='int', # angry = 0, happy = 1, sad = 2. use sparse_categorical_crossentropy as a loss function
    class_names=CLASS_NAMES,
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True, 
    seed=None, # allows you to have the same shuffling each time
    validation_split=None, # allows you to split dataset into two parts for validation & training. but we don't need since the kaggle dataset already segments it into test and training for us
)

         
# DATASET PREPARATION - add prefetching to it
training_dataset = (
    train_dataset.prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    val_dataset.prefetch(tf.data.AUTOTUNE)
)


# BUILD THE MODEL - LENET MODEL

# resize_rescale() function in Sequential Model Form
resize_rescale_layers = Sequential([
    Resizing(height = CONFIGURATION["IM_SIZE"], width = CONFIGURATION["IM_SIZE"]),
    Rescaling(scale=1./255),
])

model = tf.keras.Sequential([
    layers.InputLayer(input_shape = (None, None, 3)),
    
    resize_rescale_layers,

    layers.Conv2D(filters = CONFIGURATION["N_FILTERS"], kernel_size = CONFIGURATION["KERNEL_SIZE"], strides = CONFIGURATION["N_STRIDES"], padding = "valid", activation = "relu", kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size = (CONFIGURATION["POOL_SIZE"],CONFIGURATION["POOL_SIZE"]), strides = CONFIGURATION["N_STRIDES"]),
    
    layers.Conv2D(filters = CONFIGURATION["N_FILTERS"]*2, kernel_size = CONFIGURATION["KERNEL_SIZE"], strides = CONFIGURATION["N_STRIDES"], padding = "valid", activation = "relu", kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size = (CONFIGURATION["POOL_SIZE"],CONFIGURATION["POOL_SIZE"]), strides = CONFIGURATION["N_STRIDES"]*2),
    
    layers.Flatten(),
    layers.Dense(CONFIGURATION["N_DENSE_1"], activation = "relu", kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    layers.BatchNormalization(),
    Dropout(rate = CONFIGURATION["DROPOUT_RATE"]), 
    layers.Dense(CONFIGURATION["N_DENSE_2"], activation = "relu", kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    layers.BatchNormalization(),
    layers.Dense(CONFIGURATION["NUM_CLASSES"], activation="softmax")
    ])
model.summary()

# Define Loss Function
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits = False, # A logit tensor is the raw, unnormalized output of a neural network. Softmax normalizes it for us already (by taking the % each class is in the sum. We set it to be not a logits tensor because of this)
)

# Metrics
metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name = "accuracy"), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name = "top_k_accuracy")]

# Compile Model
model.compile(
    optimizer = optimizers.Adam(learning_rate = CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics,
)

# Fit the Model 
history = model.fit(training_dataset,
                    validation_data = validation_dataset,
                    epochs = CONFIGURATION["N_EPOCHS"],
                    verbose = 1
                    )

# VISUALIZE WHAT THE MODEL IS PREDICTING TO WHAT THE ACTUAL LABEL IS 
emotions_index = {
    0: "Angry",
    1: "Happy",
    2: "Sad",
}

for images, labels in validation_dataset.take(1): # take 1 "batch" from validation dataset
    for i in range(16): #
        ax = plt.subplot(4,4, i+1)
        plt.imshow(images[i]/255.)
        plt.title("True Label: " + emotions_index[labels[i].numpy()] + "\n" + "Predicted Label: " + emotions_index[tf.argmax(model(tf.expand_dims(images[i], axis = 0)), axis = -1).numpy()[0]])
        # ^ Note: tf.argmax(Tensor, axis, name)'s second parameter of axis works like this: axis = 1 returns the index of the max arg in each column. axis = 0 returns index for each row, axis = -1s
        plt.axis("off")
        plt.show() 


