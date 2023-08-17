import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers, losses, optimizers
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import data

# LOAD DATASET 
dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])

# Split Data into Training, Validation, and Testing Batches
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO): 
    LENGTH = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO * LENGTH))
    
    val_dataset = dataset.skip(int(TRAIN_RATIO * LENGTH))
    val_dataset = val_dataset.take(int(VAL_RATIO * LENGTH))
    
    test_dataset = dataset.skip(int((TEST_RATIO + TRAIN_RATIO) * LENGTH))
    test_dataset = test_dataset.take(int(TEST_RATIO * LENGTH))
    
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

# VISUALIZE YOUR DATA (removing for quickness)
# for i, (image, label) in enumerate(train_dataset.take(16)): 
#     ax = plt.subplot(4, 4, i+1)
#     plt.imshow(image)
#     plt.title(dataset_info.features['label'].int2str(label))
#     plt.axis('off')
    # plt.show()
    
# DATA PREPROCESSING - NORMALIZE THE DATA AND STANDARDIZE FORMAT

IM_SIZE = 224
def resize_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255, label

# Map the resize_rescale() function to each element in the dataset
train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

# Shuffle and configure dataset settings 
train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

# CREATE THE MODEL - a LeNet Convolutional Neural Network Architecture
model = tf.keras.Sequential([
    layers.InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)),
    
    layers.Conv2D(filters = 6, kernel_size = 3, strides = 1, padding = "valid", activation = "relu"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size = (2,2), strides = 2),
    
    layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = "valid", activation = "relu"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size = (2,2), strides = 2),
    
    layers.Flatten(),
    layers.Dense(100, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dense(10, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dense(1, activation="sigmoid")
])
model.summary()

# COMPILE THE MODEL - Use Binary Cross Entropy Loss Function and Adam Optimizer

model.compile(optimizer = optimizers.Adam(learning_rate = 0.01),
              loss = losses.BinaryCrossentropy(),
              metrics = 'accuracy'
              )

history = model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1)

# PLOT LOSS OVER TIME 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL LOSS')
plt.ylabel('LOSS')
plt.xlabel('EPOCHS')
plt.legend(['train', "val_loss"])
plt.show()

# MODEL EVALUATION AND TESTING 
test_dataset = test_dataset.batch(1)
model.evaluate(test_dataset)

# model.predict(test_dataset.take(1))[0][0]
