import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers, losses, optimizers, models, Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import data

# FUNCTIONS
def parasite_or_not(x): 
    if(x < 0.5): 
        return str('P')
    else: 
        return str('U')

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

# CREATE THE MODEL - a LeNet Convolutional Neural Network Architecture using Model Subclassing

# Break Up the Model Into Two Parts - Feature Extraction and Output

# FEATURE EXTRACTION LAYER CLASS

class FeatureExtractor(Layer): 
    def __init__(self, filters, kernel_size, strides, padding, activation, pool_size): 
        super(FeatureExtractor, self).__init__()
        
        self.conv_1 = layers.Conv2D(filters = filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)
        self.batch_1 = layers.BatchNormalization()
        self.pool_1 = layers.MaxPool2D(pool_size=pool_size, strides=2*strides)
        
        self.conv_2 = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)
        self.batch_2 = layers.BatchNormalization()
        self.pool_2 = layers.MaxPool2D(pool_size=pool_size, strides=2*strides)
    
    def call(self, x, training): 
        
        x = self.conv_1(x)
        x = self.batch_1(x, training = training)
        x = self.pool_1(x)
        
        x = self.conv_2(x)
        x = self.batch_2(x, training = training)
        x = self.pool_2(x)
        
        return x

    
# FEATURE EXTRACTOR MODEL CLASS 

class LenetModel(Model): 
    def __init__(self): 
        super(LenetModel, self).__init__()

        self.feature_extractor = FeatureExtractor(8, 3, 1, "valid", "relu", 2)
    
    def call(self, x): 
        
        x = self.feature_extractor(x, training = False)
        
        return x

# FINAL OUTPUT MODEL CLASS 

class FinalModel(Model): 
    def __init__(self, activation): 
        super(FinalModel, self).__init__()
        
        # self.lenet_model = LenetModel()
        
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(100, activation = activation)
        self.batch_normalize_1 = layers.BatchNormalization()
        
        self.dense_2 = layers.Dense(10, activation = activation)
        self.batch_normalize_2 = layers.BatchNormalization()
        
        self.final_output = layers.Dense(1, activation = "sigmoid")
        
    def call(self, x, training): 
        # x = self.lenet_model.call(x = Input(shape = (IM_SIZE, IM_SIZE, 3)))
        
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.batch_normalize_1(x, training = training)
        
        x = self.dense_2(x)
        x = self.batch_normalize_2(x, training = training)
        
        x = self.final_output(x)
        
        return x
    
# FINAL OUTPUT GATHERING MODEL WITH MODEL SUBCLASSING
lenet_model = LenetModel()
final_model = FinalModel("relu")

# Connect Inputs and Outputs of Different Models 
inputs = Input(shape = (IM_SIZE, IM_SIZE, 3))
lenet_output = lenet_model.call(inputs)
final_output = final_model.call(lenet_output, training = False)

# Create a new model that takes inputs and outputs final_output
full_model = Model(inputs, final_output)
full_model.summary()

# COMPILE THE MODEL - Use Binary Cross Entropy Loss Function and Adam Optimizer

full_model.compile(optimizer = optimizers.Adam(learning_rate = 0.01),
              loss = losses.BinaryCrossentropy(),
              metrics = 'accuracy'
              )

history = full_model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1)

# # PLOT LOSS OVER TIME 

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL LOSS')
plt.ylabel('LOSS')
plt.xlabel('EPOCHS')
plt.legend(['train', "val_loss"])
plt.show()

# MODEL EVALUATION AND TESTING 
# test_dataset = test_dataset.batch(1)
# final_model.evaluate(test_dataset)

# VISUALIZE YOUR DATA
# for i, (image, label) in enumerate(test_dataset.take(9)): 
#     ax = plt.subplot(3, 3, i+1)
#     plt.imshow(image[0]) # take the 0th element of image object because that's where the link to the image actually is    
#     plt.title(str(parasite_or_not(label.numpy()[0])) + ":" + parasite_or_not(feed_forward_model.predict(image)[0][0]))
#     plt.axis('off')
#     plt.show()
