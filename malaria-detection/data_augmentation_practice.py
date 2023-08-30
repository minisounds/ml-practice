import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import cv2
import albumentations as A
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
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import data


# LOAD DATASET 
dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])

# Split Data into Training, Validation, and Testing Batches
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO): 
    LENGTH = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO * LENGTH))
    
    val_dataset = dataset.skip(int(TRAIN_RATIO * LENGTH))
    val_dataset = val_dataset.take(int(VAL_RATIO * LENGTH))
    
    test_dataset = dataset.skip(int((TEST_RATIO + TRAIN_RATIO+0.18) * LENGTH))
    test_dataset = test_dataset.take(int(TEST_RATIO * LENGTH))
    
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

# DATA PREPROCESSING - NORMALIZE THE DATA AND STANDARDIZE FORMAT

IM_SIZE = 224

# DEFINE RESIZE AND RESCALE IMAGE FUNCTION FOR DATA PREPROCESSING & BATCHING IMAGES SO THEIR TENSORSHAPES MATCH
def resize_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255, label

# resize_rescale() function in Sequential Model Form
resize_rescale_layers = Sequential([
    Resizing(height = IM_SIZE, width = IM_SIZE),
    Rescaling(scale=1./255),
])


# MIX-UP DATA AUGMENTATION

lamda = tfp.distributions.Beta(2.0, 2.0) # creates a beta probability distribution with equal probability density on both the lower and upper bounds of [0, 1]
lamda = lamda.sample(1)[0]
print(lamda)

image_1 = cv2.resize(cv2.imread("black_cat.jpg"), (224, 224))
image_2 = cv2.resize(cv2.imread("dog.jpg"), (224, 224))

label_1 = 0
label_2 = 1

image = lamda*image_1 + (1-lamda)*image_2
label = lamda*label_1 * (1-lamda)*image_2
print(label)
plt.title("Mixup Data Augmentation Following a Beta Probability Distribution")
plt.imshow(image/255)
plt.show()

# DEFINE DATA AUGMENTATION FUNCTION (using this for data preprocessing for the train dataset)

def augment(image, label): 
    image, label = resize_rescale(image, label)
    image = tf.image.rot90(image, k = tf.random.uniform(shape=[], minval = 0, maxval = 2, dtype=tf.int32))
    image = tf.image.stateless_random_flip_left_right(image, seed = (1,2))
    return image, label

# Create Custom Rotation Layer for Data Augmentation Model (not going to use due to batch size)

class RotNinety(Layer): 
    def __init__(self): 
        super().__init__()
    
    def call(self, image): 
        return tf.image.rot90(image, k = tf.random.uniform(shape=[], minval = 0, maxval = 2, dtype=tf.int32))
    

# Create Data Augmentation Sequential Model (not going to use due to batch size) 

augment_layers = Sequential([
    RotNinety(),
    tf.keras.layers.RandomFlip(mode = "horizontal")
])

def augment_layer(image, label): 
    return augment_layers(resize_rescale_layers(image), training = True), label



# Shuffle and configure dataset settings 

train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).map(augment).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).map(resize_rescale).batch(32).prefetch(tf.data.AUTOTUNE)
