import tensorflow as tf
import tensorflow_datasets as tfds # for creating tensorflow dataset objects 
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


# LOAD DATASET 
dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])

# Split Data into Training, Validation, and Testing Batches
TRAIN_RATIO = 0.1
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO): 
    LENGTH = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO * LENGTH))
    
    val_dataset = dataset.skip(int(TRAIN_RATIO * LENGTH))
    val_dataset = val_dataset.take(int(VAL_RATIO * LENGTH))
    
    test_dataset = dataset.skip(int((TEST_RATIO + TRAIN_RATIO) * LENGTH))
    test_dataset = test_dataset.take(int(TEST_RATIO * LENGTH))
    
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

# DATA PREPROCESSING AND AUGMENTATION
    
# DATA PREPROCESSING - NORMALIZE THE DATA AND STANDARDIZE FORMAT

IM_SIZE = 224

# DEFINE RESIZE AND RESCALE IMAGE FUNCTION FOR DATA PREPROCESSING & BATCHING IMAGES SO THEIR TENSORSHAPES MATCH
def resize_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255, label

# ALBUMENTATION DATA AUGMENTATION 

transforms = A.Compose([
    A.Resize(IM_SIZE, IM_SIZE),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
])

def aug_albument(image):
    data = {"image": image} # create a dictionary with a single key-value pair
    image = transforms(**data) # **data unpacks dictionary data's key-value pairs to pass into transforms(). Dictionaries are commonly used in ALbumentations
    image = image["image"] # upon successful completion of transforms, the transformed image is extracted from the image from the dictionary using the key "image" and assigned back to the variable image
    image = tf.cast(image/255., tf.float32) # scales pixel values from [0,255] to the range [0,1]
    return image

def process_data(image, label): 
    aug_img = tf.numpy_function(func = aug_albument, inp = [image], Tout = tf.float32) # numpy_function converts a python function into a tensor operations
    return aug_img, label

# Shuffle and configure dataset settings 

train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).map(process_data).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).map(resize_rescale).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(resize_rescale)
# Visualize transforms() function

plt.figure(figsize=(15,15))

im, _ = next(iter(train_dataset))

for i in range(1, 32): 
    plt.subplot(8,4,i)
    plt.imshow(im[i])

# CALLBACKS 

# Create a Tensorboard Callback that creates a web interface displaying metrics w/o having to use matplotlib code
CURRENT_TIME = datetime.datetime.now().strftime('%d%m%y - %h%m%s')
METRIC_DIR = './tensorboard_logs/' + CURRENT_TIME + '/metrics' # create a folder in the tensorboard data folder
LOG_DIR = './tensorboard_logs/' + CURRENT_TIME

train_writer = tf.summary.create_file_writer(METRIC_DIR) # create something to write a file to tensorboard
tensorboard_callback = TensorBoard(log_dir=LOG_DIR) # log everything in date folder within tensorboard folder

# Create a Custom Callback Class to Display Loss Values after each Epoch
class LossCallback(Callback): 
    def on_epoch_end(self, epochs, logs): 
        print("/n For Epoch Number {} the Loss Function is {}".format(epochs+1, logs["loss"]))

# Log Confusion Matrix to Tensorboard 

class LogImagesCallback(Callback):
    def on_epoch_end(self, epoch, logs): 
        # VISUALIZING CONFUSION MATRIX 
        labels = [] # create list of labels (ground truths) to store from dataset
        inp = [] # intake the inputs from the dataset 

        for x,y in test_dataset.as_numpy_iterator(): # fill up the lists with data
            labels.append(y) 
            inp.append(x)

        inp = tf.expand_dims(inp, axis = 0)  #returns shape (1, 2757, 244, 244, 3)
        inp = inp[0,...] # removes the first dimension to return shape (2757, 244, 244, 3)
        print(inp.shape)
        predicted = model.predict(inp) # feeds in preprocessed input list into the model
        # predicted = predicted[:,0]
        # print(predicted[:,0].shape)

        threshold = 0.5

        cm = confusion_matrix(labels, predicted > threshold) # plugs in confusion_matrix(y_true, y_pred) where y_pred is the list with 1s and 0s for if the predicted value was > 0.5 or not

        # Plot Confusion Matrix

        plt.figure(figsize=(8,8))

        sns.heatmap(cm, annot=True)
        plt.title("Confusion Matrix - {}".format(threshold))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.axis('off')
        
        buffer = io.BytesIO() 
        plt.savefig(buffer, format = 'png')
        
        image = tf.image.decode_png(buffer.getvalue(), channels = 3)
        image = tf.expand_dims(image, axis = 0)
        
        IMAGE_DIR = './custom_tensorboard_logs/' + CURRENT_TIME + '/images'
        image_writer = tf.summary.create_file_writer(IMAGE_DIR)
        
        with image_writer.as_default(): 
            tf.summary.image("Training Confusion Matrix", image, step=epoch) 
        

# Create a CSV Logger to Move Log Data into a CSV File after each Epoch

csvlog = CSVLogger(
    'logs.csv', separator=",", append = False
)

# Create an EarlyStopping Callback to Prevent Overfitting to Training Data

early_stop = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto', restore_best_weights=True, start_from_epoch=0
)

# Create a LearningRateScheduler Callback to Dynamically adjust Learning Rates by Epoch # 

def scheduler(epoch, lr): # this is the FUNCTION, not the CALLBACK 
    learning_rate = lr
    if epoch >= 1: 
        # if epoch > 3, return learning rate * e^-0.1, which is 1/1.105, a value less than 1
        learning_rate = lr * tf.math.exp(-0.1)
    with train_writer.as_default():
        tf.summary.scalar('Learning Rate', data = learning_rate, step = epoch)
    return learning_rate

learning_scheduler_callback = LearningRateScheduler(scheduler, verbose = 1)

# Create a Reduced Learning Rate on Plateau Callback Function

reduce_callback = ReduceLROnPlateau(
    monitor='val_bin accuracy',
    factor=0.1,
    patience=2,
    verbose=1,
)

# Create a Model Checkpointing Callback to Save Model Weights and Model Parameters After Best Validation Loss

checkpoint_filepath = "model_checkpoint.h5"

checkpoint_callback = ModelCheckpoint(
    filepath = checkpoint_filepath,
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    mode = 'auto',
    save_freq = 'epoch',
    initial_value_threshold = None,
)

# Create a customizable dense layer for use in Sequential API model

class NeuraLearnDense(Layer): 
    def __init__(self, output_units, activation, l2_rate): 
        super(NeuraLearnDense, self).__init__()
        
        self.output_units = output_units
        self.activation = activation
        self.l2_rate = l2_rate
        
    def build(self, input_feature_shape): 
        self.w = self.add_weight(shape = (input_feature_shape[-1], self.output_units), initializer = "glorot_normal", regularizer = L2(self.l2_rate), trainable = True)
        self.b = self.add_weight(shape = (self.output_units, ), initializer = "zeros", trainable = True)
        
    def call(self, input_features): 
        pre_output = tf.matmul(input_features, self.w) + self.b
        
        if self.activation == "relu":
            return tf.nn.relu(pre_output)
        elif self.activation == "sigmoid": 
            return tf.math.sigmoid(pre_output)
        else: 
            return pre_output
    
# Create Custom Loss Function 

def custom_bce(y_true, y_pred): 
    bce = losses.BinaryCrossentropy()
    return bce(y_true, y_pred)
        
# CREATE THE MODEL - a LeNet Convolutional Neural Network Architecture using SEQUENTIAL API
regularization_rate = 0.001

model = tf.keras.Sequential([
    layers.InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)),

    layers.Conv2D(filters = 6, kernel_size = 3, strides = 1, padding = "valid", activation = "relu", kernel_regularizer = L2(regularization_rate)),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size = (2,2), strides = 2),
    
    layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = "valid", activation = "relu", kernel_regularizer = L2(regularization_rate)),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size = (2,2), strides = 2),
    
    layers.Flatten(),
    layers.Dense(16, activation = "relu", kernel_regularizer = L2(regularization_rate)),
    layers.BatchNormalization(),
    Dropout(rate = 0.1), 
    layers.Dense(32, activation = "relu", kernel_regularizer = L2(regularization_rate)),
    layers.BatchNormalization(),
    layers.Dense(1, activation="sigmoid")
    ])
model.summary()

# COMPILE THE MODEL - Use Binary Cross Entropy Loss Function and Adam Optimizer

# DEFINE YOUR METRICS 

metrics = [BinaryAccuracy(name = "bin accuracy"), AUC(name = "auc"), Precision(name = "precision"), Recall(name = "recall"), TruePositives(name = "tp"), TrueNegatives(name = "tn"), FalsePositives(name = "fp"), FalseNegatives(name = "fn")]

model.compile(optimizer = optimizers.Adam(learning_rate = 0.01),
            loss = custom_bce,
            metrics = metrics,
            run_eagerly = False
            )

model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1, callbacks = [])


# CREATE CUSTOM TRAINING LOOP IN PLACE OF MODEL.FIT

# PREPARE CUSTOM TRAINING LOOP TENSORBOARD LOGS
CUSTOM_METRIC_DIR = './custom_tensorboard_logs/' + CURRENT_TIME + '/metrics'
CUSTOM_TRAIN_DIR = './custom_tensorboard_logs/' + CURRENT_TIME + '/train'
CUSTOM_VAL_DIR = './custom_tensorboard_logs/' + CURRENT_TIME + '/validation' # Have to Create a Writer to Log Validation Data, as well. Model.fit() normally handles this automatically 

custom_train_writer = tf.summary.create_file_writer(CUSTOM_TRAIN_DIR)
custom_validation_writer = tf.summary.create_file_writer(CUSTOM_VAL_DIR) # Create two validation writers to log changes to different files 

# SET CONSTANTS 
OPTIMIZER = optimizers.Adam(learning_rate=0.01)
METRIC = BinaryAccuracy()
METRIC_VAL = BinaryAccuracy()
EPOCHS = 2
tf.config.run_functions_eagerly(False)

@tf.function #use graph mode to compute this for faster training times 
def training_block(x_batch, y_batch): 
    with tf.GradientTape() as recorder: # record the gradients in this tape recorder (to make partial derivative)
        y_pred = model(x_batch, training = True) 
        loss = custom_bce(y_batch, y_pred) # use custom loss function (binary cross entropy) to calc loss
        
    partial_derivatives = recorder.gradient(loss, model.trainable_weights) # uses recorded losses to calculate the partial derivative between the loss and each of the model's trainable weights
    OPTIMIZER.apply_gradients(zip(partial_derivatives, model.trainable_weights)) # uses the ADAM optimizer to apply the gradients 
    # zip() function uses seperate derivatives [deriv1, deriv2] and weights [weight1, weight2] into [(grad1, weight1), (grad2, weight2)]
        
    METRIC.update_state(y_batch, y_pred) # takes in y_batch as a true value, y_pred as predicted variable. update_state() adds to the counter of total correct predictions avs total predictions, used to calculate the final accuracy of the model

    return loss

@tf.function #convert to graph mode
def val_block(x_batch_val, y_batch_val):
    y_pred_val = model(x_batch_val, training = False) # important: set training = False
    loss_val = custom_bce(y_batch_val, y_pred_val)
    METRIC_VAL.update_state(y_batch_val, y_pred_val) 
    
    return loss_val

# CREATE TRAINING FUNCTION
def neuralearn(model, loss_function, METRIC, VAL_METRIC, train_dataset, val_dataset, EPOCHS, OPTIMIZER):
    for epoch in range(EPOCHS): 
        print("Training Begins for Epoch number {}".format(epoch+1))
        for (x_batch, y_batch) in train_dataset: # enumerate train_dataset to help keep track of how many steps we've gotten through
            loss = training_block(x_batch, y_batch)
                
        print("The Loss is: ", loss.numpy())        
        print("The accuracy is: ", METRIC.result().numpy())
        with custom_train_writer.as_default(): # Log Training Loss and Accuracy to the Training Writer File 
            tf.summary.scalar('Training Loss', data = loss, step = epoch)
        with custom_train_writer.as_default():
            tf.summary.scalar('Training Accuracy', data = METRIC.result(), step = epoch)
        METRIC.reset_states()
        
        for (x_batch_val, y_batch_val) in val_dataset: # calculate validation loss after going through epoch x
            loss_val = val_block(x_batch_val, y_batch_val)
            
        print("Validation loss", loss_val.numpy())
        print ("The Validation Accuracy is: ", METRIC_VAL.result().numpy())
        with custom_validation_writer.as_default(): # Log Validation Loss and Accuracy to the Validation Writer File
            tf.summary.scalar('Validation Loss', data = loss_val, step = epoch)
        with custom_validation_writer.as_default():
            tf.summary.scalar('Validation Accuracy', data = METRIC_VAL.result(), step = epoch)
        METRIC_VAL.reset_states()
        
    print("Training Complete!")

# RUN THE TRAINING FUNCTION HERE (Commented out to use tensorboard)
# neuralearn(model = model, loss_function=custom_bce, METRIC=METRIC, VAL_METRIC=METRIC_VAL, train_dataset = train_dataset, val_dataset=val_dataset, EPOCHS = EPOCHS, OPTIMIZER = OPTIMIZER)

# # PLOT LOSS OVER TIME 

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('MODEL LOSS')
# plt.ylabel('LOSS')
# plt.xlabel('EPOCHS')
# plt.legend(['train', "val_loss"])
# plt.show()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('MODEL ACCURACY')
# plt.ylabel('ACCURACY')
# plt.xlabel('EPOCHS')
# plt.legend(['train', "val_accuracy"])
# plt.show()


# MODEL EVALUATION AND TESTING 

test_dataset = test_dataset.batch(1)
model.evaluate(test_dataset)

# # Plotting ROC Curve

# fp, tp, threshold = roc_curve(labels, predicted)
# plt.plot(fp, tp)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")

# plt.grid()

# skip = 20

# for i in range(0, len(threshold), skip): 
#     plt.text(fp[i], tp[i], threshold[i])

# plt.show()

# # VISUALIZE YOUR DATA

# def parasite_or_not(x): 
#     if(x < 0.5): 
#         return str('P')
#     else: 
#         return str('U')
    
# for i, (image, label) in enumerate(test_dataset.take(9)): 
#     ax = plt.subplot(3, 3, i+1)
#     plt.imshow(image[0]) # take the 0th element of image object because that's where the link to the image actually is    
#     plt.title(str(parasite_or_not(label.numpy()[0])) + ":" + parasite_or_not(model.predict(image)[0][0]))
#     plt.axis('off')
#     plt.show()