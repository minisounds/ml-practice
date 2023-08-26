import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers, losses, optimizers, models
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import AUC, BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler
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
    
    test_dataset = dataset.skip(int((TEST_RATIO + TRAIN_RATIO+0.18) * LENGTH))
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

# Create a Custom Callback Class to Display Loss Vallues after each Epoch
class LossCallback(Callback): 
    def on_epoch_end(self, epochs, logs): 
        print("/n For Epoch Number {} the Loss Function is {}".format(epochs+1, logs["loss"]))

# Create a CSV Logger to Move Log Data into a CSV File after each Epoch

csvlog = CSVLogger(
    'logs.csv', separator=",", append = False
)

# Create an EarlyStopping Callback to Prevent Overfitting to Training Data

early_stop = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto', restore_best_weights=True, start_from_epoch=0
)

# Create a LearningRateScheduler Callback to Dynamically adjust Learning Rates by Epoch # 

def scheduler(epochs, lr): 
    if epochs < 2: 
        return lr
    else: 
        # if epoch > 3, return learning rate * e^-0.1, which is 1/1.105, a value less than 1
        return lr * tf.math.exp(-0.1)

learning_scheduler = LearningRateScheduler(scheduler, verbose = 1)


# Create a customizable dense layer for use in Sequential API model

class NeuraLearnDense(Layer): 
    def __init__(self, output_units, activation): 
        super(NeuraLearnDense, self).__init__()
        
        self.output_units = output_units
        self.activation = activation
        
    def build(self, input_feature_shape): 
        self.w = self.add_weight(shape = (input_feature_shape[-1], self.output_units), initializer = "glorot_normal", trainable = True)
        self.b = self.add_weight(shape = (self.output_units, ), initializer = "zeros", trainable = True)
        
    def call(self, input_features): 
        pre_output = tf.matmul(input_features, self.w) + self.b
        
        if self.activation == "relu":
            return tf.nn.relu(pre_output)
        elif self.activation == "sigmoid": 
            return tf.math.sigmoid(pre_output)
        else: 
            return pre_output
    

# CREATE THE MODEL - a LeNet Convolutional Neural Network Architecture using SEQUENTIAL API
model = tf.keras.Sequential([
    layers.InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)),
    
    layers.Conv2D(filters = 6, kernel_size = 3, strides = 1, padding = "valid", activation = "relu"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size = (2,2), strides = 2),
    
    layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = "valid", activation = "relu"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size = (2,2), strides = 2),
    
    layers.Flatten(),
    NeuraLearnDense(100, activation = "relu"),
    layers.BatchNormalization(),
    NeuraLearnDense(10, activation = "relu"),
    layers.BatchNormalization(),
    NeuraLearnDense(1, activation="sigmoid")
])
model.summary()


# COMPILE THE MODEL - Use Binary Cross Entropy Loss Function and Adam Optimizer

# DEFINE YOUR METRICS 

metrics = [BinaryAccuracy(name = "bin accuracy"), AUC(name = "auc"), Precision(name = "precision"), Recall(name = "recall"), TruePositives(name = "tp"), TrueNegatives(name = "tn"), FalsePositives(name = "fp"), FalseNegatives(name = "fn")]

model.compile(optimizer = optimizers.Adam(learning_rate = 0.01),
              loss = losses.BinaryCrossentropy(),
              metrics = metrics
              )

history = model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1, callbacks = [learning_scheduler])

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

# VISUALIZING CONFUSION MATRIX 

labels = []
inp = []

for x,y in test_dataset.as_numpy_iterator(): 
    labels.append(y)
    inp.append(x)
 
inp = np.array(inp)
npy_inputs = np.squeeze(inp, axis = 1)

predicted = model.predict(npy_inputs)

threshold = 0.5
cm = confusion_matrix(labels, predicted > threshold)
print(cm)

# Plot Confusion Matrix

# plt.figure(figsize=(8,8))

# sns.heatmap(cm, annot=True)
# plt.title("Confusion Matrix - {}".format(threshold))
# plt.ylabel('Actual')
# plt.xlabel('Predicted')

# plt.show()

# Plotting ROC Curve

fp, tp, threshold = roc_curve(labels, predicted)
plt.plot(fp, tp)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.grid()

skip = 20

for i in range(0, len(threshold), skip): 
    plt.text(fp[i], tp[i], threshold[i])

plt.show()

# VISUALIZE YOUR DATA

# for i, (image, label) in enumerate(test_dataset.take(9)): 
#     ax = plt.subplot(3, 3, i+1)
#     plt.imshow(image[0]) # take the 0th element of image object because that's where the link to the image actually is    
#     plt.title(str(parasite_or_not(label.numpy()[0])) + ":" + parasite_or_not(model.predict(image)[0][0]))
#     plt.axis('off')
#     plt.show()