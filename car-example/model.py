import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import data


data = pd.read_csv("train.csv")

# VISUALIZING THE DATA
# print(data.head())
# pair_plot = sns.pairplot(data[['years','km','rating','condition','economy','top speed', 'hp', 'torque', 'current price']], diag_kind='kde')
# plt.show()

# MOVE THE DATA FROM CSV TO A TENSOR
tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, tf.float32)
tensor_data = tf.random.shuffle(tensor_data, seed=5)


# # GET INPUT VALUES AND EXCLUDE FIRST 3 COLUMNS
X_raw = tensor_data[:, 3:-1]
y = tensor_data[:, -1]
y = tf.expand_dims(y, axis = -1)
y = tf.cast(y, tf.float32)


# # NORMALIZE THE DATA 
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X_raw)
X = normalizer(X_raw)
# normalizer.adapt(y)
# y = normalizer(y)


# Section Off Data for Training, Validation, and Testing
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(X)

X_train_data = X[:int(DATASET_SIZE*TRAIN_RATIO)]
Y_train_data = y[:int(DATASET_SIZE*TRAIN_RATIO)]
X_val_data = X[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO + VALIDATION_RATIO))]
Y_val_data = y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO + VALIDATION_RATIO))]
X_test_data = X[int(DATASET_SIZE*(TRAIN_RATIO + VALIDATION_RATIO)):]
Y_test_data = y[int(DATASET_SIZE*(TRAIN_RATIO + VALIDATION_RATIO)):]


# Use TensorFlow Dataset API to create Datasets for Training, Validation, and Testing 
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_data, Y_train_data))
train_dataset = train_dataset.shuffle(buffer_size = 10, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_data, Y_val_data))
val_dataset = val_dataset.shuffle(buffer_size = 5, reshuffle_each_iteration = True).batch(16).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_data, Y_test_data))
test_dataset = test_dataset.shuffle(buffer_size = 5, reshuffle_each_iteration = True).batch(16).prefetch(tf.data.AUTOTUNE)

for x,y in train_dataset: 
    print(x,y)
    break

# CREATE A SEQUENTIAL MODEL
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(8,)),
    normalizer,
    layers.Dense(128, activation = "relu"),
    layers.Dense(128, activation = "relu"),
    layers.Dense(128, activation = "relu"),
    layers.Dense(1)
])

# provides a summary of the model, the dimensions of its input and outputs of each layer 
model.summary()
tf.keras.utils.plot_model(model, to_file = "model.png", show_shapes = True, show_layer_names = True)

#configures the model to have a Means Squared Error loss function to optimize the data. 
model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)) 
history = model.fit(train_dataset, validation_data = val_dataset, epochs = 500, verbose = 0)

# PLOT LOSS OVER TIME 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL LOSS')
plt.ylabel('LOSS')
plt.xlabel('EPOCHS')
plt.legend(['train', "val_loss"])
plt.show()


# START TESTING MODEL 

y_true = list(Y_test_data[:,0].numpy())
y_pred = list(model.predict(X_test_data)[:, 0])

# PLOT PREDICTED vs. ACTUAL DATA OVER TIME 
# index = np.arange(100)
# plt.figure(figsize=(40,20))

# width = 0.4

# plt.bar(index, y_pred, width, label = 'Predicted Car Price')
# plt.bar(index + width, y_true, width, label = 'Actual Car Price')
# plt.xlabel('Actual vs Predicted Prices')
# plt.ylabel('Car Price Prices')
# plt.title('Bar Graph')

# plt.show()
