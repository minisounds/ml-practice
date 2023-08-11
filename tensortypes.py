import tensorflow as tf
import numpy as np

tf.random.set_seed(5)

print(tf.random.uniform([3,3], 0, 15, dtype = tf.float16, seed = 10))
print(tf.random.uniform([3,3], 0, 15, dtype = tf.float16, seed = 10))

print(tf.random.uniform([3,3], 0, 15, dtype = tf.float16, seed=5))
print(tf.random.uniform([3,3], 0, 15, dtype = tf.float16, seed=5))

