import numpy as np
import tensorflow as tf

new_model = tf.keras.models.load_model("fnn_handwritten_digit_classfication.h5")
new_model.summary()
new_model.predict()