import numpy as np
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, datasets

# image = load_img("6.PNG")
image_size = 28
x_image = tf.io.read_file('3.PNG')
decode_img = tf.image.decode_png(x_image, channels=1)
print(decode_img.shape)     # (28,28,1)
decode_img = tf.image.convert_image_dtype(decode_img, tf.float32)
test_img = tf.image.resize(decode_img, [image_size, image_size])
test_img = tf.reshape(test_img, [-1, image_size, image_size, 1])
test_img = tf.keras.utils.normalize(test_img, axis=1)
test_img = test_img.numpy()
test_label = np.array([3])
print(decode_img.shape)

plt.imshow(np.squeeze(decode_img), cmap=plt.cm.binary)
plt.show()


new_model = tf.keras.models.load_model("fnn_handwritten_digit_classfication.h5")
new_model.summary()
predictions = new_model.predict(test_img)
print(predictions)
print("the prediction result is ：{}".format(np.argmax(predictions[0])))

