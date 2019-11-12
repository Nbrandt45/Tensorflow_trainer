import cv2
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt


data_dir = r".\test_images"
types = ["fork_images", "spoon_images", "knife_images"]

categories = ["Fork", "Spoon", "Knife"]


def load_data(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    #plt.imshow(new_array, cmap="gray")
    #plt.show()
    return new_array


model = tf.keras.models.load_model('image_recognition.h5')

path = os.path.join(data_dir, "fork.jpg")
input_test_data = load_data(path)

#model.summary()

print(input_test_data.shape)

img = (np.expand_dims(input_test_data, 0))

print(img.shape)

prediction_img = model.predict(img)

train_size = 80
data = np.load('data_features.npy')

label = np.load('labels_list.npy')

data = data/255.0

testX = data[train_size, :]
testy = label[train_size:]

testX = testX/255.0

#predict = model.predict([input_test_data])

#print(predict)

#print(categories[int(predict[0][0])])








