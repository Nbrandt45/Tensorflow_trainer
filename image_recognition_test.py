import cv2
import os
import tensorflow as tf
from keras.models import load_model


data_dir = r".\test_images"
types = ["fork_images", "spoon_images", "knife_images"]

categories = ["Fork", "Spoon", "Knife"]


def load_data(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array


model = tf.keras.models.load_model('image_recognition.h5')

path = os.path.join(data_dir, "fork.jpg")

print(load_data(path))

predict = model.predict([load_data(path)])

print(predict)

model.summary()







