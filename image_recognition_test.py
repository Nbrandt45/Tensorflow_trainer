import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.datasets.samples_generator import make_blobs
from tensorflow.keras.utils import to_categorical
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

#print(input_test_data)

#prediction_img = model.predict([[input_test_data]])

input_img = (np.expand_dims(input_test_data, 0))


input_data_img = np.array([[input_test_data]])

#prediction_img = model.predict([input_test_data])
#prediction_img = model.predict([input_data_img])

train_size = 80

data = np.load('data_features.npy')

label = np.load('labels_list.npy')

data = data/255.0

# generate 2d classification dataset
data, label = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
# encode output variable
label = to_categorical(label)


# train on 10 images

testX = data[train_size:, :]
testy = label[train_size:]



_, test_acc = model.evaluate(testX, testy, verbose=0)

print('Test Accuraccy: %.3f' % test_acc)

#print(predict)

#print(categories[int(predict[0][0])])








