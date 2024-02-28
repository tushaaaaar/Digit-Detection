import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# first we need to train the model using the inbuilt dataset of tensorflow library

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

# saving the trained model
model.save('handwritten.model')

#========================================================================

# opening the trained model
model = tf.keras.models.load_model('handwritten.model')

# # checking the accuracy of the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: ", accuracy*100, "%")

#========================================================================
# # as the model has alrady been trained we can use it in other projects too
# # using the trained model to detect digits from images
image_number = 1
# checking if the files exist in the following folder
while os.path.isfile(f"Samples/digit{image_number}.png"):
    try:
        # reading the images 
        img = cv2.imread(f"Samples/digit{image_number}.png")[:,:,0]
        # converting them into a numpy array
        img = np.invert(np.array([img]))
        # predicting the digit
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1


# as not all models are 100% accurate, this model too have some inaccuracy
        