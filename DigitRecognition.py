from keras.datasets import mnist
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

import numpy as np


(x_train, y_train),(x_test, y_test) = mnist.load_data()
# print(x_train[0])

plt.imshow(x_train[0])
plt.show()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
print(x_train[0])

# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.imshow(x_train[0])
# plt.show()



model = Sequential()
model.add(Conv2D(28, (3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(28, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[29]:


model.compile(optimizer='nadam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_split=0.3)


# In[5]:


# test = cv2.imread("/home/anupriya/Documents/Acads/DistributedSystem/eight.png")
# test = tf.keras.utils.normalize(test, axis=1)
# test = cv2.resize(test, (28, 28))
# plt.imshow(x_test[0])
# plt.show()


# test = np.array(test).reshape(-1, 28, 28, 1)
# test

plt.imshow(x_test[2])
plt.show()

test = x_test[2].reshape(1, 28, 28, 1)


prediction = model.predict(test)
np.argmax(prediction)

