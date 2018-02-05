'''MNIST Classification'''

import keras
from keras.layers import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.datasets import mnist
from keras.callbacks import History
from keras.utils import np_utils
from PIL import Image
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Data reading
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Sample Image
img = Image.fromarray(x_train[6], mode = None)
img.show()
x_train.shape[1]
# Procces variables
x_train = np.reshape(x_train , newshape=(-1,x_train.shape[1],x_train.shape[2],1))
x_test = np.reshape(x_test , newshape=(-1,x_train.shape[1],x_train.shape[2],1))
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#%%

img_height = 28
img_length = 28
chanels = 1
# Model
model = keras.models.Sequential()
model.add(Convolution2D(20,3,input_shape = (img_height, img_length, chanels)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(20,3,input_shape = (img_height, img_length, chanels)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(40, bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Nadam',
              metrics=['categorical_accuracy'])
# Train
history = History()
model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=100,
                    validation_split=0.1,
                    callbacks=[history])


plt.figure()
plt.plot(history.history['categorical_accuracy'], label = 'train')
plt.plot(history.history['val_categorical_accuracy'], label = 'validation')
plt.title('Categorical accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.title('Loss')
plt.legend()
plt.show()

# Predict some values
y_pred = model.predict(x_test, batch_size=1)

def categorical_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=-1) ==
                   np.argmax(y_pred, axis=-1))

categorical_accuracy(y_test, y_pred)
log_loss(y_test,y_pred)



img = Image.fromarray(x_test[0][:,:,0], mode = None)
np.argmax(y_test[0])
img.show()
