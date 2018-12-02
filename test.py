from matplotlib import pyplot as plt
import make_data
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import *
from keras.utils import np_utils
import numpy as np


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(30,125,3)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1215, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1215, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#model lears to predict one specific integer, and nothing else. 
prediction = model.predict_generator(make_data.generator_wow_hp_images("full"), steps=1, verbose=1) 
for i in prediction:
	print(np.argmax(i))

exit()
model.fit_generator(make_data.generator_wow_hp_images("full"), 
	verbose=1, steps_per_epoch=500, epochs=1, validation_data=make_data.generator_wow_hp_images("full"), validation_steps=50, shuffle=True)
prediction = model.predict_generator(make_data.generator_wow_hp_images("full"), steps=1, verbose=1) 
# for i in prediction:
# 	print(np.argmax(i))
print(np.argmax(prediction[0]))
print("length of predictions = {}".format(len(prediction)))

# prediction = model.predict(make_data.predict_wow_hp(), batch_size=1, verbose=1)
# print(prediction)