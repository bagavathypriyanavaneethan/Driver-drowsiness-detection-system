# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:43:28 2020

@author: Bagavathi Priya
"""



from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

classifier=Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (24, 24, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'D:/Education/Drowsiness detection/dataset/train',
                                                 target_size = (24, 24),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'D:/Education/Drowsiness detection/dataset/test',
                                            target_size = (24, 24),
                                            batch_size = 32,
                                            class_mode = 'binary')
#print(np.shape(training_set))

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)


classifier.save('D:/Education/Drowsiness detection/model2.h5', overwrite=True)

from keras.models import load_model

model1 = load_model('model.h5')

model1.summary()

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

image=cv2.imread("D:/Education/Drowsiness detection/dataset/test/open/s0001_02237_0_0_1_0_0_01.png")

image = cv2.resize(image, (24, 24))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


value=model1.predict(image)
print(value)














