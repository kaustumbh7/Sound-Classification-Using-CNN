#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 23:37:45 2018

@author: KJ
"""


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Classifier 1 trained to identify Chainsaw sounds

# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)



###############################################################################

# CLASSIFIER 2 trained to identify Gunshots sound

# Initialising the CNN
classifier_2 = Sequential()

# Convolution
classifier_2.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier_2.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier_2.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier_2.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier_2.add(Flatten())

# Full connection
classifier_2.add(Dense(output_dim = 128, activation = 'relu'))
classifier_2.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier_2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen_2 = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen_2 = ImageDataGenerator(rescale = 1./255)

training_set_2 = train_datagen_2.flow_from_directory('train_2',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set_2 = test_datagen_2.flow_from_directory('test_2',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier_2.fit_generator(training_set_2,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set_2,
                         nb_val_samples = 2000)




###############################################################################


###############################################################################

# Recording Test Sample
import pyaudio
import wave
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "test_file.wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print ("recording...")
frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print ("finished recording")
 
 
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()



# *******************************************************************************

# Creating graph for the test sample
import os
import pandas as pd
import librosa
import librosa.display
import glob 
import matplotlib.pyplot as plt

data, sampling_rate = librosa.load('test_file.wav')

fig = plt.figure(figsize=(12, 4))
fig.savefig('test.jpg')
graph_test = librosa.display.waveplot(data, sr=sampling_rate)



###############################################################################

# Making predictions on the test sample

import numpy as np
from keras.preprocessing import image
test_image1 = image.load_img('test.jpg', target_size= (64,64))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis=0)
result = classifier.predict(test_image1)

if result == 0:
    test_image2 = image.load_img('ttt2.jpg', target_size= (64,64))
    test_image2 = image.img_to_array(test_image2)
    test_image2 = np.expand_dims(test_image2, axis=0)
    result = classifier_2.predict(test_image2)
    if result == 1:
        result = 2
    
