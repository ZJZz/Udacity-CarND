import os
import csv
from scipy import ndimage

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(len(train_samples))
print(len(validation_samples))

import cv2
import numpy as np
import sklearn
from random import shuffle

def generator(samples, batch_size=32, validation = False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                current_path = '../data/IMG/' + filename
                
                # regular image
                center_image = ndimage.imread(current_path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                if not validation:
                    # flipped image
                    image_flipped = np.fliplr(center_image)
                    measurement_flipped = -center_angle
                    images.append(image_flipped)
                    angles.append(measurement_flipped)
                
                    # create adjusted steering measurements for the side camera images
                    correction = 0.2 # this is a parameter to tune
                    steering_left = center_angle + correction
                    steering_right = center_angle - correction
                
                    # read in images from center, left and right cameras
                    source_path_left = batch_sample[1]
                    filename_left = source_path_left.split('/')[-1]
                    current_path_left = '../data/IMG/' + filename_left
                    image_left = ndimage.imread(current_path_left)
                    images.append(image_left)
                    angles.append(steering_left)
                
                    source_path_right = batch_sample[2]
                    filename_right = source_path_right.split('/')[-1]
                    current_path_right = '../data/IMG/' + filename_right
                    image_right = ndimage.imread(current_path_right)
                    images.append(image_right)
                    angles.append(steering_right)
                
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32, validation=True )


# Model

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D, Convolution2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))
model.summary()

model.compile(loss='mse', optimizer='adam')

# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=15)
# model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=15, verbose=1)

# model.fit_generator(train_generator,samples_per_epoch= len(train_samples), 
#                     validation_data=validation_generator,nb_val_samples=len(validation_samples),
#                     epochs=3)

model.fit_generator(train_generator,steps_per_epoch = len(train_samples), 
                    validation_data=validation_generator,validation_steps=len(validation_samples),
                    epochs=3)

model.save('PiloNet_model_5Conv_Generator.h5')