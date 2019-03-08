import math
import csv
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# Required portions of Keras for building a CNN
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


def generator(lines, batch_size=32, augment=False):
    """Python data generator for feeding Keras network

    Generator provides a subset of the data every time it is iterated. Will
    also selectively apply data augmentation

    Input:
        lines: list of lists with each outer element corresponding to a list
            of image path and steering angle, respectively
        batch_size: how many examples from the dataset to generate at once
        augment: boolean representing whether or not to apply data
            augmentation.  This must be false for any validation or testing set

    Output:
        X_data: batch_size number of images in a numpy array
        y_data: corresponding batch_size number of labels for X_data, also as
            a numpy array

    """

    num_samples = len(lines)
    while 1:    # Loop generator indefinitely
        shuffle(lines)  # Shuffle data between epochs
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset: offset + batch_size]

            images = []
            steer_angles = []

            for line in batch_samples:
                image = ndimage.imread(line[0])
                steer_angle = line[1]

                # TODO: Add image augmentation
                # if augment:
                #     image, steer_angle = image_augment(image, steer_angle)

                images.append(image)
                steer_angles.append(steer_angle)

            # Convert lists to numpy arrays for use with Keras
            X_data = np.array(images)
            y_data = np.array(steer_angles)

            yield shuffle(X_data, y_data)


# Load recorded driving data from the simulator
# This is based on the code found in the Udacity Self-driving
# Nanodegree lecture on behavior cloning

# Extract the csv with dataset information into a more usable list
csv_path = '../data/driving_log.csv'
lines = []
line_count = 0
with open(csv_path) as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip column headings
    for line in csvreader:
        lines.append(line)
        line_count = line_count + 1
print("Lines read from the CSV file: " + str(line_count))

# Create a new list with a single image path and steering angle per line
# Original list had 3 images (center, left, right) and a single steer angle
local_img_path = '../data/IMG/'
camera_correction = 0.2  # Steering angle correction for non-center images
single_lines = []
for line in lines:
    center_path = local_img_path + line[0].split('/')[-1]
    left_path = local_img_path + line[1].split('/')[-1]
    right_path = local_img_path + line[2].split('/')[-1]

    center_angle = float(line[3])
    left_angle = center_angle + camera_correction
    right_angle = center_angle - camera_correction

    single_lines.extend([[center_path, center_angle], [left_path, left_angle],
                        [right_path, right_angle]])
print("Total images: " + str(len(single_lines)))

# Split training and validation data
train_lines, validation_lines = train_test_split(single_lines, test_size=0.2)

# Create generators to reduce memory requirements while training
batch_size = 32
train_generator = generator(train_lines, batch_size=batch_size, augment=True)
validation_generator = generator(validation_lines, batch_size=batch_size)

# Construct a CNN in Keras to output steering angle based on camera image
model = Sequential()
# Normalize the data and mean shift to zero
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Crop top and bottom of image
top_crop_px = 70
bot_crop_px = 20
model.add(Cropping2D(cropping=((top_crop_px, bot_crop_px), (0, 0))))
# Add two convolutional layers
pool_size = 2
num_filters = 10
filter_size = 5
model.add(Conv2D(num_filters, filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(num_filters, filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
# Fully connected layers for output
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# Train the newly constructed model
epochs = 5
model.fit_generator(train_generator,
                    steps_per_epoch=math.ceil(len(train_lines) / batch_size),
                    validation_data=validation_generator,
                    validation_steps=math.ceil(len(validation_lines) /
                                               batch_size),
                    epochs=epochs, verbose=1)

model.save('network.h5')
