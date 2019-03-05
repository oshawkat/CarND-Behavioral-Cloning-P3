import csv
import cv2
import numpy as np
from scipy import ndimage

# Load recorded driving data from the simulator
# This is based on the code found in the Udacity Self-driving
# Nanodegree lecture on behavior cloning

# Extract the csv to a more usable list
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

# Construct lists of corresponding image data and steering angle measurements
local_img_path = '../data/IMG/'
images = []
measurements = []
for line in lines:
    # File paths will likely differ from the data collection machine
    # so update to local paths
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = local_img_path + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
print("Loaded dataset examples: " + str(len(images)))
if len(measurements) != len(images):
    print("WARNING - there is a mismatch in the number of training images \
         and labels")

# Augment data by flipping image horizontally
# Should eliminate left or right bias from training data
augmented_images, augmented_measures = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measures.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measures.append(measurement * -1.0)

# Convert training data to numpy arrays for use with Keras/TensorFlow
X_train = np.array(augmented_images)
y_train = np.array(augmented_measures)
print("Data set converted to numpy arrays")

# Import required portions of Keras for building a CNN
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Create a very basic model to output steering angle
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
model.fit(X_train, y_train, validation_split=.2, shuffle=True, nb_epoch=7)

model.save('network.h5')
