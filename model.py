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
print(str(line_count) + " lines read from the CSV")

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
print("Loaded dataset has " + str(len(images)) + " examples")
if len(measurements) != len(images):
    print("WARNING - there is a mismatch in the number of training images \
         and labels")

# Convert training data to numpy arrays for use with Keras/TensorFlow
X_train = np.array(images)
y_train = np.array(measurements)
print("Data set converted to numpy arrays")

# Create a very basic model to output steering angle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D

model = Sequential()
# Normalize the data and mean shift to zero
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=.2, shuffle=True, nb_epoch=7)

model.save('network.h5')