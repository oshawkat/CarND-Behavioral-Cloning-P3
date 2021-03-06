import os
import math
import csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
import augment


# Set Save paths for output (and create directory as needed)
output_dir = 'output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load recorded driving data from the simulator
# This is based on the code found in the Udacity Self-driving
# Nanodegree lecture on behavior cloning
# Extract the csv with dataset information into a more usable list
csv_path = './data/driving_log.csv'
local_img_path = './data/IMG/'
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
camera_correction = 0.25  # Steering angle correction for non-center images
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
train_generator = augment.generator(train_lines, batch_size=batch_size,
                                    augment=True)
validation_generator = augment.generator(validation_lines,
                                         batch_size=batch_size)

# Save a visualization of the dataset (random images sample of images)
augment.save_dataset_visual(train_lines, output_dir)
print("Saved sample training set visualization")

# Save a visualization of sample data augmentation
augment.save_augmentation_visual(train_lines, output_dir)
print("Saves data augmentation example")

# Create a very basic model to output steering angle
model = Sequential()
# Normalize the data and mean shift to zero
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Crop top and bottom of image
top_crop_px = 70
bot_crop_px = 20
model.add(Cropping2D(cropping=((top_crop_px, bot_crop_px), (0, 0))))
# Add a single convolutional layer with max pooling and dropout
drop_rate = 0.2
pool_size = 2
num_filters = 20
filter_size = 5
model.add(Conv2D(num_filters, filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
# Fully Connected layers
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# Train the newly constructed model
epochs = 3
model.fit_generator(train_generator,
                    steps_per_epoch=math.ceil(len(train_lines) / batch_size),
                    validation_data=validation_generator,
                    validation_steps=math.ceil(len(validation_lines) /
                                               batch_size),
                    epochs=epochs, verbose=1)

model.save(output_dir + 'model.h5')
print("Model saved successfully: " + output_dir + 'model.h5')
