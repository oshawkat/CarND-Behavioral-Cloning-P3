# **Behavioral Cloning**

This aims to construct a neural network that can drive a car around a simulator with minimal training data.  The End-to-End model takes in an image from the simulator, representing a camera view from above the car, and outputs the appropriate steering angle to stay on the track

The primary steps of this project are:
* Build a neural network in Keras that predicts steering angles from images
* Enhance the available training with data augmentation
* Train and validate the model with a training and validation set
* Test that the model successfully drives around Track One without leaving the road

[//]: # (Image References)

[image1]: ./output/training_data_sample.png "Training Data Sample"
[image2]: ./output/data_histogram.png "Data Histogram"
[image3]: ./output/data_augmentation.png "Data Augmentation"
[image4]: ./output/model_architecture.png "Model Diagram"
---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

While I experimented with models of varying architecture and complexity, I settled on a fairly simple one that was able to perform just as well (and sometimes better than) the others.  It consists of a Normalization layer followed by Cropping to remove parts of the top and bottom of the image.  There is a single convolutional layer with 5x5 filters to produce a depth of 20.  Three fully connected layers follow the RELU activation and Max Pooling layer

#### 2. Attempts to reduce overfitting in the model

Though the more complex models I experimented with required changes to reduce overfitting (eg dropout, L2 regularization), this simple model did not require it for the 5 epochs on which it was trained.

#### 3. Model parameter tuning

The model used an Adam optimizer, using the default initial rate of 0.001, so the learning rate was not tuned manually

#### 4. Appropriate training data

The model was trained on the pre-recorded data provided by Udacity, which consists of the car going around Track One, in the counter-clockwise direction, keeping to the center of the lane.

I performed some additional data augmentation to expand both the quantity and diversity of the training data.  Side camera images, from the provided data, are also utilized by modifying their steering angles

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My approach is to start simple and build up from there.  To that end, my original model was as basic as possible - just a single neuron (with normalized input).  Training is performed on only the center camera images from the Udacity provided dataset.  Driving performance was understandably poor

Next, I increased the complexity of my model architecture by including two convolutional layers followed by three dense layers.  Training on the same data, performance improved significantly in the nominal case but, if the car neared a road boundary, it would not take any corrective action, resulting in the car completely leaving the driving lane

The prior result is understandable as the model was never exposed to any recovery scenarios; the training data only included images of the car in the center of the lane.  To make the training data more comprehensive, I started to use the side camera data, left and right, from the Udacity dataset.  These images do not have steering angles of their own so I estimated them based on the center-camera steering angle by adding or subtracting 0.25, found through empirical testing

The Udacity dataset consists of the car driving in a counter-clockwise motion around the track, which instills a left bias on the steering angle.  To counteract this, I augmented the dataset with horizontally flipped versions, with steering angles to match, for each original image.

Image cropping is used to remove extraneous parts of the image.  Given the 320 x 160 images in the Udacity dataset, I cut off the top 70px and bottom 20px as those regions of the image consisted mostly of sky and foliage or the car hood, respectively

I implemented a data generator to enable additional data augmentation.  Without the generator, training the model on anything much larger than the horizontal-flip augmented dataset would crash due to running out of memory.

I also experimented with much larger and more complex network architectures, including one inspired by the *[End to End Learning for Self-Driving Cars]*(http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper.  Their 5 layer CNN ultimately proved ineffective for my task as the car generally ended up under-steering in the curves.  Liberal application of dropout and L2 regularization helped to reduce overfitting but training times were still quite long (~4 minutes per epoch).  Both the training and validation accuracy, even without any overfitting prevention techniques, were much higher than even a single layer CNN.

#### 2. Final Model Architecture

In the end, I returned to a simple, single layer CNN architecture as it provided similar or better driving performance relative to other tested architectures while being the simplest and fastest to train (~42 s/epoch).  No dropout or regularization was required to prevent overfitting

**TODO** add model diagram

![alt text][image4]

#### 3. Creation of the Training Set & Training Process

I did not record any new driving data in the simulator and instead chose to use the Udacity provided dataset along with my own augmentation.  20% of the data was reserved for the validation set to ensure that the models do no overfit.  A sample of the original training data, along with steering angles, is shown below

![alt text][image1]

As noted above, I utilized the two side-camera images, along with estimated corresponding steering angles, to effectively triple the size of the dataset and provide recovery data.  Similarly, horizontally flipping the images doubled the dataset size and eliminated the left steering bias inherent in driving counter-clockwise around a loop.  A histogram of steering angles in this dataset is provided below, showcasing a wider and more symmetric distribution.  Further augmentation required the use of a data generator so the machine would not run out of memory while training.

![alt text][image2]

I tested additional augmentation techniques, shown below, to increase the diversity of the training dataset and better mimic scenarios that may occur on other tracks or environments (eg shadows).  Translating the images also provided a wider range of steering angles on which to train, reducing the bias of the model to drive straight ahead.  In the end, I found that these augmentations did not generally improve model performance and, more often than not, significantly reduced it.  It may be possible to better leverage these additions, which are still included in the code in [augment.py](./augment.py), with more complex networks or further turning of the augmentations themselves (eg translation steering angle correction)

![alt text][image3]

All data is shuffled between epochs.  The optimal number of epochs to train varied widely based on model architecture, data augmentation, and regularization techniques.  5 epochs was sufficient for most models but complex ones that relied on more data augmentation and regularization required more.  Additional training would either overfit the training dataset (better training accuracy than validation) or not result in any significant improvement in model performance