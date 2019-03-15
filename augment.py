import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

                # Apply data augmentation as necessary
                if augment:
                    image, steer_angle = random_horizontal_flip(
                        image, steer_angle)
                    # image, steer_angle = random_all(image, steer_angle)
                    # image = random_shadows(image)
                    # image = random_gaussian(image)

                images.append(image)
                steer_angles.append(steer_angle)

            # Convert lists to numpy arrays for use with Keras
            X_data = np.array(images)
            y_data = np.array(steer_angles)

            yield shuffle(X_data, y_data)


def random_horizontal_flip(img, angle, prob=0.5):
    """Randomly flip an image horizontally

    Input:
        img: image to flip
        angle: steering angle corresponding to image
        prob: probability of flipping the image

    Output:
        img, angle

    """

    if np.random.rand() <= prob:
        img = cv2.flip(img, 1)
        angle = -angle

    return img, angle


def random_shadows(img, prob=0.5, max_scale=0.8):
    """Add random shadows or highlights to an image

    Input:
        img: RGB image on which to add shadows
        prob: probability any shadows will be added to the image
        max_scale: maximum factor by which to adjust brightness in
            shadow/highlight region

    Output:
        image with random shadows added
    """

    # Leave the image unmodified, with some probability
    if np.random.rand() > prob:
        return img

    # Find random scale factor by which to adjust image brightness
    scale = (np.random.random() - 0.5) * (max_scale * 2) + 1.0

    # Generate random polygon to act as the shadow/highlight
    num_poly_points = 6  # Number of vertices for polygon shadow
    height, width, depth = img.shape
    mask = np.zeros((height, width), np.int32)   # Create 2D mask
    x_poly_points = np.reshape(np.random.randint(0, width, num_poly_points),
                                                (num_poly_points, 1))
    y_poly_points = np.reshape(np.random.randint(0, height, num_poly_points),
                                                (num_poly_points, 1))
    poly_pts = np.array([np.hstack((x_poly_points, y_poly_points))])
    mask = cv2.fillPoly(mask, poly_pts, 1.0)

    # Apply shadow/highlight in HSV space to mimic real light changes
    float_img = np.copy(img.astype(np.float32))
    hsv_img = cv2.cvtColor(float_img, cv2.COLOR_RGB2HSV)
    hsv_img[:, :, 2] = np.where(mask, hsv_img[:, :, 2] * scale,
                                hsv_img[:, :, 2])

    # Convert back to RGB color space
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    rgb_img[rgb_img > 255] = 255
    rgb_img[rgb_img < 0] = 0
    rgb_img = rgb_img.astype(np.uint8)

    return rgb_img


def random_brightness(img, max_scale=0.5):
    """Randomly adjust the brightness of the image

    Input:
        img: RGB image to adjust brightness on
        max_scale: maximum factor by which to adjust brightness

    Output:
        Brightness adjusted image

    """

    # Find random scale factor by which to adjust image brightness
    scale = (np.random.random() - 0.5) * (max_scale * 2) + 1.0

    # Adjust the brightness of the V channel of HSV to mimic daylight changes
    float_img = np.copy(img.astype(np.float32))
    hsv_img = cv2.cvtColor(float_img, cv2.COLOR_RGB2HSV)
    hsv_img[:, :, 2] = hsv_img[:, :, 2] * scale

    # Convert back to RGB color space
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    rgb_img[rgb_img > 255] = 255
    rgb_img[rgb_img < 0] = 0
    rgb_img = rgb_img.astype(np.uint8)

    return rgb_img


def random_translation(img, angle, max_tx=80, max_ty=20, x_steer_corr_px=0.01,
                       y_steer_corr_px=0.001):
    """Randomly shift image in view

    Input:
        img: image to translate
        angle: steering angle corresponding to image
        max_tx: maximum number of pixels to translate by in the horizontal
            direction
        max_ty: maximum number of pixels to translate by in the vertical
            direction
        x_steer_corr_px: Amount to adjust the steering angle per pixel of
            horizontal translation
        y_steer_corr_px: Amount to adjust the steering angle per pixel of
            vertical translation

    Output:
        translated image with reflected border

    """

    # Shift image content and fill with reflected border
    rows, cols, depth = img.shape
    if max_tx == 0:
        tx = 0
    else:
        tx = np.random.randint(-max_tx, max_tx)
    if max_ty == 0:
        ty = 0
    else:
        ty = np.random.randint(-max_ty, max_ty)

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    out = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # Adjust steering angle based on horizontal translation
    angle = angle + (tx * x_steer_corr_px) + (ty * y_steer_corr_px)

    return out, angle


def random_gaussian(img, mu=0.0, sigma=4.0):
    """Add random Gaussian noise to an image

    Input:
        img: image to add noise to
        mu: mean of the Gaussian noise to add to image
        sigma: deviation of the Gaussian noise to add to image

    """

    out = np.copy(img.astype(np.float))
    rows, cols, depth = img.shape
    noise = np.random.normal(mu, sigma, (rows, cols))
    for dim in range(depth):
        out[:, :, dim] = img[:, :, dim] + noise
    out[out > 255] = 255
    out[out < 0] = 0
    out = out.astype(np.uint8)

    return out


def random_all(image, angle):
    """Perform all image randomizations

    Apply rotation, translation, scale, salt & pepper,
    Gaussian, and brightness randomization to the input image

    Input:
        image: image to translate
        angle: steering angle corresponding to image

    Output:
        img: randomly transformed image of same shape as input img
        angle: angle corresponding to the newly transformed image
    """

    # Create a copy of the image to prevent changing the original
    img = np.copy(image)

    # Apply augmentations that can impact steering angle first
    y_steer_corr_px = 0.001
    x_steer_corr_px = 0.01   # Initial value taken from Vivek Yadav's blog:
    max_tx = 30
    max_ty = 20
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-
    # 496b569760a9
    img, angle = random_horizontal_flip(img, angle)
    img, angle = random_translation(img, angle, max_tx=max_tx, max_ty=max_ty,
                                    x_steer_corr_px=x_steer_corr_px,
                                    y_steer_corr_px=y_steer_corr_px)

    # Apply remaining augmentations
    img = random_gaussian(random_shadows(random_brightness(img)))

    return img, angle


def visualize_augmentation(image, angle):
    """Show data augmentation performed on an image

    Input:
        image: image to perform augmentation transforms on
        angle: steering angle corresponding to image
    Output:
        Pyplot graph with two columns, original image and transformed, with
            each row representing an example augmentation tranformation.
            Steering angle is printed above each image
    """

    # Create a copy of the image to prevent changing the original
    img = np.copy(image)

    cols = 2
    rows = 6
    fig_size = (7 * cols, 4 * rows)   # Figure width and height, in inches

    fig, ax = plt.subplots(rows, cols, figsize=fig_size)
    # Plot original images in the left column
    for idx in range(rows):
        ax[idx, 0].imshow(img)
        ax[idx, 0].set_title("Original, Angle = " + str(round(angle, 3)))
    # Horizontal Flip
    tmp_img, tmp_angle = random_horizontal_flip(img, angle, 1.0)
    ax[0, 1].imshow(tmp_img)
    ax[0, 1].set_title("Horizontal Flip, Angle = " + str(round(tmp_angle, 3)))
    # Translation
    tmp_img, tmp_angle = random_translation(img, angle)
    ax[1, 1].imshow(tmp_img)
    ax[1, 1].set_title("Translation, Angle = " + str(round(tmp_angle, 3)))
    # Gaussian Noise
    tmp_img = random_gaussian(img)
    ax[2, 1].imshow(tmp_img)
    ax[2, 1].set_title("Gaussian Noise, Angle = " + str(round(angle, 3)))
    # Shadows
    tmp_img = random_shadows(img, 1.0, 0.9)
    ax[3, 1].imshow(tmp_img)
    ax[3, 1].set_title("Shadows, Angle = " + str(round(angle, 3)))
    # Brightness
    tmp_img = random_brightness(img)
    ax[4, 1].imshow(tmp_img)
    ax[4, 1].set_title("Brightness, Angle = " + str(round(angle, 3)))
    # All Augmentation
    tmp_img, tmp_angle = random_all(img, angle)
    ax[5, 1].imshow(tmp_img)
    ax[5, 1].set_title("All Randomization, Angle = " +
                       str(round(tmp_angle, 3)))

    return fig


def save_augmentation_visual(lines, output_dir):
    """Save a sample of data augmentation to disk

    Input:
        lines: list of lists of size n x 2  with first column specifying image
            path and second column corresponding steering angle
        output_dir: where to save output visualization images
    Output:
        None
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_name = 'data_augmentation.png'
    idx = np.random.randint(0, len(lines))
    image = ndimage.imread(lines[idx][0])
    angle = lines[idx][1]
    fig = visualize_augmentation(image, angle)
    fig.savefig(output_dir + save_name, bbox_inches='tight')


def save_dataset_visual(lines, output_dir):
    """Visualize and save dataset

    Save a random sampling of dataset images and a histogram of label
    distributions

    Input:
        lines: list of lists of size n x 2  with first column specifying image
            path and second column corresponding steering angle
        output_dir: where to save output visualization images
    Output:
        None
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cols = 3
    rows = 3
    fig_size = (7 * cols, 4 * rows)     # Figure width and height, in inches
    # Random sample of images
    save_name = "training_data_sample.png"
    fig, ax = plt.subplots(rows, cols, figsize=fig_size)
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(0, len(lines))
            ax[row, col].imshow(ndimage.imread(lines[idx][0]))
            ax[row, col].set_title("Angle = " + str(round(lines[idx][1], 3)))
    plt.savefig(output_dir + save_name, bbox_inches='tight')
    # Distribution of steering angles
    save_name = "data_histogram.png"
    fig_size = (5, 3)       # Figure width and height, in inches
    num_bins = 100
    angles = np.array([line[1] for line in lines])
    hist, bins = np.histogram(angles, bins=num_bins)
    fig = plt.figure(figsize=fig_size)
    plt.bar(bins[:-1], hist)
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.title("Distribution of Steering Angles in Training Data")
    plt.savefig(output_dir + save_name, bbox_inches='tight')
