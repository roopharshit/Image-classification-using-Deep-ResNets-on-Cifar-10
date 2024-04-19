import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:

        # Resize the image to add four extra pixels on each side.

        image = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)

        # Randomly crop a [32, 32] section of the image.

        crop_start_height = np.random.randint(0, 9)
        crop_start_width = np.random.randint(0, 9)
        image = image[crop_start_height:crop_start_height+32, crop_start_width:crop_start_width+32, :]

        # Randomly flip the image horizontally.

        if np.random.rand() < 0.5:
            image = np.fliplr(image)

    # Subtract off the mean and divide by the standard deviation of the pixels.

    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    image = (image - mean) / std

    return image