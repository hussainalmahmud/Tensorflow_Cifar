## load cifar 10 data from keras
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


# def read_data(dataset="cifar100", validation_split=0.1):
#     """Returns the CIFAR-10 dataset"""
#     if dataset == "cifar10":
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()

#         x_train = x_train.astype("float32") / 255.0
#         x_test = x_test.astype("float32") / 255.0

#         # Splitting out validation data from training data
#         validation_length = int(validation_split * len(x_train))
#         x_val, y_val = x_train[:validation_length], y_train[:validation_length]
#         x_train, y_train = x_train[validation_length:], y_train[validation_length:]

#     elif dataset == "cifar100":
#         # load cifar100
#         (x_train, y_train), (x_test, y_test) = cifar100.load_data()

#         x_train = x_train.astype("float32") / 255.0
#         x_test = x_test.astype("float32") / 255.0

#         # Splitting out validation data from training data
#         validation_length = int(validation_split * len(x_train))
#         x_val, y_val = x_train[:validation_length], y_train[:validation_length]
#         x_train, y_train = x_train[validation_length:], y_train[validation_length:]
#     else:
#         raise ValueError("dataset must be either 'cifar10' or 'cifar100'")

#     return x_train, y_train, x_val, y_val, x_test, y_test


def read_data(dataset="cifar100", validation_split=0.1):
    """Returns the CIFAR dataset"""
    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise ValueError("dataset must be either 'cifar10' or 'cifar100'")

    # Normalize pixel values to the range [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Compute mean and standard deviation
    mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
    
    # Apply normalization
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Splitting out validation data from training data
    validation_length = int(validation_split * len(x_train))
    x_val, y_val = x_train[:validation_length], y_train[:validation_length]
    x_train, y_train = x_train[validation_length:], y_train[validation_length:]

    # Apply the same normalization to validation data
    x_val = (x_val - mean) / std

    return x_train, y_train, x_val, y_val, x_test, y_test


def data_augmentation_generator():
    """Returns a data augmentation generator"""
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    return datagen
