import numpy as np

# Data used for training and testing: http://yann.lecun.com/exdb/mnist/
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = f'{path}/{kind}-labels-idx1-ubyte'
    images_path = f'{path}/{kind}-images-idx3-ubyte'

    # 'rb' argument stands for 'read binary'
    with open(labels_path, 'rb') as labels_file:
        # Extract data from the labels file and turn it into a numpy array
        # dtype argument: Data-type of the returned array elements. In this case, 'np.uint8' means an 8 bit integer or byte (max 255)
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as images_file:
        # Extract data from the images file and turn it into a numpy array
        # dtype argument: Data-type of the returned array elements. In this case, 'np.uint8' means an 8 bit integer or byte (max 255)
        # .reshape turn the images array into a 2D array. Each item in the array contains the 784 pixels of an image
        images = np.frombuffer(images_file.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels