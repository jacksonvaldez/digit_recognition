import numpy as np
import matplotlib.pyplot as plt





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





# pixels is a 1D numpy array with 784 pixels, each being a byte
def show_image(pixels):
    img = pixels.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.savefig('mnist_digit.png')
    return





class Neural_Network:
    # CREATE Neural Network and set up parameters
    def __init__():
        return

    # TRAIN the model (learning, backward propagation). Creates the most optimized sets of weights and biases
    def train():
        return

    # USE neural network to make predictions (forward propagation). Takes in the pixels of an image and creates a prediction of what the digit is.
    def query():
        return





# Load the MNIST dataset
images_train, labels_train = load_mnist('mnist_data', kind='train')
# images_test, labels_test = load_mnist('mnist_data', kind='t10k')


pixels = images_train[569]
show_image(pixels)