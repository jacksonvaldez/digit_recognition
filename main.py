import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from mnist_data.load import load_mnist




# pixels is a 1D numpy array with 784 pixels, each being a byte
def save_image(pixels):
    img = pixels.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.savefig('mnist_digit.png')
    return


# Load the MNIST dataset
images_test, labels_test = load_mnist('mnist_data', kind='t10k')
save_image(images_test[0])


print('-------------- TESTING RESULTS --------------')
weights1 = np.load('trained_params/weights1.npy')
weights2 = np.load('trained_params/weights2.npy')
biases1 = np.load('trained_params/biases1.npy')
biases2 = np.load('trained_params/biases2.npy')

neural_net = NeuralNetwork(weights1, weights2, biases1, biases2)
print(neural_net.testing_accuracy(images_test, labels_test), 'out of 10,000 testing examples correct!')