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

pixels = images_test[7623] # Gets the pixel values of the 487th image in the testing set
save_image(pixels) # Saves the 69th image of the testing set as mnist_digit.png


print('-------------- RESULT --------------')
neural_net = NeuralNetwork()
query = neural_net.query(pixels)
number = np.argmax(query[4])
print(query[4])
print('This number is a ', number, '!')
print('-------------- TEST RESULTS --------------')
print(neural_net.test_model(images_test, labels_test), 'out of 10,000 correct')



