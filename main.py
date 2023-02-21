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
def save_image(pixels):
    img = pixels.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.savefig('mnist_digit.png')
    return



def probability(x):
    sum = np.sum(x)
    return x / sum

def probability_derivative(x):
    sum = np.sum(x)

    numerator = sum - x
    denominator = sum * sum
    result = numerator / denominator
    return result




def reLU(x):
    return np.maximum(0, x)

def reLU_derivative(x):
    return np.where(x <= 0, 0, 1)    



# Neural network layers: 784(input) - 16(hidden) - 10(output)
class Neural_Network:
    # CREATE Neural Network and set up parameters

    def __init__(self):
        # Creates a random set of weights
        self.weights1 = np.random.uniform(-0.5, 0.5, (16, 784)) # Weights connecting the input layer and the hidden layer (16 x 784)
        self.weights2 = np.random.uniform(-0.5, 0.5, (10, 16)) # Weights connecting the hidden layer and the output layer (10 x 16)

        # Creates a set of biases, all 0 to start
        self.biases1 = np.full(16, 0).reshape(16, 1) # Biases connecting the input layer and the hidden layer (16 x 1)
        self.biases2 = np.full(10, 0).reshape(10, 1) # Biases connecting the hidden layer and the output layer (10 x 1)
        return

    # TRAIN the model (learning, backward propagation). Creates the most optimized sets of weights and biases
    def train(self, images_train, labels_train, learn_rate, a):

        assert len(images_train) == len(labels_train) # The number of training images should match the number of labels for all the images

        for training_example_index in range(len(images_train)):
            query = self.query(images_train[training_example_index])
            desired_output = np.full(10, 0).reshape(10, 1)
            desired_output[labels_train[training_example_index]] = 1

            print('---------------------------------') if a == False else ''
            term1 = 2 * (query[4] - desired_output)
            term2 = probability_derivative(query[3])
            term3 = query[2].reshape(1, 16)
            print('term1 shape: ', term1.shape) if a == False else ''
            print('term2 shape: ', term2.shape) if a == False else ''
            print('term3 shape: ', term3.shape) if a == False else ''
            weights2_gradient = term3 * term2 * term1
            print('---------------------------------') if a == False else ''


            term1 = query[0].reshape(1, 1, 784)
            term2 = reLU_derivative(query[1]).reshape(1, 16)
            term3 = self.weights2
            term4 = probability_derivative(query[3])
            term5 = 2 * (query[4] - desired_output)
            print('term1 shape: ', term1.shape) if a == False else ''
            print('term2 shape: ', term2.shape) if a == False else ''
            print('term3 shape: ', term3.shape) if a == False else ''
            print('term4 shape: ', term4.shape) if a == False else ''
            print('term5 shape: ', term5.shape) if a == False else ''
            weights1_gradient = (term5 * term4 * term3 * term2).reshape(10, 16, 1)
            weights1_gradient = weights1_gradient * term1
            weights1_gradient = np.sum(weights1_gradient, axis=0)
            print('---------------------------------') if a == False else ''


            term1 = 2 * (query[4] - desired_output)
            term2 = probability_derivative(query[3])
            print('term1 shape: ', term1.shape) if a == False else ''
            print('term2 shape: ', term2.shape) if a == False else ''
            biases2_gradient = term2 * term1
            print('---------------------------------') if a == False else ''


            term1 = reLU_derivative(query[1]).reshape(1, 16)
            term2 = self.weights2
            term3 = probability_derivative(query[3])
            term4 = 2 * (query[4] - desired_output)
            print('term1 shape: ', term1.shape) if a == False else ''
            print('term2 shape: ', term2.shape) if a == False else ''
            print('term3 shape: ', term3.shape) if a == False else ''
            print('term4 shape: ', term4.shape) if a == False else ''
            biases1_gradient = term4 * term3 * term2 * term1
            biases1_gradient = np.sum(biases1_gradient, axis=0).reshape(16, 1)
            print('---------------------------------') if a == False else ''


            # Update weights and biases
            self.weights1 = self.weights1 - learn_rate * weights1_gradient
            self.weights2 = self.weights2 - learn_rate * weights2_gradient
            self.biases1 = self.biases1 - learn_rate * biases1_gradient
            self.biases2 = self.biases2 - learn_rate * biases2_gradient
            if a == False:
                return
        return

    # USE neural network to make predictions (forward propagation). Takes in the pixels of an image and creates a prediction of what the digit is.
    def query(self, pixels):

        # Compute the 16 neuron values of the hidden layer 'h'
        unactive_h = self.weights1 * pixels.reshape(1, len(pixels))
        unactive_h = np.sum(unactive_h, axis=1).reshape(16, 1)
        unactive_h = unactive_h + self.biases1
        active_h = reLU(unactive_h) # Uses ReLU(Rectified Linear Unit) to create activated neurons.

        # Compute the 10 neuron values of the output layer 'o'
        unactive_o = self.weights2 * active_h.reshape(1, 16)
        unactive_o = np.sum(unactive_o, axis=1).reshape(10, 1)
        unactive_o = unactive_o + self.biases2
        active_o = probability(unactive_o) # Uses probability to compute activated neurons of output layer

        return pixels, unactive_h, active_h, unactive_o, active_o

    # Tests the accurancy of the trained weights and biases using testing data
    def test_model(images_test, labels_test):
        return



# Load the MNIST dataset
images_train, labels_train = load_mnist('mnist_data', kind='train')
images_test, labels_test = load_mnist('mnist_data', kind='t10k')

pixels = images_test[485] # Gets the pixel values of the 487th image in the testing set
save_image(pixels) # Saves the 69th image of the testing set as mnist_digit.png


neural_net = Neural_Network()
query = neural_net.query(pixels)
print('Untrained Output')
print(query[4])
print(query[4].sum()) # should equal 100

print('Trained Output')
neural_net.train(images_train, labels_train, 0.001, True)
query = neural_net.query(pixels)
print(query[4])
print(query[4].sum()) # should equal 100