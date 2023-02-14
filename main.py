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


# This function takes in an input x, which is a numpy array of real numbers, and returns the softmax of x. The softmax of x is a numpy array with the same shape as x, where each element is the corresponding softmax value.
def softmax(x):
    # Scales the values down to a range from -1 to 1 but still proportional so there is no infinity error
    max_val = np.linalg.norm(x, ord=np.inf)
    normalized = x / max_val
    x = np.clip(normalized, -1, 1)


    # Calculates softmax
    result = np.exp(x)/np.exp(x).sum()

    return(result * 100) # Returns as percentages so output is easier to read.

# This function takes in an input x, which is a numpy array of real numbers, and returns the softmax of x. The reLU of x is a numpy array with the same shape as x, where each element is the corresponding reLU value.
def reLU(array):
    return np.maximum(0, array)




# Neural network layers: 784(input) - 16(hidden) - 10(output)
class Neural_Network:
    # CREATE Neural Network and set up parameters

    def __init__(self):
        # Creates a random set of weights
        self.w_i_h = np.random.uniform(-0.5, 0.5, (16, 784)) # Weights connecting the input layer 'i', and the hidden layer 'h'
        self.w_h_o = np.random.uniform(-0.5, 0.5, (10, 16)) # Weights connecting the hidden layer 'h', and the output layer 'o'

        # Creates a set of biases, all 0 to start
        self.b_i_h = np.full(16, 0).reshape(16, 1) # Creates an array of 16 elements, each with the value of 0
        self.b_h_o = np.full(10, 0).reshape(10, 1) # Creates an array of 10 elements, each with the value of 0
        return

    # TRAIN the model (learning, backward propagation). Creates the most optimized sets of weights and biases
    def train(images_train, labels_train):
        
        return

    # USE neural network to make predictions (forward propagation). Takes in the pixels of an image and creates a prediction of what the digit is.
    def query(self, pixels):

        # Reshape certain matrices so they can be added or multiplied together
        pixels = pixels.reshape(1, len(pixels))


        # Compute the 16 neuron values of the hidden layer 'h'
        unactive_h = self.w_i_h * pixels
        unactive_h = np.sum(unactive_h, axis=1).reshape(16, 1)
        unactive_h = unactive_h + self.b_i_h
        active_h = reLU(unactive_h) # Uses ReLU(Rectified Linear Unit) to create activated neurons.

        # Compute the 10 neuron values of the output layer
        active_h = active_h.reshape(1, 16)
        unactive_o = self.w_h_o * active_h
        unactive_o = np.sum(unactive_o, axis=1).reshape(10, 1)
        active_o = softmax(unactive_o) # Uses a Softmax function to turn the 10 neuron values to probabilites ranging from 0 - 1
        return active_o



# Load the MNIST dataset
images_train, labels_train = load_mnist('mnist_data', kind='train')
images_test, labels_test = load_mnist('mnist_data', kind='t10k')

pixels = images_test[487] # Gets the pixel values of the 487th image in the testing set
save_image(pixels) # Saves the 69th image of the testing set as mnist_digit.png


neural_net = Neural_Network()
results = neural_net.query(pixels)
print(results)
print(results.sum())


