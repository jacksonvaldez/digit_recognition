import numpy as np
import pdb

# Neural network layers: 784(input) - 85(hidden) - 10(output)
class NeuralNetwork:
    # CREATE Neural Network and set up parameters

    def __init__(self, weights1, weights2, biases1, biases2):
        self.weights1 = weights1
        self.weights2 = weights2
        self.biases1 = biases1
        self.biases2 = biases2
        return

    def cost(self, images, labels):
        assert len(images) == len(labels)
        length = len(labels)
        indices = np.random.permutation(np.arange(length))
        cost = 0

        for training_example_index in indices:
            query = self.query(images[training_example_index])
            desired_output = np.full(10, 0, dtype=np.float64).reshape(10, 1)
            desired_output[labels[training_example_index]] = 1

            cost -= (desired_output * np.log(query[4])).sum() # Cross Entropy Loss

        cost /= length
        return round(cost, 5)
        
    def softmax(self, x):
        # Subtract the maximum value from each element to avoid overflow
        x = x - np.max(x)
        # Compute the exponentials of each element
        exp_x = np.exp(x)
        # Normalize by dividing each row by the sum of its elements
        return exp_x / np.sum(exp_x)

    def reLU(self, x):
        return np.maximum(0, x)

    def reLU_derivative(self, x):
        return np.where(x <= 0, 0, 1)


    # TRAIN the model (learning, backward propagation). Creates the most optimized sets of weights and biases
    def train(self, images_train, labels_train, learn_rate):

        assert len(images_train) & len(labels_train) == 60000 # The number of training images and labels should be 60000

        indices = np.random.permutation(np.arange(60000))
        progress = 0

        for training_example_index in indices:
            query = self.query(images_train[training_example_index])
            desired_output = np.full(10, 0, dtype=np.float64).reshape(10, 1)
            desired_output[labels_train[training_example_index]] = 1

            term1 = query[4] - desired_output
            term2 = query[2]
            term3 = 1
            term4 = self.weights2
            term5 = self.reLU_derivative(query[1])
            term6 = query[0]
            term7 = 1

            # term1, term2
            weights2_gradient = term2.reshape(1, 85) * term1
            assert weights2_gradient.shape == self.weights2.shape
            

            # term1, term4, term5, term6
            weights1_gradient = term1 * term4 * term5.reshape(1, 85)
            weights1_gradient = weights1_gradient.reshape(10, 85, 1) * term6.reshape(1, 1, 784)
            weights1_gradient = weights1_gradient.sum(axis=0)
            assert weights1_gradient.shape == self.weights1.shape

            # term1, term3
            biases2_gradient = term1 * term3
            assert biases2_gradient.shape == self.biases2.shape

            #term1, term4, term5, term7
            biases1_gradient = term1 * term4 * term5.reshape(1, 85) * term7
            biases1_gradient = biases1_gradient.sum(axis=0).reshape(85, 1)
            assert biases1_gradient.shape == self.biases1.shape

            self.weights1 -= learn_rate * weights1_gradient
            self.weights2 -= learn_rate * weights2_gradient
            self.biases1 -= learn_rate * biases1_gradient
            self.biases2 -= learn_rate * biases2_gradient
            
            progress += 1
            print(f'Progress: {progress}/60000', end='\r')

        print('')
        return

    # USE neural network to make predictions (forward propagation). Takes in the pixels of an image and creates a prediction of what the digit is.
    def query(self, pixels):

        # Compute the 85 neuron values of the hidden layer 'h'
        unactive_h = self.weights1 * pixels.reshape(1, len(pixels))
        unactive_h = np.sum(unactive_h, axis=1).reshape(85, 1)
        unactive_h = unactive_h + self.biases1
        active_h = self.reLU(unactive_h) # Uses ReLU(Rectified Linear Unit) to create activated neurons.

        # Compute the 10 neuron values of the output layer 'o'
        unactive_o = self.weights2 * active_h.reshape(1, 85)
        unactive_o = np.sum(unactive_o, axis=1).reshape(10, 1)
        unactive_o = unactive_o + self.biases2
        active_o = self.softmax(unactive_o) # Uses probability to compute activated neurons of output layer

        return pixels, unactive_h, active_h, unactive_o, active_o

    # Tests the accurancy of the trained weights and biases using testing data
    def accuracy(self, images, labels):
        assert len(images) == len(labels)

        accuracy = 0

        for testing_example_index in range(len(images)):
            query = self.query(images[testing_example_index])
            if np.argmax(query[4]) == labels[testing_example_index]:
                accuracy += 1

        accuracy = round(((accuracy / len(labels)) * 100), 2) # Turn accuracy into a percentage and round to 1 decimal place

        return accuracy
