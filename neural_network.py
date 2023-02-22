import numpy as np

# Neural network layers: 784(input) - 16(hidden) - 10(output)
class NeuralNetwork:
    # CREATE Neural Network and set up parameters

    def __init__(self):

        return

    def probability(self, x):
        sum = np.sum(x)
        return x / sum

    def probability_derivative(self, x):
        sum = np.sum(x)

        numerator = sum - x
        denominator = sum * sum
        result = numerator / denominator
        return result

    def reLU(self, x):
        return np.maximum(0, x)

    def reLU_derivative(self, x):
        return np.where(x <= 0, 0, 1)



    # TRAIN the model (learning, backward propagation). Creates the most optimized sets of weights and biases
    def train(self, images_train, labels_train, learn_rate):

        assert len(images_train) == len(labels_train) # The number of training images should match the number of labels for all the images

        weights1 = np.load('trained_params/weights1.npy')
        weights2 = np.load('trained_params/weights2.npy')
        biases1 = np.load('trained_params/biases1.npy')
        biases2 = np.load('trained_params/biases2.npy')

        weights1_gradient_final = np.zeros([16, 784])
        weights2_gradient_final = np.zeros([10, 16])
        biases1_gradient_final = np.zeros([16, 1])
        biases2_gradient_final = np.zeros([10, 1])

        for training_example_index in range(len(images_train)):
            query = self.query(images_train[training_example_index])
            desired_output = np.full(10, 0).reshape(10, 1)
            desired_output[labels_train[training_example_index]] = 1


            term1 = 2 * (query[4] - desired_output)
            term2 = self.probability_derivative(query[3])
            term3 = query[2].reshape(1, 16)
            weights2_gradient = term3 * term2 * term1

            term1 = 2 * (query[4] - desired_output)
            term2 = self.probability_derivative(query[3])
            term3 = np.sum(weights2, axis=0).reshape(1, 16)
            term4 = self.reLU_derivative(query[1]).reshape(1, 16)
            term5 = query[0].reshape(784, 1)
            weights1_gradient = (term1 * term2 * term3 * term4).reshape(10, 16, 1)
            term5 = term5.reshape(1, 1, 784)
            weights1_gradient = weights1_gradient * term5
            weights1_gradient = np.sum(weights1_gradient, axis=0)

            term1 = 2 * (query[4] - desired_output)
            term2 = self.probability_derivative(query[3])
            biases2_gradient = term2 * term1

            term1 = 2 * (query[4] - desired_output)
            term2 = self.probability_derivative(query[3])
            term3 = np.sum(weights2, axis=0).reshape(1, 16)
            term4 = self.reLU_derivative(query[1]).reshape(1, 16)
            biases1_gradient = term4 * term3 * term2 * term1
            biases1_gradient = np.sum(biases1_gradient, axis=0).reshape(16, 1)


            weights1_gradient_final += weights1_gradient
            weights2_gradient_final += weights2_gradient
            biases1_gradient_final += biases1_gradient
            biases2_gradient_final += biases2_gradient

        weights1_gradient_final /= 60000
        weights2_gradient_final /= 60000
        biases1_gradient_final /= 60000
        biases2_gradient_final /= 60000

        weights1 = weights1 - learn_rate * weights1_gradient_final
        weights2 = weights2 - learn_rate * weights2_gradient_final
        biases1 = biases1 - learn_rate * biases1_gradient_final
        biases2 = biases2 - learn_rate * biases2_gradient_final

        return weights1, weights2, biases1, biases2

    # USE neural network to make predictions (forward propagation). Takes in the pixels of an image and creates a prediction of what the digit is.
    def query(self, pixels):
        weights1 = np.load('trained_params/weights1.npy')
        weights2 = np.load('trained_params/weights2.npy')
        biases1 = np.load('trained_params/biases1.npy')
        biases2 = np.load('trained_params/biases2.npy')

        # Compute the 16 neuron values of the hidden layer 'h'
        unactive_h = weights1 * pixels.reshape(1, len(pixels))
        unactive_h = np.sum(unactive_h, axis=1).reshape(16, 1)
        unactive_h = unactive_h + biases1
        active_h = self.reLU(unactive_h) # Uses ReLU(Rectified Linear Unit) to create activated neurons.

        # Compute the 10 neuron values of the output layer 'o'
        unactive_o = weights2 * active_h.reshape(1, 16)
        unactive_o = np.sum(unactive_o, axis=1).reshape(10, 1)
        unactive_o = unactive_o + biases2
        active_o = self.probability(unactive_o) # Uses probability to compute activated neurons of output layer

        return pixels, unactive_h, active_h, unactive_o, active_o

    # Tests the accurancy of the trained weights and biases using testing data
    def test_model(self, images_test, labels_test):

        accuracy = 0

        for testing_example_index in range(len(images_test)):
            query = self.query(images_test[testing_example_index])
            if np.argmax(query[4]) == labels_test[testing_example_index]:
                accuracy += 1

        return accuracy
