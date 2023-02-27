import numpy as np
from mnist_data.load import load_mnist
from neural_network import NeuralNetwork


weights1 = np.load('trained_params/weights1.npy')
weights2 = np.load('trained_params/weights2.npy')
biases1 = np.load('trained_params/biases1.npy')
biases2 = np.load('trained_params/biases2.npy')

# print(weights2)
# print(biases2)

images_train, labels_train = load_mnist('mnist_data', kind='train')
neural_net = NeuralNetwork()

print('Computing Cost...')
cost = neural_net.compute_cost(images_train, labels_train)
print('Cost: ', cost)