from neural_network import NeuralNetwork
from mnist_data.load import load_mnist
import numpy as np


print('Model Training .....')

images_train, labels_train = load_mnist('mnist_data', kind='train')
neural_net = NeuralNetwork()

trained_params = neural_net.train(images_train, labels_train, 0.001)
np.save('trained_params/weights1.npy', trained_params[0])
np.save('trained_params/weights2.npy', trained_params[1])
np.save('trained_params/biases1.npy', trained_params[2])
np.save('trained_params/biases2.npy', trained_params[3])

print('Model Trained!')