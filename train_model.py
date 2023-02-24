from neural_network import NeuralNetwork
from mnist_data.load import load_mnist
import numpy as np


print('Model Training .....')

images_train, labels_train = load_mnist('mnist_data', kind='train')
neural_net = NeuralNetwork()
trained_params = neural_net.train(images_train, labels_train, 0.001)

# print('weights1:', trained_params[0].shape, trained_params[0].dtype)
# print('weights2:', trained_params[1].shape, trained_params[1].dtype)
# print('biases1:', trained_params[2].shape, trained_params[2].dtype)
# print('biases2:', trained_params[3].shape, trained_params[3].dtype)
np.save('trained_params/weights1.npy', trained_params[0])
np.save('trained_params/weights2.npy', trained_params[1])
np.save('trained_params/biases1.npy', trained_params[2])
np.save('trained_params/biases2.npy', trained_params[3])

print('Model Trained!')