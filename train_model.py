from neural_network import NeuralNetwork
from mnist_data.load import load_mnist
import numpy as np
import pdb

def get_learning_rate(epoch):
    base_lr = 0.1  # Starting learning rate
    lr = base_lr * (0.1 ** (epoch // 10))  # Decrease learning rate by a factor of 10 every 10 epochs
    return lr

for x in range(500):
	print('Model Training .....')

	images_train, labels_train = load_mnist('mnist_data', kind='train')
	neural_net = NeuralNetwork()

	epochs = np.load('epochs.npy')
	learn_rate = get_learning_rate(epochs[0])
	print('Epoch:', epochs[0])
	print('Learning Rate', learn_rate)

	trained_params = neural_net.train(images_train, labels_train, learn_rate)
	epochs += 1

	np.save('trained_params/weights1.npy', trained_params[0])
	np.save('trained_params/weights2.npy', trained_params[1])
	np.save('trained_params/biases1.npy', trained_params[2])
	np.save('trained_params/biases2.npy', trained_params[3])
	np.save('epochs.npy', epochs)

	print('Model Trained!')
	# print('Computing Cost...')
	# new_cost = neural_net.compute_cost(images_train, labels_train)
	# print('Cost:', new_cost)
