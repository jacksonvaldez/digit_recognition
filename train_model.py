from neural_network import NeuralNetwork
from mnist_data.load import load_mnist
import numpy as np
import pdb

def get_learn_rate(epoch):
	lr = 1 * (0.1 ** (epoch / 20))
	return lr


for x in range(500):
	print('------------------- Model Training -------------------')
	epoch = np.load('epoch.npy')[0]
	learn_rate = get_learn_rate(epoch)
	print('Epoch:', epoch)
	print('Learn Rate:', learn_rate)

	images_train, labels_train = load_mnist('mnist_data', kind='train')
	neural_net = NeuralNetwork()

	trained_params = neural_net.train(images_train, labels_train, learn_rate)

	epoch += 1
	np.save('epoch.npy', [epoch])

	np.save('trained_params/weights1.npy', trained_params[0])
	np.save('trained_params/weights2.npy', trained_params[1])
	np.save('trained_params/biases1.npy', trained_params[2])
	np.save('trained_params/biases2.npy', trained_params[3])

	print('Model Trained!')
