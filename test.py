import numpy as np
from controller import Controller

dna = np.array([[3, 1, 30, 2], [3, 1, 30, 2], [3, 1, 40, 2]])
controller = Controller()

controller.train_child_network(dna, 'test')