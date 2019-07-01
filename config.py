import os

child_network_params = {
    "learning_rate": 1e-3,
    "max_epochs": 15,
    "beta": 1e-3,
    "batch_size": 45
}

controller_params = {
    "max_layers": 3,
    "components_per_layer": 4,
    'beta': 3e-4,
    'max_episodes': 2000,
    "num_children_per_episode": 5
}

MAIN_DIR = os.getcwd()


