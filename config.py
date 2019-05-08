import os

child_network_params = {
    "learning_rate": 1e-3,
    "max_epochs": 100,
    "beta": 1e-3,
    "batch_size": 32
}

controller_params = {
    "max_layers": 4,
    "components_per_layer": 4,
    'beta': 1e-4,
    'max_episodes': 2000,
    "num_children_per_episode": 5
}

MAIN_DIR = os.getcwd()
LOGS_DIR = os.path.join(MAIN_DIR, 'logs')