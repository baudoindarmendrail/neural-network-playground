from neuron_class import Network
from network_config import NetworkConfig
from dataset_creation import generate_all_datasets

from numpy import array, linspace, zeros, ndarray
from typing import Tuple
from tqdm import tqdm

from network_trainer import train_fast, train, evaluate_performance, end_train_check
from network_visualizer import create_performance_heatmap, display_training_results, display_search_results

# Set up a generic configuration for the rest of the notebook
default_config = NetworkConfig(dimensions=[2, 3, 2, 1], # Defines the number of neurons in each layer
                               lrng_rate=0.025, #Learning rate
                               nb_epochs=1000, #Number of epochs
                               r=0.02,         #Regularization coefficient
                               end_train=True) #Early stopping

def grid_search(data: ndarray,
                model: ndarray,
                architecture: list[int],
                learning_rate_range: Tuple[float, float, int] = (0.01, 0.1, 20),
                epoch_range: Tuple[int, int, int] = (100, 1000, 10),
                regularization: float = 0.02,
                nb_tests: int = 20,
                print_result_array: bool = False) -> None:
    """
        Studies on a grid the efficiency of configurations by varying the learning rate and the number of iterations.
    """
    h_min, h_max, nb_h = learning_rate_range
    epoch_min, epoch_max, nb_epochs = epoch_range

    h_values = linspace(h_min, h_max, nb_h)
    epoch_values = linspace(epoch_min, epoch_max, nb_epochs, dtype=int)

    nw = Network()
    nw.build_network(architecture)
    results = zeros((nb_epochs, nb_h))

    for i, lrng_rate in enumerate(tqdm(h_values, desc="progress of mapping")):
        for j, epoch in enumerate(epoch_values):
            results[j, i] = evaluate_performance(data, model, nw, epoch, lrng_rate, regularization, nb_tests)
    if print_result_array:
        for l in results:
            print(l)

    # Visualisation automatique
    create_performance_heatmap(
        results, learning_rate_range, epoch_range,
        f"Performance - Architecture {architecture}",
    )

def search_radius_solutions(nb: int,
                            data: ndarray,
                            model: ndarray,
                            config: NetworkConfig = default_config):
    r_init, r_sol, r_b_sol, r_w_sol, result = [], [], [], [], []

    n = Network()
    n.build_network(config.dimensions)
    nb_epochs, lrng_rate, r, end_train = config.nb_epochs, config.lrng_rate, config.r, config.end_train

    for _ in tqdm(range(nb)):
        Network.re_initialize(n.layers)
        r_init.append(n.compute_norm()[0])
        train_fast(n, data, model, nb_epochs, lrng_rate, r, end_train)

        result.append(end_train_check(n, data, model))
        t = n.compute_norm()
        r_sol.append(t[0])
        r_w_sol.append(t[1])
        r_b_sol.append(t[2])
    display_search_results(array(r_init), array(r_sol), array(r_w_sol), array(r_b_sol), array(result), config)


# Generate the point clouds used
datasets = generate_all_datasets(seed=1)
x, y = datasets["circles"]

# Exemple d'utilisation intégrée avec votre workflow existant
def exemple_of_use():
    # Création du réseau
    n_ex = Network()
    n_ex.build_network(default_config.dimensions)
    history_dict = train(n_ex, x, y, default_config)

    # Affichage de la frontière de décision seulement
    display_training_results(n_ex, x, y, default_config, history_dict)

    # Évaluation de performance
    success_rate_ex = evaluate_performance(x, y, n_ex, default_config.nb_epochs, default_config.lrng_rate, default_config.r, nb_tests=50)
    print(f"For 50 similarly trained MLPs, perfect reproduction in {success_rate_ex:.2%} of cases.")

