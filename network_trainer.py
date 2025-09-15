from neuron_class import Network
from network_config import NetworkConfig
from numba_functions import log_loss_numba
from tqdm import tqdm

from numpy import dot, sum, ndarray, array_equal
from typing import Dict, Optional

def train_fast(n: Network,
               data: ndarray,
               model: ndarray,
               nb_epochs: int,
               lrng_rate: float,
               r : float,
               end_train: bool = True,
               verbose: bool = False) -> int:
    """
    Fast training with early stopping.
    Returns the final epoch reached.

    Args :
        n: Network to train
        data: Training data
        model: Expected training results
        nb_epochs : number of epochs
        lrng_rate : learning rate
        config: Training configuration
        verbose: Display training prog
    """
    m = model.shape[0]
    layers, p = n.layers, len(n.layers) - 1

    for k in range(nb_epochs):
        # Forward (pas besoin de stocker la sortie)
        n.forward_propagation(data)
        dz = layers[-1].a - model

        # Backward
        for j in range(p):
            x = p - j
            a_prev = layers[x - 1].a
            layer = layers[x]

            dw = dz @ a_prev.T / m
            db = dz.sum(axis=1, keepdims=True) / m

            if x != 1:
                dz = (layer.weights.T @ dz) * a_prev * (1 - a_prev)

            layer.weights -= lrng_rate * (dw + r * layer.weights)
            layer.biases -= lrng_rate * (db + r * layer.biases)

        # Vérification convergence
        if end_train and end_train_check(n, data, model):
            if verbose:
                print(f"Convergence reached at epoch {k}.")
            return k  # stop immédiat

    return nb_epochs + 1  # pas convergé avant la fin

def end_train_check(n: Network, data: ndarray, model: ndarray) -> bool:
    n.forward_propagation(data)
    prediction = (n.layers[-1].a[0] >= 0.5).astype(int)
    return array_equal(prediction, model[0])

def evaluate_performance(data: ndarray,
                         model: ndarray,
                         nw: Network,
                         nb_epochs: int,
                         lrng_rate: float,
                         regularization: float,
                         nb_tests: int = 20,
                         end_train: bool = True) -> float:
    """
    Args :
        data: Training data
        model: Expected training results
        nw : an MLP that is trained several times
        nb_epochs : number of training epochs
        lrng_rate : learning rate
        nb_tests : number of test
    Returns :
        Success rate (between 0 and 1)
    """
    counter = 0

    for _ in range(nb_tests):
        Network.re_initialize(nw.layers)
        k = train_fast(nw, data, model, nb_epochs, lrng_rate, regularization, end_train)

        # Vérification de la convergence
        if k <= nb_epochs:
            counter += 1

    return float(counter / nb_tests)

def train(n: Network,
          data: ndarray,
          model: ndarray,
          config: 'NetworkConfig',
          verbose: bool = False) -> Dict:
    """
    Args :
        n: Network to train
        data: Training data
        model: Expected training results
        config: Training configuration
        x_t: Optional test data
        y_t: Optional test labels
        verbose: Display training progress
    Returns :
        Dictionary with training history
    """
    n.build_network(config.dimensions)
    h, nb_epochs, rp, end_train = config.lrng_rate, config.nb_epochs, 2 * config.r, config.end_train
    m = model.shape[0]
    layers, p = n.layers, len(n.layers) - 1

    training_history_dict = {
        'loss': [],
        'iterations': [],
        'final_iteration': None,
        'converged': False
    }
    l, epoch_list = [], []

    p = len(layers) - 1  # Profondeur du réseau (votre variable)
    epoch_range = tqdm(range(nb_epochs), desc="Training") if verbose else range(nb_epochs)

    for k in epoch_range:
        # Forward propagation
        v = n.forward_propagation(data)

        # Calcul du gradient de sortie
        dz = layers[-1].a - model

        # Backpropagation
        for j in range(p):
            x = p - j
            prev_x = x - 1
            # Calcul des gradients
            a_prev = layers[prev_x].a
            dw = dot(dz, a_prev.T) / m
            db = sum(dz, axis=1, keepdims=True) / m

            # Gradient pour la couche suivante (sauf la dernière)
            if x != 1:
                dz = dot(layers[x].weights.T, dz) * a_prev * (1 - a_prev)

            # Mise à jour des poids et biais (avec régularisation)
            layers[x].weights -= h * (dw + rp * layers[x].weights)
            layers[x].biases -= h * (db + rp * layers[x].biases)

        # Logging toutes les 10 itérations
        if k % 10 == 0:
            c = log_loss_numba(v, model)
            l.append(c)
            epoch_list.append(k)

            # Arrêt anticipé si demandé (votre logique end_train)
            if end_train and end_train_check(n, data, model):
                training_history_dict['final_iteration'] = k
                if verbose:
                    print(f"Convergence atteinte à l'itération {k}")
                break

    # Stockage de l'historique
    training_history_dict['loss'] = l
    training_history_dict['iterations'] = epoch_list
    if training_history_dict['final_iteration'] is None:
        training_history_dict['final_iteration'] = nb_epochs

    # Test final de convergence
    training_end = end_train_check(n, data, model)
    training_history_dict['converged'] = training_end

    return training_history_dict

def test_accuracy(n: Network, data: ndarray, model: ndarray, prefix: str = "") -> str:
    n.forward_propagation(data)
    vrai = 0
    for k in range(model.shape[1]):
        rep = (n.layers[-1].a[0][k] >= 0.5)
        if (rep and model[0][k] == 1) or (not rep and model[0][k] == 0):
            vrai += 1
    return prefix + str(vrai) + '/' + str(model.shape[1])
