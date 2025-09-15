from typing import Optional
from numpy import random, ndarray, zeros, sqrt, linalg
from numba_functions import sigmoid_numba

class NeuralLayer:
    __slots__ = ('nb_neurons', 'nb_ants', 'z', 'a', 'weights', 'biases')

    def __init__(self, nb_neurons: int, nb_ant: int, seed: Optional[int] = None):
        self.nb_neurons = int(nb_neurons)
        self.nb_ants = int(nb_ant)
        self.z = zeros((self.nb_neurons, 1), dtype=float)
        self.a = zeros((self.nb_neurons, 1), dtype=float)

        if self.nb_ants > 0:
            rng = random.default_rng(seed)
            self.weights = rng.normal(size=(nb_neurons, nb_ant))
            self.biases = rng.normal(size=(nb_neurons, 1))
        else:
            self.weights = None
            self.biases = None

    def forward(self, inputs: ndarray) -> ndarray:
        w = self.weights
        if w is None:
            self.a = inputs
            return inputs
        else:
            self.z = self.weights @ inputs + self.biases
            self.a = sigmoid_numba(self.z)
            return self.a

    def initialize(self, seed: Optional[int] = None):
        if self.nb_ants > 0:
            rng = random.default_rng(seed)
            self.weights = rng.normal(size=(self.nb_neurons, self.nb_ants))
            self.biases = rng.normal(size=(self.nb_neurons, 1))

class Network:
    def __init__(self):
        self.layers: list[NeuralLayer] = []
        self.architecture: list[int] = []
        self.training_history = {'loss':[], 'accuracy':[]}

    def add_layer(self, nb_neurons: int) -> None:
        nb_ant = self.layers[-1].nb_neurons if self.layers else 0
        layer = NeuralLayer(int(nb_neurons), int(nb_ant))
        self.layers.append(layer)
        self.architecture.append(int(nb_neurons))

    def build_network(self, architecture: list[int]) -> None:
        self.layers.clear()
        self.architecture.clear()
        for nb in architecture:
            self.add_layer(nb)

    @staticmethod
    def re_initialize(layers: list[NeuralLayer]):
        for layer in layers:
            layer.initialize()

    def forward_propagation(self, data: ndarray) -> ndarray:
        if not self.architecture or data.shape[0] != self.architecture[0]:
            raise ValueError(f"Dimension d'entrée incorrecte: {data.shape[0]} vs {self.architecture[0] if self.architecture else '??'}")

        layers = self.layers
        layers[0].a = data
        prev = data
        for layer in layers[1:]:
            prev = layer.forward(prev)
        return prev

    def compute_norm(self) -> tuple[float, float, float]:
        """
        Returns:
            float: The Euclidean norm of the network := sqrt(sum of squares of weights and biases)
        """
        w_norms_squared = []
        b_norms_squared = []

        for layer in self.layers:
            if layer.weights is not None:
                w_norms_squared.append(linalg.norm(layer.weights) ** 2)
            else:
                w_norms_squared.append(0)
            if layer.biases is not None:
                b_norms_squared.append(linalg.norm(layer.biases) ** 2)
            else:
                b_norms_squared.append(0)
            a, b = sum(w_norms_squared), sum(b_norms_squared)
        return sqrt(a+b), sqrt(a), sqrt(b)