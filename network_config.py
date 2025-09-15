from dataclasses import dataclass, field
@dataclass
class NetworkConfig:
    """
    Complete setup for training a multi-layer perceptron
    """
    dimensions: list[int]
    lrng_rate: float = 0.01
    nb_epochs: int = 500
    r: float = 0.02  # regularization coefficient

    # Paramètres d'arrêt
    end_train: bool = False  # early stopping

    def __post_init__(self):
        """Validation des paramètres après initialisation."""
        if len(self.dimensions) < 2:
            raise ValueError("L'architecture doit avoir au moins 2 couches (entrée et sortie)")

        if self.lrng_rate <= 0:
            raise ValueError("Le taux d'apprentissage doit être positif")

        if self.nb_epochs <= 0:
            raise ValueError("Le nombre d'itérations doit être positif")