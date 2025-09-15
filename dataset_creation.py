from numpy import (linspace, random, pi,
                   cos, sin, vstack, column_stack, hstack, zeros, array, ones, unique, bincount)
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles

def generate_spiral_data(n_samples=300, noise=0.001, n_turns=1.5):
    n_per_class = n_samples // 2

    # Spiral 1 (classe 0)
    t1 = linspace(0, n_turns * 2 * pi, n_per_class)
    r1 = linspace(0.1, 2, n_per_class)
    x1 = r1 * cos(t1) + random.normal(0, noise, n_per_class)
    y1 = r1 * sin(t1) + random.normal(0, noise, n_per_class)

    # Spiral 2
    t2 = linspace(0, n_turns * 2 * pi, n_per_class) + pi
    r2 = linspace(0.1, 2, n_per_class)
    x2 = r2 * cos(t2) + random.normal(0, noise, n_per_class)
    y2 = r2 * sin(t2) + random.normal(0, noise, n_per_class)

    # Combinaison
    x = vstack([column_stack([x1, y1]), column_stack([x2, y2])])
    y = hstack([zeros(n_per_class), ones(n_per_class)])

    return x, y


def generate_checkerboard_data(n_samples=400, grid_size=4, noise=0.1):
    x = []
    y = []

    for i in range(n_samples):
        a = random.uniform(-2, 2)
        y_coord = random.uniform(-2, 2)

        # Ajouter du bruit
        a += random.normal(0, noise)
        y_coord += random.normal(0, noise)

        # Déterminer la classe basée sur la position dans la grille
        grid_x = int((a + 2) * grid_size / 4)
        grid_y = int((y_coord + 2) * grid_size / 4)

        # Motif damier
        if (grid_x + grid_y) % 2 == 0:
            label = 0
        else:
            label = 1

        x.append([a, y_coord])
        y.append(label)

    return array(x), array(y)

def generate_xor_pattern(n_samples=400, noise=0.2):
    x = []
    y = []

    for i in range(n_samples):
        # Générer des points dans les 4 quadrants
        quadrant = random.randint(0, 4)

        if quadrant == 0:  # Quadrant I (classe 0)
            a = random.uniform(0.2, 2) + random.normal(0, noise)
            y_coord = random.uniform(0.2, 2) + random.normal(0, noise)
            label = 0
        elif quadrant == 1:  # Quadrant II (classe 1)
            a = random.uniform(-2, -0.2) + random.normal(0, noise)
            y_coord = random.uniform(0.2, 2) + random.normal(0, noise)
            label = 1
        elif quadrant == 2:  # Quadrant III (classe 0)
            a = random.uniform(-2, -0.2) + random.normal(0, noise)
            y_coord = random.uniform(-2, -0.2) + random.normal(0, noise)
            label = 0
        else:  # Quadrant IV (classe 1)
            a = random.uniform(0.2, 2) + random.normal(0, noise)
            y_coord = random.uniform(-2, -0.2) + random.normal(0, noise)
            label = 1

        x.append([a, y_coord])
        y.append(label)

    return array(x), array(y)


def generate_wave_pattern(n_samples=400, frequency=2, noise=0.1):
    x = []
    y = []

    for i in range(n_samples):
        a = random.uniform(-3, 3)

        # Ligne sinusoïdale
        wave_y = sin(frequency * a)

        # Point au-dessus ou en-dessous de la vague
        if random.random() < 0.5:
            y_coord = wave_y + random.uniform(0.2, 1.5) + random.normal(0, noise)
            label = 1
        else:
            y_coord = wave_y - random.uniform(0.2, 1.5) + random.normal(0, noise)
            label = 0

        x.append([a, y_coord])
        y.append(label)

    return array(x), array(y)


# Fonction principale pour générer différents datasets
def create_complex_dataset(dataset_type='spiral', n_samples=400, noise=0.1, seed: int = 42, **kwargs):
    """
    Create a complex dataset for training neural networksParameters:
    Args:
        dataset_type: 'spiral', 'checkerboard', 'xor', 'wave', 'moons', 'circles'
        n_samples: number of samples
        noise: level of noise
        seed: random seed
    """

    if dataset_type == 'spiral':
        x, y = generate_spiral_data(n_samples, noise/5, kwargs.get('n_turns', 1.5))
    elif dataset_type == 'checkerboard':
        x, y = generate_checkerboard_data(n_samples, kwargs.get('grid_size', 4), noise)
    elif dataset_type == 'xor':
        x, y = generate_xor_pattern(n_samples, noise)
    elif dataset_type == 'wave':
        x, y = generate_wave_pattern(n_samples, kwargs.get('frequency', 2), noise)
    elif dataset_type == 'moons':
        x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif dataset_type == 'circles':

        x, y = make_circles(n_samples=n_samples, noise=noise, factor=0.2, random_state=seed)
    else:
        raise ValueError(f"Type de dataset non reconnu: {dataset_type}")

    # Conversion au format requis (transposé)
    y = array([[y[k]] for k in range(len(y))])
    x, y = x.T, y.T

    return x, y


# Fonction de visualisation
def visualize_dataset(x, y, title="Dataset Complexe"):
    """Visualise le dataset généré"""
    plt.figure(figsize=(10, 8))

    # Reconvertir pour la visualisation
    x_vis = x.T
    y_vis = y.T.flatten()

    # Colormap pour différentes classes
    colors = plt.cm.Set1(linspace(0, 1, len(unique(y_vis))))

    for i, class_val in enumerate(unique(y_vis)):
        mask = y_vis == class_val
        plt.scatter(x_vis[mask, 0], x_vis[mask, 1],
                    c=[colors[i]], label=f'Classe {int(class_val)}',
                    alpha=0.7, s=50)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate_all_datasets(nb_samples: int =100, seed: int = 42):
    # Créer différents datasets
    datasets_params = {
        'spiral': {'n_turns': 2},
        'checkerboard': {'grid_size': 3},
        'xor': {},
        'wave': {'frequency': 1.5},
        'moons': {},
        'circles': {},
    }
    datasets = {}
    for name, params in datasets_params.items():
        data, model = create_complex_dataset(name, n_samples=nb_samples, noise=0.1, seed=seed, **params)
        datasets[name] = (data, model)
    return datasets