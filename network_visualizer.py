from network_config import NetworkConfig
from neuron_class import Network
import matplotlib.pyplot as plt

from numpy import array, linspace, ndarray, meshgrid, median, where, asarray, mean
from typing import Dict, Tuple
from math import floor
from network_trainer import test_accuracy

def display_training_results(nw: Network,
                             data: ndarray,
                             model: ndarray,
                             config: 'NetworkConfig',
                             history: Dict,
                             title: str = None):
    """
    Displays the training results: loss + decision boundary.

    Args:
        nw: The neural network
        data: Training data (2 x N)
        model: Corresponding labels
        config: Configuration (for auto-title and scaling)
        history: Dict with 'iterations' and 'loss'
        title: Custom title
    """
    # Construction du titre
    default_title = f'MLP_dimensions_{config.dimensions}_lrng_rate_{config.lrng_rate}_epochs_{history["final_iteration"]}'

    # Données pour les graphiques
    x_loss = history['iterations']
    y_loss = history['loss']

    # Utilise vos paramètres de cadrage
    cadre_x = config.cadre_x if hasattr(config, 'cadre_x') else (-2, 2, 41)
    cadre_y = config.cadre_y if hasattr(config, 'cadre_y') else (-2, 2, 41)

    # Évaluation de la fonction sur l'espace de travail
    x_min, x_max, nb_x = cadre_x
    y_min, y_max, nb_y = cadre_y
    x = linspace(x_min, x_max, nb_x)
    y = linspace(y_min, y_max, nb_y)

    xx, yy = meshgrid(x, y)
    zz = array([nw.forward_propagation(array([xx[i], yy[i]]))[0]
                for i in range(len(y))])

    # Configuration de la figure
    plt.figure(3, figsize=(17, 9))
    plt.gcf().subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                              wspace=0.1, hspace=0.1)

    # Graphique de perte
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(x_loss, y_loss, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss evolution')
    plt.legend()

    # Contour de décision
    plt.subplot(1, 2, 2)
    plt.contourf(x, y, zz)
    plt.title(test_accuracy(nw, data, model, "Performance : "))

    if title is not None:
        t = title
    else:
        t = default_title
    plt.title(t)
    plt.axis('scaled')
    plt.colorbar()

    # Affichage des points de données
    if data is not None:
        plt.scatter(data[0, :], data[1, :], c=model, cmap='summer', s=50, alpha=0.8)
    plt.grid()

    plt.show()

def create_performance_heatmap(
    results: ndarray,
    h_range: Tuple[float, float, int],
    it_range: Tuple[int, int, int],
    title: str = "Performance map",
):
    """
    Affiche une carte de performance sous forme de heatmap.

    Args :
        results (ndarray) : Matrice des résultats de performance.
        h_range (Tuple[float, float, int]) : (min, max, nb_points) pour le taux d'apprentissage.
        epoch_range (Tuple[int, int, int]) : (min, max, nb_points) for the epochs
        title (str) : graph title
    """
    h_min, h_max, _ = h_range
    it_min, it_max, _ = it_range
    extent = (h_min, h_max, it_min, it_max)

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        results,
        extent=extent,
        vmin=0,
        vmax=1,
        aspect='auto',
        origin="lower",
    )
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Number of epochs")

    # Barre de couleur
    fig.colorbar(image, ax=ax, label='Success rate')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def display_search_results(r_init: ndarray,
                           r_sol: ndarray,
                           r_w_sol: ndarray,
                           r_b_sol: ndarray,
                           result: ndarray,
                           config: NetworkConfig,
                           title: str = "Radius_search"):
    """
    Displays the norms of the trained MLPs based on the initial norms.

    Args:
        r_init: List of initial norms
        r_sol: List of final norms
        result: List of training results (True = success, False = failure)
        title: Display title
    """
    r_init, r_sol = asarray(r_init), asarray(r_sol)
    colors = where(result, '#3498db', '#e67e22')  # blue for success, orange for failure

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axes
    informations = f'MLP_dimensions_{config.dimensions}_lrng_rate_{config.lrng_rate}_epochs_{config.nb_epochs}_regularization_{config.r}'
    fig.suptitle(f'{title} - {len(r_init)} trials - {informations}', fontsize=16, fontweight='bold')

    # --- 2. Initial vs final norm ---
    ax1.scatter(r_w_sol, r_b_sol, c=colors, alpha=0.7, s=2)
    ax1.plot([r_w_sol.min(), r_w_sol.max()], [r_w_sol.min(), r_w_sol.max()],
             'k--', alpha=0.5, label='y=x')
    ax1.set(title="Final weight vs final bias norm", xlabel="weight norm", ylabel="bias norm")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- 2. Initial vs final norm ---
    ax2.scatter(r_init, r_sol, c=colors, alpha=0.7, s=2)
    ax2.plot([r_init.min(), r_init.max()], [r_init.min(), r_init.max()],
             'k--', alpha=0.5, label='y=x')
    ax2.set(title="Initial vs Final norm", xlabel="Initial norm", ylabel="Final norm")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- 3. Histograms ---
    bins = max(10, len(r_init) // 3)
    ax3.hist(r_init, bins=bins, alpha=0.7, color='#2980b9', label='Initial', density=True)
    ax3.hist(r_sol, bins=bins, alpha=0.7, color='#8e44ad', label='Final', density=True)
    ax3.set(title="Norm distributions", xlabel="Norm value", ylabel="Density")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # --- 4. Statistics ---
    ax4.axis("off")
    delta = r_sol - r_init
    p = floor(mean(result == 1) * 100)
    stats = (
        f"STATISTICS\n\n"
        f"Success : {p}%\n"
        f"INITIAL NORMS:\n"
        f"• Min: {r_init.min():.3f} | Max: {r_init.max():.3f} | Mean: {r_init.mean():.3f}\n\n"
        f"FINAL NORMS:\n"
        f"• Min: {r_sol.min():.3f} | Max: {r_sol.max():.3f} | Mean: {r_sol.mean():.3f}\n\n"
        f"EVOLUTION:\n"
        f"• Mean Δ: {delta.mean():+.3f}\n"
        f"• Median Δ: {median(delta):+.3f}"
    )
    ax4.text(0.05, 0.95, stats, transform=ax4.transAxes,
             fontsize=11, va='top', family='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()