# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import config


'''
TODO: 
1) legenda z licznikiem a) osobników aseksualnych b) osobników płciowych 
2) kolorowanie osobników na bazie ich rozmnażania
'''
def plot_population(population, alpha, generation, save_path=None, show_plot=False):
    """
    Rysuje populację w 2D wraz z optymalnym fenotypem alpha.
    Można zarówno wyświetlać (show_plot=True),
    jak i zapisywać obraz (save_path != None).
    """
    individuals = population.get_individuals()
    ind_reversed = []
    for i in range(len(individuals)-1, -1, -1):
        ind_reversed.append(individuals[i])
    x = np.array([ind.get_phenotype()[0] for ind in ind_reversed])
    y = np.array([ind.get_phenotype()[1] for ind in ind_reversed])
    sex = np.array([ind.get_phenotype()[-1] for ind in ind_reversed])

    '''colors = []
    for ind in population.get_individuals():
        if ind.get_pair() is ind:  # Paired with self
            colors.append("blue")
        elif ind.get_pair() is None:  # No pair
            colors.append("gray")
        else:  # Paired with another individual
            colors.append("orange")'''
    colors = np.array([
        "blue" if ind.get_pair() is ind else  # Asexual reproduction
        "yellow" if ind.get_age() == 0 else
        "gray" if ind.get_pair() is None else  # Did not find a mate
        "green"  # Successfully mated

        for ind in ind_reversed
    ])
    # Define legend handles
    legend_patches = [
        mpatches.Patch(color="blue", label="Asexual reproduction"),
        mpatches.Patch(color="gray", label="No mate found"),
        mpatches.Patch(color="green", label="Mated successfully"),
        mpatches.Patch(color="red", label="Optimum")
    ]


    #plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(figsize=(5, 5))




    # Males (Triangles)
    male_mask = (sex == 1)
    ax.scatter(x[male_mask], y[male_mask], c=colors[male_mask], marker="^", alpha=0.5, label="Males")

    # Females (Circles)
    female_mask = (sex == 0)
    ax.scatter(x[female_mask], y[female_mask], c=colors[female_mask], marker="o", alpha=0.5, label="Females")

    legend_markers = [
        mlines.Line2D([], [], color='black', marker="^", linestyle="None", markersize=8, label="Males"),
        mlines.Line2D([], [], color='black', marker="o", linestyle="None", markersize=8, label="Females")
    ]
    #plt.scatter(x, y, label="Populacja", alpha=0.5, c=colors)
    ax.scatter([alpha[0]], [alpha[1]], color='red', label="Optimum", marker="x")
    # Reproduction barriers:
    circle_asex = mpatches.Circle((alpha[0], alpha[1]), np.sqrt(-2 * config.sigma ** 2 * np.log(config.threshold_asex)),
                                  fill=False, edgecolor='red',
                                  linestyle='--', label='Asexual threshold')
    circle_surv = mpatches.Circle((alpha[0], alpha[1]), np.sqrt(-2 * config.sigma ** 2 * np.log(config.threshold_surv)),
                                  fill=False, edgecolor='green',
                                  linestyle='--', label='Survival threshold')

    # Add circles to the plot
    ax.add_patch(circle_asex)
    ax.add_patch(circle_surv)


    ax.set_title(f"Pokolenie: {generation}, liczba osobników: {len(individuals)}")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.legend(handles=legend_patches+legend_markers)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)  # Zapis do pliku
    if show_plot:
        plt.show()
    else:
        # Jeśli nie chcesz pokazywać, to zamykaj figurę, 
        # żeby nie zapełniać pamięci
        plt.close()
