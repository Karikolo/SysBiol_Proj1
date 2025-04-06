import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import config
from environment import Environment
from population import Population
from mutation import mutate_population
from selection import proportional_selection, threshold_selection
from reproduction import reproduction

# Wykres 3 - średnia częstość rozmnażania płciowego w zależności od częstości mutacji

def boxplot_sexvsmut(mean_matrix, labels):

    fig, ax  = plt.subplots()
    ax.set_ylabel('Średnia częstość rozmnażania płciowego')
    ax.set_xlabel('Częstość mutacji')
    ax.set_title('Średnia częstosć rozmnażania płciowego w zależności od częstosci mutacji')

    colors = ['green', 'blue', 'yellow', 'magenta', 'grey']

    bplot = ax.boxplot(mean_matrix,
                       patch_artist=True,
                       tick_labels = labels)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    fig.tight_layout()

    plt.savefig(f'C:\\Users\\Anastazja\\Desktop\\boxplots\\boxplot_sexual_kier_zm{config.c[0]}.png', dpi=300, bbox_inches = 'tight')
    plt.show()


def one_simulation(mut):

    count_increase = 0 # licznik przyrostu populacji
    result_increase = []

    env = Environment(alpha_init=config.alpha0, c=config.c, delta=config.delta)
    pop = Population(size=config.N, n_dim=config.n)
    finish_gif = False  # zmienna służąca do zapisania w gifie momentu wymarcia populacji

    print("Starting simulation")

    last_gen = config.N
    gen = 0
    sex_success = 0

    for generation in range(1, config.max_generations):

        gen+=1

        # 1. Mutacja
        mutate_population(pop, mu=mut, mu_c=config.mu_c, xi=config.xi)

        # 2. Selekcja
        survivors = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_surv)
        for individual in survivors:
            individual.set_pair(None)
            individual.set_sex_reproduction(False)
            #if individual.get_age() > config.lifespan:
            #    dying += individual
        pop.set_individuals(survivors)
        #survivors = [ind for ind in survivors if ind not in dying]

        if len(survivors) <= 0:
            print(f"Wszyscy wymarli w pokoleniu {generation}. Kończę symulację.")
            finish_gif = True
        if finish_gif: break
        # print("Survivors:", survivors, len(survivors))

        # 3. Reprodukcja
        # Bezpłciowa
        asexuals = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_asex)
        # zmiana atrybutu - rozmnaża się sam
        asex_paired = []
        asex_female = []
        for individual in asexuals:
            # remove all males from asexual reproduction:
            if individual.get_phenotype()[-1] == 0:
                individual.set_pair(individual)
                asex_paired += [(individual, individual)]
                asex_female.append(individual)
        # print("Asexuals paired (not males):", len(asex_paired))

        # Płciowa
        # Dobieranie w pary (ten, kto się nie dobierze, ten się nie rozmnaża)
        sex_to_pair = [s for s in survivors if
                       s not in asex_female]  # lista osobników, które w tej generacji rozmnażają się płciowo
        # parowanie osobników - rozmnażanie płciowe
        sex_paired = pop.set_pairs(sex_to_pair)
        # print("Sexual paired:", sex_paired)

        # lista wszystkich par w populacji - płciowe i bezpłciowe
        all_paired = sex_paired + asex_paired
        # print("All paired:", all_paired)

        children_phenotypes = reproduction(all_paired, env.get_optimal_phenotype(), config.sigma, len(survivors))
        # print("Children phenotypes:", children_phenotypes, len(children_phenotypes))
        pop.add_individuals(children_phenotypes)
        # print("New population:", pop.get_individuals(), len(pop.get_individuals()))

        for individual in pop.get_individuals():
            if individual.get_sex_reproduction(): sex_success+=1


        # 4. Zmiana środowiska
        env.update()

    print("Symulacja zakończona.")
    return round(sex_success / 2*gen, 2)

def multi_simulations(mut_v):

    matrix_success=[] # wiersze dla mut

    for mut in mut_v:
        success = []
        for i in range(20):
            success.append(one_simulation(mut))
        matrix_success.append(success)

    return matrix_success

matrix = multi_simulations([0.1, 0.3, 0.5, 0.7, 0.9])
boxplot_sexvsmut(matrix, [0.1, 0.3, 0.5, 0.7, 0.9])