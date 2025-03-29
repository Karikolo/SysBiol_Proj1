import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import statistics

import config
from environment import Environment
from population import Population
from mutation import mutate_population
from selection import proportional_selection, threshold_selection


'''
    mut_rates - wektor z wartosciami parametru mutation rate
    sigma_val - wektor z wartościami parametru selection strength
    mean_pop - mediana ze średnich liczności populacji we wszystkich generacjach
    
'''
# Wykres pierwszy

# TODO: zliczanie dzieci zamiast różnica między generacjami

def heatmap_mut_sel(mut_rates, sigma_val, increase_pop):

    fig, ax = plt.subplots()
    im = ax.imshow(increase_pop)

    ax.set_xticks(range(len(sigma_val)), labels = sigma_val) # ustawienie podpisów jest default
    ax.set_yticks(range(len(mut_rates)), labels = mut_rates)

    for i in range(len(sigma_val)):
        for j in range(len(mut_rates)):
            # pierwszy indeks to numer kolumny, drugi to numer wiersza
            text = ax.text(i, j, increase_pop[i,j], ha='center', va='center', color='w') # indeksy upewnić się

    ax.set_title('Mediana średniego przyrostu populacji')
    fig.tight_layout()

    plt.savefig('C:\\Users\\Anastazja\\Desktop\\heatmapa_mut_sel.png', dpi=300, bbox_inches = 'tight')
    plt.show()


def one_simulation(mut, sigma):

    count_increase = 0 # licznik przyrostu populacji
    result_increase = []

    env = Environment(alpha_init=config.alpha0, c=config.c, delta=config.delta)
    pop = Population(size=config.N, n_dim=config.n)
    finish_gif = False  # zmienna służąca do zapisania w gifie momentu wymarcia populacji

    print("Starting simulation")


    last_gen = config.N

    for generation in range(1, config.max_generations):
        # 1. Mutacja
        mutate_population(pop, mu=mut, mu_c=config.mu_c, xi=config.xi)
        # 2. Selekcja
        survivors = threshold_selection(pop, env.get_optimal_phenotype(), sigma, config.threshold_surv)
        for individual in survivors:
            individual.set_pair(None)
        pop.set_individuals(survivors)

        if len(survivors) > 0:
            proportional_selection(pop, env.get_optimal_phenotype(), sigma, config.N)
        else:
            print(f"Wszyscy wymarli w pokoleniu {generation}. Kończę symulację.")
            finish_gif = True

        # 3. Reprodukcja
        # Dobieranie w pary (ten, kto się nie dobierze, ten się nie rozmnaża)
        asexuals = threshold_selection(pop, env.get_optimal_phenotype(), sigma, config.threshold_asex)
        # zmiana atrybutu - rozmnaża się sam
        for individual in asexuals:
            individual.set_pair(individual)

        sex_to_pair = [s for s in survivors if
                       s not in asexuals]  # lista osobników, które w tej generacji rozmnażają się płciowo
        # parowanie osobników - rozmnażanie płciowe
        sex_paired = pop.set_pairs(sex_to_pair)
        asex_paired = [(ind, ind) for ind in asexuals]
        # lista wszystkich par w populacji - płciowe i bezpłciowe
        all_paired = sex_paired + asex_paired



        # 4. Zmiana środowiska
        env.update()


        increase = len(pop.get_individuals()) - last_gen # wywalić, że osobniki wymierają po 5 latach
        count_increase += increase
        last_gen = len(pop.get_individuals())

        if finish_gif:
            result_increase.append(count_increase/generation)
            break

    print("Symulacja zakończona.")
    return statistics.median(result_increase)


def multi_simulations(mut_v, sigma_v):

    matrix_increase=[] # wiersze dla mut, kolumny dla sigma

    for mut in mut_v:
        v = []
        for sigma in sigma_v:
            increase = one_simulation(mut, sigma)
            v.append(increase)
        matrix_increase.append(v)

    return matrix_increase

increase_pop = multi_simulations([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
heatmap_mut_sel([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9], increase_pop)

