import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import statistics

import config
from environment import Environment
from population import Population
from mutation import mutate_population
from selection import proportional_selection, threshold_selection
from reproduction import reproduction


'''
    mut_rates - wektor z wartosciami parametru mutation rate
    sigma_val - wektor z wartościami parametru selection strength
    mean_pop - mediana ze średnich liczności populacji we wszystkich generacjach
    
'''
# Wykres pierwszy - HEATMAPA
# Mediana średniego przyrostu populacji



def heatmap_mut_sel(mut_rates, sigma_val, increase_pop):

    fig, ax = plt.subplots()
    im = ax.imshow(increase_pop)
    mut_vr = sorted(mut_rates, reverse=True)

    ax.set_xticks(range(len(sigma_val)), labels = sigma_val) # ustawienie podpisów jest default
    ax.set_yticks(range(len(mut_vr)), labels = mut_vr)

    for i in range(len(sigma_val)):
        for j in range(len(mut_vr)):
            text_color = 'white' if increase_pop[i][j] < 0.5 else 'black'
            # pierwszy indeks to numer kolumny, drugi to numer wiersza
            text = ax.text(j,i, increase_pop[i][j], ha='center', va='center', color=text_color) # indeksy upewnić się

    ax.set_title('Mediana średniego przyrostu populacji')
    ax.set_xlabel('Siła selekcji')
    ax.set_ylabel('Częstość mutacji')
    fig.tight_layout()

    plt.savefig(f'C:\\Users\\Anastazja\\Desktop\\heatmaps\\heatmapa_mut_sel__kier_zm{config.c[0]}.png', dpi=300, bbox_inches = 'tight')
    plt.show()


def one_simulation(mut, sigma):

    count_increase = 0 # licznik przyrostu populacji
    result_increase = []

    env = Environment(alpha_init=config.alpha0, c=config.c, delta=config.delta)
    pop = Population(size=config.N, n_dim=config.n)
    finish_gif = False  # zmienna służąca do zapisania w gifie momentu wymarcia populacji

    print("Starting simulation")

    last_gen = config.N
    gen = 0

    for generation in range(1, config.max_generations):
        dying = []
        gen+=1

        # 1. Mutacja
        mutate_population(pop, mu=mut, mu_c=config.mu_c, xi=config.xi)

        # 2. Selekcja
        survivors = threshold_selection(pop, env.get_optimal_phenotype(), sigma, config.threshold_surv)
        for individual in survivors:
            individual.set_pair(None)
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
        asexuals = threshold_selection(pop, env.get_optimal_phenotype(), sigma, config.threshold_asex)
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

        children_phenotypes = reproduction(all_paired, env.get_optimal_phenotype(), sigma, len(survivors))
        # print("Children phenotypes:", children_phenotypes, len(children_phenotypes))
        pop.add_individuals(children_phenotypes)
        # print("New population:", pop.get_individuals(), len(pop.get_individuals()))


        # 4. Zmiana środowiska
        env.update()


        increase = len(pop.get_individuals()) - last_gen # wywalić, że osobniki wymierają po 5 latach
        count_increase += increase
        last_gen = len(pop.get_individuals())




    print("Symulacja zakończona.")
    return round(count_increase / gen, 2)


def multi_simulations(mut_v, sigma_v):

    matrix_increase=[] # wiersze dla mut, kolumny dla sigma
    mut_vr = sorted(mut_v, reverse=True)

    for mut in mut_vr:
        v = []
        for sigma in sigma_v:
            for_med = []
            for i in range(5):
                increase = one_simulation(mut, sigma)
                for_med.append(increase)
            v.append(statistics.median(for_med))
        matrix_increase.append(v)

    return matrix_increase

increase_pop = multi_simulations([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
#increase_pop = multi_simulations([0.1,0.5], [0.1,0.5])
heatmap_mut_sel([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9], increase_pop)
#heatmap_mut_sel([0.1,0.5], [0.1,0.5], increase_pop)

