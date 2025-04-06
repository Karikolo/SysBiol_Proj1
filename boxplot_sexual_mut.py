import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import os
import config
from environment import Environment
from population import Population
from mutation import mutate_population
from selection import proportional_selection, threshold_selection
from reproduction import reproduction

# Wykres 3 - średnia częstość rozmnażania płciowego w zależności od częstości mutacji

def boxplot_sexvsmut(mean_matrix, labels):

    fig, ax  = plt.subplots()
    ax.set_ylabel('Częstość rozmnażania płciowego')
    ax.set_xlabel('Częstość mutacji')
    ax.set_title('Częstosć rozmnażania płciowego w zależności od częstosci mutacji')

    colors = ['green', 'blue', 'yellow', 'magenta', 'grey']

    bplot = ax.boxplot(mean_matrix,
                       patch_artist=True,
                       tick_labels = labels)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    fig.tight_layout()
    path =f'C:\\Users\\Anastazja\\Desktop\\boxplots\\boxplot11.png'
    if not os.path.exists(path): path = "boxplot11.png"
    plt.savefig(path, dpi=300, bbox_inches = 'tight')
    plt.show()


def one_simulation(mut):

    sex_success = 0
    asex_success = 0
    env = Environment(alpha_init=config.alpha0, c=config.c, delta=config.delta)
    pop = Population(size=config.N, n_dim=config.n)
    finish_gif = False  # zmienna służąca do zapisania w gifie momentu wymarcia populacji

    print("Starting simulation")

    last_gen = config.N
    gen = 0
    pop_quantity = []

    for generation in range(1, config.max_generations):
        gen+=1
        pop_quantity.append(len(pop.get_individuals()))
        dying = []

        # 1. Mutacja
        mutate_population(pop, mu=config.mu, mu_c=mut, xi=config.xi)

        # 2. Selekcja
        survivors = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_surv)
        for individual in survivors:
            individual.set_pair(None)
            individual.set_sex_reproduction(False)
            individual.set_age(individual.get_age() + 1)
            if individual.get_age() > config.lifespan:
                dying.append(individual)
        survivors = [ind for ind in survivors if ind not in dying]
        pop.set_individuals(survivors)

        if len(survivors) <= 0:
            print(f"Wszyscy wymarli w pokoleniu {generation}. Kończę symulację.")
            finish_gif = True
        if finish_gif: break
        # print("Survivors:", survivors, len(survivors))

        # 3. Reprodukcja
        # Bezpłciowa
        asexuals = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_asex)
        # print("Asexuals:",  [ind.get_phenotype() for ind in asexuals], len(asexuals))

        # zmiana atrybutu - rozmnaża się sam
        asex_paired = []
        # asex_female = []
        for individual in asexuals:
            # remove all males from asexual reproduction:
            individual.set_pair(individual)
            asex_paired += [(individual, individual)]
            '''if individual.get_phenotype()[-1] == 0:
                individual.set_pair(individual)
                asex_paired +=  [(individual,individual)]
                asex_female.append(individual)'''
        # print("Asexuals paired (not males):", len(asex_paired))

        # Płciowa
        # Dobieranie w pary (ten, kto się nie dobierze, ten się nie rozmnaża)
        # sex_to_pair = [s for s in survivors if s not in asex_female] # lista osobników, które w tej generacji rozmnażają się płciowo
        sex_to_pair = [s for s in survivors if
                       s not in asexuals]  # lista osobników, które w tej generacji rozmnażają się płciowo

        # parowanie osobników - rozmnażanie płciowe
        sex_paired = pop.set_pairs(sex_to_pair)
        # print("Sexual paired:", sex_paired)

        # lista wszystkich par w populacji - płciowe i bezpłciowe
        all_paired = sex_paired + asex_paired
        # print("All paired:", all_paired)

        children_phenotypes = reproduction(all_paired, len(survivors), env.get_optimal_phenotype(), config.sigma)
        # print("Children phenotypes:", children_phenotypes, len(children_phenotypes))
        pop.add_individuals(children_phenotypes)
        # print("New population:", pop.get_individuals(), len(pop.get_individuals()))

        for individual in pop.get_individuals():
            if individual.get_sex_reproduction():
                if individual.get_pair()!=individual:
                    sex_success+=1
                else:
                    asex_success+=1

        '''
        odgórnie osobniki dobierane w pary, zapamiętują z kim są w parze lub gdy w ogóle się nie rozmnażają
        reproduction robi update populacji: nie rozmnażające się pozostają, aseksualne się klonują, 
        płciowe samice i jej dzieci wchodzą do populacji, samce umierają? (na początku wszyscy przeżywają)
        '''

        # 4. Zmiana środowiska
        # TODO: add nurture, environmental adaptation (ensure the asexual individuals aren't stacked in one dot)
        env.update()
    #if gen<100: return None
    #else:

    all_success = sex_success/2 + asex_success
    print("Symulacja zakończona.")
    if all_success == 0:
        return 0
    else:
        return (sex_success/all_success)




def multi_simulations(mut_v):

    matrix_success=[] # wiersze dla mut

    for mut in mut_v:
        np.random.seed(config.seed)
        success = []
        for i in range(10):
            sim = one_simulation(mut)
            if sim is not None:
                success.append(sim)
            else: continue
        matrix_success.append(success)

    return matrix_success

matrix = multi_simulations([0.1, 0.3, 0.5, 0.7, 0.9])
boxplot_sexvsmut(matrix, [0.1, 0.3, 0.5, 0.7, 0.9])