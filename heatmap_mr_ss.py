import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import statistics
from decimal import Decimal, ROUND_HALF_UP
from multiprocessing import Pool

import os
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
            text_color = 'white' if increase_pop[i][j] < 0.2 else 'black'
            # pierwszy indeks to numer kolumny, drugi to numer wiersza
            text = ax.text(j,i, increase_pop[i][j], ha='center', va='center', color=text_color) # indeksy upewnić się

    ax.set_title('Mediana średniej liczby urodzeń w populacji')
    ax.set_xlabel('Siła selekcji')
    ax.set_ylabel('Częstość mutacji')
    fig.tight_layout()

    path = f'C:\\Users\\Anastazja\\Desktop\\heatmaps\\heatmapa6.png'
    if not os.path.exists(path): path = "heatmapa6.png"
    plt.savefig(path, dpi=300, bbox_inches = 'tight')
    plt.show()


def one_simulation(mut, sigma):

    for_small = True
    final_result = 0
    count_increase = 0 # licznik przyrostu populacji
    result_increase = []

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
        survivors = threshold_selection(pop, env.get_optimal_phenotype(), sigma, config.threshold_surv)
        for individual in survivors:
            individual.set_pair(None)
            individual.set_age(individual.get_age() + 1)
            #if individual.get_age() > config.lifespan:
                #dying.append(individual)
        #survivors = [ind for ind in survivors if ind not in dying]
        pop.set_individuals(survivors)

        if len(survivors) <= 0:
            print(f"Wszyscy wymarli w pokoleniu {generation}. Kończę symulację.")
            finish_gif = True
        if finish_gif: break
        # print("Survivors:", survivors, len(survivors))

        # 3. Reprodukcja
        # Bezpłciowa
        asexuals = threshold_selection(pop, env.get_optimal_phenotype(), sigma, config.threshold_asex)
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

        children_phenotypes = reproduction(all_paired, len(survivors), env.get_optimal_phenotype(), sigma)
        # print("Children phenotypes:", children_phenotypes, len(children_phenotypes))
        pop.add_individuals(children_phenotypes)
        # print("New population:", pop.get_individuals(), len(pop.get_individuals()))

        '''
        odgórnie osobniki dobierane w pary, zapamiętują z kim są w parze lub gdy w ogóle się nie rozmnażają
        reproduction robi update populacji: nie rozmnażające się pozostają, aseksualne się klonują, 
        płciowe samice i jej dzieci wchodzą do populacji, samce umierają? (na początku wszyscy przeżywają)
        '''

        # 4. Zmiana środowiska
        # TODO: add nurture, environmental adaptation (ensure the asexual individuals aren't stacked in one dot)
        env.update()


        #count_increase = len(pop.get_individuals()) - config.N
        increase = len(pop.get_individuals()) - last_gen
        count_increase += increase
        last_gen = len(pop.get_individuals())
    mean_quantity = statistics.mean(pop_quantity)



    if gen > 100 and for_small == False:
        final_result = Decimal(str(count_increase / gen)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    elif gen>100 and for_small==True:
        final_result = Decimal(str(count_increase / gen)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    else:
        final_result = None

    print("Symulacja zakończona.")
    return final_result


def multi_simulations(mut_v, sigma_v):
    np.random.seed(config.seed)
    matrix_increase=[] # wiersze dla mut, kolumny dla sigma
    mut_vr = sorted(mut_v, reverse=True)

    for mut in mut_vr:
        v = []
        for sigma in sigma_v:
            for_med = []
            for i in range(5):
                increase = one_simulation(mut, sigma)
                if increase != None:
                    for_med.append(increase)
                else:continue
            if len(for_med) == 0: v.append(0)
            else:
                v.append(float(statistics.median(for_med)))
        matrix_increase.append(v)

    return matrix_increase

#increase_pop = multi_simulations([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
#heatmap_mut_sel([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9], increase_pop)

increase_pop = multi_simulations([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.15, 0.2, 0.25, 0.3])
heatmap_mut_sel([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.15, 0.2, 0.25, 0.3], increase_pop)


