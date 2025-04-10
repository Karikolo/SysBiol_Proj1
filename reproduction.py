# reproduction.py

import copy
import numpy as np
import config
from selection import fitness_function


def create_children(parents, number):
    children = []
    for child in range(number):
        # randomly choose the values of traits from among parental alleles
        phenotype = [np.random.choice([x,y]) for (x,y) in zip(parents[0].get_phenotype()[:-1], parents[1].get_phenotype()[:-1])]
        #print("Create children, parent's phenotype: ", parents[0].get_phenotype() , " and sex: ", parents[0].get_phenotype()[-1], )
        sex = [np.random.choice([int(parents[0].get_phenotype()[-1]), int(parents[1].get_phenotype()[-1])])]
        phenotype = phenotype + sex
        #print("Phenotype:", phenotype)
        children.append(phenotype)

    return children





def reproduction(all_paired, N, alpha, sigma):
    """
    Wersja bezpłciowa (klonowanie):
    - Zakładamy, że potomków będzie tyle, aby utrzymać rozmiar populacji = N.
    - W najprostszej wersji: jeżeli mamy M ocalałych, 
      a M < N, to klonujemy ich losowo aż do uzyskania N osobników.
    """
    #print("N in reproduction: ", N)
    new_population = []
    if len(all_paired) == 0:
        # Zabezpieczenie: jeśli wszyscy wymarli, inicjujemy od nowa (albo zatrzymujemy symulację).
        return []

    # Mieszanie par, żeby możliwość posiadania dzieci nie zależało od fitness
    np.random.shuffle(all_paired)

    #print("All paired: ", [(pair[0].get_phenotype(), pair[1].get_phenotype()) for pair in all_paired])
    # Wyliczenie średniego fitness dla par w populacji
    #print("Sigma in repr:", sigma)
    #print("Parent's fitnesses: ",(fitness_function(all_paired[0][0].get_phenotype(),alpha, sigma), fitness_function(all_paired[0][1].get_phenotype(), alpha, sigma)))

    fitnesses = [((fitness_function(pair[0].get_phenotype(),alpha, sigma) +
                      fitness_function(pair[1].get_phenotype(), alpha, sigma))/2) for pair in all_paired]
    #print("Fitnesses: ", fitnesses)

    total_fitness = sum(fitnesses)*2
    #print("Total fitnesses: ", total_fitness)


    # Ustalenie możliwej liczby dzieci i liczby dzieci, które para ostatecznie urodziła
    free_spots = config.K - N
    #print("First free spots: ", free_spots)
    for i,pair in enumerate(all_paired):
        #print("Free spots: ", free_spots)

        ex_value = min(config.avg_children,free_spots/len(all_paired)) # oczekiwana liczba dzieci dla pary
        #p = 1 / (ex_value + 1)

        pair_fitness = fitnesses[i]
        lambda_values = (1 - pair_fitness / total_fitness) * ex_value
        if total_fitness == 0:
            # Jeśli całkowite fitness jest 0, (każdy osobnik jest w optimum), to każdy osobnik dostaje równą szansę
            pair_fitness = 1.0 / len(all_paired)
            lambda_values = pair_fitness * ex_value


        spots_children = np.minimum(np.random.poisson(lambda_values), free_spots)  # Ensure we don't exceed free spots
        #print("Spots children: ", spots_children)
        total_children = np.sum(spots_children)
        if total_children > 0 :
            pair[0].set_sex_reproduction(True)
            pair[1].set_sex_reproduction(True)

        '''spots_children = min(np.random.poisson(ex_value), free_spots) #TODO: Poisson może lepiej?
        total_children = 0
        for i in range(spots_children):
            prob = 1 - (pair_fitness[i] / total_fitness)  # good fitness -> low value -> pair_fitness[i] / total_fitness low, so we
            yes = np.random.choice([0,1], p=[1-prob, prob])
            if yes: total_children+=1'''

        # aktualizujemy liczbe dostępnych miejsc dla potomstwa pozostałych par
        free_spots-=total_children

        # tworzymy osobniki potomne dla pary i dodajemy je do listy
        pairs_children = create_children(pair, total_children)
        new_population += pairs_children
    return new_population

"""
    while len(new_population) < N:
        # parent = copy.deepcopy(survivors[0])  # np. zawsze klonuj pierwszego (do testów)
        # W praktyce można klonować losowo: 
        parent = copy.deepcopy(np.random.choice(survivors)) # , p=probabilities
        new_population.append(parent)

    return new_population[:N]  # przycinamy, gdyby było za dużo

"""


def sexual_reproduction(survivors, N):
    """
    Wesja płciowa:
    - 
    """
    if len(survivors) == 0:
        # Zabezpieczenie: jeśli wszyscy wymarli, inicjujemy od nowa (albo zatrzymujemy symulację).
        return []
    
