# reproduction.py

import copy
import numpy as np
import config
from selection import fitness_function


def create_children(parents, number):
    children = []
    for child in range(number):
        # randomly choose the values of traits from among parental alleles
        phenotype = [np.random.choice([x,y]) for (x,y) in zip(parents[0].get_phenotype()[:-1], parents[0].get_phenotype()[:-1])]
        sex = [np.random.choice(parents[0].get_phenotype()[-1], parents[0].get_phenotype()[-1])]
        phenotype = np.append(phenotype, sex)
        children.append(phenotype)

    return children



def create_children(parents, number):
    children = []
    for child in range(number):
        # randomly choose the values of traits from among parental alleles
        phenotype = [np.random.choice([x,y]) for (x,y) in zip(parents[0].get_phenotype()[:-1], parents[0].get_phenotype()[:-1])]
        sex = [np.random.choice(parents[0].get_phenotype()[-1], parents[0].get_phenotype()[-1])]
        phenotype = np.append(phenotype, sex)
        children.append(phenotype)
    return children


def asexual_reproduction(all_paired, N, alpha, sigma):
    """
    - Zakładamy, że populacja nie musi zapełnić wszystkich dostępnych miejsc w siedlisku


    """
    new_population = []
    if len(all_paired)*2 == 0:
        # Zabezpieczenie: jeśli wszyscy wymarli, inicjujemy od nowa (albo zatrzymujemy symulację).
        return []
    # Mieszanie par, żeby posiadanie dzieci nie zależało od fitness
    np.random.shuffle(all_paired)
    # Wyliczenie fitness par w populacji
    fitnesses = [(pair[0].get_phenotype()[:-1] + pair[1].get_phentype()[:-1]) / 2
                 for pair in all_paired]
    total_fitness = sum(fitnesses)

    # Ustalenie miejsc dla dzieci i ile para urodziła dzieci ostatecznie
    free_spots = config.K - 2 * len(all_paired)
    for pair in all_paired:
        ex_value = free_spots/len(all_paired)
        p = 1 / (ex_value + 1)

        if total_fitness == 0:
            # Jeśli całkowite fitness jest 0, to każdy osobnik dostaje równą szansę
            pair_fitness = [1.0 / len(all_paired)] * len(all_paired)
        else:
            # fitness pary - średnia ich fitnessów
            pair_fitness = (pair[0].get_phenotype()[:-1] + pair[1].get_phenotype()[:-1]) / 2

        spots_children = min(np.random.geometric(p), free_spots) #TODO: Poisson może lepiej?
        total_children = 0
        for i in range(spots_children):
            yes = np.random.choice([0,1], p=pair_fitness/total_fitness)
            if yes: total_children+=1
        # aktualizujemy liczbe dostępnych miejsc dla potomstwa pozostałych par
        free_spots-=total_children




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
    
