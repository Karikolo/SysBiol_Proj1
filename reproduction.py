# reproduction.py

import copy
import numpy as np


def create_children(parents, number):
    children = []
    for child in range(number):
        # randomly choose the values of traits from among parental alleles
        phenotype = [np.random.choice([x,y]) for (x,y) in zip(parents[0].get_phenotype()[:-1], parents[0].get_phenotype()[:-1])]
        sex = [np.random.choice(parents[0].get_phenotype()[-1], parents[0].get_phenotype()[-1])]
        phenotype = np.append(phenotype, sex)
        children.append(phenotype)

    return children


def asexual_reproduction(survivors, N):
    """
    Wersja bezpłciowa (klonowanie):
    - Zakładamy, że potomków będzie tyle, aby utrzymać rozmiar populacji = N.
    - W najprostszej wersji: jeżeli mamy M ocalałych, 
      a M < N, to klonujemy ich losowo aż do uzyskania N osobników.
    """
    new_population = []
    if len(survivors) == 0:
        # Zabezpieczenie: jeśli wszyscy wymarli, inicjujemy od nowa (albo zatrzymujemy symulację).
        return []

    while len(new_population) < N:
        # parent = copy.deepcopy(survivors[0])  # np. zawsze klonuj pierwszego (do testów)
        # W praktyce można klonować losowo: 
        parent = copy.deepcopy(np.random.choice(survivors)) # , p=probabilities
        new_population.append(parent)

    return new_population[:N]  # przycinamy, gdyby było za dużo


def sexual_reproduction(survivors, N):
    """
    Wesja płciowa:
    - 
    """
    if len(survivors) == 0:
        # Zabezpieczenie: jeśli wszyscy wymarli, inicjujemy od nowa (albo zatrzymujemy symulację).
        return []
    
