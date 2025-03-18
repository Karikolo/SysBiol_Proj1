# reproduction.py

import copy
import numpy as np

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
    
