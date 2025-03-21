# population.py

import numpy as np
from individual import Individual

class Population:
    """
    Klasa przechowuje listę osobników (Individual)
    oraz pomaga w obsłudze różnych operacji na populacji.
    """
    def __init__(self, size, n_dim):
        """
        Inicjalizuje populację losowymi fenotypami w n-wymiarach.
        :param size: liczba osobników (N)
        :param n_dim: wymiar fenotypu (n) + 1 (płeć)
        """
        self.individuals = []
        for _ in range(size):
            # przykładowo inicjalizujemy fenotypy w okolicach [0, 0, ..., 0]
            phenotype = np.random.normal(loc=0.0, scale=1.0, size=n_dim)
            sex = np.random.choice([0,1], size=1)
            phenotype = np.append(phenotype, sex) # dimention n_dim + 1
            self.individuals.append(Individual(phenotype))

    def get_individuals(self):
        return self.individuals

    def set_individuals(self, new_individuals):
        self.individuals = new_individuals

    @staticmethod
    def set_pairs(individuals):
        # Separate males and females and change each of their pair attributes to the other
        females = [ind for ind in individuals if ind.get_pheontype()[-1] == 0]
        males = [ind for ind in individuals if ind.get_pheontype()[-1] == 1]

        np.random.shuffle(females)
        np.random.shuffle(males)

        paired = list(zip(females, males))
        for pair in paired:
            pair[0].set_pair(pair[1])
            pair[1].set_pair(pair[0])
        return paired