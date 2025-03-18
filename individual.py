# individual.py

import numpy as np

class Individual:
    """
    Klasa opisująca pojedynczego osobnika.
    Przechowuje wektor fenotypu w n-wymiarowej przestrzeni.
    """
    def __init__(self, phenotype):
        self.phenotype = phenotype
        self.pair = self
        self.sex = 1

    def get_phenotype(self):
        return self.phenotype

    def set_phenotype(self, new_phenotype):
        self.phenotype = new_phenotype
    
    def get_pair(self):
        return self.pair
    
    # def set_pairs(individuals):
    # [(A, B), ]

    def get_sex(self):
        return self.sex