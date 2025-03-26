# individual.py

import numpy as np

class Individual:
    """
    Klasa opisująca pojedynczego osobnika.
    Przechowuje wektor fenotypu w n-wymiarowej przestrzeni.
    """
    def __init__(self, phenotype):
        self.phenotype = phenotype
        self.pair = None

    def get_phenotype(self):
        return self.phenotype

    def set_phenotype(self, new_phenotype):
        self.phenotype = new_phenotype
    
    def get_pair(self):
        return self.pair
    
    def set_pair(self, individual):
        self.pair = individual

    def get_sex(self):
        return self.phenotype[-1]

    def print_phenotype(self):
        list = []
        for value in self.phenotype:
            list.append(value)
        return list