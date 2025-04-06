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
        self.age = 0
        self.sex_reproduction = False

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

    def set_age(self, age):
        self.age = age

    def get_age(self):
        return self.age

    def set_sex_reproduction(self, sex_reproduction):
        self.sex_reproduction = sex_reproduction

    def get_sex_reproduction(self):
        return self.sex_reproduction