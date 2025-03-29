# config.py

import numpy as np

# -------------------
# PARAMETRY POPULACJI
# -------------------
K = 200           # limit pojemności siedliska
N = 20           # liczba osobników w populacji
n = 2            # wymiar przestrzeni fenotypowej
lifespan = 4

# --------------------
# PARAMETRY MUTACJI
# --------------------
mu = 0.5         # prawdopodobieństwo mutacji dla osobnika
mu_c = 0.7       # prawdopodobieństwo mutacji konkretnej cechy, jeśli osobnik mutuje
xi = 0.1         # odchylenie standardowe w rozkładzie normalnym mutacji

# --------------------
# PARAMETRY SELEKCJI
# --------------------
sigma = 0.5     # parametr w funkcji fitness (kontroluje siłę selekcji)
threshold_surv = 0.2  # przykładowy próg do selekcji progowej (do ewentualnego użycia)
threshold_asex = 0.9 # próg do selekcji progowej osobników mogących się rozmnażać bezpłciowo
#threshold_sex = 0.2
# --------------------
# PARAMETRY ŚRODOWISKA
# --------------------
# Początkowe alpha(t)
alpha0 = np.array([0.0, 0.0])  
# Wektor kierunkowej zmiany c
c = np.array([0.01, 0.01])     
delta = 0.1    # odchylenie standardowe dla fluktuacji
max_generations = 20  # liczba pokoleń do zasymulowania

# ----------------------
# PARAMETRY REPRODUKCJI
# ----------------------
avg_children = 3
# W wersji bezpłciowej zakładamy klonowanie z uwzględnieniem mutacji.
# Jeśli chcemy modelować płciowo, trzeba dodać odpowiednie parametry.
