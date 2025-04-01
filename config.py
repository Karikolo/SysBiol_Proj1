# config.py

import numpy as np

# -------------------
# PARAMETRY POPULACJI
# -------------------
K = 250           # 200 limit pojemności siedliska
N = 20           # liczba osobników w populacji
n = 2            # wymiar przestrzeni fenotypowej
lifespan = 5

# --------------------
# PARAMETRY MUTACJI
# --------------------
mu = 0.8         # prawdopodobieństwo mutacji dla osobnika
mu_c = 0.7       # 0,7 prawdopodobieństwo mutacji konkretnej cechy, jeśli osobnik mutuje
xi = 0.3        # odchylenie standardowe w rozkładzie normalnym mutacji

# --------------------
# PARAMETRY SELEKCJI
# --------------------
sigma = 0.5     # parametr w funkcji fitness (kontroluje siłę selekcji)
threshold_surv = 0.2  # przykładowy próg do selekcji progowej (do ewentualnego użycia)
threshold_asex = 0.9 # 0,9próg do selekcji progowej osobników mogących się rozmnażać bezpłciowo
#threshold_sex = 0.2
# --------------------
# PARAMETRY ŚRODOWISKA
# --------------------
# Początkowe alpha(t)
alpha0 = np.array([0.0, 0.0])  
# Wektor kierunkowej zmiany c
c = np.array([0.01, 0.01])     # [0.01, 0.01]
delta = 0.15    # odchylenie standardowe dla fluktuacji
max_generations = 200  # 20 liczba pokoleń do zasymulowania
seed = 10 # używane dla porównywania tych samych symulacji w analizach

# ----------------------
# PARAMETRY REPRODUKCJI
# ----------------------
avg_children = 2
# W wersji bezpłciowej zakładamy klonowanie z uwzględnieniem mutacji.
# Jeśli chcemy modelować płciowo, trzeba dodać odpowiednie parametry.
