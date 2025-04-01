# main.py

import os
import shutil
import numpy as np
import config
from environment import Environment
from population import Population
from mutation import mutate_population
from selection import proportional_selection, threshold_selection
from reproduction import reproduction
from visualization import plot_population

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def simulation(env, mut):
    c = env[0]
    delta = env[1]
    env = Environment(alpha_init=config.alpha0, c=c, delta=delta)
    pop = Population(size=config.N, n_dim=config.n)
    finish_gif = False  # zmienna służąca do zapisania w gifie momentu wymarcia populacji

    print("Starting simulation")

    # Zapis aktualnego stanu populacji do pliku PNG od pierwszego pokolenia
    plot_population(pop, env.get_optimal_phenotype(), 0, save_path=frame_filename, show_plot=False)
    # dying = []
    for generation in range(1, config.max_generations):
        # print("Generation: ", generation, "\n", "Population: ", [ind.get_phenotype() for ind in pop.get_individuals()], "\n")
        dying = []

        # 1. Mutacja
        mutate_population(pop, mu=config.mu, mu_c=config.mu_c, xi=config.xi)

        # 2. Selekcja
        survivors = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_surv)
        for individual in survivors:
            individual.set_pair(None)
            individual.set_age(individual.get_age() + 1)
            if individual.get_age() > config.lifespan:
                dying.append(individual)
        survivors = [ind for ind in survivors if ind not in dying]
        pop.set_individuals(survivors)
        # print("Survivors: ", [ind.get_phenotype() for ind in survivors])

        if len(survivors) <= 0:
            print(f"Wszyscy wymarli w pokoleniu {generation}. Kończę symulację.")
            finish_gif = True
        if finish_gif: break
        # print("Survivors:", survivors, len(survivors))

        # 3. Reprodukcja
        # Bezpłciowa
        asexuals = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_asex)
        # print("Asexuals:",  [ind.get_phenotype() for ind in asexuals], len(asexuals))

        # zmiana atrybutu - rozmnaża się sam
        asex_paired = []
        # asex_female = []
        for individual in asexuals:
            # remove all males from asexual reproduction:
            individual.set_pair(individual)
            asex_paired += [(individual, individual)]
            '''if individual.get_phenotype()[-1] == 0:
                individual.set_pair(individual)
                asex_paired +=  [(individual,individual)]
                asex_female.append(individual)'''
        # print("Asexuals paired (not males):", len(asex_paired))

        # Płciowa
        # Dobieranie w pary (ten, kto się nie dobierze, ten się nie rozmnaża)
        # sex_to_pair = [s for s in survivors if s not in asex_female] # lista osobników, które w tej generacji rozmnażają się płciowo
        sex_to_pair = [s for s in survivors if
                       s not in asexuals]  # lista osobników, które w tej generacji rozmnażają się płciowo

        # parowanie osobników - rozmnażanie płciowe
        sex_paired = pop.set_pairs(sex_to_pair)
        # print("Sexual paired:", sex_paired)

        # lista wszystkich par w populacji - płciowe i bezpłciowe
        all_paired = sex_paired + asex_paired
        # print("All paired:", all_paired)

        children_phenotypes = reproduction(all_paired, len(survivors), env.get_optimal_phenotype(), config.sigma)
        # print("Children phenotypes:", children_phenotypes, len(children_phenotypes))
        pop.add_individuals(children_phenotypes)
        # print("New population:", pop.get_individuals(), len(pop.get_individuals()))

        '''
        odgórnie osobniki dobierane w pary, zapamiętują z kim są w parze lub gdy w ogóle się nie rozmnażają
        reproduction robi update populacji: nie rozmnażające się pozostają, aseksualne się klonują, 
        płciowe samice i jej dzieci wchodzą do populacji, samce umierają? (na początku wszyscy przeżywają)
        '''

        # 4. Zmiana środowiska
        # TODO: add nurture, environmental adaptation (ensure the asexual individuals aren't stacked in one dot)
        env.update()

        # Zapis aktualnego stanu populacji do pliku PNG
        frame_filename = os.path.join(frames_dir, f"frame_{generation:03d}.png")
        plot_population(pop, env.get_optimal_phenotype(), generation, save_path=frame_filename, show_plot=False)

    print("Symulacja zakończona. Tworzenie GIF-a...")

    # Tutaj wywołujemy funkcję, która połączy zapisane klatki w animację
    mut_vs_env_plot(frames_dir, "simulation.gif")
    print("GIF zapisany jako simulation.gif")


def mut_vs_env_plot(population, alpha, generation, save_path=None, show_plot=False):


    # create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(start_year, start_year + populations.shape[1] - 1)
    ax.set_ylim(0, populations.values.max() / 1_000_000 * 1.3)
    ax.set_xlabel("Year", fontsize = 20)
    ax.set_ylabel("Population (millions)", fontsize = 20)
    ax.set_title(f"Population Growth Over Time for {task_dict[task][0]}", fontsize = 20)

    # initialize line plots
    lines = []
    for i in range(len(countries)):
        line, = ax.plot([], [], label=countries.iloc[i])
        lines.append(line)
    ax.legend(loc="upper left", title="Countries", fontsize=12)

    display_year = ax.text(0.5, 0.9, str(start_year), transform=ax.transAxes, fontsize=20, ha="center")

def main():
    """
    script creating the mutation rate vs environmental conditions violin plots
    (several subplots, one subplot per environment, one violin plot per mutation rate)
    as well as the sexual reproduction to all reproduction proportion vs encironmental conditions line plot
    (one plot, for each set of conditions one line, shows progression of sexual reproduction percentage over time???)
    TODO: check if the last one really is the best way to present the data, maybe if the change was happening faster? bar plot?
    Returns
    -------

    """
    np.random.seed(10)

    # weak, medium, strong direction, strong sudden change, strong both:
    envs = [[np.array([0.01, 0.01]), 0.01], [np.array([0.1, 0.1]), 0.1], [np.array([0.01, 0.01]), 0.5],
            [np.array([0.5, 0.5]), 0.01], [np.array([0.5, 0.5]), 0.5]]

    # only mutation rate is considered: no mutation, very weak,  weak, medium, high (half of the population)
    mutations = [0, 0.05, 0.1, 0.3, 0.5]
    ''' Parametry mutacji:
    mu = 0.1
    mu_c = 0.5       # prawdopodobieństwo mutacji konkretnej cechy, jeśli osobnik mutuje
    xi = 0.3        # odchylenie standardowe w rozkładzie normalnym mutacji
    Parametry środowiska:
    c = np.array([0.01, 0.01])     # [0.01, 0.01]
    delta = 0.15  
    '''
    settings = {}
    for env in envs:
        for mut in mutations:
            popul, sex_to_repr_proport = simulation(env, mut)
            # popul =
            settings[(env, mut)] = popul, sex_to_repr_proport

if __name__ == "__main__":
    main()
