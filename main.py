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

def main():
    env = Environment(alpha_init=config.alpha0, c=config.c, delta=config.delta)
    pop = Population(size=config.N, n_dim=config.n)
    finish_gif = False # zmienna służąca do zapisania w gifie momentu wymarcia populacji

    print("Starting simulation")
    # Katalog, w którym zapisujemy obrazki (możesz nazwać np. "frames/")
    frames_dir = "frames"
    if os.path.exists(frames_dir): shutil.rmtree(frames_dir) # upewnia się, że frames z poprzedniej symulacji nie wejdą do nowego gifa
    os.makedirs(frames_dir, exist_ok=True)  # tworzy folder

    for generation in range(config.max_generations):
        # 1. Mutacja
        mutate_population(pop, mu=config.mu, mu_c=config.mu_c, xi=config.xi)

        # 2. Selekcja
        survivors = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_surv)
        for individual in survivors:
            individual.set_pair(None)
        pop.set_individuals(survivors) # what does this mean?

        if len(survivors) <= 0:
            print(f"Wszyscy wymarli w pokoleniu {generation}. Kończę symulację.")
            finish_gif = True
        if finish_gif: break
        print("Survivors:", survivors)

        # 3. Reprodukcja
        # Bezpłciowa
        asexuals = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_asex)
        # zmiana atrybutu - rozmnaża się sam
        asex_paired = []
        for individual in asexuals:
            individual.set_pair(individual)
            asex_paired +=  [(individual,individual)]
        print("Asexuals paired:", asex_paired)

        # Płciowa
        # Dobieranie w pary (ten, kto się nie dobierze, ten się nie rozmnaża)
        sex_to_pair = [s for s in survivors if s not in asexuals] # lista osobników, które w tej generacji rozmnażają się płciowo
        # parowanie osobników - rozmnażanie płciowe
        sex_paired = pop.set_pairs(sex_to_pair)
        print("Sexual paired:", sex_paired)

        # lista wszystkich par w populacji - płciowe i bezpłciowe
        all_paired = sex_paired + asex_paired
        print("All paired:", all_paired)

        new_population = np.append(survivors, reproduction(all_paired, env.get_optimal_phenotype(), config.sigma, len(survivors)))
        pop.set_individuals(new_population) # wszyscy rodzice i potomkowie

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
    create_gif_from_frames(frames_dir, "simulation.gif")
    print("GIF zapisany jako simulation.gif")

def create_gif_from_frames(frames_dir, gif_filename, duration=0.5):
    """
    Łączy wszystkie obrazki z katalogu `frames_dir` w jeden plik GIF.
    Wymaga biblioteki imageio (pip install imageio).
    :param frames_dir: folder z plikami .png
    :param gif_filename: nazwa pliku wyjściowego GIF
    :param duration: czas wyświetlania jednej klatki w sekundach
    """
    import imageio.v2 as imageio
    import os

    # Sortujemy pliki po nazwach, żeby zachować kolejność generacji
    filenames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    
    with imageio.get_writer(gif_filename, mode='I', duration=duration) as writer:
        for file_name in filenames:
            path = os.path.join(frames_dir, file_name)
            image = imageio.imread(path)
            '''
            /home/karina/Documents/Bioinf_III/Systems_biology/SysBiol_Proj1/main.py:84: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
  image = imageio.imread(path)
            '''
            writer.append_data(image)


if __name__ == "__main__":
    main()
