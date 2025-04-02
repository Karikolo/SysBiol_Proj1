import numpy as np
import config
from concurrent.futures import ProcessPoolExecutor
import time

# functions and classes locally saved:
from environment import Environment
from population import Population
from mutation import mutate_population
from selection import threshold_selection
from reproduction import reproduction
from visualization import mut_vs_env_plot_with_iterations


def run_simulation(params):
    count_increase = 0 # licznik przyrostu populacji
    result_increase = []
    env_params, mut = params

    env = Environment(alpha_init=config.alpha0, c=env_params[0], delta=env_params[1])
    pop = Population(size=config.N, n_dim=config.n)
    finish_sim = False  # zmienna służąca do zapisania w gifie momentu wymarcia populacji

    #print(f"Start symulacji z parametrami: c, delta = {env_params} mu = {mut}")

    last_gen = config.N
    sex_success = 0

    for generation in range(1, config.max_generations+1):
        gen = generation
        # 1. Mutacja
        mutate_population(pop, mu=mut, mu_c=config.mu_c, xi=config.xi)

        # 2. Selekcja
        survivors = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_surv)
        for individual in survivors:
            individual.set_pair(None)
            individual.set_sex_reproduction(False)
            #if individual.get_age() > config.lifespan:
            #    dying += individual
        pop.set_individuals(survivors)
        #survivors = [ind for ind in survivors if ind not in dying]

        if len(survivors) <= 0:
            print(f"Wszyscy wymarli w pokoleniu {generation}. Kończę symulację.")
            finish_sim = True
        if finish_sim: break
        # print("Survivors:", survivors, len(survivors))

        # 3. Reprodukcja
        # Bezpłciowa
        asexuals = threshold_selection(pop, env.get_optimal_phenotype(), config.sigma, config.threshold_asex)
        # zmiana atrybutu - rozmnaża się sam
        asex_paired = []
        asex_female = []
        for individual in asexuals:
            # remove all males from asexual reproduction:
            if individual.get_phenotype()[-1] == 0:
                individual.set_pair(individual)
                asex_paired += [(individual, individual)]
                asex_female.append(individual)
        # print("Asexuals paired (not males):", len(asex_paired))

        # Płciowa
        # Dobieranie w pary (ten, kto się nie dobierze, ten się nie rozmnaża)
        sex_to_pair = [s for s in survivors if
                       s not in asex_female]  # lista osobników, które w tej generacji rozmnażają się płciowo
        # parowanie osobników - rozmnażanie płciowe
        sex_paired = pop.set_pairs(sex_to_pair)
        # print("Sexual paired:", sex_paired)

        # lista wszystkich par w populacji - płciowe i bezpłciowe
        all_paired = sex_paired + asex_paired
        # print("All paired:", all_paired)

        children_phenotypes = reproduction(all_paired,  len(survivors), env.get_optimal_phenotype(), config.sigma)
        # print("Children phenotypes:", children_phenotypes, len(children_phenotypes))
        pop.add_individuals(children_phenotypes)
        # print("New population:", pop.get_individuals(), len(pop.get_individuals()))

        for individual in pop.get_individuals():
            if individual.get_sex_reproduction(): sex_success+=1

        # 4. Zmiana środowiska
        env.update()

    #print("Symulacja zakończona.")
    return ((env_params[0], env_params[1]), mut), gen


"""
        settings = {}
    for env in envs:
        for mut in mutations:
            popul, sex_to_repr_proport = simulation(env, mut)
            # popul =
            settings[(env, mut)] = popul, sex_to_repr_proport


matrix = multi_simulations([0.1, 0.3, 0.5, 0.7, 0.9])
boxplot_sexvsmut(matrix, [0.1, 0.3, 0.5, 0.7, 0.9])

    """
def run_parallel(envs, mutations):
    start_time = time.time()
    envs_mutations = [(env, mut) for env in envs for mut in mutations]
    #print("Envs and mut: ", envs_mutations)
    with ProcessPoolExecutor() as executor:
        print("Starting parallel")
        # eg. [(((array([0.01, 0.01]), 0.01), 0), 114), (((array([0.01, 0.01]), 0.01), 0.05), 199), (((array([0.01, 0.01]), 0.01), 0.1), 199), (((array([0.01, 0.01]), 0.01), 0.3), 199), (((array([0.01, 0.01]), 0.01), 0.5), 199), (((array([0.1, 0.1]), 0.1), 0), 14), (((array([0.1, 0.1]), 0.1), 0.05), 14), (((array([0.1, 0.1]), 0.1), 0.1), 14), (((array([0.1, 0.1]), 0.1), 0.3), 11), (((array([0.1, 0.1]), 0.1), 0.5), 8), (((array([0.01, 0.01]), 0.5), 0), 5), (((array([0.01, 0.01]), 0.5), 0.05), 4), (((array([0.01, 0.01]), 0.5), 0.1), 5), (((array([0.01, 0.01]), 0.5), 0.3), 9), (((array([0.01, 0.01]), 0.5), 0.5), 2), (((array([0.5, 0.5]), 0.01), 0), 2), (((array([0.5, 0.5]), 0.01), 0.05), 3), (((array([0.5, 0.5]), 0.01), 0.1), 3), (((array([0.5, 0.5]), 0.01), 0.3), 2), (((array([0.5, 0.5]), 0.01), 0.5), 3), (((array([0.5, 0.5]), 0.5), 0), 3), (((array([0.5, 0.5]), 0.5), 0.05), 3), (((array([0.5, 0.5]), 0.5), 0.1), 4), (((array([0.5, 0.5]), 0.5), 0.3), 4), (((array([0.5, 0.5]), 0.5), 0.5), 2)]
        results_list = list(executor.map(run_simulation, envs_mutations))
        #print("Results list: ", results_list, "\n length: ", len(results_list))
        #results_dict = {key: value for key, value in results_list}
    end_time = time.time()
    print(f"Parallel execution time: {end_time - start_time:.2f} seconds")
    return results_list
'''
    def run_simulation(env_mut):
        env, mut = env_mut
        popul, sex_to_repr_proport = multi_simulations(env, mut)
        return ((env, mut), popul) # TODO - make it work with dict

    envs_mutations = [(env, mut) for env in envs for mut in mutations]

    with ProcessPoolExecutor() as executor:
        results = dict(executor.map(run_simulation, envs_mutations))



'''

'''
def run_multiple_times(iterations):
    np.random.seed(10)
    # weak, medium, strong direction, strong sudden change, strong both:
    envs = [[np.array([0.01, 0.01]), 0.01], [np.array([0.1, 0.1]), 0.1], [np.array([0.01, 0.01]), 0.5],
            [np.array([0.5, 0.5]), 0.01], [np.array([0.5, 0.5]), 0.5]]

    # only mutation rate is considered: no mutation, very weak,  weak, medium, high (half of the population)
    mutations = [0, 0.05, 0.1, 0.3, 0.5]

    print("\nRunning on all available cores...")
    for i in range (iterations):
        print(f"Iteration {i+1}/{iterations}")
        parallel_results = run_parallel(envs, mutations)
        mut_vs_env_plot(parallel_results)
'''
def run_multiple_times(iterations):
    np.random.seed(10)
    # Environment parameters
    envs = [[np.array([0.01, 0.01]), 0.01], [np.array([0.05, 0.05]), 0.05], [np.array([0.1, 0.1]), 0.2],
            [np.array([0.2, 0.2]), 0.1], [np.array([0.2, 0.2]), 0.2]]
    # Mutation parameters
    mutations = [0, 0.01, 0.05, 0.1, 0.2]

    # Store results by iteration
    results_by_iteration = {}

    print("\nRunning on all available cores...")
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")
        iteration_results = run_parallel(envs, mutations)
        results_by_iteration[i + 1] = iteration_results

    # Generate visualization with all iterations
    mut_vs_env_plot_with_iterations(results_by_iteration, save_path=f"bubble_plot_all_iterations_{iterations}.png")

    # Calculate average generations across iterations
    aggregated_results = {}
    '''for key in results_by_iteration[1].keys():  # Assumes all iterations have the same keys
        aggregated_results[key] = sum(results_by_iteration[i + 1][key] for i in range(iterations)) / iterations
'''
    print(f"Completed {iterations} iterations.")
    return results_by_iteration, aggregated_results

def main():
    print("Start of a script returning bubble plots for different environmental conditions and mutation rates.")
    res_by_it, aggr_res = run_multiple_times(3)
    print("Results by iteration: ", res_by_it)
    #print("Results aggregated: ", aggr_res)
    print("End of script.")


if __name__ == "__main__":
    main()
