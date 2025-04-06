# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import config
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


'''
TODO: 
1) legenda z licznikiem a) osobników aseksualnych b) osobników płciowych 
2) kolorowanie osobników na bazie ich rozmnażania
'''
def plot_population(population, alpha, generation, save_path=None, show_plot=False):
    """
    Rysuje populację w 2D wraz z optymalnym fenotypem alpha.
    Można zarówno wyświetlać (show_plot=True),
    jak i zapisywać obraz (save_path != None).
    """
    individuals = population.get_individuals()
    ind_reversed = []
    for i in range(len(individuals)-1, -1, -1):
        ind_reversed.append(individuals[i])
    x = np.array([ind.get_phenotype()[0] for ind in ind_reversed])
    y = np.array([ind.get_phenotype()[1] for ind in ind_reversed])
    sex = np.array([ind.get_phenotype()[-1] for ind in ind_reversed])

    '''colors = []
    for ind in population.get_individuals():
        if ind.get_pair() is ind:  # Paired with self
            colors.append("blue")
        elif ind.get_pair() is None:  # No pair
            colors.append("gray")
        else:  # Paired with another individual
            colors.append("orange")'''
    colors = np.array([
        "blue" if ind.get_pair() is ind else  # Asexual reproduction
        "yellow" if ind.get_age() == 0 else  # Only joined in this generation
        "gray" if ind.get_pair() is None else  # Did not find a mate
        "green"  # Successfully mated

        for ind in ind_reversed
    ])
    # Define legend handles
    legend_patches = [
        mpatches.Patch(color="blue", label="Asexual reproduction"),
        mpatches.Patch(color="gray", label="No mate found"),
        mpatches.Patch(color="green", label="Mated successfully"),
        mpatches.Patch(color="yellow", label="Offspring"),
        mpatches.Patch(color="red", label="Optimum")
    ]


    #plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(figsize=(5, 5))




    # Males (Triangles)
    male_mask = (sex == 1)
    ax.scatter(x[male_mask], y[male_mask], c=colors[male_mask], marker="^", alpha=0.5, label="Males")

    # Females (Circles)
    female_mask = (sex == 0)
    ax.scatter(x[female_mask], y[female_mask], c=colors[female_mask], marker="o", alpha=0.5, label="Females")

    legend_markers = [
        mlines.Line2D([], [], color='black', marker="^", linestyle="None", markersize=8, label="Males"),
        mlines.Line2D([], [], color='black', marker="o", linestyle="None", markersize=8, label="Females")
    ]
    #plt.scatter(x, y, label="Populacja", alpha=0.5, c=colors)
    ax.scatter([alpha[0]], [alpha[1]], color='red', label="Optimum", marker="x")
    # Reproduction barriers:
    circle_asex = mpatches.Circle((alpha[0], alpha[1]), np.sqrt(-2 * config.sigma ** 2 * np.log(config.threshold_asex)),
                                  fill=False, edgecolor='red',
                                  linestyle='--', label='Asexual threshold')
    circle_surv = mpatches.Circle((alpha[0], alpha[1]), np.sqrt(-2 * config.sigma ** 2 * np.log(config.threshold_surv)),
                                  fill=False, edgecolor='green',
                                  linestyle='--', label='Survival threshold')

    # Add circles to the plot
    ax.add_patch(circle_asex)
    ax.add_patch(circle_surv)


    ax.set_title(f"Pokolenie: {generation}, liczba osobników: {len(individuals)}")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.legend(handles=legend_patches+legend_markers)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)  # Zapis do pliku
    if show_plot:
        plt.show()
    else:
        # Jeśli nie chcesz pokazywać, to zamykaj figurę, 
        # żeby nie zapełniać pamięci
        plt.close()


'''
# Wykres 3 - średnia częstość rozmnażania płciowego w zależności od częstości mutacji

def boxplot_sexvsmut(mean_matrix, labels):

    fig, ax  = plt.subplots()
    ax.set_ylabel('Średnia częstość rozmnażania płciowego')
    ax.set_xlabel('Częstość mutacji')
    ax.set_title('Średnia częstosć rozmnażania płciowego w zależności od częstosci mutacji')

    colors = ['green', 'blue', 'yellow', 'magenta', 'grey']

    bplot = ax.boxplot(mean_matrix,
                       patch_artist=True,
                       tick_labels = labels)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    fig.tight_layout()

    plt.savefig(f'C:\\Users\\Anastazja\\Desktop\\boxplots\\boxplot_sexual_kier_zm{config.c[0]}.png', dpi=300, bbox_inches = 'tight')
    plt.show()
'''

# Working:
def mut_vs_env_plot_with_iterations(results_by_iteration, save_path=None):
    """
    Create a bubble plot showing environment parameters, mutation rates, and generations,
    with individual iterations displayed as smaller points to visualize variation.

    Parameters:
    -----------
    results_by_iteration : dict
        Dictionary with iteration number as keys and simulation results as values
        Each simulation result is a dict with ((c, delta), mut) tuples as keys and generation counts as values
    save_path : str, optional
        Path to save the figure, if None the figure is displayed
    """
    # Convert all results to a single DataFrame
    all_data = []
    iterations = []

    for iteration, results in results_by_iteration.items():
        iterations.append(iteration)
        for (env_params, mut), gen in results:
            c, delta = list(env_params)
            all_data.append({
                'c': tuple(c),
                'delta': delta,
                'environment': f"c={c}, δ={delta:.2f}",
                'mutation': mut,
                'mutation rate':f"mu={mut}",
                'generations': gen,
                'iteration': iteration
            })

    df = pd.DataFrame(all_data)

    # Create a figure and customize appearance
    fig, ax = plt.subplots(figsize=(15, 10))  # Create a figure and axis

    # Create custom colormap for mutation rates
    cmap = LinearSegmentedColormap.from_list('mutation_cmap', ['#2E86C1', '#D4AC0D', '#CB4335'], N=100)

    # Determine unique environment combinations for x-axis positions
    cond_x = df[['c', 'delta', 'environment']].drop_duplicates()
    cond_x_pos = dict(zip(cond_x['environment'], range(len(cond_x))))

    # Determine unique environment combinations for y-axis positions
    cond_y = df[['mutation rate', "mutation"]].drop_duplicates()
    cond_y_pos = dict(zip(cond_y['mutation rate'], range(len(cond_y))))

    # Map environment strings to x positions in both dataframes
    df['x_pos'] = df['environment'].map(cond_x_pos)
    df['y_pos'] = df['mutation rate'].map(cond_y_pos)


    # Plot individual iterations as small semi-transparent points
    for iter in df['iteration'].unique():
        iter_df = df[df['iteration'] == iter]
        size = 50 + (iter_df['generations'] / iter_df['generations'].max()) * 500
       # Approximate radius in data coordinates
        r_data = (df['iteration'] - np.median(iterations)) * np.sqrt(iter_df['generations'].max()) / 100  # Adjust 10 as needed

        # Move bubbles by radius
        iter_df['x_jitter'] = iter_df['x_pos'] + r_data
        ax.scatter(iter_df['x_jitter'], iter_df['y_pos'],
                    s=size, alpha=0.8, color=cmap(iter / len(iterations)),
                    marker='o', edgecolors='none',
                    label=f'Iteration {iter}')

        # Add text annotations for generation values
        for _, row in iter_df.iterrows():
            ax.text(row['x_jitter'], row['y_pos'] + 0.1,  # Adjust y position slightly for visibility
                    str(int(row['generations'])),  # Display generations as an integer
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    # Add environment labels to x-axis
    ax.set_xticks(list(cond_x_pos.values()))
    ax.set_xticklabels(list(cond_x_pos.keys()), rotation=45, ha='right')

    ax.set_yticks(list(cond_y_pos.values()))
    ax.set_yticklabels(list(cond_y_pos.keys()), rotation=45)

    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Create a custom color bar
    norm = plt.Normalize(0, len(iterations))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])


    # Set axis labels and title
    ax.set_xlabel('Environment Parameters', fontsize=14)
    ax.set_ylabel('Mutation rate', fontsize=14)
    ax.set_title(
        'Impact of Environment Parameters and Mutation Rates on Population Survival\n(bubble size = number of generations survived)',
        fontsize=16,
        pad = 20
    )

    # Add a text box with simulation parameters
    param_text = (
        "Simulation Parameters:\n"
        f"Population Size: {getattr(config, 'N', 'N/A')}\n"
        f"Max Generations: {getattr(config, 'max_generations', 'N/A')}\n"
        f"Survival Threshold: {getattr(config, 'threshold_surv', 'N/A')}\n"
        f"Asexual Threshold: {getattr(config, 'threshold_asex', 'N/A')}\n"
        f"Avg nr of children: {getattr(config, 'avg_children', 'N/A')}\n"
        f"Iterations: {len(results_by_iteration)}"
    )
    fig.text(0.87, 0.03, param_text, fontsize=10,
                bbox=dict(facecolor='white', edgecolor="lightgray", alpha=0.8, pad=10.0))

    # Add legend for mutation rates
    legend = ax.legend(title="Iterations", fontsize=10, loc='upper right', bbox_to_anchor=(1.12, 1), markerscale=0.6 )
    legend.get_title().set_fontsize(12)

    fig.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return df  # Return the DataFrame for further analysis if needed


def sexrepr_vs_env_plot(results, save_path=None):
    """
    Creates line plot with multiple lines representing the sexual reproduction/all reproduction ratio in various
    environmental conditions.
    Parameters
    ----------
    results =
        (env_params[0], env_params[1]) #(line colour and legend)
        , sexrepr_vs_repr #(y values)
        , all_gens #(x values)
    save_path # if save path == None, show the plot, else save it to save_path

    Returns
    -------

    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 6))

    # Color map for different environment parameter combinations
    colours = plt.cm.plasma(np.linspace(0, 1, len(results)))


    # Plot each line
    for i, (env_params, sexrepr_vs_repr) in enumerate(results):
        label = f"Env Params: climate warming - {env_params[0]}, sudden change - {env_params[1]})"
        ax.plot(np.linspace(0,len(sexrepr_vs_repr)+1, len(sexrepr_vs_repr)), sexrepr_vs_repr,
                label=label, color=colours[i], linewidth=2)

    # Set labels and title
    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel('Sexual Reproduction / All Reproduction Ratio', fontsize=12)
    ax.set_title('Sexual Reproduction Ratio Across Different Environmental Conditions', fontsize=14)

    # Add grid, legend, and adjust layout
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=10)
    # Add a text box with simulation parameters
    param_text = (
        "Simulation Parameters:\n"
        f"Population Size: {getattr(config, 'N', 'N/A')}\n"
        f"Max Generations: {getattr(config, 'max_generations', 'N/A')}\n"
        f"Mutation rate: {getattr(config, 'mu', 'N/A')}\n"
        f"Survival Threshold: {getattr(config, 'threshold_surv', 'N/A')}\n"
        f"Asexual Threshold: {getattr(config, 'threshold_asex', 'N/A')}\n"
        f"Avg nr of children: {getattr(config, 'avg_children', 'N/A')}"
    )
    fig.text(0.93, 0.41, param_text, fontsize=10,
                bbox=dict(facecolor='white', edgecolor="lightgray", alpha=0.8, pad=10.0))
    #plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig, ax




# Archive (old codes for reference):
'''# Calculate aggregated statistics
    agg_df = df.groupby(['c', 'delta', 'environment', 'mutation']).agg(
        mean_generations=('generations', 'mean'),
        std_generations=('generations', 'std'),
        min_generations=('generations', 'min'),
        max_generations=('generations', 'max')
    ).reset_index()'''
# cbar = plt.colorbar(sm, ax=ax)  # Explicitly associate the colorbar with 'ax'
# cbar.set_label('Iteration', fontsize=12)
'''agg_df['x_pos'] = agg_df['environment'].map(cond_x_pos)'''
# Calculate jitter for individual iterations to avoid overlap
#df['x_jitter'] = df['x_pos'] + (df['iteration'] - 2) * (df['generations'] / df['generations'].max())
'''# Plot aggregated results as larger, more opaque bubbles
    for mut in agg_df['mutation'].unique():
        mut_agg = agg_df[agg_df['mutation'] == mut]
        # Slightly offset x position by mutation rate to avoid overlap
        x_pos = mut_agg['x_pos'] + (mut - 0.25) * 0.1

        # Size based on mean generations
        size = 100 + (mut_agg['mean_generations'] / agg_df['mean_generations'].max()) * 500

        scatter = ax.scatter(x_pos, mut_agg['mean_generations'],
                              s=size, alpha=0.8, color=cmap(mut / 0.5),
                              edgecolors=cmap(mut / 0.5), linewidth=1.5,
                              label=f'Mean μ={mut}')

        # Add error bars to show standard deviation
        ax.errorbar(x_pos, mut_agg['mean_generations'],
                     yerr=mut_agg['std_generations'],
                     fmt='none', ecolor='black', capsize=5, alpha=0.5)
'''
'''ax.xlabel('Environment Parameters', fontsize=14)
    ax.ylabel('Survived Generations', fontsize=14)
    ax.title(
        'Impact of Environment Parameters and Mutation Rates on Population Survival\n(Individual iterations shown as small points)',
        fontsize=16)'''
'''# Add annotations for average values
    for _, row in agg_df.iterrows():
        ax.annotate(f"μ={row['mutation']}",
                     (row['x_pos'] + (row['mutation'] - 0.25) * 0.1, row['mean_generations']),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, fontweight='bold')'''
'''
        # Create a secondary plot: boxplot of generations by mutation rate
        plt.figure(figsize=(10, 6))
    
        # Create boxplot
        sns.boxplot(x='mutation', y='generations', data=df,
                    palette=sns.color_palette("viridis", len(df['mutation'].unique())))
    
        # Add individual points
        sns.stripplot(x='mutation', y='generations', data=df, size=5, color='black', alpha=0.5)
    
        plt.xlabel('Mutation Rate (μ)', fontsize=12)
        plt.ylabel('Survived Generations', fontsize=12)
        plt.title('Distribution of Survived Generations by Mutation Rate', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
        # Save secondary figure if path provided
        if save_path:
            boxplot_path = save_path.replace('.', '_boxplot.')
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            print(f"Secondary figure saved to {boxplot_path}")
        else:
            plt.show()
'''
'''
def mut_vs_env_plot_with_iterations2(results_by_iteration, save_path=None):
    """
    Create a bubble plot showing environment parameters, mutation rates, and generations,
    with individual iterations displayed as smaller points to visualize variation.

    Parameters:
    -----------
    results_by_iteration : dict
        Dictionary with iteration number as keys and simulation results as values
        Each simulation result is a dict with ((c, delta), mut) tuples as keys and generation counts as values
    save_path : str, optional
        Path to save the figure, if None the figure is displayed
    """
    # Convert all results to a single DataFrame
    all_data = []

    for iteration, results in results_by_iteration.items():
        for (env_params, mut), gen in results:
            c, delta = list(env_params)
            all_data.append({
                'c': tuple(c),
                'delta': delta,
                'environment': f"c={c}, δ={delta:.2f}",
                'mutation': mut,
                'generations': gen,
                'iteration': iteration
            })

    df = pd.DataFrame(all_data)

    # Calculate aggregated statistics
    agg_df = df.groupby(['c', 'delta', 'environment', 'mutation']).agg(
        mean_generations=('generations', 'mean'),
        std_generations=('generations', 'std'),
        min_generations=('generations', 'min'),
        max_generations=('generations', 'max')
    ).reset_index()

    # Create a figure and customize appearance
    fig, ax = plt.subplots(figsize=(15, 10))  # Create a figure and axis

    # Create custom colormap for mutation rates
    cmap = LinearSegmentedColormap.from_list('mutation_cmap', ['#2E86C1', '#D4AC0D', '#CB4335'], N=100)

    # Determine unique environment combinations for x-axis positions
    cond_x = df[['c', 'delta', 'environment']].drop_duplicates()
    cond_x_pos = dict(zip(cond_x['environment'], range(len(cond_x))))

    # Determine unique environment combinations for y-axis positions
    cond_y = df['mutation'].drop_duplicates()
    cond_y_pos = dict(zip(cond_y['environment'], range(len(cond_y))))
    #f"μ={row['mutation']}"

    # Map environment strings to x positions in both dataframes
    df['x_pos'] = df['environment'].map(cond_x_pos)
    agg_df['x_pos'] = agg_df['environment'].map(cond_x_pos)

    # Calculate jitter for individual iterations to avoid overlap
    df['x_jitter'] = df['x_pos'] + (df['mutation'] - 0.25) * 0.2

    # Plot individual iterations as small semi-transparent points
    for mut in df['mutation'].unique():
        mut_df = df[df['mutation'] == mut]
        ax.scatter(mut_df['x_jitter'], mut_df['generations'],
                    s=50, alpha=0.3, color=cmap(mut / 0.5),
                    marker='o', edgecolors='none',
                    label=f'μ={mut}' if mut == df['mutation'].iloc[0] else "")

    # Plot aggregated results as larger, more opaque bubbles
    for mut in agg_df['mutation'].unique():
        mut_agg = agg_df[agg_df['mutation'] == mut]
        # Slightly offset x position by mutation rate to avoid overlap
        x_pos = mut_agg['x_pos'] + (mut - 0.25) * 0.1

        # Size based on mean generations
        size = 100 + (mut_agg['mean_generations'] / agg_df['mean_generations'].max()) * 500

        scatter = ax.scatter(x_pos, mut_agg['mean_generations'],
                              s=size, alpha=0.8, color=cmap(mut / 0.5),
                              edgecolors=cmap(mut / 0.5), linewidth=1.5,
                              label=f'Mean μ={mut}')

        # Add error bars to show standard deviation
        ax.errorbar(x_pos, mut_agg['mean_generations'],
                     yerr=mut_agg['std_generations'],
                     fmt='none', ecolor='black', capsize=5, alpha=0.5)

    # Add environment labels to x-axis
    #plt.xticks(list(env_x_pos.values()), list(env_x_pos.keys()), rotation=45, ha='right')
    ax.set_xticks(list(cond_x_pos.values()))
    ax.set_xticklabels(list(cond_x_pos.keys()), rotation=45, ha='right')

    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Create a custom color bar
    norm = plt.Normalize(0, 0.5)  # Assuming max mutation rate is 0.5
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)  # Explicitly associate the colorbar with 'ax'
    cbar.set_label('Mutation Rate (μ)', fontsize=12)

    # Set axis labels and title
    ax.set_xlabel('Environment Parameters', fontsize=14)
    ax.set_ylabel('Survived Generations', fontsize=14)
    ax.set_title(
        'Impact of Environment Parameters and Mutation Rates on Population Survival\n(Individual iterations shown as small points)',
        fontsize=16
    )

    # Add annotations for average values
    for _, row in agg_df.iterrows():
        ax.annotate(f"μ={row['mutation']}",
                     (row['x_pos'] + (row['mutation'] - 0.25) * 0.1, row['mean_generations']),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, fontweight='bold')

    # Add a text box with simulation parameters
    param_text = (
        "Simulation Parameters:\n"
        f"Population Size: {getattr(config, 'N', 'N/A')}\n"
        f"Max Generations: {getattr(config, 'max_generations', 'N/A')}\n"
        f"Survival Threshold: {getattr(config, 'threshold_surv', 'N/A')}\n"
        f"Asexual Threshold: {getattr(config, 'threshold_asex', 'N/A')}\n"
        f"Iterations: {len(results_by_iteration)}"
    )
    fig.text(0.02, 0.02, param_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

    # Add legend for mutation rates
    legend = ax.legend(title="Mutation Rates", fontsize=10, loc='upper right')
    legend.get_title().set_fontsize(12)

    fig.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    """    # Create a secondary plot: boxplot of generations by mutation rate
        plt.figure(figsize=(10, 6))
    
        # Create boxplot
        sns.boxplot(x='mutation', y='generations', data=df,
                    palette=sns.color_palette("viridis", len(df['mutation'].unique())))
    
        # Add individual points
        sns.stripplot(x='mutation', y='generations', data=df, size=5, color='black', alpha=0.5)
    
        plt.xlabel('Mutation Rate (μ)', fontsize=12)
        plt.ylabel('Survived Generations', fontsize=12)
        plt.title('Distribution of Survived Generations by Mutation Rate', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
        # Save secondary figure if path provided
        if save_path:
            boxplot_path = save_path.replace('.', '_boxplot.')
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            print(f"Secondary figure saved to {boxplot_path}")
        else:
            plt.show()
    """
    return df  # Return the DataFrame for further analysis if needed
'''

