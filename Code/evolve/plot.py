import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import os
from networkx import is_connected, from_edgelist
import torch
import umap
from generator import GraphGenerator
import pandas as pd
from utils import is_clique
from config import output_figures_directory, output_others_directory
import random
from maxcut import SDP_max_cut, brute_force_max_cut, random_SDP_max_cut, multiple_random_SDP_max_cuts, run_multiple_rqaoa
from rqaoa import RQAOA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from pandas.plotting import parallel_coordinates
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from matplotlib.lines import Line2D


def save_figure(filename):
    filepath = os.path.join(output_figures_directory, filename)
    plt.savefig(filepath)
    
def plot_mean_and_std_deviation(all_best_so_far_values, filename="Mean_and_Std_Deviation_Best_So_Far.pdf"):
    """
    Plot the mean and standard deviation of best-so-far values across runs.
    """
    max_length = max(len(trajectory) for trajectory in all_best_so_far_values) # Ensure all trajectories are of the same length
    for trajectory in all_best_so_far_values:
        trajectory.extend([trajectory[-1]] * (max_length - len(trajectory)))

    mean_values = np.mean(all_best_so_far_values, axis=0) # Calculate mean and standard deviation across runs at each step
    std_dev_values = np.std(all_best_so_far_values, axis=0)

    fig, ax = plt.subplots()
    ax.plot(mean_values, label='Mean Best-So-Far Value')
    ax.fill_between(range(max_length), mean_values - std_dev_values, mean_values + std_dev_values, color='b', alpha=0.2)
    ax.set_xlabel('Evaluation Index')
    ax.set_ylabel('Objective Function Value')
    ax.set_title('Mean and Std Deviation of Best-So-Far Values Across Runs')
    ax.legend()
    save_figure(filename)
    plt.close()
    
def visualize_connected_latent_points(all_points, sampled_latent_points, filename="latent_points_visualization.pdf"):
    """
    Visualize connected and not connected latent points using UMAP projection.
    """
    all_points_array = np.array(all_points)
    labels = np.zeros(len(all_points_array))

    for point in sampled_latent_points:
        index = np.where(np.all(all_points_array == point, axis=1))[0][0] # Find the index of the current point in the all_points array and set its label to 1
        labels[index] = 1

    umap_reducer = umap.UMAP() # Apply UMAP to project the points to a 2D space for visualization
    points_2d = umap_reducer.fit_transform(all_points_array)
    connected_points = points_2d[labels == 1] # Separate connected and not connected points based on their labels
    not_connected_points = points_2d[labels == 0]

    plt.figure(figsize=(8, 8))
    plt.scatter(not_connected_points[:, 0], not_connected_points[:, 1], color='red', alpha=0.5, label='Not connected graphs')
    plt.scatter(connected_points[:, 0], connected_points[:, 1], color='blue', alpha=0.5, label='Connected graphs')
    plt.title('UMAP visualization of latent points')
    plt.xlabel('UMAP feature 1')
    plt.ylabel('UMAP feature 2')
    plt.legend()
    save_figure(filename)
    plt.close()
    
def plot_decoded_graphs_at_best(run_index, bounds, objective_values, latent_points, final_optimized_index, initial_adjacency_matrix, generator=None):
    unique_objective_indices = np.unique(objective_values, return_index=True)[1]
    unique_indices = np.append(unique_objective_indices, final_optimized_index)
    unique_indices = np.unique(unique_indices)  # Remove any duplicates that might exist

    num_graphs = len(unique_indices)
    print("Number of graphs during the optimization process including the optimized one:", num_graphs)
    
    cols = int(np.ceil(np.sqrt(num_graphs)))
    rows = int(np.ceil(num_graphs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows > 1 or cols > 1:
        axes = axes.flatten()

    for i, index in enumerate(unique_indices):
        latent_point = latent_points[index]
        if generator:
            adjacency_matrix = generator.generate_and_plot_graph(latent_point)
        
        if isinstance(adjacency_matrix, torch.Tensor): # Save the adjacency matrix of the best so far graphs
            adjacency_matrix = adjacency_matrix.cpu().numpy()
        filename = os.path.join(output_others_directory, f"Best_So_Far_Graph_Run_{run_index}_Index_{index}.npy")
        np.save(filename, adjacency_matrix)
        
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Retrieve the corresponding objective value for the graph
        objective_value = objective_values[index] if index < len(objective_values) else objective_values[-1]

        if index == final_optimized_index:
            title = f"Optimized Sol (Obj: {objective_value:.3f})"
        else:
            title = f"Index {index} (Obj: {objective_value:.3f})"
        
        if rows == 1 and cols == 1:
            plot_graph(G, title, axes, 16)
        else:
            plot_graph(G, title, axes[i], 16)

    if rows > 1 or cols > 1:
        for i in range(num_graphs, rows * cols):
            axes[i].axis('off')

    plt.tight_layout()
    save_figure(f"Best_So_Far_Decoded_Graphs_Run_{run_index}.pdf")
    plt.close(fig)
    
def plot_histogram(last_objective_values, title="Histogram of final approximation ratio", xlabel="Approximation Ratio", ylabel="Frequency"):
    plt.figure(figsize=[10,6])
    n, bins, patches = plt.hist(last_objective_values)
    
    # Adding text for count and approximation ratio value above each bin if count is not zero
    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Compute the bin centers
    for count, bin_center, patch in zip(n, bin_centers, patches):
        if count > 0:
            height = patch.get_height()
            text_str = f'Freq: {int(count)}; Ratio: {bin_center:.2f}'
            plt.text(bin_center, height, text_str, 
                     ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=15)
    save_figure("Histogram_of_final_approximation_ratio.pdf")
    plt.close()
    
def visualize_final_latent_points(final_latent_points, filename="final_latent_points_visualization.pdf"):
    # Filter out None values
    filtered_points = [point for point in final_latent_points if point is not None]

    # Proceed only if there are at least 3 valid points to ensure n_neighbors can be greater than 1
    if len(filtered_points) >= 3:
        # Check if all points have the same length after filtering
        lengths = [len(point) for point in filtered_points]
        if len(set(lengths)) > 1:
            raise ValueError("All points in final_latent_points must have the same length")

        final_points_array = np.array(filtered_points)
        # Ensure n_neighbors is at least 2, which is the minimum required by UMAP
        # n_neighbors = max(min(15, len(filtered_points) - 1), 2)

        # Apply UMAP to project the points to a 2D space for visualization
        # umap_reducer = umap.UMAP(n_neighbors=n_neighbors)
        umap_reducer = umap.UMAP()
        points_2d = umap_reducer.fit_transform(final_points_array)

        plt.figure(figsize=(8, 8))
        plt.scatter(points_2d[:, 0], points_2d[:, 1], color='green', alpha=0.5)
        plt.title('UMAP visualization of final latent points')
        plt.xlabel('UMAP feature 1')
        plt.ylabel('UMAP feature 2')
        save_figure(filename)
        plt.close()
    else:
        print("No valid data points available for visualization. Need at least 3 points.")
        
def visualize_paired_latent(initial_latent_points, final_latent_points, filename="final_paired_latent_points.pdf", combine_first=True):
    filtered_initial = [point for point in initial_latent_points if point is not None]
    filtered_final = [point for point in final_latent_points if point is not None]
    
    if len(filtered_initial) != len(filtered_final):
        raise ValueError("Initial and final points lists must have the same number of valid points")

    initial_array = np.array(filtered_initial)
    final_array = np.array(filtered_final)

    # Decide whether to combine the data first based on combine_first parameter
    if combine_first: # Combine, apply UMAP, then split
        combined_array = np.concatenate((initial_array, final_array), axis=0)
        umap_reducer = umap.UMAP()
        combined_2d = umap_reducer.fit_transform(combined_array)
        split_point = len(filtered_initial)
        initial_2d = combined_2d[:split_point]
        final_2d = combined_2d[split_point:]
    else: # Apply UMAP separately
        umap_reducer = umap.UMAP()
        initial_2d = umap_reducer.fit_transform(initial_array)
        final_2d = umap_reducer.transform(final_array)

    plt.figure(figsize=(10, 10))
    plt.scatter(initial_2d[:, 0], initial_2d[:, 1], c='red', alpha=0.5, label='Initial Points')
    plt.scatter(final_2d[:, 0], final_2d[:, 1], c='blue', alpha=0.5, label='Final Points')

    # Draw arrows from initial to final points
    for i_point, f_point in zip(initial_2d, final_2d):
        plt.arrow(i_point[0], i_point[1], f_point[0] - i_point[0], f_point[1] - i_point[1], 
                  color='grey', alpha=0.5, length_includes_head=True, head_width=0.03, head_length=0.05)

    plt.title('UMAP Visualization of Paired Latent Points')
    plt.xlabel('UMAP feature 1')
    plt.ylabel('UMAP feature 2')
    plt.legend()
    save_figure(filename)
    plt.close()

def plot_all_step_sizes(all_recorded_step_sizes):
    num_runs = len(all_recorded_step_sizes)  # Determine the layout of the subplots
    num_columns = 3  # For example, adjust based on your preference
    num_rows = (num_runs + num_columns - 1) // num_columns
    
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5), squeeze=False)
    fig.suptitle('Step Sizes Over Iterations for Each CMA-ES Run')
    
    axs = axs.flatten()  # Always flatten axs to simplify handling
    
    for i, step_sizes in enumerate(all_recorded_step_sizes):
        ax = axs[i]
        ax.plot(step_sizes, label=f'Run {i+1}')
        ax.set_title(f'Run {i+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Step Size')
        ax.legend(loc='upper right')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the main title
    save_figure(f"Recorded_step_sizes.pdf")  # Ensure save_figure is defined or use plt.savefig
    plt.close(fig)
    
def visualize_CMAES_paths_with_umap(all_populations, final_solution):
    # Flatten the list of populations to create an array suitable for UMAP
    flattened_populations = np.vstack(all_populations)

    # Apply UMAP reduction
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(flattened_populations)

    # Find the index of the final solution within the flattened list
    final_index = np.where((flattened_populations == final_solution).all(axis=1))[0]
    if final_index.size == 0:
        print("Final solution not found in the populations. Ensure it was included in all_populations.")
        return

    # Plot the UMAP projection
    plt.figure(figsize=(12, 8))
    # Plot all points
    plt.scatter(embedding[:, 0], embedding[:, 1], color='gray', s=5)  # Removed cmap, using solid color
    # Highlight the final solution with a cross marker
    plt.scatter(embedding[final_index, 0], embedding[final_index, 1], color='red', marker='x', s=100, label='Final Solution')
    plt.title('UMAP projection of CMA-ES paths')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    save_figure(f"visualize_CMAES_path.pdf")

def visualize_multiple_CMAES_paths_with_umap(all_recorded_all_populations, all_recorded_optimized_latent_point):
    # Define a colormap to use colors for each CMA-ES run
    colors = cm.rainbow(np.linspace(0, 1, len(all_recorded_all_populations)))

    # Initialize UMAP reducer
    reducer = umap.UMAP(random_state=42)

    plt.figure(figsize=(12, 8))

    # Go through each CMA-ES run
    for i, (all_populations, final_solution) in enumerate(zip(all_recorded_all_populations, all_recorded_optimized_latent_point)):
        # Flatten the list of populations for the current run
        flattened_populations = np.vstack(all_populations)

        # Fit UMAP and transform the data
        embedding = reducer.fit_transform(flattened_populations)

        # Plot all points for the current run
        plt.scatter(embedding[:, 0], embedding[:, 1], color=colors[i], s=5, alpha=0.5)

        # Highlight the final solution for the current run
        final_index = np.where((flattened_populations == final_solution).all(axis=1))[0]
        if final_index.size > 0:
            plt.scatter(embedding[final_index, 0], embedding[final_index, 1], color=colors[i], marker='x', s=100, label=f'Final Run {i+1}')

    plt.title('UMAP projection of multiple CMA-ES paths')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    save_figure(f"visualize_multiple_CMAES_path.pdf")
    
def plot_graph(G, title, ax, fontsize, node_colors='lightblue', edge_colors='gray'):
    # Create a subgraph containing only nodes with edges
    H = G.subgraph([node for node in G if G.degree(node) > 0])
    pos = nx.spring_layout(H, seed=42)  # Fixed layout for consistent positioning
    nx.draw(H, pos, ax=ax, with_labels=True, node_color=node_colors, edge_color=edge_colors)
    ax.set_title(title, fontsize=fontsize)
    
def plot_graph_with_partitions(G, pos, ax, cut_vector, intra_edges, cut_edges, fontsize):
    # Draw nodes with colors based on their partition
    node_colors = ['green' if cut_vector[i] > 0 else 'red' for i in range(len(G.nodes()))]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    # Draw intra-partition edges with a lighter color
    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, style='solid', edge_color='black', ax=ax)
    # Draw cut edges (inter-partition) with a darker color
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='solid', edge_color='lightgray', ax=ax)
    
def compute_partition_edges(G, cut_vector):
    left_nodes = {node for idx, node in enumerate(G.nodes()) if cut_vector[idx] <= 0}
    right_nodes = set(G.nodes()) - left_nodes
    intra_edges = [(u, v) for u, v in G.edges() if (u in left_nodes and v in left_nodes) or (u in right_nodes and v in right_nodes)]
    cut_edges = [(u, v) for u, v in G.edges() if (u in left_nodes and v in right_nodes) or (u in right_nodes and v in left_nodes)]
    return intra_edges, cut_edges

def plot_diagnostics(ax, diagnostics, title_fontsize, label_fontsize):
    default_popsize = diagnostics['default_popsize']  # Extract default population size from diagnostics
    maxfevals = diagnostics['options']['maxfevals']
    num_max_restarts = diagnostics['max_restarts']
    # final_best_value = diagnostics['iterations_details'][-1]['best_value'] if diagnostics['iterations_details'] else 'Not Available'
    final_best_value = min(detail['best_value'] for detail in diagnostics['iterations_details']) if diagnostics['iterations_details'] else 'Not Available'


    # Prepare text string with the selected information
    text_str = (f"Default Population Size: {default_popsize}\n"
                f"Max Function Evaluations: {maxfevals}\n"
                f"Number of Restart for CMA-ES: {num_max_restarts}\n"
                f"Final Best Value: {final_best_value}")

    # Formatting and displaying text in the plot
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=label_fontsize, verticalalignment='top', horizontalalignment='left')
    ax.axis('off')  # Turn off axis for cleaner display
    ax.set_title('CMA-ES Internal Variables', fontsize=title_fontsize)
    
def plot_all(run_index, initial_G, optimized_G, objective_values, latent_points, SDP_max_cut, best_so_far_values, step_sizes, diagnostics):
    fig = plt.figure(figsize=(32, 24))
    gs = fig.add_gridspec(3, 4)
    ax1, ax2, ax3, ax4 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])
    ax5, ax6, ax7, ax8 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])
    ax9, ax10, ax11, ax12 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3])
    
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    
    plot_graph(initial_G, f"Initial Graph Run {run_index}", ax1, fontsize=title_fontsize) # Row 1 - Initial Graph (Original), GW, RQAOA, Brute-force (Optimal)
    _, _, initial_cut_vector, initial_edges_cut, initial_mean_edges_cut = SDP_max_cut(initial_G) # Plots for GW max cut
    initial_G, initial_pos, _, _, clique_left_initial, clique_right_initial, initial_intra, initial_cut = apply_max_cut_layout_and_filter_edges_with_clique_check(initial_G, initial_cut_vector)
    cut_edges_count = initial_edges_cut  # Number of edges cut by GW
    plot_graph_with_partitions(initial_G, initial_pos, ax2, initial_cut_vector, initial_intra, initial_cut, title_fontsize)
    initial_clique_info = f"GW Max-Cut for Initial Graph: {cut_edges_count} edges cut;\n" \
                          f"Left Partition is {'a clique' if clique_left_initial else 'not a clique'};\n" \
                          f"Right Partition is {'a clique' if clique_right_initial else 'not a clique'}"
    ax2.set_title(initial_clique_info, fontsize=title_fontsize)
    
    G = initial_G # Plots for RQAOA max cut
    grid_N, search_space, solver, batch_size = 100, [0, 2 * np.pi], 'analytic_brute', 1
    n = G.number_of_nodes()
    nc = 10 if n == 20 else 18 if n == 100 else 0
    r = RQAOA(n=n, nc=nc, d=None, G=G, batch_size=batch_size, solver=solver, grid_N=grid_N, search_space=search_space)
    batch_energies1, successful, best_cut, initial_graph_history, initial_cut_edges = r.run_rqaoa()
    intra_edges, cut_edges = compute_partition_edges(G, best_cut)
    cut_edges_count = initial_cut_edges  # Number of edges cut by RQAOA
    plot_graph_with_partitions(initial_G, initial_pos, ax3, best_cut, intra_edges, cut_edges, title_fontsize)
    initial_clique_info = f"RQAOA Max-Cut for Initial Graph: {cut_edges_count} edges cut;\n" \
                          f"Left Partition is {'a clique' if clique_left_initial else 'not a clique'};\n" \
                          f"Right Partition is {'a clique' if clique_right_initial else 'not a clique'}"
    ax3.set_title(initial_clique_info, fontsize=title_fontsize)
    
    _, initial_bf_cut = brute_force_max_cut(initial_G) # Plots for brute-force max cut
    initial_G, initial_pos, _, _, clique_left_initial, clique_right_initial, initial_intra, initial_bf_cut_edges = apply_max_cut_layout_and_filter_edges_with_clique_check(initial_G, initial_bf_cut)
    cut_edges_count = len(initial_bf_cut_edges)  # Number of edges cut by brute-force
    plot_graph_with_partitions(initial_G, initial_pos, ax4, initial_bf_cut, initial_intra, initial_bf_cut_edges, title_fontsize)
    initial_clique_info = f"Brute-force Max-Cut for Initial Graph: {cut_edges_count} edges cut;\n" \
                          f"Left Partition is {'a clique' if clique_left_initial else 'not a clique'};\n" \
                          f"Right Partition is {'a clique' if clique_right_initial else 'not a clique'}"
    ax4.set_title(initial_clique_info, fontsize=title_fontsize)
    
    plot_graph(optimized_G, f"Optimized Graph Run {run_index}", ax5, fontsize=title_fontsize) # Row 2 - Initial Graph (Original), GW, RQAOA, Brute-force (Optimal)
    _, _, optimized_cut_vector, optimized_edges_cut, optimized_mean_edges_cut = SDP_max_cut(optimized_G) # Plots for GW max cut
    optimized_G, optimized_pos, _, _, clique_left_optimized, clique_right_optimized, optimized_intra, optimized_cut = apply_max_cut_layout_and_filter_edges_with_clique_check(optimized_G, optimized_cut_vector)
    cut_edges_count = optimized_edges_cut  # Number of edges cut by GW
    plot_graph_with_partitions(optimized_G, optimized_pos, ax6, optimized_cut_vector, optimized_intra, optimized_cut, title_fontsize)
    optimized_clique_info = f"GW Max-Cut for Optimized Graph: {cut_edges_count} edges cut;\n" \
                            f"Left Partition is {'a clique' if clique_left_optimized else 'not a clique'};\n" \
                            f"Right Partition is {'a clique' if clique_right_optimized else 'not a clique'}"
    ax6.set_title(optimized_clique_info, fontsize=title_fontsize)
    
    G = optimized_G # Plots for RQAOA max cut
    grid_N, search_space, solver, batch_size = 100, [0, 2 * np.pi], 'analytic_brute', 1
    n = G.number_of_nodes()
    nc = 10 if n == 20 else 18 if n == 100 else 0
    r = RQAOA(n=n, nc=nc, d=None, G=G, batch_size=batch_size, solver=solver, grid_N=grid_N, search_space=search_space)
    batch_energies1, successful, best_cut, optimized_graph_history, optimized_cut_edges = r.run_rqaoa()
    intra_edges, cut_edges = compute_partition_edges(G, best_cut)
    cut_edges_count = optimized_cut_edges  # Number of edges cut by RQAOA
    plot_graph_with_partitions(initial_G, initial_pos, ax7, best_cut, intra_edges, cut_edges, title_fontsize)
    initial_clique_info = f"RQAOA Max-Cut for Optimized Graph: {cut_edges_count} edges cut;\n" \
                          f"Left Partition is {'a clique' if clique_left_initial else 'not a clique'};\n" \
                          f"Right Partition is {'a clique' if clique_right_initial else 'not a clique'}"
    ax7.set_title(initial_clique_info, fontsize=title_fontsize)
    
    _, optimized_bf_cut = brute_force_max_cut(optimized_G)
    optimized_G, optimized_pos, _, _, clique_left_optimized, clique_right_optimized, optimized_intra, optimized_bf_cut_edges = apply_max_cut_layout_and_filter_edges_with_clique_check(optimized_G, optimized_bf_cut)
    cut_edges_count = len(optimized_bf_cut_edges)  # Number of edges cut by brute-force
    plot_graph_with_partitions(optimized_G, optimized_pos, ax8, optimized_bf_cut, optimized_intra, optimized_bf_cut_edges, title_fontsize)
    optimized_clique_info = f"Brute-force Max-Cut for Initial Graph: {cut_edges_count} edges cut;\n" \
                          f"Left Partition is {'a clique' if clique_left_optimized else 'not a clique'};\n" \
                          f"Right Partition is {'a clique' if clique_right_optimized else 'not a clique'}"
    ax8.set_title(optimized_clique_info, fontsize=title_fontsize)
    
    # Best-So-Far Trajectory Plot, Original Trajectory Plot
    plot_trajectory(ax9, best_so_far_values, 'Best So Far', 'Evaluation Index', 'Best Objective Function Value So Far', run_index, label_fontsize, title_fontsize, tick_fontsize)

    best_index = objective_values.index(min(objective_values)) # Find the index and value of the best objective
    best_value = objective_values[best_index]
    objective_values.append(best_value)
    plot_trajectory(ax10, objective_values, 'Objective Values', 'Evaluation Index', 'Objective Function Values So Far', run_index, label_fontsize, title_fontsize, tick_fontsize)
    
    ax11.plot(step_sizes, marker='o', linestyle='-', color='blue') # New subplot for step sizes
    ax11.set_xlabel('Iteration', fontsize=label_fontsize)
    ax11.set_ylabel('Step Size', fontsize=label_fontsize)
    ax11.set_title('Step Sizes Over Iterations', fontsize=title_fontsize)
    ax11.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax11.grid(True)
    
    plot_diagnostics(ax12, diagnostics, title_fontsize, label_fontsize)
    
    plt.tight_layout()
    save_figure(f"Graphs_Run_{run_index}.pdf")
    plt.close(fig)
    
def plot_trajectory(ax, data, label, xlabel, ylabel, run_index, label_fontsize, title_fontsize, tick_fontsize):
    ax.plot(data[:-1], marker='o', label=label)
    ax.scatter(len(data) - 1, data[-1], color='red', marker='x', s=100, label='Recommended by CMA-ES')
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(f'{label} Trajectory Run {run_index}', fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(True)
    ax.legend()

def apply_max_cut_layout_and_filter_edges_with_clique_check(G, cut_vector):
    pos = nx.spring_layout(G, seed=42)  # Layout for visual separation
    left_nodes = {node for node, cut_val in zip(G.nodes(), cut_vector) if cut_val <= 0}
    right_nodes = {node for node, cut_val in zip(G.nodes(), cut_vector) if cut_val > 0}

    for node in left_nodes: # Shift nodes to the left or right to visually separate partitions
        pos[node][0] -= 1.0  # Shift nodes in the left partition to the left
    for node in right_nodes:
        pos[node][0] += 1.0  # Shift nodes in the right partition to the right

    # Identify intra-partition and inter-partition edges
    intra_edges = [(u, v) for u, v in G.edges() if (u in left_nodes and v in left_nodes) or (u in right_nodes and v in right_nodes)]
    cut_edges = [(u, v) for u, v in G.edges() if (u in left_nodes and v in right_nodes) or (u in right_nodes and v in left_nodes)]

    subgraph_left = G.subgraph(left_nodes) # Check for cliques in each partition
    subgraph_right = G.subgraph(right_nodes)
    clique_left = is_clique(subgraph_left)
    clique_right = is_clique(subgraph_right)

    return G, pos, subgraph_left, subgraph_right, clique_left, clique_right, intra_edges, cut_edges

def plot_method_comparisons(run_index, initial_G, optimized_G, SDP_max_cut, n_repetitions=10):
    n_cols = n_repetitions + 1
    n_rows = 4  # One row for each method: SDP, RQAOA
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    title_fontsize = 14
    
    plot_graph(initial_G, "Initial Graph", axs[0, 0], fontsize=title_fontsize) # Plotting initial and optimized graphs and their brute-force results in the first column
    axs[0, 0].set_title(f"Initial Graph", fontsize=title_fontsize)
    
    _, initial_bf_cut = brute_force_max_cut(initial_G) # Brute-force for initial_G
    modified_IG, pos, _, _, _, _, intra, cut = apply_max_cut_layout_and_filter_edges_with_clique_check(initial_G, initial_bf_cut)
    plot_graph_with_partitions(modified_IG, pos, axs[1, 0], initial_bf_cut, intra, cut, title_fontsize)
    cut_value = len(cut)
    axs[1, 0].set_title(f"Brute-force for Initial Graph \n {cut_value} edges in the cut", fontsize=title_fontsize)
    
    plot_graph(optimized_G, "Optimized Graph", axs[2, 0], fontsize=title_fontsize)
    axs[2, 0].set_title(f"Optimized Graph", fontsize=title_fontsize)
    
    _, optimized_bf_cut = brute_force_max_cut(optimized_G)
    modified_OG, pos, _, _, _, _, intra, cut = apply_max_cut_layout_and_filter_edges_with_clique_check(optimized_G, optimized_bf_cut)
    plot_graph_with_partitions(modified_OG, pos, axs[3, 0], optimized_bf_cut, intra, cut, title_fontsize)
    cut_value = len(cut)
    axs[3, 0].set_title(f"Brute-force for Optimized Graph \n {cut_value} edges in the cut", fontsize=title_fontsize)
    
    for j in range(n_repetitions): # Plot comparisons for each method
        _, edges_cut_initial, cut_vector_initial = random_SDP_max_cut(initial_G) # SDP Max-Cut for initial_G and optimized_G
        _, edges_cut_optimized, cut_vector_optimized = random_SDP_max_cut(optimized_G)
        
        modified_IG, pos, _, _, _, _, intra, cut = apply_max_cut_layout_and_filter_edges_with_clique_check(initial_G, cut_vector_initial)
        cut_value = edges_cut_initial
        plot_graph_with_partitions(initial_G, pos, axs[0, j+1], cut_vector_initial, intra, cut, title_fontsize) # SDP results of initial and optimized graphs
        axs[0, j+1].set_title(f"SDP(GW) for Initial Graph Run {j+1}: \n {cut_value} edges in the cut", fontsize=title_fontsize)
        
        modified_OG, pos, _, _, _, _, intra, cut = apply_max_cut_layout_and_filter_edges_with_clique_check(optimized_G, cut_vector_optimized)
        cut_value = edges_cut_optimized
        plot_graph_with_partitions(optimized_G, pos, axs[2, j+1], cut_vector_optimized, intra, cut, title_fontsize)
        axs[2, j+1].set_title(f"SDP(GW) for Optimized Graph Run {j+1}: \n {cut_value} edges in the cut", fontsize=title_fontsize)
            
        grid_N, search_space, solver, batch_size = 100, [0, 2 * np.pi], 'analytic_brute', 1 # RQAOA for initial_G and optimized_G
        initial_G = initial_G
        n = initial_G.number_of_nodes()
        nc = 10 if n == 20 else 18 if n == 100 else 0
        r = RQAOA(n=n, nc=nc, d=None, G=initial_G, batch_size=batch_size, solver=solver, grid_N=grid_N, search_space=search_space)
        batch_energies1_initial, successful_initial, best_cut_initial, graph_history_initial, num_cut_edges_initial = r.run_rqaoa()
        optimized_G = optimized_G
        optimized_n = optimized_G.number_of_nodes()
        nc = 10 if optimized_n == 20 else 18 if optimized_n == 100 else 0
        optimized_r = RQAOA(n=optimized_n, nc=nc, d=None, G=optimized_G, batch_size=batch_size, solver=solver, grid_N=grid_N, search_space=search_space)
        batch_energies1_optimized, successful_optimized, best_cut_optimized, graph_history_optimized, num_cut_edges_optimized = optimized_r.run_rqaoa()
        
        intra_edges_initial, cut_edges_initial = compute_partition_edges(initial_G, best_cut_initial) # Count cut edges and update plot for initial graph
        cut_value = num_cut_edges_initial
        plot_graph_with_partitions(initial_G, pos, axs[1, j+1], best_cut_initial, intra_edges_initial, cut_edges_initial, title_fontsize)
        axs[1, j+1].set_title(f"RQAOA for Initial Graph Run {j+1}: \n {cut_value} edges in the cut", fontsize=title_fontsize)

        intra_edges_optimized, cut_edges_optimized = compute_partition_edges(optimized_G, best_cut_optimized)
        cut_value = num_cut_edges_optimized
        plot_graph_with_partitions(optimized_G, pos, axs[3, j+1], best_cut_optimized, intra_edges_optimized, cut_edges_optimized, title_fontsize)
        axs[3, j+1].set_title(f"RQAOA for Optimized Graph Run {j+1}: \n {cut_value} edges in the cut", fontsize=title_fontsize)

    plt.tight_layout()
    save_figure(f"Comparative_MaxCut_Methods_Run_{run_index}.pdf")
    plt.close(fig)
    
def plot_SDP_50_runs(run_index, initial_G, random_SDP_max_cut, n_repetitions, tag):
    n_cols = 10
    n_rows = 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    title_fontsize = 14
    for i in range(n_repetitions):
        row_index = i // n_cols
        col_index = i % n_cols
        _, edges_cut_initial, cut_vector_initial = random_SDP_max_cut(initial_G)
        modified_IG, pos, _, _, _, _, intra, cut = apply_max_cut_layout_and_filter_edges_with_clique_check(initial_G, cut_vector_initial)
        cut_value = edges_cut_initial
        plot_graph_with_partitions(modified_IG, pos, axs[row_index, col_index], cut_vector_initial, intra, cut, title_fontsize)
        axs[row_index, col_index].set_title(f"SDP(GW) for Initial Graph Run {i+1}: \n {cut_value} edges in the cut", fontsize=title_fontsize)
    plt.tight_layout()
    save_figure(f"SDP_MaxCut_50_Runs_{run_index}_{tag}.pdf")
    plt.close(fig)
    
def plot_RQAOA_50_runs(run_index, initial_G, n_repetitions, tag):
    n_cols = 10
    n_rows = 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    title_fontsize = 14
    pos = nx.spring_layout(initial_G)
    for i in range(n_repetitions):
        row_index = i // n_cols
        col_index = i % n_cols
        grid_N, search_space, solver, batch_size = 100, [0, 2 * np.pi], 'analytic_brute', 1
        n = initial_G.number_of_nodes()
        nc = 10 if n == 20 else 18 if n == 100 else 0
        r = RQAOA(n=n, nc=nc, d=None, G=initial_G, batch_size=batch_size, solver=solver, grid_N=grid_N, search_space=search_space)
        batch_energies1_initial, successful_initial, best_cut_initial, graph_history_initial, num_cut_edges_initial = r.run_rqaoa()
        intra_edges_initial, cut_edges_initial = compute_partition_edges(initial_G, best_cut_initial)  # Calculate cut edges
        cut_value = num_cut_edges_initial
        plot_graph_with_partitions(initial_G, pos, axs[row_index, col_index], best_cut_initial, intra_edges_initial, cut_edges_initial, title_fontsize)
        axs[row_index, col_index].set_title(f"RQAOA for Initial Graph Run {i+1}: \n {cut_value} edges in the cut", fontsize=title_fontsize)
    plt.tight_layout()
    save_figure(f"RQAOA_MaxCut_50_Runs_{run_index}_{tag}.pdf")
    plt.close(fig)

def plot_rqaoa_evolution(run_index, initial_graph, graph_history, tag):
    pos = nx.spring_layout(initial_graph, seed=42)  # Fixed seed for consistent layout

    initial_state = (initial_graph.copy(), None, None)
    full_graph_history = [initial_state] + graph_history  # Append initial state to the history

    num_steps = len(full_graph_history)
    num_cols = 5  # Maximum number of columns per row
    num_rows = (num_steps + num_cols - 1) // num_cols  # Calculate required number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), constrained_layout=True)
    if num_rows > 1 or num_cols > 1:
        axes = axes.flatten()

    previous_graph = None  # Start with no previous graph

    for idx, (graph, edge_removed, node_processed) in enumerate(full_graph_history):
        ax = axes[idx]

        # Draw all nodes and standard edges
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', ax=ax)
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='gray')

        if previous_graph:
            # Find new edges by comparing current graph's edges with previous graph's edges
            new_edges = set(graph.edges()).difference(previous_graph.edges())
            # Highlight new edges in red
            if new_edges:
                nx.draw_networkx_edges(graph, pos, edgelist=new_edges, ax=ax, edge_color='red', style='solid', width=2.5)

        # Update previous_graph at the end of the loop
        previous_graph = graph.copy()

        # Additional visual settings
        nx.draw_networkx_labels(graph, pos, ax=ax, font_color='black', font_size=8)
        ax.set_title(f"Step {idx}: {'Initial' if idx == 0 else f'Node {node_processed} processed'}", fontsize=10)
        ax.axis('off')

    # Turn off unused axes
    for j in range(num_steps, len(axes)):
        axes[j].axis('off')

    save_figure(f"RQAOA_evolution_{run_index}_{tag}.pdf")
    plt.close(fig)

def SDP_GW_box_plot(graph, num):
    cut_values, edges_cut = multiple_random_SDP_max_cuts(graph, 100)
    plt.figure()
    bp = plt.boxplot(edges_cut, notch=True, patch_artist=True, showmeans=True)
    for box in bp['boxes']:
        box.set(color='#7570b3', linewidth=2)  # Set box color and line width
        box.set(facecolor='#1b9e77')           # Set fill color
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    plt.title('Box Plot of Number of Edge Cuts Over 100 Trials')
    plt.ylabel('Number of Edge Cuts')
    plt.xlabel('Trial')
    plt.grid(True)
    save_figure(f"SDP_GW_box_plot_{num}.pdf")
    plt.close()

def RQAOA_box_plot(G, num):
    num_cut_edges_results = run_multiple_rqaoa(G)
    plt.figure(figsize=(10, 5))
    bp = plt.boxplot(num_cut_edges_results, notch=True, patch_artist=True, showmeans=True)
    for box in bp['boxes']:
        box.set(color='#7570b3', linewidth=2)
        box.set(facecolor='#d95f02')
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#e6ab02', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#66a61e', alpha=0.5)
    plt.title('Box Plot of Number of Cut Edges from 100 RQAOA Runs')
    plt.ylabel('Number of Cut Edges')
    plt.xlabel('RQAOA Runs')
    plt.grid(True)
    save_figure(f"RQAOA_box_plot_{num}.pdf")
    plt.close()