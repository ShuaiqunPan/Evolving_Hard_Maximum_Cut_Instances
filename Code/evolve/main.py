import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import time
import random
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['OMP_DISPLAY_ENV'] = 'false'
from generator import GraphGenerator
from rqaoa import RQAOA
from utils import *
from plot import *
from maxcut import SDP_max_cut, random_SDP_max_cut
from concurrent import futures
from optimizer import run_CMAES_with_cma, run_CMAES_with_nevergrad
from functools import partial


def create_objective_func(objective_values, latent_points, apply_initial_penalty=True, invert_ratio=False):
    penalty_value_initial = 1e+5
    penalty_value_RQAOA = 1

    # Bind additional arguments required by the objective_func
    bound_objective_func = partial(
        objective_func, 
        objective_values=objective_values, 
        latent_points=latent_points,
        apply_initial_penalty=apply_initial_penalty, 
        penalty_value_initial=penalty_value_initial, 
        penalty_value_RQAOA=penalty_value_RQAOA, 
        invert_ratio=invert_ratio
    )
    return bound_objective_func

def objective_func(latent_point, objective_values, latent_points, apply_initial_penalty, penalty_value_initial, penalty_value_RQAOA, invert_ratio):
    latent_points.append(latent_point.copy())
    J = generator.generate_and_plot_graph(latent_point)
    if isinstance(J, torch.Tensor): # Convert PyTorch tensor to NumPy array if J is a PyTorch tensor
        J = J.numpy()
        
    # TODO: add edge weights, i.i.d. standard Gaussian on the edges for later
    G = nx.from_numpy_array(J)
    grid_N, search_space, solver, batch_size = 100, [0, 2 * np.pi], 'analytic_brute', 1
    n = G.number_of_nodes()
    
    if not nx.is_connected(G):  # Check whether the graph is disconnected for the initial latent point
        if apply_initial_penalty:
            print("Initial latent point resulted in a disconnected graph, applying penalty.")
            return penalty_value_initial
        else:
            print("The graph is disconnected. Processing without penalty.")
    
    nc = 10 if n == 20 else 20 if n == 100 else None # Configure RQAOA parameters based on the number of nodes
    
    r = RQAOA(n=n, nc=nc, d=None, G=G, batch_size=batch_size, solver=solver, grid_N=grid_N, search_space=search_space) # Initialize RQAOA and run it
    batch_energies1, successful, best_cut, graph_history, cut_edges = r.run_rqaoa()
    
    if not successful:
        print("RQAOA process is unsuccessful, applying penalty.")
        batch_energies1 = penalty_value_RQAOA  # Assign penalty if RQAOA fails
    else:
        reference_energies = SDP_max_cut(G)[0] # Get the reference SDP max-cut value
        print("GW (Goemans-Williamson) solution: ", reference_energies)
        if invert_ratio: # GW / RQAOA
            approximation_ratio = reference_energies / batch_energies1
        else: # RQAOA / GW
            approximation_ratio = batch_energies1 / reference_energies
        print("Current approximation ratio: ", approximation_ratio)
        
        # if np.isnan(approximation_ratio) or np.isinf(approximation_ratio):
        #     print("Calculated approximation ratio is invalid, applying penalty.")
        #     approximation_ratio = 1e+5
            
        batch_energies1 = approximation_ratio  # Use the calculated ratio as the objective value
    
    print(f"Returning from objective function: {batch_energies1}")
    objective_values.append(batch_energies1)
    return batch_energies1

def run_CMAES_optimization(run_index, bounds, initial_latent_point, use_nevergrad=False, invert_ratio=False):
    objective_values = []
    latent_points = []
    
    current_objective_func = create_objective_func(objective_values, latent_points, apply_initial_penalty=True, invert_ratio=invert_ratio)
    save_initial_latent_point_numpy(initial_latent_point, run_index) # Save the initial latent point for each run of the CMA-ES
    
    initial_loss = current_objective_func(initial_latent_point) # Evaluate the initial latent point
    initial_adjacency_matrix = generator.generate_and_plot_graph(initial_latent_point)
    initial_G = nx.from_numpy_array(initial_adjacency_matrix.numpy())
    num_nodes = initial_G.number_of_nodes()
    print("Number of nodes for the initial graph:", num_nodes)
    print(f"Number of latent points before optimization: {len(latent_points)}")
    if initial_loss == PENALTY_VALUE:
        print("Initial latent point resulted in penalty, skipping optimization.")
        return [PENALTY_VALUE], None, None, None, None
    
    # If optimization continues, set up without initial penalty application
    current_objective_func = create_objective_func(objective_values, latent_points, apply_initial_penalty=False, invert_ratio=invert_ratio)
    
    if use_nevergrad:
        best_solution, best_value, step_sizes, all_populations, diagnostics = run_CMAES_with_nevergrad(run_index, bounds, initial_latent_point, current_objective_func)
    else:
        best_solution, best_value, step_sizes, all_populations, diagnostics, top_20_points, bottom_20_points = run_CMAES_with_cma(run_index, bounds, initial_latent_point, current_objective_func, generator)
    
    # Save top and bottom 20 points with run_index included in filename
    save_points_to_csv(top_20_points, "top_20_points.csv", run_index)
    save_points_to_csv(bottom_20_points, "bottom_20_points.csv", run_index)
    
    # Saving adjacency matrices with run_index included
    for i, (point, value) in enumerate(top_20_points):
        adj_matrix = generator.generate_and_plot_graph(point)
        save_adjacency_matrices(adj_matrix, "top_20", i, run_index)

    for i, (point, value) in enumerate(bottom_20_points):
        adj_matrix = generator.generate_and_plot_graph(point)
        save_adjacency_matrices(adj_matrix, "bottom_20", i, run_index)

    # Each point evaluated during the optimization is added, but no additional points (such as an initial point) are included.
    print(f"Number of latent points after optimization: {len(latent_points)}")
    optimized_latent_point = best_solution
    optimized_loss = current_objective_func(optimized_latent_point)
    optimized_adjacency_matrix = generator.generate_and_plot_graph(optimized_latent_point)
    optimized_G = nx.from_numpy_array(optimized_adjacency_matrix.numpy())
    print('Optimized latent point:', optimized_latent_point)
    print(f"Number of latent points after check the best solution: {len(latent_points)}")
    
    save_optimized_latent_point_numpy(optimized_latent_point, run_index) # Save the optimized latent point for each run of the CMA-ES
    
    if isinstance(optimized_adjacency_matrix, torch.Tensor): # Save the optimized adjacency matrix
        optimized_adjacency_matrix_np = optimized_adjacency_matrix.cpu().numpy()
    else:
        optimized_adjacency_matrix_np = optimized_adjacency_matrix
    save_optimized_adjacency_matrix_numpy(optimized_adjacency_matrix_np, run_index)
    
    return objective_values, latent_points, optimized_latent_point, initial_G, optimized_G, initial_adjacency_matrix, step_sizes, all_populations, diagnostics

def run_optimization_and_plot(run_index, bounds, initial_latent_point, invert_ratio=False):
    
    objective_values, latent_points, optimized_latent_point, initial_G, optimized_G, initial_adjacency_matrix, step_sizes, all_populations, diagnostics = run_CMAES_optimization(run_index, 
        bounds, initial_latent_point, use_nevergrad=False, invert_ratio=invert_ratio)
    
    # visualize_CMAES_paths_with_umap(all_populations, optimized_latent_point)
    
    all_values = objective_values # Save all the origianl results, run_id, embeddings, approximation ratio, and show whether it is the final recommend value.
    all_embeddings = latent_points
    save_optimization_results_csv(run_index, initial_latent_point, optimized_latent_point, all_embeddings, 
                                  all_values, filename="all_results.csv")

    # Save all the best so far results in the csv file
    best_index = objective_values.index(min(objective_values))  # Get the index of the best value
    best_value = objective_values[best_index]  # This is the minimum objective value encountered
    best_solution = latent_points[best_index]  # This is the embedding corresponding to the best value
    best_so_far_values = [objective_values[0]]
    best_so_far_embeddings = [latent_points[0]]
    for index, value in enumerate(objective_values[1:], start=1): # Exclude the last item for now
        if value < best_so_far_values[-1]:
            best_so_far_values.append(value) # Save the embedding that led to the new best value
            best_so_far_embeddings.append(latent_points[index])
        else:
            best_so_far_values.append(best_so_far_values[-1]) # If not a new best, repeat the last best value and corresponding embedding
            best_so_far_embeddings.append(best_so_far_embeddings[-1])
    best_so_far_values.append(best_value) # Explicitly update with the recommended (last) value and point
    best_so_far_embeddings.append(best_solution)
    
    save_optimization_results_csv(run_index, initial_latent_point, optimized_latent_point, best_so_far_embeddings, 
                                  best_so_far_values, filename="best_so_far_results_all.csv")

    # Do the max-cut on the original and final graphs and show the plots
    plot_all(run_index, initial_G, optimized_G, objective_values, latent_points, SDP_max_cut, best_so_far_values, step_sizes, diagnostics)
    plot_method_comparisons(run_index, initial_G, optimized_G, SDP_max_cut, n_repetitions=10)
    
    # 50 runs for SDP and RQAOA
    plot_SDP_50_runs(run_index, initial_G, random_SDP_max_cut, n_repetitions=50, tag="initial")
    plot_SDP_50_runs(run_index, optimized_G, random_SDP_max_cut, n_repetitions=50, tag="optimized")
            
    plot_RQAOA_50_runs(run_index, initial_G, n_repetitions=50, tag="initial")
    plot_RQAOA_50_runs(run_index, optimized_G, n_repetitions=50, tag="optimized")
    
    final_optimized_index = len(latent_points) - 1
    plot_decoded_graphs_at_best(run_index, bounds, best_so_far_values, latent_points, 
                                final_optimized_index, initial_adjacency_matrix, generator=generator)
    
    # Save the best so far results only in the csv file    
    best_so_far_values_only = [objective_values[0]]
    best_so_far_embeddings_only = [latent_points[0]]
    for index, value in enumerate(objective_values[1:], start=1):
        if value < best_so_far_values_only[-1]:
            best_so_far_values_only.append(value)
            best_so_far_embeddings_only.append(latent_points[index])
    best_so_far_values_only.append(best_value) # Explicitly update with the recommended (last) value and point
    best_so_far_embeddings_only.append(best_solution)
    save_optimization_results_csv(run_index, initial_latent_point, optimized_latent_point, best_so_far_embeddings_only, 
                                  best_so_far_values_only, filename="best_so_far_results_only.csv")
    
    # Save internal variables/states of pycma
    save_diagnostics_to_csv(diagnostics)
    
    return best_so_far_values, best_so_far_values[-1], optimized_latent_point, best_so_far_embeddings, step_sizes, all_populations

def initialization_optimization(num_nodes, single_initial_point=True, invert_ratio=False):
    uniformed_latent_points = []
    sampled_latent_points = []
    all_best_so_far_values = []
    final_approximation_ratio = []
    initial_latent_points = []
    final_latent_points = []
    all_recorded_step_sizes = []
    all_recorded_optimized_latent_point = []
    all_recorded_all_populations = []
    
    iterations = 75 if num_nodes == 20 else 5000 # Generate initial points
    for _ in range(iterations):
        point = np.array([np.random.uniform(low, high) for low, high in bounds])
        uniformed_latent_points.append(point)
        adjacency_matrix = generator.generate_and_plot_graph(point)
        if isinstance(adjacency_matrix, torch.Tensor):
            adjacency_matrix = adjacency_matrix.cpu().numpy()
        G = nx.from_numpy_array(adjacency_matrix)
        if nx.is_connected(G):
            sampled_latent_points.append(point)
            if single_initial_point:
                break

    if not sampled_latent_points:
        raise Exception("No connected graphs found in sampled points.")
    
    if single_initial_point: # Case 1: Only the first valid initial point is used for 10 CMA-ES runs
        cma_es_repetitions = 10
        initial_points_to_use = [sampled_latent_points[0]] * cma_es_repetitions
    else: # Case 2: Each valid initial point is used for exactly 1 CMA-ES run
        cma_es_repetitions = len(sampled_latent_points)
        initial_points_to_use = sampled_latent_points

    for run_index, initial_point in enumerate(initial_points_to_use, start=1): # Optimization with CMA-ES
        print(f"\nStarting Run {run_index} with initial point {initial_point}...")
        start_time = time.time()
        best_so_far_values, last_objective_value, final_latent_point, best_so_far_embeddings, recorded_step_sizes, recorded_all_populations = run_optimization_and_plot(run_index, 
            bounds, initial_point, invert_ratio=invert_ratio)
        all_recorded_step_sizes.append(recorded_step_sizes)
        all_recorded_optimized_latent_point.append(final_latent_point)
        all_recorded_all_populations.append(recorded_all_populations)
        
        final_approximation_ratio.append(last_objective_value)
        initial_latent_points.append(initial_point)
        final_latent_points.append(final_latent_point)
        if best_so_far_values and best_so_far_values[-1] != PENALTY_VALUE:
            all_best_so_far_values.append(best_so_far_values)
            end_time = time.time()
            duration = end_time - start_time
            print(f"Run {run_index} completed. Duration: {duration:.2f} seconds. Best value so far: {best_so_far_values[-1]}")
        else:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Run {run_index} skipped due to penalty. Duration: {duration:.2f} seconds.")

    plot_all_step_sizes(all_recorded_step_sizes)
    # visualize_multiple_CMAES_paths_with_umap(all_recorded_all_populations, all_recorded_optimized_latent_point)
    
    if not single_initial_point:
        visualize_final_latent_points(final_latent_points)
        visualize_paired_latent(initial_latent_points, final_latent_points, filename="final_paired_latent_combine.pdf", combine_first=True)
        visualize_paired_latent(initial_latent_points, final_latent_points, filename="final_paired_latent_separate.pdf", combine_first=False)
        print(final_approximation_ratio)
        plot_histogram(final_approximation_ratio) # Plot the histogram of the final approximation ratio from all runs of CMA-ES.
        plot_mean_and_std_deviation(all_best_so_far_values) # Mean and std deviation across multiple runs

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    num_nodes = 100
    generator = GraphGenerator(num_nodes=num_nodes)
    dimension = 64
    bounds = [(generator.range[0][i], generator.range[1][i]) for i in range(dimension)]
    PENALTY_VALUE = 1e+5
    # If invert_ratio = True, GW/RQAOA, otherwise, RQAOA/GW.
    initialization_optimization(num_nodes=num_nodes, single_initial_point=False, invert_ratio=True)
    print("All optimizations completed.")