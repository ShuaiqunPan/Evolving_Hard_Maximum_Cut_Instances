import cma
import numpy as np
import nevergrad as ng
import heapq
from utils import dynamic_save_adjacency_matrices, dynamic_save_points_to_csv
import logging
from multiprocessing import Pool


def calculate_initial_sigma(bounds):
    # Calculate the range for each dimension
    ranges = [bound[1] - bound[0] for bound in bounds]
    # Use the average range divided by a heuristic value (e.g., 5) as the initial sigma
    sigma0 = sum(ranges) / len(ranges) / 5
    return sigma0

def parallel_evaluation(points, func, num_processes=64):
    results = []
    with Pool(processes=num_processes) as pool:
        try:
            results = pool.map(func, points)
        except Exception as e:
            logging.error(f"Error during parallel evaluation: {e}")
            results = [float('inf')] * len(points)  # Assign a large penalty value on error
    return results

def run_CMAES_with_cma(run_index, bounds, initial_latent_point, current_objective_func, generator, max_restarts=10):
    sigma0 = calculate_initial_sigma(bounds)
    print("The initial sigma: ", sigma0)
    max_total_fevals = 4000
    best_solution = None
    best_value = float('inf')
    restarts = 0
    all_populations = []
    step_sizes = []  # Initialize step_sizes outside the restart loop
    
    top_20_points = []  # Lists for top and bottom 20 points
    bottom_20_points = []
    
    # Dynamically updated lists
    dynamic_top_20_points = []
    dynamic_bottom_20_points = []
    
    options = {
        'bounds': ([b[0] for b in bounds], [b[1] for b in bounds]),
        'maxfevals': 2000,
        'tolfun': 1e-9,
        'tolfunhist': 1e-12,
        'verb_disp': 0,
        'popsize': 64
    }
    
    # Initialize CMA-ES with the specified population size
    es = cma.CMAEvolutionStrategy(initial_latent_point, sigma0, options)
    diagnostics = {
        'run_index': run_index,
        'options': options,
        'default_popsize': es.sp.popsize,  # Store the actual population size used in diagnostics
        'max_restarts': max_restarts,
        'iterations_details': []
    }
    
    while restarts < max_restarts:
        while not es.stop():
            X = es.ask()
            evaluations = parallel_evaluation(X, current_objective_func, num_processes=64)
            es.tell(X, evaluations)
            
            if es.result.fbest < best_value: # Keep track of the best solution
                best_solution = es.result.xbest
                best_value = es.result.fbest
                
            # Update the original and dynamic top and bottom point lists
            update_top_bottom_points(evaluations, X, top_20_points, bottom_20_points)
            update_top_bottom_points(evaluations, X, dynamic_top_20_points, dynamic_bottom_20_points)

            # Save dynamically at regular intervals
            if es.countevals % 960 == 0:
                dynamic_save_points_to_csv(dynamic_top_20_points, 'dynamic_top_20_points.csv', run_index)
                dynamic_save_points_to_csv(dynamic_bottom_20_points, 'dynamic_bottom_20_points.csv', run_index)
                
                # Save dynamic points and their corresponding adjacency matrices
                current_indices = [i for i, _ in enumerate(dynamic_top_20_points)]
                for i, (point, value) in enumerate(dynamic_top_20_points):
                    adj_matrix = generator.generate_and_plot_graph(point)
                    dynamic_save_adjacency_matrices(adj_matrix, "dynamic_top_20", i, run_index, current_indices)
                    
                current_indices = [i for i, _ in enumerate(dynamic_bottom_20_points)]
                for i, (point, value) in enumerate(dynamic_bottom_20_points):
                    adj_matrix = generator.generate_and_plot_graph(point)
                    dynamic_save_adjacency_matrices(adj_matrix, "dynamic_bottom_20", i, run_index, current_indices)
                    
            step_sizes.append(es.sigma)
            all_populations.append(X)

            diagnostics['iterations_details'].append({
                'iterations': es.result.iterations,  # Number of iterations.
                'evals': es.countevals,  # Number of evaluations.
                'sigma': es.sigma,  # The adaptation of the step size.
                'mean': es.mean.tolist(),  # The population mean.
                'variance': es.C.tolist(),  # The covariance matrix.
                'best_value': es.result.fbest, # The current best solution.
                'best_solution': es.result.xbest.tolist()  # The value of the current best solution.
            })

        sigma0, restarts = handle_stopping_criteria(es, options, max_total_fevals, sigma0, restarts)
        initial_latent_point = best_solution  # Restart from the best known solution
        es = cma.CMAEvolutionStrategy(initial_latent_point, sigma0, options)
        restarts += 1
        
    # Final sorting before returning
    top_20_points.sort(key=lambda x: -x[1])
    bottom_20_points.sort(key=lambda x: x[1])
    
    print("Top 20 Points and Their Values:")
    for point, value in top_20_points:
        print(f"Point: {point}, Objective Value: {value}")

    print("Bottom 20 Points and Their Values:")
    for point, value in bottom_20_points:
        print(f"Point: {point}, Objective Value: {value}")
        
    return best_solution, best_value, step_sizes, all_populations, diagnostics, top_20_points, bottom_20_points

def update_top_bottom_points(evaluations, X, top_20_points, bottom_20_points):
    for eval, x in zip(evaluations, X):
        x_list = list(x)
        if eval > 1:
            if len(top_20_points) < 50:
                top_20_points.append((x_list, eval))
                top_20_points.sort(key=lambda x: -x[1])
            elif eval > top_20_points[-1][1]:
                top_20_points[-1] = (x_list, eval)
                top_20_points.sort(key=lambda x: -x[1])
        if eval < 1:
            if len(bottom_20_points) < 50:
                bottom_20_points.append((x_list, eval))
                bottom_20_points.sort(key=lambda x: x[1])
            elif eval < bottom_20_points[-1][1]:
                bottom_20_points[-1] = (x_list, eval)
                bottom_20_points.sort(key=lambda x: x[1])
                    
def handle_stopping_criteria(es, options, max_total_fevals, sigma0, restarts):
    stopping_reasons = es.stop()
    if 'tolfun' in stopping_reasons and stopping_reasons['tolfun']:
        print(f"Restarting due to tolfun criterion (restart {restarts+1})")
        sigma0 *= np.random.uniform(0.9, 1.1)  # Mildly adjust sigma0 for finer exploration
    elif 'tolfunhist' in stopping_reasons and stopping_reasons['tolfunhist']:
        print(f"Restarting due to tolfunhist criterion (restart {restarts+1})")
        sigma0 *= np.random.uniform(0.9, 1.1)  # Mildly adjust sigma0 for finer exploration
    elif 'tolflatfitness' in stopping_reasons and stopping_reasons['tolflatfitness']:
        print(f"Restarting due to tolflatfitness criterion (restart {restarts+1}). Adjusting sigma0 more aggressively.")
        sigma0 *= np.random.uniform(0.7, 0.9)  # More aggressive adjustment to escape flat fitness
    elif 'maxfevals' in stopping_reasons and stopping_reasons['maxfevals']:
        if options['maxfevals'] + 1000 <= max_total_fevals:
            options['maxfevals'] += 1000  # Only add 1000 if it doesn't exceed the max_total_fevals
            print(f"Extended max function evaluations to {options['maxfevals']} (restart {restarts+1}).")
        else:
            remaining_capacity = max_total_fevals - options['maxfevals']
            if remaining_capacity > 0:
                options['maxfevals'] += remaining_capacity  # Add only the remaining capacity if it's positive
                print(f"Extended max function evaluations to {options['maxfevals']}, which is now capped (restart {restarts+1}).")
            else:
                print(f"Reached the maximum function evaluations cap of {max_total_fevals}. No further extensions allowed.")
    else:
        print("Stopping criteria other than the above were met, not restarting.")
        for reason, value in stopping_reasons.items():
            if value:  # If the criterion triggered stopping
                print(f" - {reason}: {value}")
    
    print("Stopping criteria met:", stopping_reasons)
    return sigma0, restarts

def run_CMAES_with_nevergrad(run_index, bounds, initial_latent_point, current_objective_func, budget=10000):
    parametrization = ng.p.Array(init=initial_latent_point).set_bounds(lower=[b[0] for b in bounds], upper=[b[1] for b in bounds])
    optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=budget)
    recommendation = optimizer.minimize(current_objective_func)
    best_solution = recommendation.value
    best_value = current_objective_func(best_solution)
    return best_solution, best_value, None, None, None