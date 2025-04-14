import networkx as nx
import numpy as np
import os
import torch
import pandas as pd
from config import output_figures_directory, output_others_directory, output_CMAES_directory 
import csv

def is_connected_by_laplacian(G):
    L = nx.laplacian_matrix(G).astype(float) # Compute the Laplacian matrix of the graph
    eigenvalues = np.linalg.eigvalsh(L.A) # Compute eigenvalues of the Laplacian
    fiedler_value = sorted(eigenvalues)[1] # The second smallest eigenvalue (Fiedler value) indicates graph connectivity
    return fiedler_value > 0
    
def calculate_default_popsize(dimension, num_workers=1):
    # Calculate the default population size for CMA-ES based on problem dimension and number of workers.
    return max(num_workers, 4 + int(3 * np.log(dimension)))

def is_clique(G):
    # A clique represents a complete subgraph within the larger graph, where every vertex in the 
    # clique is adjacent to every other vertex in the clique.
    for node in G:
        neighbors = set(G.neighbors(node))
        if not neighbors >= set(G.nodes()) - {node}:
            return False
    return True
    
def save_initial_latent_point_numpy(latent_point, run_index):
    if not isinstance(latent_point, np.ndarray):
        try:
            latent_point = np.array(latent_point)
        except Exception as e:
            raise ValueError(f"Error converting latent_point to a NumPy array: {e}")
    filename = os.path.join(output_others_directory, f"Initial_latent_point_{run_index}.npy")
    np.save(filename, latent_point)
    
def save_optimized_latent_point_numpy(latent_point, run_index):
    if not isinstance(latent_point, np.ndarray):
        try:
            latent_point = np.array(latent_point)
        except Exception as e:
            raise ValueError(f"Error converting latent_point to a NumPy array: {e}")
    filename = os.path.join(output_others_directory, f"Optimized_latent_point_{run_index}.npy")
    np.save(filename, latent_point)

def load_initial_latent_point(run_index):
    filename = os.path.join(output_others_directory, f"Initial_latent_point_{run_index}.npy")
    try:
        latent_point = np.load(filename)
        return latent_point
    except Exception as e:
        raise IOError(f"Error loading latent_point from {filename}: {e}")
    
def load_optimized_latent_point(run_index):
    filename = os.path.join(output_others_directory, f"Optimized_latent_point_{run_index}.npy")
    try:
        latent_point = np.load(filename)
        return latent_point
    except Exception as e:
        raise IOError(f"Error loading latent_point from {filename}: {e}")
    
def save_optimized_adjacency_matrix_numpy(adjacency_matrix, run_index):
    if not isinstance(adjacency_matrix, np.ndarray):
        try:
            adjacency_matrix = np.array(adjacency_matrix)
        except Exception as e:
            raise ValueError(f"Error converting adjacency_matrix to a NumPy array: {e}")
    filename = os.path.join(output_others_directory, f"Optimized_adjacency_matrix_{run_index}.npy")
    np.save(filename, adjacency_matrix)

def save_optimization_results_csv(run_index, initial_latent_point, optimized_latent_point, best_so_far_embeddings_only, 
                                  best_so_far_values_only, filename="best_so_far_results.csv"):
    filename = os.path.join(output_others_directory, filename)
    
    # Prepare data records for the current run
    data = [{
        "run_id": run_index,
        "embedding": str(embedding.tolist()),  # Convert numpy array to list then to string for serialization
        "approximation_ratio": approximation_ratio,
        "is_final_solution": False
    } for embedding, approximation_ratio in zip(best_so_far_embeddings_only, best_so_far_values_only)]
    
    # Ensure the optimized point is marked as the final solution
    if len(best_so_far_embeddings_only) > 0 and np.array_equal(best_so_far_embeddings_only[-1], optimized_latent_point):
        data[-1]["is_final_solution"] = True
        
    df = pd.DataFrame(data) # Convert the list of dictionaries to a DataFrame
    file_exists = os.path.isfile(filename) # Check if the file already exists
    # Save the DataFrame to CSV; if file exists, append without header; otherwise, write with header
    df.to_csv(filename, mode='a', index=False, header=not file_exists)
    
def save_diagnostics_to_csv(diagnostics, filename="diagnostics.csv"):
    if not diagnostics:
        print("Diagnostics data is empty.")
        return
    
    filename = os.path.join(output_others_directory, filename)

    # Extract the nested iterations details and expand them into a DataFrame
    iterations_details = diagnostics['iterations_details']
    if not iterations_details:
        print("No iteration details to save.")
        return
    
    # Convert nested diagnostic details into a DataFrame
    df = pd.DataFrame(iterations_details)
    
    df['run_index'] = diagnostics['run_index']
    df['default_popsize'] = diagnostics['default_popsize']
    df['max_restarts'] = diagnostics['max_restarts']
    
    for key, value in diagnostics['options'].items():
        if key not in ['bounds']:
            df[key] = value

    # Check if the file already exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    # Save to CSV, appending if the file exists, and writing headers only if the file does not exist
    df.to_csv(filename, mode='a', index=False, header=not file_exists)
    print(f"Diagnostics data saved to {filename}.")

def save_adjacency_matrices(adjacency_matrix, file_prefix, index, run_index):
    if not isinstance(adjacency_matrix, np.ndarray):
        try:
            adjacency_matrix = np.array(adjacency_matrix)
        except Exception as e:
            raise ValueError(f"Error converting adjacency_matrix to a NumPy array: {e}")
    
    # Include run_index in the filename
    file_name = os.path.join(output_others_directory, f"{file_prefix}_{run_index}_adj_matrix_{index}.npy")
    np.save(file_name, adjacency_matrix)
    
def save_points_to_csv(points, filename, run_index):
    file_path = os.path.join(output_others_directory, filename)
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as file:  # 'a' mode for appending
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Run Index', 'Point', 'Objective Value'])  # Adding Run Index to headers
        for point, value in points:
            writer.writerow([run_index, point, value])

def dynamic_save_adjacency_matrices(adjacency_matrix, file_prefix, index, run_index, current_indices):
    if not isinstance(adjacency_matrix, np.ndarray):
        try:
            adjacency_matrix = np.array(adjacency_matrix)
        except Exception as e:
            raise ValueError(f"Error converting adjacency_matrix to a NumPy array: {e}")

    directory_path = output_CMAES_directory
    file_name = os.path.join(directory_path, f"{file_prefix}_{run_index}_adj_matrix_{index}.npy")
    np.save(file_name, adjacency_matrix)

    # Clean up: Remove files whose indices are not in the current index list for the specific run_index
    existing_files = os.listdir(directory_path)
    for file in existing_files:
        if file.startswith(f"{file_prefix}_{run_index}") and file.endswith(".npy"):
            try:
                file_index = int(file.split('_')[-1].split('.')[0])
                if file_index not in current_indices:
                    os.remove(os.path.join(directory_path, file))
            except ValueError:
                continue  # skip files that do not follow the expected naming convention
             
def dynamic_save_points_to_csv(points, filename, run_index):
    file_path = os.path.join(output_CMAES_directory, filename)
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Run Index', 'Point', 'Objective Value'])  # Adding Run Index to headers
        for point, value in points:
            writer.writerow([run_index, point, value])
