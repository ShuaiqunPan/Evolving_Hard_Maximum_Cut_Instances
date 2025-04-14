import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import random
from plot_graph_features import *
from generator_computed_features import GraphGenerator
from collections import deque
from scipy.linalg import eigvalsh
import math
from goemens_williamson_charles_features import *

'''
Implement the computed features related to the Graph features.
'''

def welsh_powell(graph):
    
    nodes_sorted = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)
    
    color_map = {}
    
    for node in nodes_sorted:
        neighbor_colors = {color_map[n] for n in graph.neighbors(node) if n in color_map}
        color = next((c for c in range(len(nodes_sorted)) if c not in neighbor_colors), 0)
        color_map[node] = color
    
    chromatic_number = max(color_map.values()) + 1
    return chromatic_number, color_map

def matrix_similarity(matrix1, matrix2):
    return np.array_equal(matrix1, matrix2)

def compute_global_graph_features(csv_file):
    # Read the points and objectives CSV
    objectives_df = pd.read_csv(csv_file)
    
    # Prepare an empty list to collect DataFrame rows
    rows_list = []

    for index, row in objectives_df.iterrows():
        latent_points = eval(row['Point'])  # Evaluate the string to actual list
        matrix_from_csv = generator.generate_and_plot_graph(latent_points)
        current_graph = nx.from_numpy_array(matrix_from_csv)
        
        # Calculate graph features, Index is based on row index of the csv
        current_graph_features = calculate_graph_features(current_graph, f"graph_{index}")
        current_graph_features['CALCULATED_RATIO'] = row['Objective Value']  # Directly use the objective value

        # Calculate advanced SDP-based features
        sdp_features = SDP_max_cut_charles_features(current_graph)

        # Combine all features into a single dictionary
        combined_features = {**current_graph_features, **sdp_features}
        
        # Append current graph features to rows list
        rows_list.append(combined_features)
    
    graph_description = pd.DataFrame(rows_list)
    
    # Use the determine_output_file function to get the correct output file path
    output_path = determine_output_file(csv_file)
    graph_description.to_csv(output_path)
    print(f'Features saved to {output_path}')

    return graph_description

def verify_kemeny_constant_eigen(G):
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph must be connected.")
    
    A = nx.adjacency_matrix(G).toarray()
    degrees = np.array([degree for _, degree in G.degree()])
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    
    H = D_inv_sqrt @ A @ D_inv_sqrt  # Symmetric normalized Laplacian
    eigvals = eigvalsh(H)  # Compute eigenvalues of the matrix H
    eigvals = np.sort(eigvals)[:-1]  # Exclude the largest eigenvalue
    
    kemeny_constant = np.sum(1 / (1 - eigvals))
    return kemeny_constant

def calculate_girth(G):
    def bfs_shortest_cycle(subgraph, start):
        # Initial setup: visited records distances from the start node
        visited = {start: 0}
        queue = deque([(start, None)])
        shortest_cycle = math.inf  # We're looking for the minimum cycle

        while queue:
            current, parent = queue.popleft()
            current_depth = visited[current]
            for neighbor in subgraph.neighbors(current):
                if neighbor == parent:
                    continue  # Avoid going back to the parent, which would not be a valid cycle

                if neighbor in visited:
                    # A cycle is found
                    cycle_length = current_depth + visited[neighbor] + 1
                    if cycle_length < shortest_cycle:
                        shortest_cycle = cycle_length
                        # We can stop processing if we found the minimal possible cycle (a triangle)
                        if shortest_cycle == 3:
                            return shortest_cycle

                elif neighbor not in visited:
                    visited[neighbor] = current_depth + 1
                    queue.append((neighbor, current))

        return shortest_cycle

    if G.number_of_edges() == 0:
        print("Graph has no edges.")
        return math.inf

    min_girth = math.inf
    # Process each connected component separately
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(component) > 2 and nx.is_connected(subgraph):  # Check if it can potentially have a cycle
            for node in subgraph:
                cycle_length = bfs_shortest_cycle(subgraph, node)
                if cycle_length == 3:
                    return 3  # Early exit if the smallest possible girth is found
                min_girth = min(min_girth, cycle_length)
    
    return min_girth if min_girth != math.inf else math.inf

def calculate_graph_features(current_graph, filename):
    current_graph_num_nodes = current_graph.number_of_nodes() #1
    current_graph_num_edges = np.log(current_graph.number_of_edges()) #2
    current_graph_density = nx.density(current_graph)  #3
    current_graph_connected = int(nx.is_connected(current_graph)) #4

    current_graph_logratio_edgestonodes = np.log(current_graph_num_edges/current_graph_num_nodes) #5

    current_graph_chromatic_num, _ = welsh_powell(current_graph) #6
    current_graph_chromatic_num = np.log(current_graph_chromatic_num)

    normalized_current_graph_mis = len(list(nx.approximation.maximum_independent_set(current_graph))) / current_graph_num_nodes #7
    current_graph_assortativity = np.round(nx.degree_assortativity_coefficient(current_graph), 8) #8

    L = nx.normalized_laplacian_matrix(current_graph)
    e = np.linalg.eigvals(L.A)
    e.sort()

    current_graph_spectral_gap = abs(max(e) - e[len(e)-2]) #9

    epsilon = 1e-10
    current_graph_loglargesteigenval = np.log(e[len(e)-1]) #10
    current_graph_logsecondlargesteigenval = np.log(e[len(e)-2]) #11
    # current_graph_logthirdlargesteigenval = np.log(e[len(e)-3])
    # current_graph_logfourthlargesteigenval = np.log(e[len(e)-4])
    # current_graph_logfifthlargesteigenval = np.log(e[len(e)-5])
    # current_graph_logsixthlargesteigenval = np.log(e[len(e)-6])
    current_graph_logsmallesteigenval = np.log(e[1] + epsilon) #Smallest NONTRIVIAL eigenvalue #12
    # current_graph_logsmallesteigenval = e[1] # Directly use the eigenvalues without logarithmic transformation
    
    try: # eccentricity only works for connected graphs
        eccs = list(nx.eccentricity(current_graph).values())
        eccs.sort()
        current_graph_min_eccentricity = eccs[1] #13
        current_graph_max_eccentricity = eccs[len(e)-1] #14
        current_graph_ratio_maxmin_eccentricity = np.log(current_graph_max_eccentricity/current_graph_min_eccentricity) #15
    except:
        # if the graph is disconnected then we set all the eccentricity features to zero
        current_graph_min_eccentricity = 0
        current_graph_max_eccentricity = 0
        current_graph_ratio_maxmin_eccentricity = 0
    
    # if the graph is disconnected then we set to zero
    if nx.is_connected(current_graph):
        current_graph_kemeny_const = np.log(np.round(verify_kemeny_constant_eigen(current_graph), 8))
    else:
        current_graph_kemeny_const = 0

    # current_graph_girth = nx.girth(current_graph) #17
    current_graph_girth = calculate_girth(current_graph)  # Replace the nx.girth call
    current_graph_transitivity = nx.transitivity(current_graph)
    
    return {
        "GRAPH_IDX":filename,
        "LOG_NUM_NODES":current_graph_num_nodes,
        "LOG_NUM_EDGES":current_graph_num_edges,
        "IS_CONNECTED":current_graph_connected, 
        "LOGRATIO_EDGETONODES":current_graph_logratio_edgestonodes,
        "DENSITY":current_graph_density,
        "LOG_CHROMATIC_NUM":current_graph_chromatic_num,
        "NORM_MIS":normalized_current_graph_mis, 
        "GRAPH_ASSORTATIVITY":current_graph_assortativity,
        "SPECTRAL_GAP":current_graph_spectral_gap,
        "LOG_LARGESTEIGVAL":current_graph_loglargesteigenval,
        "LOG_SECONDLARGESTEIGVAL":current_graph_logsecondlargesteigenval,
        "LOG_SMALLESTEIGVAL":current_graph_logsmallesteigenval,
        "MIN_ECC":current_graph_min_eccentricity,
        "MAX_ECC":current_graph_max_eccentricity,
        "LOGRATIO_MAXMINECC":current_graph_ratio_maxmin_eccentricity,
        "GIRTH":current_graph_girth,
        "TRANSITIVITY":current_graph_transitivity
    }
    
def determine_output_file(path):
    if 'GW_RQAOA_bottom' in path:
        return "global_graph_features_GW_RQAOA_bottom.csv"
    elif 'GW_RQAOA_top' in path:
        return "global_graph_features_GW_RQAOA_top.csv"
    elif 'RQAOA_GW_bottom' in path:
        return "global_graph_features_RQAOA_GW_bottom.csv"
    elif 'RQAOA_GW_top' in path:
        return "global_graph_features_RQAOA_GW_top.csv"
    return "global_graph_features.csv"

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    
    num_nodes = 100
    generator = GraphGenerator(num_nodes=num_nodes)
    
    # Step 1 Compute the global features
    compute_global_graph_features(csv_file="GW_RQAOA_bottom_20_points.csv")
    compute_global_graph_features(csv_file="GW_RQAOA_top_20_points.csv")
    
    # Step 2 Usage
    file_paths = [
        "global_graph_features_GW_RQAOA_bottom.csv",
        "global_graph_features_GW_RQAOA_top.csv"
    ]
    
    # Combine data from specified file paths
    combined_df = combine_data(file_paths)
    scaled_df = scale_data(combined_df)
    visualize_umap_global_features(scaled_df)
    
    plot_parallel_coordinates(scaled_df)
    plot_parallel_coordinates_separate(scaled_df)
    plot_parallel_coordinates_two_figures(scaled_df)
    plot_parallel_coordinates_separate_two_pdf(scaled_df)