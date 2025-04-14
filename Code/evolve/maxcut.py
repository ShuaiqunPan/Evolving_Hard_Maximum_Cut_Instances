import cvxpy as cp
import itertools
import networkx as nx
import numpy as np
from rqaoa import RQAOA

def cut_cost(x, L):
    return 0.25 * x @ L @ x

def count_cut_edges(cut, G):
    # Calculate the number of edges between the partitions.
    # Assume cut is a binary vector with values in {1, -1}
    num_edges_cut = 0
    for u, v in G.edges():
        if cut[u] * cut[v] < 0:  # Node u and v are in different sets
            num_edges_cut += 1
    return num_edges_cut

def SDP_max_cut(G):
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes)).toarray()

    X = cp.Variable((n, n), PSD=True)
    obj = 0.25 * cp.trace(L @ X)
    constr = [cp.diag(X) == 1]
    problem = cp.Problem(cp.Maximize(obj), constraints=constr)
    problem.solve(solver=cp.SCS)

    u, s, v = np.linalg.svd(X.value)
    U = u * np.sqrt(s)

    num_trials = 100
    gw_results = np.zeros(num_trials)
    gw_edges_count = np.zeros(num_trials)
    cuts = []

    for i in range(num_trials):
        r = np.random.randn(n)
        r = r / np.linalg.norm(r)
        cut = np.sign(r @ U.T)
        cuts.append(cut)
        current_cut_value = cut_cost(cut, L)
        gw_results[i] = current_cut_value
        gw_edges_count[i] = count_cut_edges(cut, G)

    # Determine the index of the median cut value
    sorted_indices = np.argsort(gw_results)
    mid_point = len(gw_results) // 2
    if len(gw_results) % 2 == 0:
        median_index = sorted_indices[mid_point - 1]  # Choose the first one of the middle two
    else:
        median_index = sorted_indices[mid_point]

    median_cut = cuts[median_index]
    median_cut_edges_count = gw_edges_count[median_index]

    print("All the cut value of edges: ", gw_edges_count)
    print("Median cut value of edges: ", median_cut_edges_count)
    print(np.median(gw_results))
    print(np.median(gw_edges_count))

    return (np.median(gw_results), np.max(gw_results), median_cut, median_cut_edges_count, np.median(gw_edges_count))


def random_SDP_max_cut(G):
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes)).toarray()

    # Set up and solve the SDP
    X = cp.Variable((n, n), PSD=True)
    obj = 0.25 * cp.trace(L @ X)
    constr = [cp.diag(X) == 1]
    problem = cp.Problem(cp.Maximize(obj), constraints=constr)
    problem.solve(solver=cp.SCS)

    # Decompose the SDP solution matrix to generate a random cut
    u, s, v = np.linalg.svd(X.value)
    U = u * np.sqrt(s)

    # Generate a random cut
    r = np.random.randn(n)
    r = r / np.linalg.norm(r)  # Normalize the random vector
    cut = np.sign(r @ U.T)

    # Calculate the cut value and the number of cut edges
    cut_value = cut_cost(cut, L)
    num_edges_cut = count_cut_edges(cut, G)

    return cut_value, num_edges_cut, cut

def multiple_random_SDP_max_cuts(G, num_trials=100):
    """Generate multiple random max cuts from the SDP solution for statistical analysis."""
    cut_values = []
    edges_cut = []

    for _ in range(num_trials):
        cut_value, num_edges_cut, _ = random_SDP_max_cut(G)
        cut_values.append(cut_value)
        edges_cut.append(num_edges_cut)

    return cut_values, edges_cut

def run_multiple_rqaoa(G, num_runs=100):
    """Run RQAOA multiple times on a graph and collect the number of cut edges from each run."""
    num_cut_edges_list = []
    successful_runs = 0

    # Constants for RQAOA
    grid_N, search_space, solver, batch_size = 100, [0, 2 * np.pi], 'analytic_brute', 1
    n = G.number_of_nodes()
    nc = 10 if n == 20 else 18 if n == 100 else 0

    for _ in range(num_runs):
        r = RQAOA(n=n, nc=nc, d=None, G=G, batch_size=batch_size, solver=solver, grid_N=grid_N, search_space=search_space)
        _, successful, _, _, num_cut_edges = r.run_rqaoa()
        
        if successful:
            num_cut_edges_list.append(num_cut_edges)
            successful_runs += 1

    print(f"Successful RQAOA runs: {successful_runs}/{num_runs}")
    return num_cut_edges_list

def brute_force_max_cut(G):
    nodes = list(G.nodes())
    n = len(nodes)
    best_cut_value = 0
    best_cut = None

    # Iterate over all possible subsets of nodes (excluding the empty set and the full set)
    for subset in itertools.chain.from_iterable(itertools.combinations(nodes, r) for r in range(1, n)):
        # Calculate cut cost: sum of weights of edges crossing the cut
        cut_value = 0
        # Subset as a set for quick lookup
        subset_set = set(subset)
        
        # Check all edges to see if they cross the cut defined by subset
        for u, v in G.edges():
            if (u in subset_set) != (v in subset_set):  # Only true if exactly one endpoint is in the subset
                cut_value += G[u][v].get('weight', 1)  # Use edge weight if available, otherwise default to 1
                
        # Update best cut if the current cut is better
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            # Create a cut vector that marks nodes in the subset as 1, others as -1
            best_cut = np.array([1 if node in subset_set else -1 for node in nodes])

    return best_cut_value, best_cut