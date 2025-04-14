import cvxpy as cp
import itertools
import networkx as nx
import numpy as np

'''
Implement the computed features related to the GW algorithm.
'''

def cut_cost(x, L):
    return 0.25 * x @ L @ x

def count_cut_edges(cut, G):
    num_edges_cut = 0
    for u, v in G.edges():
        if cut[u] * cut[v] < 0:
            num_edges_cut += 1
    return num_edges_cut

def SDP_max_cut_charles_features(G):
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes)).toarray()
    num_edges = G.number_of_edges()

    X = cp.Variable((n, n), PSD=True)
    obj = 0.25 * cp.trace(L @ X)
    constr = [cp.diag(X) == 1]
    problem = cp.Problem(cp.Maximize(obj), constraints=constr)
    problem.solve(solver=cp.SCS)

    C_relax = problem.value
    u, s, v = np.linalg.svd(X.value)

    # we perform cholesky decomposition using SVD as np.linalg.cholesky gives an error weirdly!
    L_factor = u @ np.diag(np.sqrt(s))
    U = u * np.sqrt(s)

    # Extract the lower triangular part
    lower_triangular_part = np.tril(L_factor)

    # Calculate the percentage of positive elements
    total_elements = lower_triangular_part.size
    positive_elements = np.sum(lower_triangular_part > 0)
    percent_positive = (positive_elements / total_elements)

    # Calculate the percentage of elements close to 0
    percent_close1 = np.sum(np.abs(lower_triangular_part) < 0.1) / total_elements
    percent_close3 = np.sum(np.abs(lower_triangular_part) < 0.001) / total_elements

    num_trials = 1000
    gw_results = np.zeros(num_trials)
    cuts = []

    for i in range(num_trials):
        r = np.random.randn(n)
        r = r / np.linalg.norm(r)
        cut = np.sign(r @ U.T)
        cuts.append(cut)
        current_cut_value = cut_cost(cut, L)
        gw_results[i] = current_cut_value

    expected_costGW = np.mean(gw_results)
    std_costGW = np.std(gw_results)

    # Calculate the percent cut
    percent_cut = (C_relax / num_edges)

    # Calculate expected and standard deviation of GW cost over SDP cost
    expected_costGW_over_sdp_cost = expected_costGW / C_relax
    std_costGW_over_sdp_cost = std_costGW / C_relax

    results = {
        "percent_cut": percent_cut,
        "percent_positive_lower_triangular": percent_positive,
        "percent_close1_lower_triangular": percent_close1,
        "percent_close3_lower_triangular": percent_close3,
        "expected_costGW_over_sdp_cost": expected_costGW_over_sdp_cost,
        "std_costGW_over_sdp_cost": std_costGW_over_sdp_cost
    }

    return results

