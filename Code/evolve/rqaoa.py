import itertools
import os
import pickle
import pprint
import sys
import time
from typing import Callable, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import symengine as sym
from scipy import optimize
from scipy.spatial.distance import hamming
from tqdm import tqdm

np.set_printoptions(suppress=True)


def karloff(m, t, b):
    # ideally one would want to choose b s.t. its value is follows (m - 2b)/m = cos(\theta), where \theta = 2.331122
    # creating a list of t-element subsets of a set {1,...,n} = [n]
    # there has been some hand waving results...for more info check out Swati Gupta's new warm start QAOA paper:https://arxiv.org/pdf/2112.11354.pdf)
    vertices = list(itertools.combinations(np.arange(1, m + 1), t))
    G = nx.Graph()
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if len(list(set(vertices[i]) & set(vertices[j]))) == b:
                G.add_edge(i, j)
    return G

def karloff_alon(m, b):
    vert_cand = itertools.product([-1, 1], repeat=m)  # generates 2^m bitstrings
    bs = []
    for i in vert_cand:
        bs.append(i)
    G = nx.Graph()
    #     for i in range(len(bs)):
    #         for j in range(i+1, len(bs)):
    #             if np.inner(bs[i], bs[j]) == 1 - 2*c:
    #                 G.add_edge(i, j)
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            if hamming(bs[i], bs[j]) * len(bs[i]) == b:
                G.add_edge(i, j)
    return G

def random_weights(graph: nx.Graph,
                   rs: Optional[np.random.RandomState] = None,
                   type: str = 'bimodal',
                   weight_type=None,
                   min_weight=None,
                   max_weight=None):
    """Take a graph and make an equivalent graph with weights of plus or minus
    one on each edge.
    Args:
    graph: A graph to add weights to
    rs: A RandomState for making replicable experiments. If not provided,
        the global numpy random state will be used. Please be careful
        when generating random problems. You should construct *one*
        seeded RandomState at the beginning of your script and use
        that one RandomState (in a deterministic fashion) to generate
        all the problem instances you may need.
    """

    if rs is None:
        rs = np.random
    elif not isinstance(rs, np.random.RandomState):
        raise ValueError("Invalid random state: {}".format(rs))

    problem_graph = nx.Graph()
    for n1, n2 in graph.edges:
        if type == 'bimodal':
            problem_graph.add_edge(n1, n2, weight=rs.choice([-1, 1]))
        elif type == 'pos_gaussian':
            problem_graph.add_edge(n1, n2, weight=np.abs(rs.randn()))
        elif type == 'gaussian':
            problem_graph.add_edge(n1, n2, weight=rs.randn())
        elif type == 'sidon_567':
            problem_graph.add_edge(n1, n2, weight=rs.choice([-5, -6, -7, 5, 6, 7]))
        elif type == 'sidon_28_normalized':
            problem_graph.add_edge(n1, n2,
                                   weight=rs.choice([-8 / 28, -13 / 28, -19 / 28, -1, 8 / 28, 13 / 28, 19 / 28, 1]))
        elif type == 'one':
            problem_graph.add_edge(n1, n2, weight=rs.choice([1]))
        elif type == 'custom':
            if weight_type == 'float':
                problem_graph.add_edge(n1, n2, weight=rs.uniform(min_weight, max_weight))
            elif weight_type == 'integer':
                problem_graph.add_edge(n1, n2, weight=rs.choice(np.arange(min_weight, max_weight + 1)))
    return problem_graph


class Graph:
    def __init__(self,
                 n: int,
                 d: int = None,
                 G: nx.Graph = None):
        self.n = n
        self.d = d
        if G is None:
            G = nx.generators.random_graphs.random_regular_graph(d, n, seed=42)
            G = random_weights(graph=G, rs=np.random.RandomState(42))
        elif G == 'sk_problem':
            G = nx.complete_graph(n)
            G = random_weights(graph=G, rs=np.random.RandomState(42))
        self.G = G
        self.G0 = G.copy()
    
    def copy(self):
        # Creates a new instance of Graph with the same properties
        new_graph = Graph(self.n, self.d, self.G.copy())
        return new_graph

    def reset(self):
        self.G = self.G0.copy()

    def get_G_numpy(self,
                    nodelist: List[int] = None):
        if nodelist is None:
            nodelist = range(self.n)
        return nx.to_numpy_array(self.G, dtype=np.float64, nodelist=nodelist)
    
    def subgraph(self, nodelist):
        subgraph = self.G.subgraph(nodelist).copy()
        subgraph = nx.convert_node_labels_to_integers(subgraph, first_label=0)
        return Graph(n=len(nodelist), G=subgraph)

    def get_G_sparse(self,
                     nodelist: List[int] = None):
        if nodelist is None:
            nodelist = range(self.n)
        return nx.to_scipy_sparse_matrix(self.G, dtype=np.float, nodelist=nodelist)
    
    def nodes(self):
        # Expose the nodes of the nx.Graph
        return list(self.G.nodes())
    
    def is_connected(self):
        """Check if the graph is connected."""
        return nx.is_connected(self.G)

    def convert_labels_to_integers(self):
        self.G = nx.convert_node_labels_to_integers(self.G)
        self.G0 = self.G.copy()  # Also update G0 to maintain consistency after label conversion
        
    def eliminate(self,
                  edge: Tuple[int],
                  sign: float):
        rmv_edges = []
        updt_edges = []
        add_edges = []
        for neighb in self.G.neighbors(edge[1]):
            rmv_edges += [(edge[1], neighb)]
            if neighb not in edge:
                if (edge[0], neighb) in self.G.edges():
                    self.G[edge[0]][neighb]['weight'] += sign * self.G[edge[1]][neighb]['weight']
                    if self.G[edge[0]][neighb]['weight'] == 0:
                        self.G.remove_edge(edge[0], neighb)
                        rmv_edges += [(edge[0], neighb)]
                    else:
                        updt_edges += [(edge[0], neighb)]
                else:
                    self.G.add_edge(edge[0], neighb, weight=sign * self.G[edge[1]][neighb]['weight'])
                    add_edges += [(edge[0], neighb)]
        for e in self.G.edges(edge[0]):
            if e not in updt_edges + rmv_edges + add_edges:
                updt_edges += [e]
        for neighb in self.G.neighbors(edge[1]):
            for e in self.G.edges(neighb):
                if e not in updt_edges + rmv_edges + add_edges:
                    updt_edges += [e]
        self.G.remove_node(edge[1])
        return rmv_edges, updt_edges, add_edges


class RQAOA:
    def __init__(self,
                 n: int,
                 nc: int,
                 d: int,
                 batch_size: int,
                 grid_N: int,
                 search_space: List,
                 G: nx.Graph = None,
                 solver: str = 'analytic_brute'):
        self.n = n
        self.nc = nc
        self.graph = Graph(n, d, G)
        self.batch_size = batch_size
        self.w = self.graph.get_G_numpy()
        self.all_angles = []
        self.ref = ()
        self.solver = solver
        self.grid_N = grid_N
        self.search_space = search_space
        self.x = sym.Symbol('x', real=True)
        self.y = sym.Symbol('y', real=True)
        self.graph_history = []

    def handle_component(self, component_nodes, full_graph):
        # Create a mutable subgraph from the full graph using the specified component nodes
        subgraph_nx = nx.Graph(full_graph.subgraph(component_nodes))
        subgraph_nx = nx.convert_node_labels_to_integers(subgraph_nx)
        local_subgraph = Graph(n=len(component_nodes), G=subgraph_nx)

        local_nodelist = list(local_subgraph.G.nodes())

        if len(local_nodelist) == 1:
            # Handle single-node component case
            return [], 0, [], 0, [], [], [], True, [], 0, 0

        tmp_J = local_subgraph.get_G_numpy(local_nodelist)

        if len(local_nodelist) <= self.nc:
            # Handle small component with brute force
            max_cut, idx, zs = self.bruteforce_full_instance(tmp_J, len(local_nodelist))
            zs = zs[0].tolist()
            print("Brute force max_cut:", max_cut)
            print("Brute force zs:", zs)
            return [], max_cut, zs, 0, [], [], [], True, zs, max_cut, max_cut
        else:
            # Recursive or main RQAOA process
            return self.rqaoa(local_subgraph, local_nodelist)
    
    def run_rqaoa(self):
        batch_rqaoa_angles, batch_energies, batch_zs, exp_flags, ties_flags, ties_batch = [], [], [], [], [], []
        
        for i in range(self.batch_size):
            print("RQAOA agent " + str(i + 1) + "/" + str(self.batch_size))
            self.graph_history = []  # Reset history for each run
            self.graph.reset()
            
            components = list(nx.connected_components(self.graph.G))    # Get connected components
            components = sorted(components, key=len, reverse=True)  # Sort by descending size
            
            self.print_connected_components()
            component_energies = 0
            component_cut_edges = 0
            component_best_energy = 0
            component_rqaoa_angles = []
            component_zs = []
            component_successful = True
            print("Number of component: ", len(components))
            for component in components:
                rqaoa_angles, energy, z, te, exp_flag, ties_flag, check_ties, successful, best_cut, best_energy, cut_edges = self.handle_component(component, self.graph.G)
                
                component_energies += energy
                if component == components[0]:
                    component_zs = z  # Store z only from the largest component
                component_best_energy += best_energy
                component_cut_edges += cut_edges
                component_rqaoa_angles.extend(rqaoa_angles)
                exp_flags.append(exp_flag)
                ties_flags.append(ties_flag)
                ties_batch.append(check_ties)
            
            batch_rqaoa_angles += [component_rqaoa_angles]
            batch_energies += [component_energies]
            batch_zs += [component_zs]
            cut_edges = component_cut_edges
            print("Energy: " + str(component_energies))
            print("Best energy: ", component_best_energy)
    
        arg = np.argmax(batch_energies)
        rqaoa_angles = np.array(batch_rqaoa_angles[arg], dtype=float)
        ref = (batch_energies[arg], batch_zs[arg], batch_energies)
        
        self.all_angles = rqaoa_angles
        self.ref = ref
        print("----------------------------------------------------------------------------------------------")
        print("Final max-cut value of the RQAOA:", self.ref[0])
        print("----------------------------------------------------------------------------------------------")
        return self.ref[0], True, self.ref[1], self.graph_history, cut_edges

    def rqaoa(self, local_graph, nodelist):
        J = local_graph.get_G_numpy(nodelist)
        f_s, h, action_space = self.generate_fs_h_actions(J, nodelist)

        assignments, signs, check_exp, check_ties = [], [], [], []
        rqaoa_angles = np.array([], dtype=float)
        ts, te = time.time(), 0
        
        ep_energies = [float('inf')]  # Initialize with a placeholder value indicating failure
        z_s = [[]]  # Initialize with a placeholder empty assignment list
        cuts_per_iter = [] # To store edge cut counts from all iterations
        exp_flag = 0  # Default false
        ties_flag = 0  # Default false
        best_cut, best_energy, successful = None, float('-inf'), True
        cut_edges = 0  # Initialize cut_edges
        total_cut_edges = 0

        full_graph = local_graph.G
        components = [list(component) for component in nx.connected_components(full_graph)] # Find all connected components in the full graph
        print(f"Initial number of connected components: {len(components)}")
        print(f"Connected components: {components}")
        current_components = [list(component) for component in nx.connected_components(full_graph)]
        z_c = {node: node for node in nodelist}  # Each node starts in its own partition

        for m in tqdm(range(local_graph.n - self.nc)):
            angles, f_val = self.compute_extrema(h, solver=self.solver, search_space=self.search_space)
            rqaoa_angles = np.append(rqaoa_angles, angles)
            expectations, indcs = self.compute_expectations(f_s, angles)
            if len(expectations) == len(set(expectations)):  # check if edge correlations are unique
                check_exp.append(m)
            abs_expectations = np.abs(expectations)
            max_abs = np.flatnonzero(abs_expectations == abs_expectations.max())
            check_ties.append(len(max_abs))
            idx = np.random.choice(max_abs)
            edge, sign = action_space[idx], np.sign(expectations[idx])
            rmv_edges, updt_edges, add_edges = local_graph.eliminate(edge, sign)

            # After edge removal, update partitions
            if nx.is_connected(local_graph.G):
                z_c[edge[1]] = z_c[edge[0]]  # Merge partition labels if still connected
            else:
                # Assign new partition label to the component that includes edge[1]
                for component in nx.connected_components(local_graph.G):
                    if edge[1] in component:
                        for node in component:
                            z_c[node] = z_c[edge[1]]
                        break

            assignments += [edge]
            signs += [sign]
            nodelist = np.array(nodelist)
            nodelist = nodelist[nodelist != edge[1]]
            J = local_graph.get_G_numpy(nodelist)
            f_s, action_space = self.update(f_s, action_space, J, nodelist, rmv_edges, updt_edges, add_edges)
            h = self.compute_h(f_s, action_space, J, nodelist)
            
            full_graph = local_graph.G
            components = list(nx.connected_components(full_graph))  # Get connected components
            components = sorted(components, key=len, reverse=True)  # Sort by descending size
            current_components = [list(component) for component in nx.connected_components(full_graph)] # Find all connected components in the full graph
            print(f"Number of components after iteration {m + 1}: {len(current_components)}")
            print(f"Connected components: {current_components}")
            
            self.graph_history.append((local_graph.G.copy(), edge, edge[1])) # Track the graph state change
            
            if len(current_components) > 1:  # Check if the graph split into more components
                print("----------------------------------------------------------------------------------------------")
                print("************** More than one component in the graph at the current iteration *****************")
                print("----------------------------------------------------------------------------------------------")
                print(J.shape)
                try:
                    z_s, ep_energies, cuts_per_iter = self.expand_result_multi_component(z_c, assignments, signs, nodelist, local_graph)
                except IndexError:
                    _, z_c, z_ss = self.bruteforce_full_instance(J, self.nc)
                    z_s, ep_energies, cuts_per_iter = self.expand_result(z_c, assignments, signs, nodelist, local_graph)
                except Exception as e:
                    print("An unexpected error occurred:", e)
                    raise

                total_cut_edges = cuts_per_iter[0]
                print("----------------------------------------------------------------------------------------------")
                print("RQAOA result before handling multiple components of the graph: ", total_cut_edges)
                print("----------------------------------------------------------------------------------------------")

                exp_flags, ties_flags, ties_batch = [], [], []
                component_energies = 0
                component_cut_edges = 0
                component_best_energy = 0
                component_rqaoa_angles = []
                component_zs = []
                component_best_cut = []
                for component in current_components:
                    rqaoa_angles, energy, z, te, exp_flag, ties_flag, check_ties, successful, best_cut, best_energy, cut_edges = self.handle_component(component, full_graph)
                    component_energies += energy
                    if component == components[0]:  # Append results only from the largest component. Check if it's the largest component
                        component_zs = z  # Store z only from the largest component
                    component_best_energy += best_energy
                    component_cut_edges += cut_edges
                    component_rqaoa_angles.extend(rqaoa_angles)
                    exp_flags.append(exp_flag)
                    ties_flags.append(ties_flag)
                    ties_batch.append(check_ties)
                
                component_energies += total_cut_edges
                component_best_energy += total_cut_edges
                component_cut_edges += total_cut_edges

                print("----------------------------------------------------------------------------------------------")
                print("**************************** Results for Handling Multiple Components ************************")
                print(f"Total Energies: {component_energies}")
                print(f"Best Overall Energy: {component_best_energy}")
                print(f"Total Cut Edges Across All Components: {component_cut_edges}")
                print("----------------------------------------------------------------------------------------------")
                    
                return component_rqaoa_angles, component_energies, component_zs, 0, exp_flags, ties_flags, ties_batch, True, component_zs, component_best_energy, component_cut_edges      
        
        _, z_c, z_ss = self.bruteforce_full_instance(J, self.nc)
        z_s, ep_energies, cuts_per_iter = self.expand_result(z_c, assignments, signs, nodelist, local_graph)
        print("Energies: ", ep_energies)
        print("Cuts per iteration: ", cuts_per_iter)
        best_cut = z_s[0]
        best_energy = ep_energies[0]
        cut_edges = cuts_per_iter[0]
    
        te = time.time() - ts

        if check_exp == list(np.arange(local_graph.n - self.nc)):
            exp_flag = 1  # if edge correlations at all iterations are unique when wt's are N(0,1)
        else:
            exp_flag = 0

        if len(check_ties) == np.sum(check_ties):
            ties_flag = 1  # if there are no ties while using argmax
        else:
            ties_flag = 0
        
        print("----------------------------------------------------------------------------------------------")
        print("*********************** Final Results for Single-Component Configuration *********************")
        print(f"Total number of cut edges: {cut_edges}")
        print(f"Elapsed time: {te} seconds")
        print(f"Best energy found: {best_energy}")
        print("z_s[0]", z_s[0])
        print("----------------------------------------------------------------------------------------------")
        return rqaoa_angles, best_energy, best_cut, te, exp_flag, ties_flag, check_ties, True, best_cut, best_energy, cut_edges

    def print_connected_components(self):
        full_graph = self.graph.G   # Access the full graph directly
        components = [list(component) for component in nx.connected_components(full_graph)] # Find all connected components in the full graph
        components = sorted(components, key=len, reverse=True)  # Sort the components by size if needed
        print(f"Number of connected components: {len(components)}")
        print(f"Connected components: {components}")
    
    def store_agent(self, pickle_path=None):
        if pickle_path is None:
            pickle_path = self.pickle_path
        pickle.dump(self, open(pickle_path, 'wb'))

    def update(self,
               f_s,
               action_space: List[Tuple[int]],
               J: np.ndarray,
               nodelist: List[int],
               rmv_edges: List[Tuple[int]],
               updt_edges: List[Tuple[int]],
               add_edges: List[Tuple[int]]):
        nl = list(nodelist)
        for edge in rmv_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            indx = action_space.index(edge)
            action_space.pop(indx)
            f_s.pop(indx)

        for edge in add_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            inserted = False
            for i in range(len(action_space)):
                edge_i = action_space[i]
                if (edge_i[0] == edge[0] and edge_i[1] > edge[1]) or edge_i[0] > edge[0]:
                    action_space.insert(i, edge)
                    f_s.insert(i, self.compute_f(J, nl.index(edge[0]), nl.index(edge[1])))
                    inserted = True
                    break
            if not inserted:
                action_space += [edge]
                f_s += [self.compute_f(J, nl.index(edge[0]), nl.index(edge[1]))]

        for edge in updt_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            if edge in action_space:
                indx = action_space.index(edge)
                f_s[indx] = self.compute_f(J, nl.index(edge[0]), nl.index(edge[1]))

        return f_s, action_space

    def compute_h(self,
                  f_s,
                  action_space: List[Tuple[int]],
                  J: np.ndarray,
                  nodelist: List[int]):
        h = 0.
        count = 0
        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    if action_space[count] != (nodelist[i], nodelist[j]):
                        print("Wrong count")
                    h += J[i, j] * f_s[count]  # max-cut

                    count += 1
        return h

    def compute_expectations(self,
                             f_s,
                             angles: List[float]):
        expectations = []
        indcs = []
        x = self.x
        y = self.y

        for i, f in enumerate(f_s):
            if f in f_s[:i]:
                indx = f_s.index(f)
                expectations += [expectations[indx]]
                indcs += [indx]
            else:
                x0, y0 = angles[0], angles[1]
                expectations += [float(f.subs({x: x0, y: y0}))]
                indcs += [i]

        return expectations, indcs

    def get_binary(self,
                   x: int,
                   n: int):

        return 2 * np.array([int(b) for b in bin(x)[2:].zfill(n)], dtype=np.int32) - 1

    def bruteforce(self,
                   J: np.ndarray,
                   n: int):
        maxi = -n
        idx = []
        for i in range(2 ** n):
            z = self.get_binary(i, n)
            val = 0
            for k in range(len(J)):
                for l in range(k + 1, len(J)):
                    if z[k] != z[l]:    # if the edges are in different partition
                        val += J[k, l]
            if val > maxi:
                maxi = val
                idx = [i]
            elif val == maxi:
                idx += [i]
        return maxi, idx

    def bruteforce_full_instance(self,
                                 J: np.ndarray,
                                 n: int):
        # max_cut: Represents the maximum value of the "cut" found by the brute force method.
        # idx: Contains the indices of all partitions that yield this maximum cut value.
        # zs: Each element in zs is the binary representation of a partition indexed by idx. 
        max_cut, idx = self.bruteforce(J, n)
        zs = []
        for i in idx:
            zs.append(self.get_binary(i, n))
        return max_cut, idx, zs

    def compute_f(self,
                  J: np.ndarray,
                  i: int,
                  j: int):

        x, y = self.x, self.y
        C = sym.cos
        S = sym.sin

        prod1, prod2, prod3, prod4 = 1., 1., 1., 1.
        for k in range(len(J)):
            if k not in [i, j]:
                if J[i, k] - J[j, k]:
                    prod1 *= C(2 * x * (J[i, k] - J[j, k]))
                if J[i, k] + J[j, k]:
                    prod2 *= C(2 * x * (J[i, k] + J[j, k]))
                if J[i, k]:
                    prod3 *= C(2 * x * J[i, k])
                if J[j, k]:
                    prod4 *= C(2 * x * J[j, k])
        term = 0.5 * (S(2 * y) ** 2) * (prod1 - prod2) + 0.5 * S(4 * y) * S(2 * x * J[i, j]) * (prod3 + prod4)

        return term

    def generate_fs_h_actions(self,
                              J: np.ndarray,
                              nodelist: List[int]):
        f_s = []
        h = 0.
        action_space = []

        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    action_space += [(nodelist[i], nodelist[j])]
                    term = self.compute_f(J, i, j)
                    f_s += [term]
                    h += J[i, j] * term   # for max-cut
        return f_s, h, action_space

    def compute_extrema(self,
                        h,
                        search_space: List,
                        solver: str = 'analytic_brute',
                        polishing_optimizer: Callable = optimize.cobyla):

        x = self.x
        y = self.y
        
        if solver == 'analytic_brute':
            # \gamma is x and \beta is y
            # E(x, y) = - (p * cos(4 * y) + q * sin(4 * y) + r), where p,q,r are complicated eqn's of x. Find p,q,r.
            # A -ve sign because of max cut
            
            r = - (h.subs({x: x, y: np.pi / 8}) + h.subs({x: x, y: -np.pi / 8})) / 2
            q = - (h.subs({x: x, y: np.pi / 8}) - h.subs({x: x, y: -np.pi / 8})) / 2
            p = - h.subs({x: x, y: 0}) - r
            
            # No -ve sign for fun as we want to minimize the energy that gives us the max cut
            max_y_fun = - (r + sym.sqrt(p ** 2 + q ** 2))  # maximum of E(x,y) over all y's.
            fun = sym.Lambdify([(x)], max_y_fun, backend='llvm')

            # for 3-reg landscape is symmetrical for gamma between 0 to np.pi at np.pi/2
            param_ranges = (slice(search_space[0], search_space[1], abs(search_space[1] - search_space[0]) / self.grid_N),)
            res_brute = optimize.brute(fun, param_ranges, full_output=True, finish=polishing_optimizer)

            solution = [res_brute[0]]

            q_val = q.subs({x: solution[0]})
            p_val = p.subs({x: solution[0]})
            y_val = 1 / 4 * (sym.atan2(q_val, p_val))
            # breakpoint()
            # print(p_val * sym.cos(4 * y_val))
            
            # print(p_val)
            # print(y_val)
            
            # print(type(p_val))
            # print(type(y_val))
            
            # assert (p_val * sym.cos(4 * y_val)).n(73, real=True) >= 0
            # assert (q_val * sym.sin(4 * y_val)).n(73, real=True) >= 0
            
            solution = np.append(solution, np.float64(y_val))
            extrema = res_brute[1]
        else:
            raise ValueError(f'Optimizer {self.solver}  is not implemented')

        return solution, extrema

    def expand_result(self, z_c, assignments, signs, nodelist, local_graph):
        z_s = [self.get_binary(z, self.nc) for z in z_c]
        # print(z_s)
        # print('---------')
        z_s = [np.array([z[nodelist.tolist().index(i)] if i in nodelist else 0 for i in range(self.n)], dtype=np.int32)
               for z in z_s]
        # print(z_s)
        # print('--------')
        # print(signs)
        local_graph.reset()
        nodelist = list(local_graph.nodes())  # Update nodelist to current subgraph nodes
        J = local_graph.get_G_numpy(nodelist=nodelist)  # Generate matrix using updated nodelist

        ep_energies = []
        num_edges_cut = []  # This will now be a list to hold the count for each configuration
        for i, assgn in enumerate(assignments[::-1]):
            for j, z in enumerate(z_s):
                z[assgn[1]] = signs[-i - 1] * z[assgn[0]]
                z_s[j] = z
            val = 0
            edges_cut = 0  # Initialize edge cut counter for this iteration
            for k in range(len(J)):
                for l in range(k + 1, len(J)):
                    if J[k, l] != 0:
                        if z_s[0][k] != z_s[0][l]:
                            edges_cut += 1
                    if z_s[0][k] != z_s[0][l]:    # if the edges are in different partition
                        val += J[k, l]
                        
            # print("edges_cut: ", edges_cut)
            ep_energies.insert(0, val)
            num_edges_cut.insert(0, edges_cut)  # Record the number of edges cut
            # print(num_edges_cut)
        return z_s, ep_energies, num_edges_cut

    def expand_result_multi_component(self, z_c, assignments, signs, nodelist, local_graph):
        nodelist = list(nodelist)
        actual_max_index = max(nodelist)  # Actual max index from the nodelist
        expected_max_index = 100

        # Initialize z_s arrays with the correct expected size
        z_s = [np.zeros(expected_max_index, dtype=np.int32) for _ in z_c]
        for i, z in enumerate(z_c):
            binary_array = self.get_binary(z, expected_max_index)
            z_s[i][:len(binary_array)] = binary_array

        node_index_map = {node: idx for idx, node in enumerate(nodelist)}
        z_s = [np.array([z[node_index_map[i]] if i in node_index_map else 0
                for i in range(expected_max_index)], dtype=np.int32)  # Use expected_max_index here
                for z in z_s]

        local_graph.reset()
        nodelist = list(local_graph.nodes())  # Update nodelist to current subgraph nodes
        J = local_graph.get_G_numpy(nodelist=nodelist)  # Generate matrix using updated nodelist

        ep_energies = []
        num_edges_cut = []  # This will now be a list to hold the count for each configuration
        for i, assgn in enumerate(assignments[::-1]):
            for j, z in enumerate(z_s):
                if 0 <= assgn[0] < len(z) and 0 <= assgn[1] < len(z):
                    z[assgn[1]] = signs[-i - 1] * z[assgn[0]]
                else:
                    print(f"Assignment out of range: assgn[0]={assgn[0]}, assgn[1]={assgn[1]}, len(z)={len(z)}")
                z_s[j] = z
            val = 0
            edges_cut = 0  # Initialize edge cut counter for this iteration
            for k in range(len(J)):
                for l in range(k + 1, len(J)):
                    if k < len(z_s[0]) and l < len(z_s[0]):
                        if J[k, l] != 0:
                            if z_s[0][k] != z_s[0][l]:
                                edges_cut += 1
                        if z_s[0][k] != z_s[0][l]:    # if the edges are in different partitions
                            val += J[k, l]
                    else:
                        print(f"Attempt to access out of bound index: k={k}, l={l}, size={len(z_s[0])}")
                        
            ep_energies.insert(0, val)
            num_edges_cut.insert(0, edges_cut)  # Record the number of edges cut
        return z_s, ep_energies, num_edges_cut




