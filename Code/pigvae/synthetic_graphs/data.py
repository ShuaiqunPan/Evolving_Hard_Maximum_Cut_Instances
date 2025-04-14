import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import random
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy

from networkx.generators.random_graphs import *
from networkx.generators.ego import ego_graph
from networkx.generators.geometric import random_geometric_graph

import os
import pickle
from pytorch_lightning.utilities import rank_zero_only
from scipy.sparse.csgraph import shortest_path
import torch.nn.functional as F


class GeometricGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, samples_per_epoch=100000, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        g = random_geometric_graph(n=n, radius=0.5)
        return g


class RegularGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, samples_per_epoch=100000, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        g = random_regular_graph(n=n, d=4)
        return g


class BarabasiAlbertGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, m_min=1, m_max=5,
                 samples_per_epoch=100000, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self.n_min == self.n_max:
            n = self.m_min
        else:
            n = np.random.randint(low=self.n_min, high=self.n_max)
        if self.m_min == self.m_max:
            m = self.m_min
        else:
            m = np.random.randint(low=self.m_min, high=self.m_max)
        g = barabasi_albert_graph(n, m)
        return g

# GNP
class BinomialGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, p_min=0.4, p_max=0.6,
                 samples_per_epoch=100000, pyg=False, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.p_min = p_min
        self.p_max = p_max
        self.samples_per_epoch = samples_per_epoch
        self.pyg = pyg

    def __len__(self):
        return self.samples_per_epoch
    
    def save_graph(self, graph, file_path):
        # Convert to NetworkX graph if in PyG format
        if self.pyg:
            graph = to_networkx(graph)
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f)

    def get_largest_subgraph(self, g):
        g = g.subgraph(sorted(nx.connected_components(g), key=len, reverse=True)[0])
        g = nx.convert_node_labels_to_integers(g, first_label=0)
        return g

    def __getitem__(self, idx):
        # n = np.random.randint(low=self.n_min, high=self.n_max)
        n = self.n_min
        if self.p_min == self.p_max:
            p = self.p_min
        else:
            p = np.random.randint(low=self.p_min, high=self.p_max)
        p = np.random.uniform(low=self.p_min, high=self.p_max)
        g = binomial_graph(n, p)
        if self.pyg:
            g = from_networkx(g)
        return g


# Dataset from Network Data
class MTXGraphDataset(Dataset):
    def __init__(self, folder_path, samples_per_epoch=100000, pyg=False):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.pyg = pyg
        self.graph_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mtx') or f.endswith('.edges')]

    def load_graph(self, file_path):
        try:
            print(f"Loading graph from file: {file_path}")
            G = nx.Graph()
            first_non_comment_line = True
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('%'):
                        continue  # Skip comment lines
                    parts = line.strip().split()
                    if first_non_comment_line and len(parts) == 3:
                        first_non_comment_line = False
                        continue  # Skip the first non-comment line if it contains metadata
                    if len(parts) >= 2:
                        u, v = map(int, parts[:2])
                        G.add_edge(u - 1, v - 1)  # Add edge without setting weight
                    first_non_comment_line = False
            return G
        except Exception as e:
            print(f"Error loading graph from file {file_path}: {e}")
            raise

    def __len__(self):
        return self.samples_per_epoch
    
    def save_graph(self, graph, file_path):
        # Convert to NetworkX graph if in PyG format
        if self.pyg:
            graph = to_networkx(graph)
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f)

    def __getitem__(self, idx):
        selected_file = random.choice(self.graph_files)
        g = self.load_graph(selected_file)  # Only one value is returned

        if self.pyg:
            g = from_networkx(g)

        return g


class RandomGraphDataset(Dataset):
    def __init__(self, n_min=20, n_max=20, samples_per_epoch=100000, pyg=False, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.samples_per_epoch = samples_per_epoch
        self.graph_generator = GraphGenerator()
        self.pyg = pyg

    def __len__(self):
        return self.samples_per_epoch
    
    def save_graph(self, graph, label, params, file_path):
        # Modify to save graph, label, and parameters
        if self.pyg:
            graph = to_networkx(graph)
        data = {'graph': graph, 'label': label, 'params': params}
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def __getitem__(self, idx):
        n = self.n_max
        g, graph_type, params = self.graph_generator(n)
        # n = np.random.randint(low=self.n_min, high=self.n_max)  # Randomly select the number of nodes
        # g, graph_type, params = self.graph_generator(n)  # Generate graph with n nodes
        return g, graph_type, params  # Return them


class PyGRandomGraphDataset(RandomGraphDataset):
    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        g = self.graph_generator(n)
        g = from_networkx(g)
        if g.pos is not None:
            del g.pos
        return g

class DenseGraphBatch(Data):
    graph_types = ["binominal", "barabasi_albert", "random_regular", "watts_strogatz", 
                   "newman_watts_strogatz", "dual_barabasi_albert"]
    
    LABEL_MAPPING = {graph_type: i for i, graph_type in enumerate(graph_types)}
    
    def __init__(self, node_features, edge_features, mask, properties=None, labels=None, params=None):
        self.node_features = node_features
        self.edge_features = edge_features
        self.mask = mask
        self.properties = properties
        self.labels = labels
        self.params = params

    @classmethod
    def from_sparse_graph_list(cls, data_list, labels=False):
        node_features = []
        edge_features = []
        mask = []
        y = []  # For storing graph types or other labels
        params = []  # For storing additional parameters
        props = []
        max_num_nodes = 0

        for item in data_list:
            try:
                if isinstance(item, tuple):
                    graph, graph_type, item_params = item  # Tuple format
                elif isinstance(item, dict):
                    graph = item['graph']
                    graph_type = item.get('label')
                    item_params = item.get('params', {})
                else:
                    raise ValueError(f"Item is neither a tuple nor a dictionary: {item}")

                # print(f"Processing graph of type: {graph_type}")  # Debug print
                
                max_num_nodes = max(max_num_nodes, graph.number_of_nodes())
                if labels and graph_type is not None:
                    mapped_label = cls.LABEL_MAPPING.get(graph_type, -1)  # Use class-level LABEL_MAPPING
                    # print(f"Mapped label for '{graph_type}': {mapped_label}")  # Debug print
                    y.append(mapped_label)
                if item_params is not None:
                    params.append(item_params)
            except AttributeError as e:
                print(f"Error processing item: {e}, {item}")
                continue
            
            num_nodes = graph.number_of_nodes()
            props.append(torch.Tensor([num_nodes]))
            graph.add_nodes_from([i for i in range(num_nodes, max_num_nodes)])
            nf = torch.ones(max_num_nodes, 1)
            node_features.append(nf.unsqueeze(0))
            
            dm = torch.from_numpy(floyd_warshall_numpy(graph)).long()
            dm = torch.clamp(dm, 0, 5).unsqueeze(-1)
            num_nodes = dm.size(1)
            dm = torch.zeros((num_nodes, num_nodes, 6)).type_as(dm).scatter_(2, dm, 1).float()
            edge_features.append(dm)
            
            mask.append((torch.arange(max_num_nodes) < num_nodes).unsqueeze(0))

        node_features = torch.cat(node_features, dim=0)
        edge_features = torch.stack(edge_features, dim=0)
        mask = torch.cat(mask, dim=0)
        props = torch.cat(props, dim=0)
        
        # Convert labels list to tensor
        if labels:
            y = torch.tensor(y, dtype=torch.long)

        return cls(node_features=node_features, edge_features=edge_features, mask=mask, properties=props, labels=y if labels else None, params=params)

    def __repr__(self):
        repr_list = []
        for key, value in self.__dict__.items():
            if value is not None and hasattr(value, 'shape'):
                shape_info = list(value.shape)
            else:
                shape_info = str(type(value))
            repr_list.append(f"{key}={shape_info}")
        return "DenseGraphBatch({})".format(", ".join(repr_list))


class DenseGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, labels=False, **kwargs):
        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=lambda data_list: DenseGraphBatch.from_sparse_graph_list(data_list, labels), **kwargs)


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, graph_family, graph_kwargs=None, samples_per_epoch=100000, batch_size=32,
                 distributed_sampler=True, num_workers=0, use_saved_graphs=False, save_dir=None):
        super().__init__()
        if graph_kwargs is None:
            graph_kwargs = {}
        self.graph_family = graph_family
        self.graph_kwargs = graph_kwargs
        self.samples_per_epoch = samples_per_epoch
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.distributed_sampler = distributed_sampler
        self.train_dataset = None
        self.eval_dataset = None
        self.train_sampler = None
        self.eval_sampler = None
        
        self.use_saved_graphs = use_saved_graphs
        self.save_dir = save_dir

    def make_dataset(self, samples_per_epoch):
        if self.use_saved_graphs:
            return SavedGraphDataset(self.save_dir)
        
        if self.graph_family == "binomial":
            ds = BinomialGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == "barabasi_albert":
            ds = BarabasiAlbertGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == "regular":
            ds = RegularGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == "geometric":
            ds = GeometricGraphDataset(samples_per_epoch=samples_per_epoch)
        elif self.graph_family == "all":
            ds = RandomGraphDataset(samples_per_epoch=samples_per_epoch)
        else:
            raise NotImplementedError
        return ds

    def train_dataloader(self):
        self.train_dataset = self.make_dataset(samples_per_epoch=self.samples_per_epoch)
        if self.distributed_sampler:
            train_sampler = DistributedSampler(
                dataset=self.train_dataset,
                shuffle=False
            )
        else:
            train_sampler = None
        self.save_sampled_graphs()
        return DenseGraphDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            labels=True,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        
    def on_epoch_end(self):
        self.save_sampled_graphs(epoch_samples=60)

    def on_train_end(self):
        save_dir = "saved_training_samples_mix_6000_20nodes_1"
        os.makedirs(save_dir, exist_ok=True)
        saved_count = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
        total_required = 6000

        if saved_count < total_required:
            remaining = total_required - saved_count
            self.save_additional_samples(remaining, save_dir, saved_count)

    def save_additional_samples(self, remaining, save_dir, start_index):
        indices = torch.randint(0, self.samples_per_epoch, (remaining,))
        for idx in indices:
            graph, graph_type, params = self.train_dataset[idx]
            file_path = os.path.join(save_dir, f"graph_{start_index}.pickle")
            self.train_dataset.save_graph(graph, graph_type, params, file_path)
            start_index += 1
    
    @rank_zero_only
    def save_sampled_graphs(self, epoch_samples=60):
        # Check if the dataset has the 'save_graph' attribute
        if hasattr(self.train_dataset, 'save_graph'):
            save_dir = "saved_training_samples_mix_6000_20nodes_1"
            os.makedirs(save_dir, exist_ok=True)
            
            # Calculate already saved samples
            saved_count = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
            total_required = 6000

            if saved_count < total_required:
                remaining = total_required - saved_count
                samples_to_save = min(epoch_samples, remaining)
                indices = torch.randint(0, self.samples_per_epoch, (samples_to_save,))
                for i in indices:
                    graph, graph_type, params = self.train_dataset[i]
                    file_path = os.path.join(save_dir, f"graph_{saved_count}.pickle")
                    self.train_dataset.save_graph(graph, graph_type, params, file_path)
                    saved_count += 1
            else:
                file_indices = torch.randint(0, total_required, (epoch_samples,))
                new_indices = torch.randint(0, self.samples_per_epoch, (epoch_samples,))
                for file_idx, data_idx in zip(file_indices, new_indices):
                    graph, graph_type, params = self.train_dataset[data_idx]
                    file_path = os.path.join(save_dir, f"graph_{file_idx}.pickle")
                    self.train_dataset.save_graph(graph, graph_type, params, file_path)

    def val_dataloader(self):
        self.eval_dataset = self.make_dataset(samples_per_epoch=4096)
        if self.distributed_sampler:
            eval_sampler = DistributedSampler(
                dataset=self.eval_dataset,
                shuffle=False
            )
        else:
            eval_sampler = None
        return DenseGraphDataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            labels=True,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=eval_sampler,
        )
    
class GraphGenerator(object):
    def __init__(self):
        self.graph_params = {
            "binominal": {
                "func": binomial_graph,
                "kwargs_float_ranges": {
                    "p": (0.2, 0.6)
                }
            },
            "newman_watts_strogatz": {
                "func": newman_watts_strogatz_graph,
                "kwargs_int_ranges": {
                    "k": (2, 6),
                },
                "kwargs_float_ranges": {
                    "p": (0.2, 0.6)
                }
            },
            "watts_strogatz": {
                "func": watts_strogatz_graph,
                "kwargs_int_ranges": {
                    "k": (2, 6),
                },
                "kwargs_float_ranges": {
                    "p": (0.2, 0.6)
                }
            },
            "random_regular": {
                "func": random_regular_graph,
                "kwargs_int_ranges": {
                    "d": (3, 6),  # n*d must be even
                }
            },
            "barabasi_albert": {
                "func": barabasi_albert_graph,
                "kwargs_int_ranges": {
                    "m": (1, 6),
                }
            },
            "dual_barabasi_albert": {
                "func": dual_barabasi_albert_graph,
                "kwargs_int_ranges": {
                    "m1": (1, 6),
                    "m2": (1, 6),
                },
                "kwargs_float_ranges": {
                    "p": (0.1, 0.9)
                }
            }
        }
        self.graph_types = list(self.graph_params.keys())

    def __call__(self, n, graph_type=None):
        while True:
            if graph_type is None:
                graph_type = random.choice(self.graph_types)
            params = self.graph_params[graph_type]
            kwargs = {}
            if "kwargs" in params:
                kwargs = {**params["kwargs"]}
            if "kwargs_int_ranges" in params:
                for key, arg in params["kwargs_int_ranges"].items():
                    kwargs[key] = np.random.randint(arg[0], arg[1] + 1)
            if "kwargs_float_ranges" in params:
                for key, arg in params["kwargs_float_ranges"].items():
                    kwargs[key] = np.random.uniform(arg[0], arg[1])

            if graph_type == "random_regular" and n * kwargs.get("d", 0) % 2 != 0:
                n -= 1

            try:
                g = params["func"](n=n, **kwargs)
                return g, graph_type, kwargs
            except nx.exception.NetworkXError:
                # If an error occurs, the loop will continue and try with a new graph type
                continue


class EvalRandomGraphDataset(Dataset):
    def __init__(self, n, pyg=False):
        self.n = n
        self.pyg = pyg
        self.graph_params = {
            "binominal": {
                "func": binomial_graph,
                "kwargs": {
                    "p": (0.25, 0.35, 0.5)
                }
            },
            "newman_watts_strogatz": {
                "func": newman_watts_strogatz_graph,
                "kwargs": {
                    "k": (2, 2, 5, 5),
                    "p": (0.25, 0.75, 0.25, 0.75,)
                }
            },
            "watts_strogatz": {
                "func": watts_strogatz_graph,
                "kwargs": {
                    "k": (2, 2, 5, 5),
                    "p": (0.25, 0.75, 0.25, 0.75,)
                }
            },
            "random_regular": {
                "func": random_regular_graph,
                "kwargs": {
                    "d": (3, 4, 5, 6)
                }
            },
            "barabasi_albert": {
                "func": barabasi_albert_graph,
                "kwargs": {
                    "m": (1, 2, 3, 4),
                }
            },
            "dual_barabasi_albert": {
                "func": dual_barabasi_albert_graph,
                "kwargs": {
                    "m1": (2, 2),
                    "m2": (4, 1),
                    "p": (0.5, 0.5)
                }
            }
        }
        # no ego
        self.graph_types = ["binominal", "barabasi_albert", "random_regular", "watts_strogatz", "newman_watts_strogatz", "dual_barabasi_albert"]
        graphs, labels = self.generate_dataset()
        c = list(zip(graphs, labels))

        random.shuffle(c)

        self.graphs, self.labels = zip(*c)

    def generate_dataset(self):
        label = 0
        graphs = []
        labels = []
        for j, graph_type in enumerate(self.graph_types):
            params = self.graph_params[graph_type]
            func = params["func"]
            if "kwargs" in params:
                kwargs = params["kwargs"]
            else:
                kwargs = None
            if "kwargs_fix" in params:
                kwargs_fix = params["kwargs_fix"]
            else:
                kwargs_fix = None
            if kwargs is not None:
                num_settings = len(list(kwargs.values())[0])
            else:
                num_settings = 1
            for i in range(num_settings):
                final_kwargs = {}
                if kwargs is not None:
                    for key, args in kwargs.items():
                        if num_settings > 1:
                            final_kwargs[key] = args[i]
                        else:
                            final_kwargs[key] = args
                num_graphs = int(256 / num_settings)
                if kwargs_fix is not None:
                    final_kwargs2 = {**final_kwargs, **kwargs_fix}
                elif kwargs is None:
                    final_kwargs2 = kwargs_fix
                else:
                    final_kwargs2 = final_kwargs
                gs = [func(n=self.n, **final_kwargs2) for _ in range(num_graphs)]
                graphs.extend(gs)
                labels.extend(len(gs) * [label])
                label += 1
        return graphs, labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        if self.pyg:
            g = from_networkx(graph)
            if g.pos is not None:
                del g.pos
            if g.edge_index.dtype != torch.long:
                print(g)
            g.y = torch.Tensor([label]).long()
            return g
        else:
            return graph, label


class RandomGraphDatasetGenerate(Dataset):
    def __init__(self, n, save_dir, pyg=False):
        self.n = n
        self.pyg = pyg
        self.save_dir = save_dir
        self.graph_params = {
            "binominal": {
                "func": binomial_graph,
                "kwargs": {
                    "p": (0.25, 0.35, 0.5)
                }
            },
            "newman_watts_strogatz": {
                "func": newman_watts_strogatz_graph,
                "kwargs": {
                    "k": (2, 2, 5, 5),
                    "p": (0.25, 0.75, 0.25, 0.75,)
                }
            },
            "watts_strogatz": {
                "func": watts_strogatz_graph,
                "kwargs": {
                    "k": (2, 2, 5, 5),
                    "p": (0.25, 0.75, 0.25, 0.75,)
                }
            },
            "random_regular": {
                "func": random_regular_graph,
                "kwargs": {
                    "d": (3, 4, 5, 6)
                }
            },
            "barabasi_albert": {
                "func": barabasi_albert_graph,
                "kwargs": {
                    "m": (1, 2, 3, 4),
                }
            },
            "dual_barabasi_albert": {
                "func": dual_barabasi_albert_graph,
                "kwargs": {
                    "m1": (2, 2),
                    "m2": (4, 1),
                    "p": (0.5, 0.5)
                }
            }
        }
        self.graph_types = ["binominal", "barabasi_albert", "random_regular", "watts_strogatz", "newman_watts_strogatz", "dual_barabasi_albert"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.graph_files = self.generate_and_save_dataset()

    def generate_and_save_dataset(self):
        graph_files = []
        label = 0
        for j, graph_type in enumerate(self.graph_types):
            params = self.graph_params[graph_type]
            func = params["func"]
            kwargs = params.get("kwargs", {})
            num_settings = len(list(kwargs.values())[0]) if kwargs else 1
            for i in range(num_settings):
                final_kwargs = {key: args[i] if len(args) > 1 else args[0] for key, args in kwargs.items()}
                num_graphs = 72  # Changed to generate 500 graphs per setting
                for _ in range(num_graphs):
                    graph = func(n=self.n, **final_kwargs)
                    file_path = os.path.join(self.save_dir, f"{graph_type}_{label}_{_}.pkl")  # Modified filename to include graph count
                    with open(file_path, 'wb') as f:
                        pickle.dump({'graph': graph, 'label': graph_type, 'params': final_kwargs}, f)
                    graph_files.append(file_path)
                label += 1
        return graph_files

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        with open(self.graph_files[idx], 'rb') as f:
            data = pickle.load(f)
        return data



class EvalRandomBinomialGraphDataset(Dataset):
    def __init__(self, n_min, n_max, p_min, p_max, num_samples, pyg=False):
        self.n_min = n_min
        self.n_max = n_max
        self.p_min = p_min
        self.p_max = p_max
        self.num_samples = num_samples
        self.pyg = pyg
        self.graphs, self.labels = self.generate_dataset()

    def generate_dataset(self):
        graphs = []
        labels = []
        for i in range(self.num_samples):
            n = np.random.randint(low=self.n_min, high=self.n_max)
            p = np.random.uniform(low=self.p_min, high=self.p_max)
            g = binomial_graph(n, p)
            if self.pyg:
                g = from_networkx(g)
                g.y = p
            graphs.append(g)
            labels.append(p)
        return graphs, labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.pyg:
            return graph
        else:
            label = self.labels[idx]
            return graph, label


# class SavedGraphDataset(Dataset):
#     def __init__(self, save_dir, sample_size=6000):
#         self.save_dir = save_dir
#         all_file_names = os.listdir(save_dir)

#         # Sort the file names to ensure deterministic order
#         sorted_file_names = sorted(all_file_names, key=lambda x: int(x.split('_')[1].split('.')[0]))

#         # Ensure sample size is not greater than the number of available files
#         sample_size = min(sample_size, len(sorted_file_names))

#         # Use the first 'sample_size' files after sorting
#         self.file_names = sorted_file_names[:sample_size]

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         file_path = os.path.join(self.save_dir, self.file_names[idx])
#         with open(file_path, 'rb') as f:
#             graph = pickle.load(f)
#         # Convert graph to the desired format if necessary
#         return graph


class SavedGraphDataset(Dataset):
    def __init__(self, save_dir, sample_size=60000):
        self.save_dir = save_dir
        all_file_names = os.listdir(save_dir)

        # Ensure sample size is not greater than the number of available files
        sample_size = min(sample_size, len(all_file_names))

        # Use the first 'sample_size' files (without sorting)
        self.file_names = all_file_names[:sample_size]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.save_dir, self.file_names[idx])
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)
        return graph


class GraphDataModule_without_dynamic(pl.LightningDataModule):
    def __init__(self, graph_family, graph_kwargs=None, samples_per_epoch=100, batch_size=32,
                 distributed_sampler=False, num_workers=1, val_split=0.25, use_full_dataset=False):
        super().__init__()
        if graph_kwargs is None:
            graph_kwargs = {}
        self.graph_family = graph_family
        self.graph_kwargs = graph_kwargs
        self.samples_per_epoch = samples_per_epoch
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.distributed_sampler = distributed_sampler
        self.val_split = val_split  # Specify the validation split ratio
        self.dataset = None
        self.train_dataset = None
        self.eval_dataset = None
        self.train_sampler = None
        self.eval_sampler = None
        self.use_full_dataset = use_full_dataset
        
    def make_dataset(self, samples_per_epoch):
        
        if self.graph_family == "binomial":
            ds = BinomialGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == "barabasi_albert":
            ds = BarabasiAlbertGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == "regular":
            ds = RegularGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == "geometric":
            ds = GeometricGraphDataset(samples_per_epoch=samples_per_epoch)
        elif self.graph_family == "all":
            ds = RandomGraphDataset(samples_per_epoch=samples_per_epoch)
        else:
            raise NotImplementedError(f"Unsupported graph_family: {self.graph_family}")
        return ds

    def prepare_data(self):
        # Instantiate the dataset only once during the preparation phase
        self.dataset = self.make_dataset(samples_per_epoch=self.samples_per_epoch)

    def setup(self, stage=None):
        if self.use_full_dataset:
            # Use the entire dataset for both training and validation
            self.train_dataset = self.dataset
            self.eval_dataset = self.dataset
            # print(f"Train dataset: {self.train_dataset}")
            # print(f"Eval dataset: {self.eval_dataset}")
        else:
            # Existing logic to split the dataset
            dataset_size = len(self.dataset)
            val_size = int(self.val_split * dataset_size)
            train_size = dataset_size - val_size
            self.train_dataset, self.eval_dataset = random_split(
                self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self, distributed_sampler=False):
        if self.distributed_sampler:
            train_sampler = DistributedSampler(
                dataset=self.train_dataset,
                shuffle=False
            )
        else:
            train_sampler = None
        return DenseGraphDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )

    def val_dataloader(self, distributed_sampler=False):
        if self.distributed_sampler:
            eval_sampler = DistributedSampler(
                dataset=self.eval_dataset,
                shuffle=False
            )
        else:
            eval_sampler = None
        return DenseGraphDataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=eval_sampler,
        )


def binomial_ego_graph(n, p):
    g = ego_graph(binomial_graph(n, p), 0)
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    return g