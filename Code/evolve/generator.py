import logging
import os
import pickle
import random
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn.functional as F
import umap
from pigvae.synthetic_graphs.data import DenseGraphBatch
from pigvae.synthetic_graphs.hyperparameter import add_arguments
from networkx.algorithms.similarity import graph_edit_distance
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pigvae.ddp import MyDDP
from pigvae.modules import GraphDecoder
from pigvae.synthetic_graphs.data import (GraphDataModule,
                                          GraphDataModule_without_dynamic)
from pigvae.synthetic_graphs.hyperparameter import add_arguments
from pigvae.synthetic_graphs.metrics import Critic
from pigvae.trainer import PLGraphAE

from scipy.sparse.csgraph import floyd_warshall

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

torch.set_default_dtype(torch.double)
logging.getLogger("lightning").setLevel(logging.WARNING)


class GraphGenerator:
    def __init__(self, num_nodes):
        parser = ArgumentParser()
        parser = add_arguments(parser)
        args = parser.parse_args(args=[])
        
        self.critic = Critic
        self.model = PLGraphAE(args.__dict__, self.critic)
        
        # Set the number of nodes
        self.num_nodes = num_nodes

        # Map from number of nodes to checkpoint path and file path
        self.path_mapping = {
            20: ("epoch=442-step=20796.ckpt", "embeddings_20nodes.pkl"),
            100: ("epoch=254-step=23959.ckpt", "embeddings_100nodes.pkl")
        }
        
        if self.num_nodes in self.path_mapping:
            self.checkpoint_path, self.file_path = self.path_mapping[self.num_nodes]
        else:
            raise ValueError(f"No data available for {self.num_nodes} nodes.")
        
        # Load model from checkpoint
        self.model = PLGraphAE.load_from_checkpoint(self.checkpoint_path, critic=self.critic)
        self.model.eval()

        # Load embeddings and set latent space attributes
        with open(self.file_path, 'rb') as file:
            data = pickle.load(file)
        embeddings = data['embeddings']
        self.latent_dim = embeddings.shape[1]
        self.latent_range = (embeddings.min(axis=0), embeddings.max(axis=0))

        # Set model hyperparameters (hidden from the user)
        self.hparams = args.__dict__
        self.decoder = GraphDecoder(self.hparams)

        # Threshold for deciding edge existence
        self.threshold = 0.5

    def generate_and_plot_graph(self, latent_point: np.ndarray) -> np.ndarray:
        """
        Generates and optionally plots a graph from a given point in the latent space.

        :param latent_point: A numpy array representing a point in the latent space.
        :param plot: Boolean indicating whether to plot the graph.
        :return: A tuple containing the adjacency matrix and the NetworkX graph.
        """
        # Validate latent point dimension...
        if len(latent_point) != self.latent_dim:
            raise ValueError(f"Expected latent point of dimension {self.latent_dim}, got {len(latent_point)}")

        latent_point_tensor = torch.tensor(latent_point, dtype=torch.float).unsqueeze(0).double()

        with torch.no_grad():
            mask = torch.ones(1, self.num_nodes).bool()
            graph_pred = self.model.graph_ae.decode(latent_point_tensor, None, mask)
            edge_probabilities = torch.sigmoid(graph_pred.edge_features[0]).double()

        adjacency_matrix = (edge_probabilities[:, :, 1] > self.threshold).float()
        adjacency_matrix.fill_diagonal_(0)

        return adjacency_matrix
        # return adjacency_matrix.cpu().numpy()
    
    def adjacency_matrix_to_latent_point(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Encodes a graph represented by its adjacency matrix into a latent point using the GraphAE model.
        
        :param adjacency_matrix: A numpy array representing the adjacency matrix of a graph.
        :return: A numpy array representing the encoded latent point of the graph.
        """
        # Convert adjacency_matrix to Float tensor
        adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float)

        num_nodes = adjacency_matrix_tensor.shape[0]
        
        # Initialize node features as ones for each node, ensuring they are Float
        node_features = torch.ones((1, num_nodes, 1), dtype=torch.float)
        
        # Calculate shortest path distances using Floyd-Warshall, then cap and one-hot encode
        distances = floyd_warshall(adjacency_matrix, unweighted=False, directed=False)
        distances[distances > 5] = 5  # Cap distances at 5
        distances = np.clip(distances, 0, 5).astype(int)
        edge_features = torch.zeros((1, num_nodes, num_nodes, 6), dtype=torch.float)  # Ensure Float type here
        edge_features[0, np.arange(num_nodes), np.arange(num_nodes), 0] = 1  # Self-loops
        edge_features.scatter_(3, torch.tensor(distances).long().unsqueeze(0).unsqueeze(-1), 1)
        
        # Assume a mask of ones, indicating all nodes are valid, and ensure Float type
        mask = torch.ones((1, num_nodes), dtype=torch.bool)
        
        # Encode the graph to get the latent representation, ensuring all inputs are Float
        self.model.graph_ae = self.model.graph_ae.float()
        
        print("Node features dtype:", node_features.dtype)
        print("Edge features dtype:", edge_features.dtype)
        print("Model parameters dtype example:", next(self.model.graph_ae.parameters()).dtype)

        graph_emb, _, mu, _ = self.model.graph_ae.encode(DenseGraphBatch(
            node_features=node_features.float(),  # Convert to Float
            edge_features=edge_features.float(),  # Convert to Float
            mask=mask,
            properties=None,  # Assuming properties are not necessary for encoding
            labels=None,  # Assuming labels are not necessary for encoding
            params=None  # Assuming params are not necessary for encoding
        ))
        
        return graph_emb.squeeze().detach().numpy()  # Detach tensor and convert to numpy array
    
    @property
    def dimension(self):
        return self.latent_dim

    @property
    def range(self):
        return self.latent_range
