# Evolving Hard Maximum Cut Instances for Quantum Approximate Optimization Algorithms

The proposed method evolves with two steps. 

The first step is to train a graph autoencoder with the proposed model Permutation-Invariant Variational Autoencoder (PIGVAE) implemeted in the paper "Permutation-Invariant Variational Autoencoder for Graph-Level Representation Learning".

The second step is to search the hard maximum cut instances within the latent space of the trained graph autoencoder.

## Run

As we train the model directly based on the paper mentioned above, the built of the python environemnt requires the users to install the pytroch geometric library with the wheels locally instead of the installation command.

All the code related to the training of the model is under the directory of pigvae. 

For users who would like to train the model from scratch, we provide the list including required versions of libraries based on the origin PIGVAE paper:
- python 3.8
- pytorch=1.7
- pytorch geometric
- pytorch-lightning=1.3.1
- rdkit
- numpy
- networkx

The following commands support the reproducibility of generating hard maximum cut instances in the latent space of the graph autoencoder:

### Run the CMA-ES in the latent space of graph autoencoder:
(To run the code with CMA-ES searched in the latent space, the trained model of the graph autoencoder is also required. Please note the following checkpoints: epoch=442-step=20796.ckpt is used for the 20-node graph experiments and epoch=254-step=23959.ckpt is used for the 100-node graph experiments. You can find these checkpoints at https://zenodo.org/records/15209979.)
- cd Graph_Instance_Generation
- export PYTHONPATH="Graph_Instance_Generation:$PYTHONPATH"
- conda env create -f maxcut.yml
- source activate maxcut
- python evolve/main.py

### Compute graph features:
- python evolve/graph_features_computation.py

The Data folder contains all the generated instances by the CMA-ES search in the latent space of the graph autoencoder with its computed features. You can run them with the below automated machine leanring pipeline.

### Build machine learning model with TPOP:
- python evolve/TPOP_train.py