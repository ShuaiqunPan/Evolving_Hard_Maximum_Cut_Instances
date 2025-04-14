import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pigvae.trainer import PLGraphAE
from pigvae.synthetic_graphs.hyperparameter import add_arguments
from pigvae.synthetic_graphs.data import GraphDataModule, GraphDataModule_without_dynamic
from pigvae.ddp import MyDDP
from pigvae.synthetic_graphs.metrics import Critic

import torch
import umap
import random
from sklearn.preprocessing import StandardScaler
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import DenseGraphBatch
import networkx as nx
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from networkx.algorithms.similarity import graph_edit_distance
import pandas as pd

from pigvae.modules import Permuter

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

def is_main_process():
    # Check if SLURM environment variables are available
    if 'SLURM_PROCID' in os.environ:
        return os.environ['SLURM_PROCID'] == '0'
    # If not using SLURM or unsure, fall back to torch.distributed (if available)
    elif dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # Default to True if not in a distributed environment
    else:
        return True

def main(hparams):
    if is_main_process():
        # Perform operations that should only be done by the main process
        dir_path = hparams.save_dir + "/run{}/".format(hparams.id)
        os.makedirs(dir_path, exist_ok=True)
        print("Starting Run {}".format(hparams.id))
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + "/run{}/".format(hparams.id),
        save_last=True,
        save_top_k=1,
        monitor="val_loss"
    )
    
    early_stop_callback = EarlyStopping(
    monitor="val_loss",  # Assuming this is the metric you want to monitor
    patience=1000,
    verbose=True,
    mode="min"
    )
    
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(hparams.save_dir + "/run{}/".format(hparams.id))
    critic = Critic
    model = PLGraphAE(hparams.__dict__, critic)
    graph_kwargs = {
        "n_min": hparams.n_min,
        "n_max": hparams.n_max,
        "m_min": hparams.m_min,
        "m_max": hparams.m_max,
        "p_min": hparams.p_min,
        "p_max": hparams.p_max
    }
    datamodule = GraphDataModule(
        graph_family=hparams.graph_family,
        graph_kwargs=graph_kwargs,
        batch_size=hparams.batch_size,
        num_workers=18,
        samples_per_epoch=60000
    )

    my_ddp_plugin = MyDDP()
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        progress_bar_refresh_rate=5 if hparams.progress_bar else 0,
        logger=tb_logger,
        checkpoint_callback=True,
        val_check_interval=hparams.eval_freq if not hparams.test else 100,
        accelerator="ddp",
        plugins=[my_ddp_plugin],
        gradient_clip_val=0.1,
        callbacks=[lr_logger, checkpoint_callback, early_stop_callback],
        terminate_on_nan=True,
        replace_sampler_ddp=False,
        precision=hparams.precision,
        max_epochs=hparams.num_epochs,
        reload_dataloaders_every_epoch=True,
        resume_from_checkpoint=hparams.resume_ckpt if hparams.resume_ckpt != "" else None,
        accumulate_grad_batches=hparams.accumulate_grad_batches
    )
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
