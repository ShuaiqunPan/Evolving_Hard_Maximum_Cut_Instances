import torch
import pytorch_lightning as pl
from pigvae.modules import GraphAE
from sklearn.metrics import roc_auc_score
from pigvae.synthetic_graphs.data import DenseGraphBatch
import numpy as np
import pickle


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams, critic):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.graph_ae = GraphAE(hparams)
        self.critic = critic(hparams)

    def forward(self, graph, training):
        graph_pred, perm, mu, logvar = self.graph_ae(graph, training, tau=1.0)
        return graph_pred, perm, mu, logvar
    
    def calculate_roc_auc(self, graph_true, graph_pred):
        with torch.no_grad():
            # Ensure the shapes are correct and consistent
            assert graph_true.edge_features.shape == graph_pred.edge_features.shape

            num_graphs = graph_true.edge_features.shape[0]

            roc_auc_scores = []
            for i in range(num_graphs):
                true_edges = graph_true.edge_features[i, :, :, 1].flatten()
                pred_edges = torch.sigmoid(graph_pred.edge_features[i, :, :, 1]).flatten()

                # Optional: Apply a mask if your graphs are sparse or irregular
                # mask = ... (define your mask based on your graph structure)
                # true_edges = true_edges[mask]
                # pred_edges = pred_edges[mask]

                true_edges_np = true_edges.cpu().numpy()
                pred_edges_np = pred_edges.cpu().numpy()

                # Calculate ROC-AUC per graph and store
                roc_auc = roc_auc_score(true_edges_np, pred_edges_np)
                roc_auc_scores.append(roc_auc)

            # Aggregate ROC-AUC scores
            avg_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
            return avg_roc_auc

    def training_step(self, graph, batch_idx):
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
        )
        loss = self.critic(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
        )
        roc_auc = self.calculate_roc_auc(graph, graph_pred)
        loss['train_roc_auc'] = roc_auc
        self.log_dict(loss)
        return loss

    def validation_step(self, graph, batch_idx):
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
        )
        metrics_soft = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val",
        )
        roc_auc_soft = self.calculate_roc_auc(graph, graph_pred)
        metrics_soft['val_roc_auc'] = roc_auc_soft
        
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=False,
        )
        metrics_hard = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val_hard",
        )
        roc_auc_hard = self.calculate_roc_auc(graph, graph_pred)
        metrics_hard['val_hard_roc_auc'] = roc_auc_hard
        
        metrics = {**metrics_soft, **metrics_hard}
        self.log_dict(metrics)
        self.log_dict(metrics_soft)

    def generate_embeddings(self, data_loader, save_path):
        
        REVERSE_LABEL_MAPPING = {0: "binominal", 1: "barabasi_albert", 2: "random_regular", 3: "watts_strogatz", 
                                 4: "newman_watts_strogatz", 5: "dual_barabasi_albert"}

        self.eval()
        embeddings = []
        labels = []
        parameters = []

        with torch.no_grad():
            for batch in data_loader:
                
                # Debugging: Print batch structure
                # print("Batch structure:", batch.__dict__)
                x = batch.node_features.double()  # Convert to double
                L = batch.edge_features.double()  # Convert to double
                mask = batch.mask.double() if batch.mask is not None else None

                graph = DenseGraphBatch(node_features=x, edge_features=L, mask=mask)
                graph_emb, _, _, _ = self.graph_ae.encode(graph)
                embedding = graph_emb.cpu().numpy().flatten()
                embeddings.append(embedding)

                if hasattr(batch, 'labels') and batch.labels:
                    for label in batch.labels:
                        # print(f"Raw label from batch: {label}")  # Debug print
                        if isinstance(label, torch.Tensor):
                            # Convert tensor to numeric label, then map to string
                            label_num = label.item()
                            # print(f"Label number (tensor to int): {label_num}")  # Debug print
                            string_label = REVERSE_LABEL_MAPPING.get(label_num, "unknown")
                            # print(f"Mapped string label: {string_label}")  # Debug print

                        else:
                            # Directly use the string label
                            string_label = label
                            # print(f"Direct string label: {string_label}") 

                        labels.append(string_label)
                        
                else:
                    # If no labels are found, assign a default label
                    labels.extend(["binominal"])  # Adjust the number as needed

                if hasattr(batch, 'params'):
                    parameters.extend(batch.params)

        # Convert lists to numpy arrays
        embeddings_array = np.array(embeddings)
        labels_array = np.array(labels) if labels else None
        parameters_array = np.array(parameters) if parameters else None

        # Create a dictionary to store embeddings, labels, and parameters
        data_dict = {
            "embeddings": embeddings_array,
            "labels": labels_array,
            "parameters": parameters_array
        }

        try:
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Embeddings, labels, and parameters saved to {save_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

        return data_dict


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph_ae.parameters(), lr=self.hparams["lr"], betas=(0.9, 0.98))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.999,
        )
        if "eval_freq" in self.hparams:
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 2 * (self.hparams["eval_freq"] + 1)
            }
        else:
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'epoch'
            }
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, 
                    second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # Calculate the number of steps for accumulation
        accumulation_steps = self.trainer.accumulate_grad_batches

        # Check if it's the right time to perform an optimizer step
        # Note: Use global_step to handle proper step calculation in case of multiple epochs
        if (self.trainer.global_step + 1) % accumulation_steps == 0:
            # Warm up lr
            if self.trainer.global_step < 10000:
                lr_scale = min(1., float(self.trainer.global_step + 1) / 10000.)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.hparams.lr

            # Perform the optimizer step
            optimizer.step(closure=optimizer_closure)

            # Zero out the gradients after the step
            optimizer.zero_grad()

