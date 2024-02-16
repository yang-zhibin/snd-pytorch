import sys, os
import logging
import time
import warnings

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .utils import load_processed_datasets

import pandas as pd

class GNNBase(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        
        """
        Initialise the Lightning Module that can scan over different Equivariant GNN training regimes
        """
        # Assign hyperparameters
        self.save_hyperparameters(hparams)

        if "graph_construction" not in self.hparams: self.hparams["graph_construction"] = None
        self.trainset, self.valset, self.testset = None, None, None

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def setup(self, stage="fit"):

        data_split = self.hparams["data_split"].copy()
        '''
        if stage == "fit":
            data_split[2] = 0 # No test set in training
        elif stage == "test":
            data_split[0], data_split[1] = 0, 0 # No train or val set in testing
        '''
        if (self.trainset is None) and (self.valset is None) and (self.testset is None):
            self.trainset, self.valset, self.testset = load_processed_datasets(self.hparams["input_dir"], 
                                                        data_split,
                                                        self.hparams["graph_construction"],
                                                        self.hparams["feature_set"][0]
                                                        )
        
        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")
            self.logger.experiment.define_metric("acc" , summary="max")
            for i in range(self.hparams['nb_classes']):
                self.logger.experiment.define_metric(f"acc{i}" , summary="max")
        except Exception:
            warnings.warn("Failed to define figures of merit, due to logger unavailable")
            
        print(time.ctime())
        
    def concat_feature_set(self, batch):
        """
        Useful in all models to use all available features of size == len(x)
        """
        
        all_features = []
        for feature in self.hparams["feature_set"]:
            if len(batch[feature]) == len(batch.x):
                all_features.append(batch[feature])
            else:
                all_features.append(batch[feature][batch.batch])
        return torch.stack(all_features).T

    def get_metrics(self, targets, output):
        
        prediction = torch.sigmoid(output)

        if self.hparams['nb_classes'] == 1:
            tp = (prediction.round() == targets).sum().item()
            acc = tp / len(targets)

            try:
                auc = roc_auc_score(targets.bool().cpu().detach(), prediction.cpu().detach())
            except Exception:
                auc = 0
            accs = {0:acc}
            return acc, auc, accs

        tp = (torch.argmax(prediction, dim=1) == targets).sum().item()
        acc = tp / len(targets)

        accs = {}
        for i in range(self.hparams["nb_classes"]):
            targets_i = targets[targets == i]
            if not len(targets_i) == 0:
                accs[i] =  (torch.argmax(prediction[targets == i], dim=1) == targets_i).sum().item() / len(targets_i)
            else: 
                accs[i] = 0.5
        
        auc_prediction = torch.softmax(output, dim=1)
        if len(torch.unique(targets)) != auc_prediction.shape[1]:
            auc_prediction = torch.softmax(torch.index_select(auc_prediction, 1, torch.unique(targets)), dim=1)
        try:
            auc = roc_auc_score(targets.cpu().detach(), auc_prediction.cpu().detach(), multi_class='ovr')
        except ValueError:
            print('WARNING in gnn_base.py get_metrics, possibly inconsistent shapes', targets.shape, auc_prediction.shape)
            auc = 0.
    
        return acc, auc, accs
    
    def apply_loss_function(self, output, batch):
        if self.hparams["nb_classes"] == 1:
            return F.binary_cross_entropy_with_logits(output, batch.y) #, pos_weight=torch.tensor(self.hparams["pos_weight"]))
        else:
            return F.cross_entropy(output, batch.y, weight=torch.tensor(self.hparams["class_weights"], device=self.device)) # the version above doesn't work with multiclass and integer class labels

    def training_step(self, batch, batch_idx):
                
        output = self(batch).squeeze(-1)

        loss = self.apply_loss_function(output, batch)
        
        acc, auc, accs = self.get_metrics(batch.y, output)

        log_dict = {"train_loss": loss, "train_auc": auc, 'acc': acc} | {f'train_acc{i}': accs[i] for i in range(self.hparams['nb_classes'])}
        
        self.log_dict(log_dict, on_step=False, on_epoch=True)

        event_id = batch.event_id

        ret = {
            "loss": loss,
            "outputs": output,
            "targets": batch.y,
            "acc": acc,
            "auc": auc,
            "event_id": batch.event_id
        }

        self.train_step_outputs.append(ret)

        return loss        

    def shared_val_step(self, batch, test=True):

        output = self(batch).squeeze(-1)

        loss = self.apply_loss_function(output, batch)

        acc, auc, accs = self.get_metrics(batch.y, output)
        
        opt = self.optimizers()
        
        current_lr = self.optimizers().param_groups[0]["lr"] if opt else 1.

        log_dict = {"train_loss": loss, "train_auc": auc, 'acc': acc, "current_lr": current_lr} | {f'vsl_acc{i}': accs[i] for i in range(self.hparams['nb_classes'])}
        
        self.log_dict(log_dict, on_step=False, on_epoch=True)
        
        event_id = batch.event_id

        ret = {
            "loss": loss,
            "outputs": output,
            "targets": batch.y,
            "acc": acc,
            "auc": auc,
            "event_id": batch.event_id
        }
        
        

        if test:
            ret = ret | {f'test_acc{i}': accs[i] for i in range(self.hparams['nb_classes'])}
            self.test_step_outputs.append(ret)
        else:
            ret = ret | {f'val_acc{i}': accs[i] for i in range(self.hparams['nb_classes'])}
            self.validation_step_outputs.append(ret)

        return ret

    def validation_step(self, batch, batch_idx):
        return self.shared_val_step(batch, False)

    def test_step(self, batch, batch_idx):
        return self.shared_val_step(batch)
        
    def shared_end_step(self, step_outputs):
        # Concatenate all predictions and targets
        preds = torch.cat([output["outputs"] for output in step_outputs])
        targets = torch.cat([output["targets"] for output in step_outputs])

        # Calculate the ROC curve
        acc, auc, accs = self.get_metrics(targets, preds)

        self.log_dict({"acc": acc, "auc": auc} | {f'acc{i}': accs[i] for i in range(self.hparams['nb_classes'])})

    def output_pred(self, step_outputs, split):
            preds = torch.cat([output["outputs"] for output in step_outputs])
            targets = torch.cat([output["targets"] for output in step_outputs])
            event_ids = torch.cat([output["event_id"] for output in step_outputs])

            preds = pd.DataFrame(preds.cpu().detach().numpy())
            targets = pd.DataFrame(targets.cpu().detach().numpy())
            event_ids = pd.DataFrame(event_ids.cpu().detach().numpy())

            out_data = pd.concat([event_ids, targets, preds], axis=1)

            out_path = os.path.join(self.hparams['pred_out_path'], "{}_pred_{}.csv".format(split, self.hparams['run_name']))
            print(split, "prediction output path:", out_path)
            out_data.to_csv(out_path, index=False, header=False) 

    def on_train_epoch_end(self):
        self.output_pred(self.train_step_outputs, "train")
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_end_step(self.validation_step_outputs)
        self.output_pred(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.shared_end_step(self.test_step_outputs)
        self.output_pred(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=self.hparams["train_batch"], shuffle=True, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=self.hparams["val_batch"], num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=self.hparams["test_batch"], num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0001,
                amsgrad=True,
            )
        ]
        if "scheduler" not in self.hparams or self.hparams["scheduler"] is None or self.hparams["scheduler"] == "StepLR":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.StepLR(
                        optimizer[0],
                        step_size=self.hparams["patience"],
                        gamma=self.hparams["factor"],
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            ]
        elif self.hparams["scheduler"] == "CosineWarmLR":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer[0],
                        T_0 = self.hparams["patience"], 
                        T_mult=2,
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
        return optimizer, scheduler
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        
