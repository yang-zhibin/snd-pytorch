import torch.nn as nn
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNBase
from ..utils import make_mlp
from .fancyconv import FancyConv

class GravNetExt(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct architecture
        # -------------------------

        if "spatial_channels" in hparams and hparams["spatial_channels"] is not None:
            self.spatial_channels = hparams["spatial_channels"]
        else:
            self.spatial_channels = len(self.hparams["feature_set"])

        # Encode input features to hidden features
        self.get_layer_structure()
        self.feature_encoder = make_mlp(
            self.spatial_channels,
            [self.layer_structure[0][0]] * hparams["nb_encoder_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        self.n_grav_heads = hparams['n_grav_heads']

        # Construct the GravNetExt convolution modules 
        self.grav_convs = nn.ModuleList()
        for _ in range(self.n_grav_heads):
            for input_size, output_size in self.layer_structure:
                self.grav_convs.append(FancyConv(hparams, input_size, output_size))

        # Decode hidden features to output features
        self.get_output_structure()
        self.output_network = make_mlp(
            self.aggregation_factor*self.n_grav_heads*self.output_size,
            [self.output_size] * hparams["nb_decoder_layer"] + [hparams["nb_classes"]],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"]
        )

    def output_step(self, all_x, batch):

        # all_x = torch.cat(all_x, dim=-1)

        graph_level_inputs = []

        for x in all_x:
            if self.hparams["aggregation"] == "mean_sum":
                graph_level_inputs.append(torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))
            elif self.hparams["aggregation"] == "sum":
                graph_level_inputs.append(global_add_pool(x, batch))
            elif self.hparams["aggregation"] == "mean":
                graph_level_inputs.append(global_mean_pool(x, batch))
        
        graph_level_inputs = torch.cat(graph_level_inputs, dim=-1)

        # Add dropout
        if "final_dropout" in self.hparams and self.hparams["final_dropout"] > 0.0:
            graph_level_inputs = F.dropout(graph_level_inputs, p=self.hparams["final_dropout"], training=self.training)

        return self.output_network(graph_level_inputs)

    def forward(self, batch, log_attention=False):

        x = self.concat_feature_set(batch)

        # Encode all features
        encoded_features = self.feature_encoder(x)

        # If concatenating, keep list of all output features
        all_hidden_features = []
        i = 0
        for _ in range(self.n_grav_heads):
            hidden_features = encoded_features
            for _ in range(len(self.layer_structure)):
                grav_conv = self.grav_convs[i]
                i += 1
                
                hidden_features, spatial_edges = checkpoint(grav_conv, hidden_features, batch.batch, self.current_epoch)

                self.log_dict({f"nbhood_sizes/nb_size_{i}": spatial_edges.shape[1] / hidden_features.shape[0]}, on_step=False, on_epoch=True)

                if self.hparams["concat_all_layers"]:
                    all_hidden_features.append(hidden_features)
            if not self.hparams["concat_all_layers"]:
                all_hidden_features.append(hidden_features)

        return self.output_step(all_hidden_features, batch.batch)

    def get_layer_structure(self):
        """
        Construct a list of [input_size, output_size] for each layer (assuming nodes are already encoded).
        For a flat structure, 3 layers, and a hidden size of 64, this would be:
        [[64, 64], [64, 64], [64, 64]]
        For a pyramid structure, 3 layers, and a hidden size of 64, this would be:
        [[64, 32], [32, 16], [16, 8]]
        For an antipyramid structure, 3 layers, and a hidden size of 64, this would be:
        [[64, 128], [128, 256], [256, 512]]
        """

        if "layer_shape" not in self.hparams or self.hparams["layer_shape"]=="flat":
            self.layer_structure = [[self.hparams["hidden"]] * 2] * self.hparams["n_graph_iters"]
        elif self.hparams["layer_shape"] == "pyramid":
            self.layer_structure = [ [max(self.hparams["hidden"] // 2**i , 8), max(self.hparams["hidden"] // 2**(i+1), 8)] for i in range(self.hparams["n_graph_iters"]) ]
        elif self.hparams["layer_shape"] == "antipyramid":
            self.layer_structure = [ [max(self.hparams["hidden"] // 2**i , 2) , max(self.hparams["hidden"] // 2**(i-1), 2)] for i in range(self.hparams["n_graph_iters"], 0, -1) ]

    def get_output_structure(self):
        """
        Calculate the size of the final encoded layer that needs to be decoded.
        If we don't concat all layers, then it is simply the size of the final layer.
        If we do concat all layers, then it is the sum of all layer output sizes (the second entry in each layer shape pair).
        """

        if "concat_all_layers" in self.hparams and self.hparams["concat_all_layers"]:
            self.output_size = sum(layer[1] for layer in self.layer_structure)
        else:
            self.output_size = self.layer_structure[-1][1]

        if self.hparams["aggregation"] == "mean_sum":
            self.aggregation_factor = 3
        elif self.hparams["aggregation"] in ["mean", "sum"]:
            self.aggregation_factor = 1